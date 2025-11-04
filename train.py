import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, TensorDataset, RandomSampler
from tqdm import tqdm
from rotor_network import NeuralNetwork, RandomPermutation, RotorLayer
from baselines import LowRankLinear, BHLinear

from rotor_layer import Rotor
import os
# import wandb
import copy

# ========================
# 1. Parse Command Line Args
# ========================
parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, required=False)
parser.add_argument('--chunk_size', type=int, required=False)
parser.add_argument('--hidden_layers', type=int, required=False)
parser.add_argument('--breadth_hidden', type=int, required=False)
parser.add_argument('--lr', type=float, required=True)
parser.add_argument('--weight_decay', type=float, default=0, help="Weight decay (L2 regularization)")
parser.add_argument('--use_perm', action='store_true', help='Insert RandomPermutation layers after each Rotor')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--cos_annealing', action='store_true', help="Use cosine annealing scheduler if set.")
parser.add_argument('--proj_parallel', action='store_true')
parser.add_argument('--single_rotor', action='store_true')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--nonlinear', type=str, choices=["relu", "prelu", "gelu", "silu", "leakyrelu", "elu"], default="relu")
parser.add_argument('--normalize', action='store_true')
parser.add_argument('--residual', action='store_true')
parser.add_argument('--adjust_chunk_size', action='store_true', help='If set, hidden rotor layers use chunk_size = hidden_dim.')
parser.add_argument('--task', type=str, choices=["key", "query", "value"], required=True, help="Which projection to train: key, query, or value.")
parser.add_argument('--layer', type=str, required=True, help="Layer to train on (e.g., layer15, layer14, etc.)")
parser.add_argument('--x_train_path', type=str)
parser.add_argument('--y_train_path', type=str)
parser.add_argument('--x_test_path', type=str)
parser.add_argument('--y_test_path', type=str)
parser.add_argument('--output_dir', type=str, required=True, help='Output path')

parser.add_argument('--rank', type=int)
parser.add_argument('--replacement_type', type=str, choices=["rotor", "lowrank_linear", "bh_linear"])

args = parser.parse_args()

# wandb.init(
#     project="rotor-Llama",
#     config=vars(args)
# )

# ========================
# 2. Dataset Loading
# ========================
default_device ="cuda" if torch.cuda.is_available() else "cpu"

if not args.x_train_path or not args.y_train_path:
    raise ValueError("Both --x_train_path and --y_train_path must be specified.")
if not args.x_test_path or not args.y_test_path:
    raise ValueError("Both --x_test_path and --y_test_path must be specified.")

if not os.path.exists(args.x_train_path):
    raise FileNotFoundError(f"Missing x train file: {args.x_train_path}")
if not os.path.exists(args.y_train_path):
    raise FileNotFoundError(f"Missing y train file: {args.y_train_path}")
if not os.path.exists(args.x_test_path):
    raise FileNotFoundError(f"Missing x test file: {args.x_test_path}")
if not os.path.exists(args.y_test_path):
    raise FileNotFoundError(f"Missing y test file: {args.y_test_path}")

# Check if model already exists
os.makedirs(args.output_dir, exist_ok=True)

if args.replacement_type == "rotor":
    model_save_path = os.path.join(
        args.output_dir,
        f"model_chunk={args.chunk_size}_layers={args.hidden_layers}_breadth={args.breadth_hidden}_lr={args.lr}_weightdecay={args.weight_decay if args.weight_decay != 0 else 0}_batchsize={args.batch_size}_nonlinear={args.nonlinear}_normalize={args.normalize}_residual={args.residual}_useperm={args.use_perm}_epochs={args.epochs}_cosanneal={args.cos_annealing}_projparallel={args.proj_parallel}_singlerotor={args.single_rotor}.pth"
    )
elif args.replacement_type == "lowrank_linear":
    model_save_path = os.path.join(
        args.output_dir,
        f"lowrank_linear_rank={args.rank}_lr={args.lr}_batchsize={args.batch_size}_epochs={args.epochs}_cosanneal={args.cos_annealing}.pth"
    )
elif args.replacement_type == "bh_linear":
    model_save_path = os.path.join(
        args.output_dir,
        f"bh_linear_lr={args.lr}_batchsize={args.batch_size}_epochs={args.epochs}_cosanneal={args.cos_annealing}.pth"
    )
else:
    raise TypeError("No replacement type given")

if os.path.exists(model_save_path):
    print(f"[SKIP] Model already exists")
    exit(0)

x_train = torch.load(args.x_train_path, weights_only=True)
y_train = torch.load(args.y_train_path, weights_only=True)
train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

x_test = torch.load(args.x_test_path, weights_only=True)
y_test = torch.load(args.y_test_path, weights_only=True)
test_dataset = TensorDataset(x_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

input_dim = x_train.shape[1]
output_dim = y_train.shape[1]

# ========================
# 3. Training Functions
# ========================
def train_loop(dataloader, test_loader, model, loss_fn, optimizer, scheduler, epoch, device=default_device):
    model.train()
    counter = 0
    for X, y in tqdm(dataloader, desc=f"Training Epoch {epoch}", leave=False):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
        model._update_rotors()
        if counter % 250 == 0:
            avg_test_loss = test_loop(test_loader, model, loss_fn)
            tqdm.write(f"Batch {counter}/{len(dataloader)}: test loss = {avg_test_loss:.6f}")
        counter += 1

def test_loop(dataloader, model, loss_fn, device=default_device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            total_loss += loss_fn(pred, y).item()
    return total_loss / len(dataloader)

# ========================
# 5. Train and Save
# ========================

if args.replacement_type == "rotor":
    model = NeuralNetwork(
        input_dim=input_dim,
        output_dim=output_dim,
        chunk_size=args.chunk_size,
        extra_rotor_layers=args.hidden_layers,
        hidden_breadth=args.breadth_hidden,
        proj_parallel=args.proj_parallel,
        single_rotor=args.single_rotor,
        use_perm=args.use_perm,
        nonlinear=args.nonlinear,
        normalize=args.normalize,
        residual=args.residual,
        device=default_device,
        dtype=torch.float32
    ).to(default_device)
elif args.replacement_type == "lowrank_linear":
    model = LowRankLinear(in_features=input_dim, out_features=output_dim, rank=args.rank).to(default_device)
    # print(model)
elif args.replacement_type == "bh_linear":
    model = BHLinear(
        in_dim=input_dim,
        out_dim=output_dim,
    ).to(default_device)
else:
    raise TypeError("No replacement type given")


# optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs) if args.cos_annealing else None
loss_fn = nn.MSELoss()

best_loss = float('inf')

for epoch in range(1, args.epochs + 1):
    train_loop(train_loader, test_loader, model, loss_fn, optimizer, scheduler, epoch)
    avg_test_loss = test_loop(test_loader, model, loss_fn)

    if avg_test_loss < best_loss:
        best_loss = avg_test_loss
        torch.save(model.state_dict(), model_save_path)

    print(f"Epoch {epoch}: test loss = {avg_test_loss:.6f}")

# Final test loss
model.load_state_dict(torch.load(model_save_path, map_location=default_device, weights_only=False))
model.eval()
model._update_rotors()
final_loss = test_loop(test_loader, model, loss_fn)
print(f"Final test loss: {final_loss:.6f}")

with open(os.path.join(args.output_dir, "result.txt"), "a") as f:
    if args.replacement_type == "rotor":
        f.write(
            f"chunk_size={args.chunk_size}, "
            f"hidden_layers={args.hidden_layers}, "
            f"breadth={args.breadth_hidden}, "
            f"lr={args.lr}, "
            f"weightdecay={args.weight_decay}, "
            f"batch_size={args.batch_size}, "
            f"nonlinear={args.nonlinear}, "
            f"normalize={args.normalize}, "
            f"residual={args.residual}, "
            f"use_perm={args.use_perm}, "
            f"epochs={args.epochs}, "
            f"cos_annealing={args.cos_annealing}, "
            f"final_loss={final_loss:.6f}\n"
        )
    elif args.replacement_type == "lowrank_linear":
        f.write(f"rank={args.rank}, "
                f"lr={args.lr}, "
                f"batch_size={args.batch_size}, "
                f"epochs={args.epochs}, "
                f"cos_annealing={args.cos_annealing}, "
            )
    elif args.replacement_type == "bh_linear":
        f.write(f"lr={args.lr}, "
                f"batch_size={args.batch_size}, "
                f"epochs={args.epochs}, "
                f"cos_annealing={args.cos_annealing}, "
            )
    else:
        raise TypeError("No replacement type given")

print(f"Saved model to {model_save_path}")
# wandb.finish()