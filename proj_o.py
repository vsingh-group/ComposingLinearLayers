import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from tqdm import tqdm
import os

SEED = 0
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class ProjO(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return self.linear(x)

def train_proj_o(x_train, y_train, x_test, y_test, dim, device, output_path, ckpt, epochs=5):
    if os.path.exists(os.path.join(output_path, ckpt)):
        print(f"[SKIP] Proj_O model already exists")
        return

    dataset = TensorDataset(x_train, y_train)
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(dataset, batch_size=256, shuffle=True, generator=generator)

    # Define model
    proj_o_model = ProjO(dim=dim).to(device)
    optimizer = torch.optim.AdamW(proj_o_model.parameters(), lr=1e-4, weight_decay=1e-2)
    loss_fn = nn.MSELoss()

    # Train
    proj_o_model.train()
    for _ in tqdm(range(epochs)):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = proj_o_model(xb)
            loss = loss_fn(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Test
    proj_o_model.eval()
    with torch.no_grad():
        x_test, y_test = x_test.to(device), y_test.to(device)
        pred = proj_o_model(x_test)
        test_loss = loss_fn(pred, y_test).item()

    print(f"Test loss for Proj_o: {test_loss:.4f}")
    os.makedirs(output_path, exist_ok=True)
    torch.save(proj_o_model.state_dict(), f'{output_path}/{ckpt}')
    return proj_o_model

def load_proj_o_model(dim, output_path, ckpt, device):
    model = ProjO(dim=dim).to(device)
    with torch.no_grad():
        model.load_state_dict(torch.load(f'{output_path}/{ckpt}', map_location=device))
        model.eval()
    return model