import torch
import torch.nn as nn

from rotor_layer import Rotor

from utils import parse_model_ckpt_name

device = "cuda" if torch.cuda.is_available() else "cpu"

class RandomPermutation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        perm = torch.randperm(dim)
        self.register_buffer("perm", perm)

    def forward(self, x):
        return x[:, self.perm.to(x.device)]

class ParallelBlock(nn.Module):
    def __init__(self, *modules, device, dtype):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.branches = nn.ModuleList(modules).to(device=self.device, dtype=self.dtype)
        self.alphas = nn.Parameter(torch.ones(len(modules), device=self.device, dtype=self.dtype))

    def forward(self, x):
        outputs = [branch(x) for branch in self.branches]
        out_stack = torch.stack(outputs, dim=0)
        return torch.einsum('n,nbd->bd', self.alphas, out_stack)

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, chunk_size, extra_rotor_layers, hidden_breadth, proj_parallel, single_rotor, use_perm=False, nonlinear="relu", normalize=False, residual=False, device='cuda', dtype=torch.float32):
        super().__init__()
        self.flatten = nn.Flatten()
        self.residual = residual
        self.layers = nn.ModuleList()
        hidden_dim = chunk_size

        # Choose nonlinearity instance (with parentheses)
        if nonlinear == "relu":
            nonlinearity_fn = nn.ReLU()
        elif nonlinear == "prelu":
            nonlinearity_fn = nn.PReLU()
        elif nonlinear == "gelu":
            nonlinearity_fn = nn.GELU()
        elif nonlinear == "silu":
            nonlinearity_fn = nn.SiLU()
        elif nonlinear == "leakyrelu":
            nonlinearity_fn = nn.LeakyReLU(0.01)
        elif nonlinear == "elu":
            nonlinearity_fn = nn.ELU()
        else:
            raise ValueError("Unsupported nonlinearity")

        # Input layer
        block = []
        block.append(Rotor(input_dim, hidden_dim, alpha_param=True, bias_param=False, single_rotor=single_rotor, device=device, dtype=dtype))
        if use_perm:
            block.append(RandomPermutation(hidden_dim))
        if normalize:
            block.append(nn.LayerNorm(hidden_dim, elementwise_affine=False))
        block.append(nonlinearity_fn)
        self.layers.append(nn.Sequential(*block))
        # blocks = []
        # for _ in range(hidden_breadth):
        #     block = []
        #     block.append(Rotor(input_dim, hidden_dim, alpha_param=True, bias_param=False, device=device, dtype=dtype))
        #     if use_perm:
        #         block.append(RandomPermutation(hidden_dim))
        #     if normalize:
        #         block.append(nn.LayerNorm(hidden_dim, elementwise_affine=False))
        #     block.append(nonlinearity_fn)
        #     blocks.append(nn.Sequential(*block))
        # self.layers.append(ParallelBlock(*blocks, device=device, dtype=dtype))

        # Hidden layers
        for _ in range(extra_rotor_layers):
            blocks = []
            for _ in range(hidden_breadth):
                block = []
                # _chunk_size = hidden_dim if adjust_chunk_size else chunk_size
                block.append(Rotor(hidden_dim, hidden_dim, alpha_param=True, bias_param=False, single_rotor=single_rotor, device=device, dtype=dtype))
                if use_perm:
                    block.append(RandomPermutation(hidden_dim))
                if normalize:
                    block.append(nn.LayerNorm(hidden_dim, elementwise_affine=False))
                block.append(nonlinearity_fn)
                blocks.append(nn.Sequential(*block))
            self.layers.append(ParallelBlock(*blocks, device=device, dtype=dtype))

        # Output layer
        if proj_parallel:
            blocks = []
            for _ in range(hidden_breadth):
                block = []
                block.append(Rotor(hidden_dim, output_dim, alpha_param=True, bias_param=False, single_rotor=single_rotor, device=device))
                blocks.append(nn.Sequential(*block))
            self.layers.append(ParallelBlock(*blocks, device=device, dtype=dtype))
        else:
            final_block = []
            final_block.append(Rotor(hidden_dim, output_dim, alpha_param=True, bias_param=False, single_rotor=single_rotor, device=device))
            self.layers.append(nn.Sequential(*final_block))

        # final_block = []
        # final_block.append(Rotor(hidden_dim, 1024, alpha_param=True, bias_param=False, device=device))
        # if use_perm:
        #     block.append(RandomPermutation(hidden_dim))
        # if normalize:
        #     block.append(nn.LayerNorm(hidden_dim, elementwise_affine=False))
        # block.append(nonlinearity_fn)
        # self.layers.append(nn.Sequential(*final_block))

        # final_block = []
        # final_block.append(Rotor(1024, 512, alpha_param=True, bias_param=False, device=device))
        # if use_perm:
        #     block.append(RandomPermutation(hidden_dim))
        # if normalize:
        #     block.append(nn.LayerNorm(hidden_dim, elementwise_affine=False))
        # block.append(nonlinearity_fn)
        # self.layers.append(nn.Sequential(*final_block))

        # final_block = []
        # final_block.append(Rotor(512, 256, alpha_param=True, bias_param=False, device=device))
        # self.layers.append(nn.Sequential(*final_block))


    def forward(self, x):
        x = self.flatten(x)
        for layer in self.layers:
            if isinstance(layer, nn.Sequential) and self.residual:
                skip = x
                x = layer(x)
                if x.shape == skip.shape:
                    x = x + skip  # Add skip if shape matches
            else:
                x = layer(x)
        return x

    def _update_rotors(self):
        for module in self.modules():
            if isinstance(module, Rotor):
                module._update_rotors()

    def to(self, device=None, dtype=None):
        super().to(device=device, dtype=dtype)

        for module in self.modules():
            if isinstance(module, Rotor):
                module.to(device=device, dtype=dtype)

        return self

#### Load rotors ####

class RotorLayer(nn.Module):
    def __init__(self, rotor_net, subseq_size=10):
        super().__init__()
        self.rotor_net = rotor_net
        self.subseq_size = subseq_size

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        x = x.view(-1, hidden_dim)
        outputs = [self.rotor_net(x[i:i + self.subseq_size]) for i in range(0, x.shape[0], self.subseq_size)]
        return torch.cat(outputs, dim=0).view(batch_size, seq_len, -1)

def load_rotor_layers(replaced_layers, input_dim, value_output_dim, model_dir, model_ckpt, device, dtype=torch.float32):
    rotor_nets = {}
    with torch.no_grad():
        for layer in replaced_layers:
            layer_dict = {}
            for out_dim, proj in zip([input_dim, value_output_dim, value_output_dim],["query", "key", "value"]):
                # Load model
                model_path = f"{model_dir}/{layer}/{proj}/{model_ckpt}"
                hparams = parse_model_ckpt_name(model_ckpt, replacement_type="rotor")
                rotor_net = NeuralNetwork(
                    input_dim=input_dim,
                    output_dim=out_dim,
                    chunk_size=hparams["chunk_size"],
                    extra_rotor_layers=hparams["hidden_layers"],
                    hidden_breadth=hparams["breadth_hidden"],
                    proj_parallel=hparams["proj_parallel"],
                    single_rotor=hparams["single_rotor"],
                    use_perm=hparams["use_perm"],
                    nonlinear=hparams["nonlinear"],
                    normalize=hparams["normalize"],
                    residual=hparams["residual"],
                    device=device
                ).to(device)
                rotor_net.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
                rotor_net.eval()
                rotor_net._update_rotors()
                rotor_net.to(dtype=dtype)

                with torch.no_grad():
                    layer_dict[proj] = RotorLayer(rotor_net, subseq_size=50).to(device)
            rotor_nets[layer] = layer_dict
    return rotor_nets