import torch.nn as nn
import torch
from utils import parse_model_ckpt_name

class LowRankLinear(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.B = nn.Linear(in_features, rank, bias=False)
        self.A = nn.Linear(rank, out_features, bias=False)

    def forward(self, x):
        orig_shape = x.shape
        x = x.view(-1, orig_shape[-1])
        x = self.A(self.B(x))
        return x.view(*orig_shape[:-1], -1)

    def _update_rotors(self):
        pass

def load_lowranklinear_layers(replaced_layers, input_dim, value_output_dim, model_dir, model_ckpt, device, dtype=torch.float32):
    rotor_nets = {}
    with torch.no_grad():
        for layer in replaced_layers:
            layer_dict = {}
            for out_dim, proj in zip([input_dim, value_output_dim, value_output_dim],["query", "key", "value"]):
                # Load model
                model_path = f"{model_dir}/{layer}/{proj}/{model_ckpt}"
                hparams = parse_model_ckpt_name(model_ckpt, replacement_type="lowrank_linear")
                lowranklinear_net = LowRankLinear(in_features=input_dim, out_features=out_dim, rank=hparams["rank"]).to(device)
                lowranklinear_net.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
                lowranklinear_net.eval()
                lowranklinear_net.to(dtype=dtype)

                with torch.no_grad():
                    layer_dict[proj] = lowranklinear_net
            rotor_nets[layer] = layer_dict
    return rotor_nets

from fast_hadamard_transform import hadamard_transform
import torch
import torch.nn as nn
import math

class BHLinear(nn.Module):
    def __init__(self, in_dim, out_dim, depth=0, block_size=64, rectangle_num=128):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.depth = depth
        self.block_size = block_size
        self.rectangle_num = rectangle_num

        # pad input to next power of 2
        self.padded_dim = 2 ** math.ceil(math.log2(in_dim))
        self.pad = self.padded_dim - in_dim
        # print(self.padded_dim)
        # print(self.pad)

        # checks
        assert self.padded_dim % block_size == 0, "padded_dim must be divisible by block_size"
        assert self.padded_dim % rectangle_num == 0, "padded_dim must be divisible by rectangle_num"
        assert out_dim % rectangle_num == 0, "out_dim must be divisible by rectangle_num"

        self.num_block_groups = self.padded_dim // block_size  # for internal B's
        self.block_width = self.padded_dim // rectangle_num
        self.block_height = out_dim // rectangle_num
        # print(self.num_block_groups, self.block_width, self.block_height)

        # Internal block-diagonal matrices: (num_blocks, B, B)
        self.inner_B = nn.ParameterList([
            nn.Parameter(torch.randn(self.num_block_groups, block_size, block_size))
            for _ in range(depth - 1)
        ])

        # Final rectangular block-diagonal matrix: (rectangle_num, W, H)
        self.final_B = nn.Parameter(0.1 * torch.randn(rectangle_num, self.block_width, self.block_height))
        # print([b.shape for b in self.inner_B])
        # print(self.final_B.shape)

    def forward(self, x):
        orig_shape = x.shape
        x = x.view(-1, self.in_dim)

        if self.pad > 0:
            pad = torch.zeros(x.size(0), self.pad, device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=-1)

        B, D = x.shape
        assert D == self.padded_dim

        # Internal transforms: (block_size × block_size) + Hadamard
        for B_i in self.inner_B:
            x = x.view(B, self.num_block_groups, self.block_size)
            x = torch.einsum("bni,nio->bno", x, B_i)
            x = x.reshape(B, D)
            x = hadamard_transform(x) / math.sqrt(x.shape[-1])

        # Final rectangular blocks
        x = x.view(B, self.rectangle_num, self.block_width)
        x = torch.einsum("bni,nio->bno", x, self.final_B)
        return x.reshape(*orig_shape[:-1], self.out_dim)

    def _update_rotors(self):
        pass

def load_bh_layers(replaced_layers, input_dim, value_output_dim, model_dir, model_ckpt, device, dtype=torch.float32):
    bh_nets = {}
    with torch.no_grad():
        for layer in replaced_layers:
            layer_dict = {}
            for out_dim, proj in zip([input_dim, value_output_dim, value_output_dim],["query", "key", "value"]):
                # Load model
                model_path = f"{model_dir}/{layer}/{proj}/{model_ckpt}"
                bh_net = BHLinear(in_dim=input_dim, out_dim=out_dim).to(device)
                bh_net.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
                bh_net.eval()
                bh_net.to(dtype=dtype)

                with torch.no_grad():
                    layer_dict[proj] = bh_net
            bh_nets[layer] = layer_dict
    return bh_nets