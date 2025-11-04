import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_ga import GeometricAlgebra

from typing import Tuple

import math
import os

# import numpy as np
# np.set_printoptions(precision=5, suppress=True, linewidth=200)

class Rotor(nn.Module):
    shared_algebras = dict()

    def __init__(self, in_dim, out_dim, in_chunks = None, out_chunks = None, chunk_size = None, single_rotor = True, alpha_param = False, bias_param = False, device = 'cuda', dtype = torch.float32):
        super().__init__()

        in_chunks, out_chunks, chunk_size = self._validate_algebra_input(in_dim, out_dim, in_chunks, out_chunks, chunk_size, alpha_param)

        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.in_chunks = int(in_chunks)
        self.out_chunks = int(out_chunks)
        self.chunk_size = int(chunk_size)
        self.single_rotor = single_rotor
        self.device = device
        self.dtype = dtype

        self.counter = 0

        # Indices of the chunks
        self.in_chunk_indices = torch.stack([
            (torch.arange(self.chunk_size, device=self.device) + start) % self.in_dim
            for start in [(i * self.in_dim) // self.in_chunks for i in range(self.in_chunks)]
        ], dim=0)
        self.out_chunk_indices = torch.stack([
            (torch.arange(self.chunk_size, device=self.device) + start) % self.out_dim
            for start in [(i * self.out_dim) // self.out_chunks for i in range(self.out_chunks)]
        ], dim=0)

        # For averaging purposes
        count = torch.zeros(self.out_dim, device=self.device, dtype=self.dtype)
        count.scatter_add_(0, self.out_chunk_indices.reshape(-1), torch.full_like(self.out_chunk_indices.reshape(-1), fill_value=self.in_chunks, device=self.device, dtype=self.dtype).reshape(-1))
        self.register_buffer("count", count)

        # Creates the algebra based on chunk size
        num_basis_vecs = int(math.log2(chunk_size))
        if Rotor.shared_algebras.get(num_basis_vecs) is None:
            current_dir = os.path.dirname(__file__)
            algebra_path = os.path.join(current_dir, "algebras", f"algebra_{num_basis_vecs}.pt")
            if os.path.exists(algebra_path):
                alg = torch.load(algebra_path, map_location=self.device, weights_only=False)
            else:
                alg = GeometricAlgebra([1] * num_basis_vecs, full_cayley_table=False, device = self.device, dtype = self.dtype)
                os.makedirs(os.path.dirname(algebra_path), exist_ok=True)
                torch.save(alg, algebra_path)
            Rotor.shared_algebras[num_basis_vecs] = alg.to(dtype=self.dtype)
        self.algebra = Rotor.shared_algebras[num_basis_vecs]

        self.k = math.floor(self.algebra._num_bases // 2)
        self.num_basis_biv = math.comb(self.algebra._num_bases , 2)

        self.bivectors_left = nn.Parameter(torch.randn(self.in_chunks * self.out_chunks, self.num_basis_biv, device=self.device, dtype=self.dtype))
        self.v_init_left = torch.randn(self.k - 1, self.bivectors_left.shape[0], self.algebra._num_bases, device=self.device, dtype=self.dtype)
        # self.bivectors = nn.Parameter(torch.tensor([[1.5, -.5, 2, 3.5, -3, 1]], device=self.device, dtype=self.dtype))

        if not single_rotor:
            self.bivectors_right = nn.Parameter(torch.randn(self.in_chunks * self.out_chunks, self.num_basis_biv, device=self.device, dtype=self.dtype))
            self.v_init_right = torch.randn(self.k - 1, self.bivectors_left.shape[0], self.algebra._num_bases, device=self.device, dtype=self.dtype)

        if alpha_param:
            self.alpha = nn.Parameter(torch.ones(self.in_chunks * self.out_chunks, device=self.device, dtype=self.dtype))
        else:
            self.register_parameter("alpha", None)

        if bias_param:
            self.bias = nn.Parameter(torch.randn(self.out_dim, device=self.device, dtype=self.dtype))
        else:
            self.register_parameter("bias", None)

        self._update_rotors()

    def _validate_algebra_input(self, in_dim, out_dim, in_chunks, out_chunks, chunk_size, alpha_param):
        # Largest available chunk size
        if chunk_size is None:
            chunk_size = min(2**int(math.log2(in_dim)), 2**int(math.log2(out_dim)))
        if not math.log2(chunk_size).is_integer():
            raise ValueError("Chunk size invalid. Cannot create an algebra that does not have a power of 2 number of elements")

        if chunk_size > in_dim:
            raise ValueError("Chunk size is invalid. Size of algebra cannot be larger than input dimension")
        if chunk_size > out_dim:
            raise ValueError("Chunk size is invalid. Size of algebra cannot be larger than output dimension")

        # Minimum number of chunks needed to fully cover
        if in_chunks is None:
            in_chunks = math.ceil(in_dim / chunk_size)
        if out_chunks is None:
            out_chunks = math.ceil(out_dim / chunk_size)

        if chunk_size * in_chunks < in_dim:
            raise ValueError("Combination of chunk size, in_chunks, and in_dim is invalid. Not enough coverage for input dimension. Must have chunk_size * in_chunks >= in_dim")
        elif chunk_size * out_chunks < out_dim:
            raise ValueError("Combination of chunk size, out_chunks, and out_dim is invalid. Not enough coverage for output dimension. Must have chunk_size * out_chunks >= out_dim")

        if in_chunks == 1 and out_chunks == 1 and alpha_param:
            pass
            # print("Warning: Alpha not enabled for one input and one output chunk")

        return in_chunks, out_chunks, chunk_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.in_chunks == 1 and self.out_chunks == 1:
            # (batch_size, chunk_size)

            ## PyTroch Full Dense Matmul
            # return F.linear(x, self.sandwich_product_matrix[0], self.bias)
            # output = (self.sandwich_sparse @ x.T).T
            # output = torch.matmul(x, self.sandwich_product_matrix).squeeze(0)

            ## PyTorch Sparse
            # output = torch.sparse.mm(self.sandwich_sparse, x.T).T

            ## Block-wise sequential Sparse
            # x_blocks = torch.split(x, [b.shape[1] for b in self.blocks], dim=1)
            # output = torch.cat([F.linear(xb, b) for xb, b in zip(x_blocks, self.blocks)], dim=-1)

            # if self.bias is not None:
            #     output += self.bias
            # return output
            res = F.linear(x, self.sandwich_product_matrix, self.bias)
            if self.alpha is not None:
                res = self.alpha * res
            return res

        batch_size = x.shape[0]

        # (batch_size, out_dim)
        output = torch.zeros(batch_size, self.out_dim, device=x.device, dtype=x.dtype)
        # (batch_size, in_chunks, 1, 1, chunk_size)
        x_chunks = x[:, self.in_chunk_indices].unsqueeze(2).unsqueeze(3)

        # (batch_size, in_chunks, out_chunks, chunk_size)
        processed_chunks = torch.matmul(x_chunks, self.sandwich_product_matrix).squeeze(3)

        if self.alpha is not None:
            processed_chunks = self.alpha.view(1, self.in_chunks, self.out_chunks, 1) * processed_chunks

        # (batch_size, out_chunks * chunk_size)
        processed_chunks_sum_flat = processed_chunks.sum(dim=1).reshape(batch_size, -1)

        batched_out_chunk_indices = self.out_chunk_indices.view(1, -1).expand(batch_size, -1)

        output.scatter_add_(1, batched_out_chunk_indices, processed_chunks_sum_flat)
        output = output / self.count

        if self.bias is not None:
            output += self.bias

        return output

    def _update_rotors(self) -> None:
        with torch.no_grad():
            _, self.v_init_left = self._simple_decomp(self.bivectors_left, self.v_init_left)
            if not self.single_rotor:
                _, self.v_init_right = self._simple_decomp(self.bivectors_right, self.v_init_right)

        # (k, in_chunks * out_chunks, num_basis_biv)
        simple_decomp_left, _ = self._simple_decomp(self.bivectors_left, self.v_init_left)
        if not self.single_rotor:
            simple_decomp_right, _ = self._simple_decomp(self.bivectors_right, self.v_init_right)

        # (k, in_chunks * out_chunks, num_basis_biv + 1)
        rotor_decomp_left = self.algebra.exp_simple_batched(simple_decomp_left.reshape(-1, self.num_basis_biv)).reshape(self.k, self.bivectors_left.shape[0], self.num_basis_biv + 1)
        if not self.single_rotor:
            rotor_decomp_right = self.algebra.exp_simple_batched(simple_decomp_right.reshape(-1, self.num_basis_biv)).reshape(self.k, self.bivectors_left.shape[0], self.num_basis_biv + 1)

        # (1, chunk_size, chunk_size)
        if not self.single_rotor:
            sandwich_product_matrix = self.algebra.precompute_sandwich_product_doublerotor(rotor_decomp_left, rotor_decomp_right)
        else:
            sandwich_product_matrix = self.algebra.precompute_sandwich_product(rotor_decomp_left)

        if self.in_chunks != 1 or self.out_chunks != 1:
            # (1, in_chunks, out_chunks, chunk_size, chunk_size)
            self.sandwich_product_matrix = (sandwich_product_matrix
                                            .view(self.in_chunks, self.out_chunks, self.chunk_size, self.chunk_size)
                                            .unsqueeze(0))
        else:
            self.sandwich_product_matrix = sandwich_product_matrix[0]

            # sandwich_dense = sandwich_product_matrix[0]
            # n = int(math.log2(self.chunk_size))
            # self.block_sizes = [math.comb(n, k) for k in range(n + 1)]
            # self.blocks = []
            # start = 0
            # for size in self.block_sizes:
            #     end = start + size
            #     block = sandwich_dense[start:end, start:end].contiguous()
            #     self.blocks.append(block)
            #     start = end

    def _simple_decomp(self, B: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        simple_decomp = torch.ones(self.k, B.shape[0], self.num_basis_biv, device=self.device, dtype=self.dtype)
        singular_vectors = torch.empty(self.k - 1, self.bivectors_left.shape[0], self.algebra._num_bases, device=self.device, dtype=self.dtype, requires_grad=False)

        for i in range(self.k - 1):
            b_simple, singular_vectors[i] = self._projection(B, v[i])
            B = B - b_simple
            simple_decomp[i] = b_simple
        else:
            # If there is bivector left over on last iteration, projection is identity
            simple_decomp[-1] = B

        return simple_decomp, singular_vectors

    def _projection(self, B: torch.Tensor, v: torch.Tensor, threshold: float = .00001, max_iterations: int = 20) -> Tuple[torch.Tensor, torch.Tensor]:
        v_prev = v

        for _ in range(max_iterations):
            v = self.algebra.inner_prod_vecbiv(self.algebra.inner_prod_vecbiv(v, B), B)
            v = F.normalize(v, p=2, dim=1, eps=1e-12)

            # If singular vector changed by less than threshold, stop
            if torch.all(torch.norm(v + v_prev, dim=1) < threshold):
                # print("Break", v)
                break
            v_prev = v
            # print(v)

        # Compute paired singular vector scaled by singular value
        u_coeff = self.algebra.inner_prod_vecbiv(v, B)
        simple_bivector = self.algebra.ext_prod_vecvec(v, u_coeff)
        return simple_bivector, v

    def extra_repr(self):
        # This string is appended to the built-in representation.
        return f'in_features={self.in_dim}, out_features={self.out_dim}, alpha={self.alpha is not None}, bias={self.bias is not None}, in_chunks={self.in_chunks}, out_chunks={self.out_chunks}, chunk_size={self.chunk_size}'

    def to(self, device=None, dtype=None):
        super().to(device=device, dtype=dtype)
        if device is not None:
            self.device = device
            self.sandwich_product_matrix = self.sandwich_product_matrix.to(device=device)
        if dtype is not None:
            self.dtype = dtype
            self.sandwich_product_matrix = self.sandwich_product_matrix.to(dtype=dtype)

        return self

# with torch.no_grad():
#     n = 11
#     in_dim  = 2**n
#     out_dim = 2**n
#     f = Rotor(in_dim, out_dim, in_chunks=1, out_chunks=1, chunk_size=2**n, single_rotor=True, alpha_param=False, bias_param=False, device='cuda', dtype=torch.float32)
    # x = torch.arange(in_dim, device='cuda', dtype=torch.float32, requires_grad=True) + 1
    # x = x.repeat(7, 1)
    # x = torch.randn(1, in_dim, device='cpu', dtype=torch.bfloat16, requires_grad=True)
    # print(x)
    # bivectors = torch.tensor([[1.5, -.5, 2, 3.5, -3, 1]], device='cuda', dtype=torch.float32)
    # v = torch.randn(f.k - 1, bivectors.shape[0], f.algebra._num_bases, device=bivectors.device, dtype=bivectors.dtype)
    # proj, v_res = f._projection(bivectors, v[0])
    # print("----------------------------")
    # print("Proj:", proj.cpu().numpy())
    # print("v:", v_res.cpu().numpy())

# f.to(device='cpu', dtype=torch.bfloat16)
# # print(torch.equal(f.algebra._cayley_rotor_left_sparse.to_dense(), f.algebra._cayley_rotor_left))

# res = f(x)
# print(res)

# torch.cuda.manual_seed(42)
# torch.cuda.reset_peak_memory_stats()


# loss = res.norm()

# loss.backward()

# print(f"Peak reserved: {torch.cuda.max_memory_reserved() / 1024**3:.3f} GB")
# print(f"Peak GPU memory allocated: {torch.cuda.max_memory_allocated()/1024**3:.3f} GB")