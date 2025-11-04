import torch
import torch.nn.functional as F

from typing import Tuple

from rotor_layer import Rotor
import types

import matplotlib.pyplot as plt

def _update_rotors(self) -> None:
        with torch.no_grad():
            _, self.v_init_left = self._simple_decomp(self.bivectors_left, self.v_init_left, True)
            if self.two_rotor:
                _, self.v_init_right = self._simple_decomp(self.bivectors_right, self.v_init_right)

        # (k, in_chunks * out_chunks, num_basis_biv)
        simple_decomp_left, _ = self._simple_decomp(self.bivectors_left, self.v_init_left, False)
        if self.two_rotor:
            simple_decomp_right, _ = self._simple_decomp(self.bivectors_right, self.v_init_right)

        # (k, in_chunks * out_chunks, num_basis_biv + 1)
        rotor_decomp_left = self.algebra.exp_simple_batched(simple_decomp_left.reshape(-1, self.num_basis_biv)).reshape(self.k, self.bivectors_left.shape[0], self.num_basis_biv + 1)
        if self.two_rotor:
            rotor_decomp_right = self.algebra.exp_simple_batched(simple_decomp_right.reshape(-1, self.num_basis_biv)).reshape(self.k, self.bivectors_left.shape[0], self.num_basis_biv + 1)

        # (1, chunk_size, chunk_size)
        if self.two_rotor:
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

def _simple_decomp(self, B: torch.Tensor, v: torch.Tensor, log: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        simple_decomp = torch.ones(self.k, B.shape[0], self.num_basis_biv, device=self.device, dtype=self.dtype)
        singular_vectors = torch.empty(self.k - 1, self.bivectors_left.shape[0], self.algebra._num_bases, device=self.device, dtype=self.dtype, requires_grad=False)

        for i in range(self.k - 1):
            b_simple, singular_vectors[i] = self._projection(B, v[i], log, i)
            B = B - b_simple
            simple_decomp[i] = b_simple
        else:
            # If there is bivector left over on last iteration, projection is identity
            simple_decomp[-1] = B

        return simple_decomp, singular_vectors

def _projection(self, B: torch.Tensor, v: torch.Tensor, log: bool, indx: int, threshold: float = .001, max_iterations: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        v_prev = v

        for counter in range(1, max_iterations + 1):
            v = self.algebra.inner_prod_vecbiv(self.algebra.inner_prod_vecbiv(v, B), B)
            v = F.normalize(v, p=2, dim=1, eps=1e-12)

            # If singular vector changed by less than threshold, stop
            if torch.all(torch.norm(v + v_prev, dim=1) < threshold):
                if log:
                    self.logs[indx].append(counter)
                break
            v_prev = v
            # print(v)
        else:
            if log:
                self.logs[indx].append(counter)

        # Compute paired singular vector scaled by singular value
        u_coeff = self.algebra.inner_prod_vecbiv(v, B)
        simple_bivector = self.algebra.ext_prod_vecvec(v, u_coeff)
        return simple_bivector, v


def test_convergence(dim, repeats=50, steps=10):
    logs = []
    for _ in range(repeats):
        f = Rotor(dim, dim)
        f._projection = types.MethodType(_projection, f)
        f._simple_decomp = types.MethodType(_simple_decomp, f)
        f._update_rotors = types.MethodType(_update_rotors, f)
        f.logs = [[] for _ in range(f.k - 1)]
        with torch.no_grad():
            target_f = Rotor(dim, dim)
            x = torch.randn(64, dim, device='cuda')
            y = target_f(x)

        x.requires_grad_(True)
        optimizer = torch.optim.Adam(f.parameters(), lr=.05)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)

        for _ in range(steps):
            pred_y = f(x)
            loss = F.mse_loss(pred_y, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            f._update_rotors()
        logs.append(f.logs)
    return logs

dims = [2**6, 2**7, 2**8, 2**9, 2**10, 2**11, 2**12]

data = []
for dim in dims:
    logs = test_convergence(dim)
    tensor = torch.tensor(logs, dtype=torch.float32)
    data.append({'dim': dim, 'tensor': tensor})

torch.save({
    'data': data
}, 'projection_results.pt')