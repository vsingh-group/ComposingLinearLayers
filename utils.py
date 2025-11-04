# utils.py
import os
import random
import numpy as np
import torch

def set_global_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

import re

def parse_model_ckpt_name(ckpt_name, replacement_type):
    if replacement_type == "rotor":
        pattern = (
            r"model_chunk=(\d+)_layers=(\d+)_breadth=(\d+)"
            r"_lr=([0-9.]+)_weightdecay=([0-9]+(?:\.[0-9]+)?)_batchsize=(\d+)"
            r"_nonlinear=([a-zA-Z]+)_normalize=(True|False)_residual=(True|False)"
            r"_useperm=(True|False)_epochs=(\d+)_cosanneal=(True|False)_projparallel=(True|False)_singlerotor=(True|False)"
        )
        match = re.match(pattern, ckpt_name.split(".pth")[0])
        if not match:
            raise ValueError("Checkpoint filename does not match expected format")

        (chunk_size, layers, breadth,
        lr, weight_decay, batch_size,
        nonlinear, normalize, residual,
        use_perm, epochs, cos_anneal, proj_parallel, single_rotor) = match.groups()

        return {
            "chunk_size": int(chunk_size),
            "hidden_layers": int(layers),
            "breadth_hidden": int(breadth),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "batch_size": int(batch_size),
            "nonlinear": nonlinear,
            "normalize": normalize == "True",
            "residual": residual == "True",
            "use_perm": use_perm == "True",
            "epochs": int(epochs),
            "cos_annealing": cos_anneal == "True",
            "proj_parallel": proj_parallel == "True",
            "single_rotor": single_rotor == "True",
        }
    elif replacement_type == "lowrank_linear":
        pattern = (
            r"lowrank_linear_rank=(\d+)_lr=([0-9.]+)_batchsize=(\d+)_"
            r"epochs=(\d+)_cosanneal=(True|False)"
        )

        match = re.match(pattern, ckpt_name.split(".pth")[0])
        if not match:
            raise ValueError("Checkpoint filename does not match expected format")

        (rank, lr, batch_size, epochs, cos_annealing) = match.groups()

        return {
            "rank": int(rank),
            "lr": float(lr),
            "batch_size": int(batch_size),
            "epochs": int(epochs),
            "cos_annealing": cos_annealing == "True"
        }