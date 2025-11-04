# from transformers import AutoTokenizer, AutoModelForCausalLM

# cache_dir = "/data/models"
# model_name = "meta-llama/Llama-3.2-1B-Instruct"

# tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
# model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir).to("cuda")

# print(model)

import torch
import torch.utils.benchmark as benchmark

# A = torch.randn(512, 2048, device='cuda')
# B = torch.randn(2048, 2048, device='cuda')

A = torch.randn(512, 462, device='cuda')
B = torch.randn(462, 462, device='cuda')

t0 = benchmark.Timer(
    stmt="A @ B",
    setup="from __main__ import A, B",
    num_threads=1,
    label="Matrix Multiply",
    description="GPU matmul"
)

print(t0.timeit(10000))