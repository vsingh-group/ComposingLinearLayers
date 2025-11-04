#!/bin/bash

# Optional: activate your Python environment
# source ~/venvs/rotor_venv/bin/activate

# layers=(7,11,13 8,13,15 9,11,14 4,5,13 4,6,10)
layers=(9,11,14)
for layer in "${layers[@]}"
do
  echo "Processing layer $layer"
  # python main.py --layers "$layer" --root result --replacement_type rotor --config config  --dataset wikitext --model llama1B --llm_batch_size 16
  python main.py --layers "$layer" --root result_normed --replacement_type lowrank_linear --rank 4 --config config  --dataset wikitext --model llama1B --train_projo --llm_batch_size 16
done
# python /workspace/emailme/emailme.py --GPU "$gpu_id"
