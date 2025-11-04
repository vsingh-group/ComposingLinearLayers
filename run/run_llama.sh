#!/bin/bash

# Optional: activate your Python environment
# source ~/venvs/rotor_venv/bin/activate

datasets=("wikitext" "c4" "ptb" "arc_challenge" "hellaswag_chat")
layers=(15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 0 10,11 10,12 10,13 10,14 10,15 11,12 11,13 11,14 11,15 12,13 12,14 12,15 13,14 13,15 14,15)
for dataset in "${datasets[@]}"
do
  for layer in "${layers[@]}"
  do
    echo "Processing layer $layer on dataset $dataset"
    python main.py --layers "$layer" --root result --replacement_type bh_linear               --config config_files/config  --dataset "$dataset" --train_projo --model llama1B
    python main.py --layers "$layer" --root result --replacement_type lowrank_linear --rank 1 --config config_files/config  --dataset "$dataset" --train_projo --model llama1B
    python main.py --layers "$layer" --root result --replacement_type lowrank_linear --rank 4 --config config_files/config  --dataset "$dataset" --train_projo --model llama1B
    python main.py --layers "$layer" --root result --replacement_type rotor                   --config config_files/config  --dataset "$dataset" --train_projo --model llama1B --remove
  done
done

layers=(4,15 7,8 11,13 8,12 5,9 10,12,13 7,11,14 6,10,13 4,5,9 6,13,15)
for layer in "${layers[@]}"
do
  echo "Processing layer $layer on dataset ptb"
  python main.py --layers "$layer" --root result --replacement_type bh_linear               --config config_files/config  --dataset ptb --train_projo --model llama1B
  python main.py --layers "$layer" --root result --replacement_type lowrank_linear --rank 1 --config config_files/config  --dataset ptb --train_projo --model llama1B
  python main.py --layers "$layer" --root result --replacement_type lowrank_linear --rank 4 --config config_files/config  --dataset ptb --train_projo --model llama1B
  python main.py --layers "$layer" --root result --replacement_type rotor                   --config config_files/config  --dataset ptb --train_projo --model llama1B --remove
done

layers=(10,13 11,14 5,9 6,8 6,14 7,11,13 8,13,15 9,11,14 4,5,13 4,6,10)
for layer in "${layers[@]}"
do
  echo "Processing layer $layer on dataset ptb"
  python main.py --layers "$layer" --root result --replacement_type bh_linear               --config config_files/config  --dataset wikitext --train_projo --model llama1B
  python main.py --layers "$layer" --root result --replacement_type lowrank_linear --rank 1 --config config_files/config  --dataset wikitext --train_projo --model llama1B
  python main.py --layers "$layer" --root result --replacement_type lowrank_linear --rank 4 --config config_files/config  --dataset wikitext --train_projo --model llama1B
  python main.py --layers "$layer" --root result --replacement_type rotor                   --config config_files/config  --dataset wikitext --train_projo --model llama1B --remove
done

layers=(13,15 11,15 4,6 8,13 4,14 9,13,14 7,10,12 4,6,15 4,7,9 7,8,10)
for layer in "${layers[@]}"
do
  echo "Processing layer $layer on dataset ptb"
  python main.py --layers "$layer" --root result --replacement_type bh_linear               --config config_files/config  --dataset c4 --train_projo --model llama1B
  python main.py --layers "$layer" --root result --replacement_type lowrank_linear --rank 1 --config config_files/config  --dataset c4 --train_projo --model llama1B
  python main.py --layers "$layer" --root result --replacement_type lowrank_linear --rank 4 --config config_files/config  --dataset c4 --train_projo --model llama1B
  python main.py --layers "$layer" --root result --replacement_type rotor                   --config config_files/config  --dataset c4 --train_projo --model llama1B --remove
done

layers=(10,15 12,15 4,14 5,11 8,12 4,13,15 9,11,13 5,14,15 6,9,14 4,7,9)
for layer in "${layers[@]}"
do
  echo "Processing layer $layer on dataset ptb"
  python main.py --layers "$layer" --root result --replacement_type bh_linear               --config config_files/config  --dataset arc_challenge --train_projo --model llama1B
  python main.py --layers "$layer" --root result --replacement_type lowrank_linear --rank 1 --config config_files/config  --dataset arc_challenge --train_projo --model llama1B
  python main.py --layers "$layer" --root result --replacement_type lowrank_linear --rank 4 --config config_files/config  --dataset arc_challenge --train_projo --model llama1B
  python main.py --layers "$layer" --root result --replacement_type rotor                   --config config_files/config  --dataset arc_challenge --train_projo --model llama1B --remove
done

layers=(7,14 6,9 9,12 5,7 10,14 11,13,14 7,9,14 4,9,15 6,3,15 5,8,11)
for layer in "${layers[@]}"
do
  echo "Processing layer $layer on dataset ptb"
  python main.py --layers "$layer" --root result --replacement_type bh_linear               --config config_files/config  --dataset hellaswag_chat --train_projo --model llama1B
  python main.py --layers "$layer" --root result --replacement_type lowrank_linear --rank 1 --config config_files/config  --dataset hellaswag_chat --train_projo --model llama1B
  python main.py --layers "$layer" --root result --replacement_type lowrank_linear --rank 4 --config config_files/config  --dataset hellaswag_chat --train_projo --model llama1B
  python main.py --layers "$layer" --root result --replacement_type rotor                   --config config_files/config  --dataset hellaswag_chat --train_projo --model llama1B --remove
done
# python /workspace/emailme/emailme.py --GPU "$gpu_id"