#!/bin/bash

# Optional: activate your Python environment
# source ~/venvs/rotor_venv/bin/activate

datasets=("wikitext" "c4" "ptb" "arc_challenge" "hellaswag_chat")
layers=(27 26 25 24 23 22 21 20 19 18 17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 0 10,11 10,12 10,13 10,14 10,15 11,12 11,13 11,14 11,15 12,13 12,14 12,15 13,14 13,15 14,15)
for dataset in "${datasets[@]}"
do
  for layer in "${layers[@]}"
  do
    echo "Processing layer $layer on dataset $dataset"
    python main.py --layers "$layer" --root result --replacement_type bh_linear               --config config_files/config  --dataset "$dataset" --train_projo --model Qwen2.5-1.5B
    python main.py --layers "$layer" --root result --replacement_type lowrank_linear --rank 1 --config config_files/config  --dataset "$dataset" --train_projo --model Qwen2.5-1.5B
    python main.py --layers "$layer" --root result --replacement_type lowrank_linear --rank 4 --config config_files/config  --dataset "$dataset" --train_projo --model Qwen2.5-1.5B
    python main.py --layers "$layer" --root result --replacement_type rotor                   --config config_files/config  --dataset "$dataset" --train_projo --model Qwen2.5-1.5B --remove
  done
done

layers=(6,27 8,10 6,13 3,18 4,20 10,16,27 8,12,26 3,18,20 12,19,20 4,6,13)
for layer in "${layers[@]}"
do
  echo "Processing layer $layer on dataset ptb"
  python main.py --layers "$layer" --root result --replacement_type bh_linear               --config config_files/config  --dataset ptb --train_projo --model Qwen2.5-1.5B
  python main.py --layers "$layer" --root result --replacement_type lowrank_linear --rank 1 --config config_files/config  --dataset ptb --train_projo --model Qwen2.5-1.5B
  python main.py --layers "$layer" --root result --replacement_type lowrank_linear --rank 4 --config config_files/config  --dataset ptb --train_projo --model Qwen2.5-1.5B
  python main.py --layers "$layer" --root result --replacement_type rotor                   --config config_files/config  --dataset ptb --train_projo --model Qwen2.5-1.5B --remove
done

layers=(11,25 6,26 12,23 7,18 17,22 7,13,18 8,23,24 13,15,20 7,14,23 13,18,21)
for layer in "${layers[@]}"
do
  echo "Processing layer $layer on dataset ptb"
  python main.py --layers "$layer" --root result --replacement_type bh_linear               --config config_files/config  --dataset wikitext --train_projo --model Qwen2.5-1.5B
  python main.py --layers "$layer" --root result --replacement_type lowrank_linear --rank 1 --config config_files/config  --dataset wikitext --train_projo --model Qwen2.5-1.5B
  python main.py --layers "$layer" --root result --replacement_type lowrank_linear --rank 4 --config config_files/config  --dataset wikitext --train_projo --model Qwen2.5-1.5B
  python main.py --layers "$layer" --root result --replacement_type rotor                   --config config_files/config  --dataset wikitext --train_projo --model Qwen2.5-1.5B --remove
done

layers=(9,23 10,14 19,22 8,15 8,11 9,10,16 13,20,24 7,11,22 9,12,15 11,20,23)
for layer in "${layers[@]}"
do
  echo "Processing layer $layer on dataset ptb"
  python main.py --layers "$layer" --root result --replacement_type bh_linear               --config config_files/config  --dataset c4 --train_projo --model Qwen2.5-1.5B
  python main.py --layers "$layer" --root result --replacement_type lowrank_linear --rank 1 --config config_files/config  --dataset c4 --train_projo --model Qwen2.5-1.5B
  python main.py --layers "$layer" --root result --replacement_type lowrank_linear --rank 4 --config config_files/config  --dataset c4 --train_projo --model Qwen2.5-1.5B
  python main.py --layers "$layer" --root result --replacement_type rotor                   --config config_files/config  --dataset c4 --train_projo --model Qwen2.5-1.5B --remove
done

layers=(20,24 4,19 3,23 7,26 5,16 6,18,25 3,19,23 2,22,27 3,15,18)
for layer in "${layers[@]}"
do
  echo "Processing layer $layer on dataset ptb"
  python main.py --layers "$layer" --root result --replacement_type bh_linear               --config config_files/config  --dataset arc_challenge --train_projo --model Qwen2.5-1.5B
  python main.py --layers "$layer" --root result --replacement_type lowrank_linear --rank 1 --config config_files/config  --dataset arc_challenge --train_projo --model Qwen2.5-1.5B
  python main.py --layers "$layer" --root result --replacement_type lowrank_linear --rank 4 --config config_files/config  --dataset arc_challenge --train_projo --model Qwen2.5-1.5B
  python main.py --layers "$layer" --root result --replacement_type rotor                   --config config_files/config  --dataset arc_challenge --train_projo --model Qwen2.5-1.5B --remove
done

layers=(2,26 5,9 8,21 4,7 17,27 8,12,21 14,15,20 5,14,16 11,16,22 9,13,17)
for layer in "${layers[@]}"
do
  echo "Processing layer $layer on dataset ptb"
  python main.py --layers "$layer" --root result --replacement_type bh_linear               --config config_files/config  --dataset hellaswag_chat --train_projo --model Qwen2.5-1.5B
  python main.py --layers "$layer" --root result --replacement_type lowrank_linear --rank 1 --config config_files/config  --dataset hellaswag_chat --train_projo --model Qwen2.5-1.5B
  python main.py --layers "$layer" --root result --replacement_type lowrank_linear --rank 4 --config config_files/config  --dataset hellaswag_chat --train_projo --model Qwen2.5-1.5B
  python main.py --layers "$layer" --root result --replacement_type rotor                   --config config_files/config  --dataset hellaswag_chat --train_projo --model Qwen2.5-1.5B --remove
done

# python /workspace/emailme/emailme.py --GPU "$gpu_id"