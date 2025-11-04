#!/bin/bash

# Optional: activate your Python environment
# source ~/venvs/rotor_venv/bin/activate

datasets=("wikitext" "c4" "ptb" "arc_challenge" "hellaswag_chat")
layers=(31 30 29 28 27 26 25 24 23 22 21 20 19 18 17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 0)
for dataset in "${datasets[@]}"
do
  for layer in "${layers[@]}"
  do
    echo "Processing layer $layer on dataset $dataset"
    python main.py --layers "$layer" --root result --replacement_type bh_linear               --config config_files/config  --dataset "$dataset" --train_projo --model fox
    python main.py --layers "$layer" --root result --replacement_type lowrank_linear --rank 1 --config config_files/config  --dataset "$dataset" --train_projo --model fox
    python main.py --layers "$layer" --root result --replacement_type lowrank_linear --rank 4 --config config_files/config  --dataset "$dataset" --train_projo --model fox
    python main.py --layers "$layer" --root result --replacement_type rotor                   --config config_files/config  --dataset "$dataset" --train_projo --model fox --remove
  done
done

layers=(8,12 3,30 16,17 16,20 3,28)
for layer in "${layers[@]}"
do
  echo "Processing layer $layer on dataset ptb"
  python main.py --layers "$layer" --root result --replacement_type bh_linear               --config config_files/config  --dataset ptb --train_projo --model fox
  python main.py --layers "$layer" --root result --replacement_type lowrank_linear --rank 1 --config config_files/config  --dataset ptb --train_projo --model fox
  python main.py --layers "$layer" --root result --replacement_type lowrank_linear --rank 4 --config config_files/config  --dataset ptb --train_projo --model fox
  python main.py --layers "$layer" --root result --replacement_type rotor                   --config config_files/config  --dataset ptb --train_projo --model fox --remove
done

layers=(9,31 24,27 19,31 4,15 11,16)
for layer in "${layers[@]}"
do
  echo "Processing layer $layer on dataset ptb"
  python main.py --layers "$layer" --root result --replacement_type bh_linear               --config config_files/config  --dataset wikitext --train_projo --model fox
  python main.py --layers "$layer" --root result --replacement_type lowrank_linear --rank 1 --config config_files/config  --dataset wikitext --train_projo --model fox
  python main.py --layers "$layer" --root result --replacement_type lowrank_linear --rank 4 --config config_files/config  --dataset wikitext --train_projo --model fox
  python main.py --layers "$layer" --root result --replacement_type rotor                   --config config_files/config  --dataset wikitext --train_projo --model fox --remove
done

layers=(12,31 19,27 10,31 26,29 14,29)
for layer in "${layers[@]}"
do
  echo "Processing layer $layer on dataset ptb"
  python main.py --layers "$layer" --root result --replacement_type bh_linear               --config config_files/config  --dataset c4 --train_projo --model fox
  python main.py --layers "$layer" --root result --replacement_type lowrank_linear --rank 1 --config config_files/config  --dataset c4 --train_projo --model fox
  python main.py --layers "$layer" --root result --replacement_type lowrank_linear --rank 4 --config config_files/config  --dataset c4 --train_projo --model fox
  python main.py --layers "$layer" --root result --replacement_type rotor                   --config config_files/config  --dataset c4 --train_projo --model fox --remove
done

layers=(7,8 28,30 23,31 28,23 5,23)
for layer in "${layers[@]}"
do
  echo "Processing layer $layer on dataset ptb"
  python main.py --layers "$layer" --root result --replacement_type bh_linear               --config config_files/config  --dataset arc_challenge --train_projo --model fox
  python main.py --layers "$layer" --root result --replacement_type lowrank_linear --rank 1 --config config_files/config  --dataset arc_challenge --train_projo --model fox
  python main.py --layers "$layer" --root result --replacement_type lowrank_linear --rank 4 --config config_files/config  --dataset arc_challenge --train_projo --model fox
  python main.py --layers "$layer" --root result --replacement_type rotor                   --config config_files/config  --dataset arc_challenge --train_projo --model fox --remove
done

layers=(14,26 14,15 5,9 6,17 18,27)
for layer in "${layers[@]}"
do
  echo "Processing layer $layer on dataset ptb"
  python main.py --layers "$layer" --root result --replacement_type bh_linear               --config config_files/config  --dataset hellaswag_chat --train_projo --model fox
  python main.py --layers "$layer" --root result --replacement_type lowrank_linear --rank 1 --config config_files/config  --dataset hellaswag_chat --train_projo --model fox
  python main.py --layers "$layer" --root result --replacement_type lowrank_linear --rank 4 --config config_files/config  --dataset hellaswag_chat --train_projo --model fox
  python main.py --layers "$layer" --root result --replacement_type rotor                   --config config_files/config  --dataset hellaswag_chat --train_projo --model fox --remove
done

# python /workspace/emailme/emailme.py --GPU "$gpu_id"