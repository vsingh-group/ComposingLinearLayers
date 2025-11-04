SEED = 0
from utils import set_global_seed
set_global_seed(SEED)


import os
import subprocess
import argparse
import yaml
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import shutil

from data import extract_qkv_data, extract_o_data, get_dataset
from evaluate import run_accuracy_eval, run_ppl_eval
import proj_o

# from emailme import execute_function_with_notification


device = "cuda" if torch.cuda.is_available() else "cpu"

def main(layer_ids, root_dir, config_name):
    # get config
    with open(f"{config_name}.yaml", "r") as f:
        config = yaml.safe_load(f)[args.replacement_type]

    # model
    if args.model == "llama1B":
        model_name = "meta-llama/Llama-3.2-1B-Instruct"
    elif args.model == "llama3B":
        model_name = "meta-llama/Llama-3.2-3B-Instruct"
    elif args.model == "Qwen2.5-1.5B":
        model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    elif args.model == "fox":
        model_name = "tensoropera/Fox-1-1.6B-Instruct-v0.1"
    else:
        raise TypeError(f"{args.model} is not implemented")

    print(model_name)
    print(args.replacement_type)
    print(args.dataset)
    token = os.getenv("HUGGINGTOKEN")
    if token is None:
        raise RuntimeError("HUGGINGTOKEN environment variable is not set")

    def base_model(dtype=torch.float32):
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, token=token)
        model.config.use_cache = False
        return model.eval().to(device), model.model.layers[0].self_attn.v_proj.in_features, model.model.layers[0].self_attn.v_proj.out_features

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw_dataset, dataset, metric, batch_size = get_dataset(args.dataset, args.llm_batch_size, tokenizer, token)

    # run
    layer_dir = "layer" + "".join(str(i) for i in layer_ids)
    rotor_ckpt = None
    replaced_layers = []

    for layer in layer_ids:
        layer_tag = f"layer{layer}"
        data_dir = f"{args.dataset}/{layer_dir}"
        output_dir = f"{args.dataset}/{layer_dir}/{args.replacement_type}"
        print(f"\n==== Processing {layer_tag} ====")

        # 1. Extract data (x, y_query/key/value)
        model, proj_o_dim, value_output_dim = base_model()
        extract_qkv_data(
            model=model,
            dataset=dataset,
            target_layer=layer,
            replaced_layers=replaced_layers,
            rotor_path=f"{root_dir}/out/{output_dir}",
            rotor_ckpt=rotor_ckpt,
            output_path=f"{root_dir}/data/{data_dir}",
            batch_size=batch_size,
            replacement_type=args.replacement_type,
            input_dim=proj_o_dim,
            value_output_dim=value_output_dim,
            device=device,
        )

        # Delete model
        del model
        torch.cuda.empty_cache()

        # 2. Train rotor using train.py
        for task in ["key", "query", "value"]:
            cmd = [
                "python", "train.py",
                "--lr", str(config["lr"]),
                "--epochs", str(config["epochs"]),
                "--batch_size", str(config["batch_size"]),
                "--task", task,
                "--layer", layer_tag,
                "--replacement_type", args.replacement_type,
                "--x_train_path", f"{root_dir}/data/{data_dir}/train/{layer_tag}/x_{layer_tag}.pt",
                "--y_train_path", f"{root_dir}/data/{data_dir}/train/{layer_tag}/y_{layer_tag}_{task}.pt",
                "--x_test_path", f"{root_dir}/data/{data_dir}/test/{layer_tag}/x_{layer_tag}.pt",
                "--y_test_path", f"{root_dir}/data/{data_dir}/test/{layer_tag}/y_{layer_tag}_{task}.pt",
                "--output_dir", f"{root_dir}/out/{output_dir}/{layer_tag}/{task}",
            ]
            if args.replacement_type == "rotor":
                cmd.extend(["--chunk_size", str(config["chunk_size"]),
                "--hidden_layers", str(config["hidden_layers"]),
                "--breadth_hidden", str(config["breadth_hidden"]),
                "--weight_decay", str(config["weight_decay"]),
                "--nonlinear", config["nonlinear"]])
                if config["normalize"]: cmd.append("--normalize")
                if config["residual"]: cmd.append("--residual")
                if config["use_perm"]: cmd.append("--use_perm")
                if config["proj_parallel"]: cmd.append("--proj_parallel")
                if config["single_rotor"]: cmd.append("--single_rotor")
            elif args.replacement_type == "lowrank_linear":
                cmd.extend(["--rank", str(args.rank)])
            elif args.replacement_type == "bh_linear":
                pass
            else:
                raise TypeError("No replacement type given")
            if config["cos_annealing"]: cmd.append("--cos_annealing")

            print(f"Running training for {layer_tag}/{task}")
            subprocess.run(cmd, check=True)
        torch.cuda.empty_cache()

        # 3. Adjust Proj_o Layer
        if args.replacement_type == "rotor":
            rotor_ckpt = (
                f"model_chunk={config['chunk_size']}_layers={config['hidden_layers']}_breadth={config['breadth_hidden']}"
                f"_lr={config['lr']}_weightdecay={config['weight_decay']}_batchsize={config['batch_size']}_nonlinear={config['nonlinear']}_"
                f"normalize={config['normalize']}_residual={config['residual']}_useperm={config['use_perm']}_"
                f"epochs={config['epochs']}_cosanneal={config['cos_annealing']}_projparallel={config['proj_parallel']}_singlerotor={config['single_rotor']}.pth"
            )
        elif args.replacement_type == "lowrank_linear":
            rotor_ckpt = (
                f"lowrank_linear_rank={args.rank}_lr={config['lr']}_batchsize={config['batch_size']}_"
                f"epochs={config['epochs']}_cosanneal={config['cos_annealing']}.pth"
            )
        elif args.replacement_type == "bh_linear":
                rotor_ckpt = (
                    f"bh_linear_lr={config['lr']}_batchsize={config['batch_size']}_"
                    f"epochs={config['epochs']}_cosanneal={config['cos_annealing']}.pth"
                )
        else:
            raise TypeError("No replacement type given")
        replaced_layers.append(layer_tag)

        if args.eval_datatype == "bfloat16":
            eval_datatype = torch.bfloat16
        else:
            eval_datatype = torch.float32

        if args.train_projo:
            if os.path.exists(os.path.join(f"{root_dir}/out/{output_dir}/{layer_tag}/output", rotor_ckpt)):
                print(f"[SKIP] Proj_O model already exists")
            else:
                model, proj_o_dim, value_output_dim = base_model(eval_datatype)
                xo_train, yo_train, xo_test, yo_test = extract_o_data(
                    model=model,
                    dataset=dataset,
                    target_layer=layer,
                    replaced_layers=replaced_layers,
                    rotor_path=f"{root_dir}/out/{output_dir}",
                    rotor_ckpt=rotor_ckpt,
                    output_path=f"{root_dir}/data/{data_dir}",
                    batch_size=batch_size,
                    replacement_type=args.replacement_type,
                    input_dim=proj_o_dim,
                    value_output_dim=value_output_dim,
                    device=device,
                    dtype=eval_datatype
                )
                del model
                torch.cuda.empty_cache()

                proj_o.train_proj_o(
                    x_train=xo_train,
                    y_train=yo_train,
                    x_test=xo_test,
                    y_test=yo_test,
                    dim=proj_o_dim,
                    device=device,
                    output_path=f"{root_dir}/out/{output_dir}/{layer_tag}/output",
                    ckpt=rotor_ckpt,
                    epochs=15
                )

        model, proj_o_dim, value_output_dim = base_model(eval_datatype)

        if metric == "accuracy":
            # Baseline
            run_accuracy_eval(
                model=model,
                tokenizer=tokenizer,
                dataset=raw_dataset,
                input_dim=proj_o_dim,
                value_output_dim=value_output_dim,
                train_projo=args.train_projo,
                dataset_name=args.dataset,
                print_model=False,
                log=False,
            )

            run_accuracy_eval(
                model=model,
                tokenizer=tokenizer,
                dataset=raw_dataset,
                input_dim=proj_o_dim,
                value_output_dim=value_output_dim,
                train_projo=args.train_projo,
                dataset_name=args.dataset,
                replacement_type=args.replacement_type,
                replaced_layers=replaced_layers,
                rotor_path=f"{root_dir}/out/{output_dir}",
                rotor_ckpt=rotor_ckpt,
                print_model=False,
                dtype=eval_datatype
            )

        elif metric == "perplexity":
            # Baseline
            run_ppl_eval(
                model=model,
                dataset=dataset,
                stride=256,
                input_dim=proj_o_dim,
                value_output_dim=value_output_dim,
                train_projo=args.train_projo,
                log=False,
            )

            run_ppl_eval(
                model=model,
                dataset=dataset,
                stride=256,
                input_dim=proj_o_dim,
                value_output_dim=value_output_dim,
                train_projo=args.train_projo,
                replacement_type=args.replacement_type,
                rank=args.rank,
                replaced_layers=replaced_layers,
                rotor_path=f"{root_dir}/out/{output_dir}",
                rotor_ckpt=rotor_ckpt,
                print_model=False,
            )

        if args.remove:
            for layer in layer_ids:
                layer_tag = f"layer{layer}"
                if os.path.exists(f"{root_dir}/data/{data_dir}"):
                    shutil.rmtree(f"{root_dir}/data/{data_dir}")

        # Delete model
        del model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=str, required=True, help="e.g., 12,13,14")
    parser.add_argument("--root", type=str, required=True, help="Root directory")
    parser.add_argument("--config", type=str, required=True, help="Config file")
    parser.add_argument("--dataset", type=str, required=True, help="e.g., arc_challenge, wikitext")
    parser.add_argument("--train_projo", action="store_true", help="If set, trains the o_proj projection")
    parser.add_argument("--model", type=str, required=True, choices=['llama1B', 'llama3B', 'Qwen2.5-1.5B', 'fox'], help="LLM model")
    parser.add_argument("--eval_datatype", type=str, default="float32", choices=['bfloat16', 'float32'], help="Data type of evaluate data in. Unsure if working?")
    parser.add_argument("--replacement_type", type=str, required=True, choices=['rotor', 'lowrank_linear', 'bh_linear'], help="Replacement type for layer")
    parser.add_argument("--rank", type=int, help="Rank for lowrank linear")
    parser.add_argument("--llm_batch_size", type=int, help="Number of prompts processed at once")
    parser.add_argument("--remove", action="store_true", help="If set, do not save data")

    args = parser.parse_args()
    layer_ids = [int(x) for x in args.layers.split(",")]
    # if len(layer_ids) > 1 and not args.remove:
    #     raise RuntimeError("Must use --remove option when replacing multiple layers")
    main(layer_ids, root_dir=f'{args.root}/{args.model}', config_name=args.config)
    # execute_function_with_notification(main, layer_ids, root_dir=f'/data/{args.root}/{args.model}', config_name=args.config)