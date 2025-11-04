import os
import torch
from tqdm import tqdm
from rotor_network import load_rotor_layers
from torch.utils.data import random_split

from custom_layers.llama_layers import CustomLlamaAttention, CustomLlamaModel
from custom_layers.qwen_layers import CustomQwen2Attention
import types

from proj_o import load_proj_o_model

import gc

from baselines import load_lowranklinear_layers, load_bh_layers

SEED = 0
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_custom_attention(attn_type, config, layer_idx):
    if attn_type == "llama" or attn_type == "fox":
        CustomAttention = CustomLlamaAttention
    elif attn_type == "qwen2":
        CustomAttention = CustomQwen2Attention
    else:
        raise ValueError(f"Unknown attention type: {attn_type}")

    return CustomAttention(config, layer_idx)

def get_custom_forward_functions(attn_type):
    if attn_type == "llama" or attn_type == "fox":
        input_forward_function, stop_after_target_function = CustomLlamaModel.forward_getinputQKV, CustomLlamaModel.forward_stop_after_target_layer

    return input_forward_function, stop_after_target_function

def get_loader_fn(replacement_type):
    if replacement_type == "rotor":
        return load_rotor_layers
    elif replacement_type == "lowrank_linear":
        return load_lowranklinear_layers
    elif replacement_type == "bh_linear":
        return load_bh_layers
    elif replacement_type is not None:
        raise TypeError(f"Unknown replacement type: {replacement_type}")

def split_data(dataset_dict):
    dataset = [
        {"input_ids": input_id, "attention_mask": attn_mask}
        for input_id, attn_mask in zip(dataset_dict["input_ids"], dataset_dict["attention_mask"])
    ]
    train_size = int(0.8 * len(dataset))
    generator = torch.Generator().manual_seed(SEED)
    train_data, test_data = random_split(dataset, [train_size, len(dataset) - train_size], generator=generator)
    return train_data, test_data

#### Extract Data ####

def get_qkv_output(
    target_layer: int,
    model,
    dataset,
    output_dir,
    device,
    batch_size,

):

    save_dir = os.path.join(output_dir, f'layer{target_layer}')
    if all(os.path.exists(os.path.join(save_dir, f"y_layer{target_layer}_{p}.pt")) for p in ["query", "key", "value"]):
        print(f"[SKIP] QKV outputs for layer{target_layer} already exist")
        return

    # Patch attention layers
    layer = model.model.layers[target_layer]
    # for i, layer in enumerate(model.model.layers):
    old_attn = layer.self_attn
    custom_attn = get_custom_attention(model.config.model_type, old_attn.config, old_attn.layer_idx)
    custom_attn.load_state_dict(old_attn.state_dict())
    layer.self_attn = custom_attn.to(device)

    all_q, all_k, all_v = [], [], []

    for start in tqdm(range(0, len(dataset), batch_size), desc=f"Extracting outputs to layer {target_layer}"):
        end = min(start + batch_size, len(dataset))
        batch = [dataset[i] for i in range(start, end)]
        input_ids = torch.stack([item["input_ids"] for item in batch]).to(device)
        attention_mask = torch.stack([item["attention_mask"] for item in batch]).to(device)

        with torch.no_grad():
            # model.model(
            #     input_ids=input_ids,
            #     attention_mask=attention_mask,
            #     target_layer=target_layer
            # )
            model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        layer_attn = model.model.layers[target_layer].self_attn

        all_q.append(layer_attn.q_proj_output.cpu().reshape(-1, layer_attn.q_proj_output.shape[-1]))
        all_k.append(layer_attn.k_proj_output.cpu().reshape(-1, layer_attn.k_proj_output.shape[-1]))
        all_v.append(layer_attn.v_proj_output.cpu().reshape(-1, layer_attn.v_proj_output.shape[-1]))

    os.makedirs(save_dir, exist_ok=True)
    torch.save(torch.cat(all_q, dim=0), os.path.join(save_dir, f"y_layer{target_layer}_query.pt"))
    torch.save(torch.cat(all_k, dim=0), os.path.join(save_dir, f"y_layer{target_layer}_key.pt"))
    torch.save(torch.cat(all_v, dim=0), os.path.join(save_dir, f"y_layer{target_layer}_value.pt"))

    # Clear memory
    del all_q
    del all_k
    del all_v
    gc.collect()
    torch.cuda.empty_cache()

def get_qkv_input(
    target_layer: int,
    model,
    dataset,
    output_dir,
    device,
    batch_size,
):

    save_path = os.path.join(output_dir, f"layer{target_layer}", f"x_layer{target_layer}.pt")
    if os.path.exists(save_path):
        print(f"[SKIP] Input to layer{target_layer} already exists")
        return

    input_forward_function, _ = get_custom_forward_functions(model.config.model_type)
    model.model.forward = types.MethodType(input_forward_function, model.model)

    extracted_inputs = []
    for start in tqdm(range(0, len(dataset), batch_size), desc=f"Extracting inputs to layer {target_layer}"):
        end = min(start + batch_size, len(dataset))
        batch = [dataset[i] for i in range(start, end)]
        input_ids = torch.stack([item["input_ids"] for item in batch]).to(device)
        attention_mask = torch.stack([item["attention_mask"] for item in batch]).to(device)

        with torch.no_grad():
            model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                target_layer=target_layer
            )
            x = model.model.saved_normed_input.cpu().reshape(-1, model.model.saved_normed_input.shape[-1])
            extracted_inputs.append(x)

    os.makedirs(output_dir, exist_ok=True)
    torch.save(torch.cat(extracted_inputs, dim=0), save_path)

    # Clear memory
    del extracted_inputs
    gc.collect()
    torch.cuda.empty_cache()

def extract_qkv_data(
    model,
    dataset,
    target_layer,
    replaced_layers,
    rotor_path,
    rotor_ckpt,
    output_path,
    batch_size,
    replacement_type,
    input_dim,
    value_output_dim,
    device,
):

    # --- Split Tokenized Data ---
    train_data, test_data = split_data(dataset)

    # input_forward_function, stop_after_target_function = get_custom_forward_functions(model.config.model_type)

    # Early stopping when layer is reached
    # model.model.forward = types.MethodType(stop_after_target_function, model.model)

    # --- Extract True/Original QKV Outputs ---
    # print("QKV output model\n", model)
    for dir, split in zip(['/train', '/test'], [train_data, test_data]):
        get_qkv_output(
            model=model,
            dataset=split,
            output_dir=output_path+dir,
            target_layer=target_layer,
            device=device,
            batch_size=batch_size,
        )

    # --- Extract Inputs ---
    loader_fn = get_loader_fn(replacement_type)
    rotor_nets = loader_fn(
            replaced_layers,
            input_dim=input_dim,
            value_output_dim=value_output_dim,
            model_dir=rotor_path,
            model_ckpt=rotor_ckpt,
            device=device)

    for layer_name in replaced_layers:
        idx = int(layer_name.replace("layer", ""))
        model.model.layers[idx].self_attn.q_proj = rotor_nets[layer_name]["query"]
        model.model.layers[idx].self_attn.k_proj = rotor_nets[layer_name]["key"]
        model.model.layers[idx].self_attn.v_proj = rotor_nets[layer_name]["value"]
        model.model.layers[idx].self_attn.o_proj = load_proj_o_model(dim=input_dim, output_path=f"{rotor_path}/{layer_name}/output", ckpt=rotor_ckpt, device=device)

    # print("QKV input model\n", model)

    for dir, split in zip(['/train', '/test'], [train_data, test_data]):
        get_qkv_input(
            model=model,
            dataset=split,
            output_dir=output_path+dir,
            target_layer=target_layer,
            device=device,
            batch_size=batch_size
        )

### Proj_O ###

def get_o_output(
    target_layer: int,
    model,
    dataset,
    output_dir,
    device,
    batch_size,
):
    save_dir = os.path.join(output_dir, f'layer{target_layer}')
    save_path = os.path.join(save_dir, f"y_layer{target_layer}_output.pt")
    if os.path.exists(save_path):
        print(f"[SKIP] o_proj output for layer{target_layer} already exists")
        return

    # Patch attention layers
    layer = model.model.layers[target_layer]
    # for i, layer in enumerate(model.model.layers):
    old_attn = layer.self_attn
    custom_attn = get_custom_attention(model.config.model_type, old_attn.config, old_attn.layer_idx)
    custom_attn.load_state_dict(old_attn.state_dict())
    layer.self_attn = custom_attn.to(device)

    all_o_proj = []

    for start in tqdm(range(0, len(dataset), batch_size), desc=f"Extracting o_proj output at layer {target_layer}"):
        end = min(start + batch_size, len(dataset))
        batch = [dataset[i] for i in range(start, end)]
        input_ids = torch.stack([item["input_ids"] for item in batch]).to(device)
        attention_mask = torch.stack([item["attention_mask"] for item in batch]).to(device)

        with torch.no_grad():
            # model.model(input_ids=input_ids, attention_mask=attention_mask, target_layer=target_layer)
            model(input_ids=input_ids, attention_mask=attention_mask)

        layer_attn = model.model.layers[target_layer].self_attn
        o_proj_out = layer_attn.o_proj_output.cpu().reshape(-1, layer_attn.o_proj_output.shape[-1])
        all_o_proj.append(o_proj_out)

    os.makedirs(save_dir, exist_ok=True)
    torch.save(torch.cat(all_o_proj, dim=0), save_path)

    # Clear memory
    del all_o_proj
    gc.collect()
    torch.cuda.empty_cache()

def get_o_input(
    target_layer: int,
    model,
    dataset,
    device,
    batch_size,
):
    # Extract
    extracted_inputs = []
    for start in tqdm(range(0, len(dataset), batch_size), desc=f"Extracting o_proj input at layer {target_layer}"):
        end = min(start + batch_size, len(dataset))
        batch = [dataset[i] for i in range(start, end)]
        input_ids = torch.stack([item["input_ids"] for item in batch]).to(device)
        attention_mask = torch.stack([item["attention_mask"] for item in batch]).to(device)

        with torch.no_grad():
            # model.model(input_ids=input_ids, attention_mask=attention_mask, target_layer=target_layer)
            model(input_ids=input_ids, attention_mask=attention_mask)

        layer_attn = model.model.layers[target_layer].self_attn
        inp = layer_attn.pre_o_proj_input.cpu().reshape(-1, layer_attn.pre_o_proj_input.shape[-1])
        extracted_inputs.append(inp)

    return torch.cat(extracted_inputs, dim=0)

def extract_o_data(
    model,
    dataset,
    target_layer,
    replaced_layers,
    rotor_path,
    rotor_ckpt,
    output_path,
    batch_size,
    replacement_type,
    input_dim,
    value_output_dim,
    device,
    dtype=torch.float32
):

    # --- Split Tokenized Data ---
    train_data, test_data = split_data(dataset)

    # --- Extract True/Original QKV Outputs ---
    # print("ProjO output model\n", model)
    for dir, split in zip(['/train', '/test'], [train_data, test_data]):
        get_o_output(
            model=model,
            dataset=split,
            output_dir=output_path+dir,
            target_layer=target_layer,
            device=device,
            batch_size=batch_size,
        )

    y_train = torch.load(f"{output_path}/train/layer{target_layer}/y_layer{target_layer}_output.pt", weights_only=True)
    y_test  = torch.load(f"{output_path}/test/layer{target_layer}/y_layer{target_layer}_output.pt", weights_only=True)

    # --- Extract Inputs ---
    loader_fn = get_loader_fn(replacement_type)
    rotor_nets = loader_fn(
            replaced_layers,
            input_dim=input_dim,
            value_output_dim=value_output_dim,
            model_dir=rotor_path,
            model_ckpt=rotor_ckpt,
            device=device,
            dtype=dtype)

    # Patch the attention module with the previous layers with the updated model
    for layer_name in replaced_layers:
        idx = int(layer_name.replace("layer", ""))
        if idx < target_layer:
            model.model.layers[idx].self_attn.q_proj = rotor_nets[layer_name]["query"]
            model.model.layers[idx].self_attn.k_proj = rotor_nets[layer_name]["key"]
            model.model.layers[idx].self_attn.v_proj = rotor_nets[layer_name]["value"]
            model.model.layers[idx].self_attn.o_proj = load_proj_o_model(dim=input_dim, output_path=f"{rotor_path}/{layer_name}/output", ckpt=rotor_ckpt, device=device)

    # Patch the attention module at the target layer with the updated model
    layer = model.model.layers[target_layer]
    old_attn = layer.self_attn
    custom_attn = get_custom_attention(model.config.model_type, old_attn.config, old_attn.layer_idx)
    custom_attn.load_state_dict(old_attn.state_dict())
    layer.self_attn = custom_attn.to(device)
    layer.self_attn.q_proj = rotor_nets[f"layer{target_layer}"]["query"]
    layer.self_attn.k_proj = rotor_nets[f"layer{target_layer}"]["key"]
    layer.self_attn.v_proj = rotor_nets[f"layer{target_layer}"]["value"]

    # print("ProjO input model\n", model)
    xs = []
    for dir, split in zip(['/train', '/test'], [train_data, test_data]):
        data = get_o_input(
            target_layer=target_layer,
            model=model,
            dataset=split,
            device=device,
            batch_size=batch_size
        )
        xs.append(data)

    return xs[0], y_train, xs[1], y_test

from datasets import load_dataset
SEED = 0
torch.manual_seed(SEED)

def get_dataset(dataset_name, llm_batch_size, tokenizer, token):
    if dataset_name == "arc_challenge":
        raw_dataset = load_dataset(
            "meta-llama/Llama-3.2-1B-Instruct-evals",
            "Llama-3.2-1B-Instruct-evals__arc_challenge__details",
            token=token
        )["latest"]

        flat_inputs = [p[0] if isinstance(p, list) else p for p in raw_dataset["input_final_prompts"]]
        dataset = tokenizer(flat_inputs, padding=True, return_tensors="pt")
        metric = "accuracy"
        batch_size = llm_batch_size

    elif dataset_name == "hellaswag_chat":
        raw_dataset = load_dataset(
            "meta-llama/Llama-3.2-1B-Instruct-evals",
            "Llama-3.2-1B-Instruct-evals__hellaswag_chat__details",
            token=token
        )["latest"]
        raw_dataset = raw_dataset.select(range(1500))
        flat_inputs = [p[0] if isinstance(p, list) else p for p in raw_dataset["input_final_prompts"]]
        dataset = tokenizer(flat_inputs, padding=True, return_tensors="pt")
        metric = "accuracy"
        batch_size = llm_batch_size

    elif dataset_name == "gpqa":
        raw_dataset = load_dataset(
            "meta-llama/Llama-3.2-1B-Instruct-evals",
            "Llama-3.2-1B-Instruct-evals__gpqa__details",
            token=token
        )["latest"]

        raw_dataset = raw_dataset.select(range(300))
        flat_inputs = [p[0] if isinstance(p, list) else p for p in raw_dataset["input_final_prompts"]]
        dataset = tokenizer(flat_inputs, padding=True, return_tensors="pt")
        metric = "accuracy"
        batch_size = llm_batch_size

    elif dataset_name == "wikitext":
        raw_dataset = load_dataset("wikitext", "wikitext-2-v1", token=token)
        test_text = "\n\n".join(raw_dataset["test"]["text"])
        encodings = tokenizer(test_text, return_tensors="pt")["input_ids"][0]  # shape: [seq_len]

        # Create overlapping 512-token chunks with stride 256
        max_length = 512
        stride = 256
        input_ids, attention_masks = [], []

        for i in range(0, len(encodings) - max_length + 1, stride):
            chunk = encodings[i:i + max_length]
            input_ids.append(chunk)
            attention_masks.append(torch.ones_like(chunk))  # all tokens are valid

        dataset = {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_masks)
        }
        metric = "perplexity"
        batch_size = llm_batch_size

    elif dataset_name == "ptb":
        raw_dataset = load_dataset("ptb_text_only", trust_remote_code=True, token=token)
        sentences = raw_dataset["test"]["sentence"]
        test_text = " ".join(sentences)
        encodings = tokenizer(test_text, return_tensors="pt")["input_ids"][0]

        max_length = 512
        stride = 256
        input_ids, attention_masks = [], []

        for i in range(0, len(encodings) - max_length + 1, stride):
            chunk = encodings[i:i + max_length]
            input_ids.append(chunk)
            attention_masks.append(torch.ones_like(chunk))  # all tokens are valid

        dataset = {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_masks)
        }
        metric = "perplexity"
        batch_size = llm_batch_size

    elif dataset_name == "c4":
        raw_dataset = None
        en = load_dataset("allenai/c4", "en", split="validation", streaming=True, token=token)
        en_subset= list(en.take(400))
        test_text = " ".join(example["text"] for example in en_subset if example["text"].strip())
        encodings = tokenizer(test_text, return_tensors="pt")["input_ids"][0]

        max_length = 512
        stride = 256
        input_ids, attention_masks = [], []

        for i in range(0, len(encodings) - max_length + 1, stride):
            chunk = encodings[i:i + max_length]
            input_ids.append(chunk)
            attention_masks.append(torch.ones_like(chunk))  # all tokens are valid

        dataset = {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_masks)
        }
        metric = "perplexity"
        batch_size = llm_batch_size

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return raw_dataset, dataset, metric, batch_size