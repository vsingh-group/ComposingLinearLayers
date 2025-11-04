import re
import torch
from tqdm import tqdm
import time
from torch.utils.data import random_split

from data import split_data

from data import get_loader_fn

from proj_o import load_proj_o_model

#### Accuracy ####
SEED=0
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def extract_answer_letter(text, dataset_name):
    if dataset_name in ["arc_challenge", "gpqa"]:
        match = re.search(r"The best answer is\s*([A-D])", text)
    elif dataset_name in ["hellaswag_chat"]:
        match = re.search(r"The best completion is\s*([A-D])", text)

    if match:
        return match.group(1)

    if dataset_name in ["arc_challenge", "gpqa"]:
        match = re.search(r"The best answer is\s*([A-D])\.", text)
    elif dataset_name in ["hellaswag_chat"]:
        match = re.search(r"The best completion is\s*([A-D])\.", text)

    return match.group(1) if match else None

# def extract_answer_letter(text):
#     match = re.search(r"The best answer is\s*([A-D])", text)
#     if match:
#         return match.group(1)
#     match = re.search(r"The best answer is\s*([A-D])\.", text)
#     return match.group(1) if match else None

def get_accuracy(
    model,
    tokenizer,
    test_data,
    device,
    dataset_name,
    print_model,
    baseline
):

    if print_model:
        print(model)

    rotor_correct = 0
    total_time = 0.0

    model.eval()
    if baseline:
        model.generation_config.do_sample = False
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None
    for example in tqdm(test_data):
        prompt = example["input_final_prompts"][0]
        correct = example["input_correct_responses"][0]
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            start = time.time()
            output = model.generate(
                **inputs,
                use_cache=False,
                pad_token_id=tokenizer.eos_token_id
            )
            total_time += time.time() - start

        pred = extract_answer_letter(tokenizer.decode(output[0], skip_special_tokens=True), dataset_name=dataset_name)
        correct = correct.strip('"')
        rotor_correct += (pred == correct)

    model.train()
    total = len(test_data)
    return rotor_correct / total, total_time

def run_accuracy_eval(
    model,
    tokenizer,
    dataset,
    input_dim,
    value_output_dim,
    train_projo,
    dataset_name,
    replacement_type=None,
    replaced_layers=[],
    rotor_path=None,
    rotor_ckpt=None,
    print_model=False,
    log=True,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    dtype=None
):

    # --- Split Tokenized Data ---
    # train_size = int(0.8 * len(dataset))
    # _, test_data = random_split(dataset, [train_size, len(dataset) - train_size])
    train_size = int(0.8 * len(dataset))
    generator = torch.Generator().manual_seed(SEED)
    _, test_data = random_split(dataset, [train_size, len(dataset) - train_size], generator=generator)

    # --- Load rotor nets only for replaced layers ---
    loader_fn = get_loader_fn(replacement_type)
    if loader_fn is not None:
        rotor_nets = loader_fn(
                replaced_layers,
                input_dim=input_dim,
                value_output_dim=value_output_dim,
                model_dir=rotor_path,
                model_ckpt=rotor_ckpt,
                device=device,
                dtype=dtype)


    for layer_name in replaced_layers:
        idx = int(layer_name.replace("layer", ""))
        model.model.layers[idx].self_attn.q_proj = rotor_nets[layer_name]["query"]
        model.model.layers[idx].self_attn.k_proj = rotor_nets[layer_name]["key"]
        model.model.layers[idx].self_attn.v_proj = rotor_nets[layer_name]["value"]

        if train_projo:
            model.model.layers[idx].self_attn.o_proj = load_proj_o_model(dim=input_dim, output_path=f"{rotor_path}/{layer_name}/output", ckpt=rotor_ckpt, device=device)

    acc, time = get_accuracy(
        model,
        tokenizer,
        test_data,
        device,
        dataset_name,
        print_model,
        baseline=(replaced_layers==[])
    )

    # --- Write results ---
    label = ",".join(replaced_layers) if replaced_layers else "[]"
    print(f"Accuracy for layers {label}: {acc:.4f}: {time:.4f}")
    if log:
        with open(rotor_path+'/accuracy.txt', "a") as f:
            f.write(f"{label}\t{acc:.4f}\t{time:.4f}\n")

### PPL ###
def get_perplexity(
    model,
    test_data,
    stride,
    print_model,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
):

    if print_model:
        print(model)

    model.eval()
    nll_sum = 0.0
    n_tokens = 0
    total_time = 0.0

    for idx, example in enumerate(tqdm(test_data, desc="Evaluating Perplexity")):
        input_ids = example["input_ids"].unsqueeze(0).to(device)  # [1, seq_len]
        attention_mask = example["attention_mask"].unsqueeze(0).to(device)
        labels = input_ids.clone()

        if idx > 0:
            labels[:, :-stride] = -100
        else:
            pass

        with torch.no_grad():
            start = time.time()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_time += time.time() - start
            neg_log_likelihood = outputs.loss * (labels != -100).sum()

        nll_sum += neg_log_likelihood
        n_tokens += (labels != -100).sum()

    model.train()
    ppl = torch.exp(nll_sum / n_tokens).item()
    return ppl, total_time

def run_ppl_eval(
    model,
    dataset,
    stride,
    input_dim,
    value_output_dim,
    train_projo,
    replacement_type=None,
    rank=None,
    replaced_layers=[],
    rotor_path=None,
    rotor_ckpt=None,
    print_model=False,
    log=True,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
):

    # --- Split Tokenized Data ---
    _, test_data = split_data(dataset)

    # --- Load rotor nets only for replaced layers ---
    loader_fn = get_loader_fn(replacement_type)
    if loader_fn is not None:
        rotor_nets = loader_fn(
                replaced_layers,
                input_dim=input_dim,
                value_output_dim=value_output_dim,
                model_dir=rotor_path,
                model_ckpt=rotor_ckpt,
                device=device
                )

    for layer_name in replaced_layers:
        idx = int(layer_name.replace("layer", ""))
        model.model.layers[idx].self_attn.q_proj = rotor_nets[layer_name]["query"]
        model.model.layers[idx].self_attn.k_proj = rotor_nets[layer_name]["key"]
        model.model.layers[idx].self_attn.v_proj = rotor_nets[layer_name]["value"]

        if train_projo:
            model.model.layers[idx].self_attn.o_proj = load_proj_o_model(dim=input_dim, output_path=f"{rotor_path}/{layer_name}/output", ckpt=rotor_ckpt, device=device)

    print("Inference model\n", model)

    ppl, time = get_perplexity(
        model,
        test_data,
        stride,
        print_model,
    )

    # --- Write results ---
    label = ",".join(replaced_layers) if replaced_layers else "[]"
    print(f"Perplexity for layers {label}: {ppl:.2f} ({time:.2f} sec total)")
    if log:
        with open(rotor_path + '/perplexity.txt', "a") as f:
            if rank is None:
                f.write(f"{label}\t{ppl:.2f}\t{time:.2f}\n")
            else:
                f.write(f"{label}\trank={rank}\t{ppl:.2f}\t{time:.2f}\n")
