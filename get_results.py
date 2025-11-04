import os

base_dir = "/data/result/fox/out/hellaswag_chat"
metric = "accuracy"
# metric= "perplexity"
layers = sorted([d for d in os.listdir(base_dir) if d.startswith("layer")])

results = {}

print(base_dir)
baselines = ["rotor", "lowrank_linear1", "lowrank_linear4", "bh_linear"]
for layer in layers:
    layer_path = os.path.join(base_dir, layer)
    results[layer] = {}
    for method in baselines:
        offset = 0
        method_for_path = method
        if method == "lowrank_linear1" or method == "lowrank_linear4":
            method_for_path = "lowrank_linear"
        if method == "lowrank_linear4":
            offset = 3
        acc_path = os.path.join(layer_path, method_for_path, f"{metric}.txt")
        try:
            with open(acc_path, "r") as f:
                line = f.read().strip()
                parts = line.split("\t")
                # print(parts)
                if len(parts) >= 2:
                    accuracy = float(parts[-2 + offset])
                    results[layer][method] = accuracy
                else:
                    results[layer][method] = None
        except FileNotFoundError:
            results[layer][method] = None

# Print the result
for layer in sorted(results.keys(), key=lambda x: int(x.replace("layer", ""))):

    if metric == "accuracy":
        for method in baselines:
            if results[layer].get(method) is not None:
                results[layer][method] *= 100

    rotor = results[layer].get("rotor")
    bh = results[layer].get("bh_linear")
    l4 = results[layer].get("lowrank_linear4")
    l1 = results[layer].get("lowrank_linear1")

    print(f'{layer}: Rotor = {rotor:.2f}' if rotor is not None else f'{layer}: Rotor = -', end=', ')
    print(f'bh = {bh:.2f}' if bh is not None else 'bh = -', end=', ')
    print(f'LowRank4 = {l4:.2f}' if l4 is not None else 'LowRank4 = -', end=', ')
    print(f'LowRank1 = {l1:.2f}' if l1 is not None else 'LowRank1 = -')

for layer in sorted(results.keys(), key=lambda x: int(x.replace("layer", ""))):

    rotor = results[layer].get("rotor")
    bh = results[layer].get("bh_linear")
    l4 = results[layer].get("lowrank_linear4")
    l1 = results[layer].get("lowrank_linear1")

    print(f"{rotor:.2f}" if rotor is not None else "-", end="\t")
    print(f"{bh:.2f}" if bh is not None else "-", end="\t")
    print(f"{l4:.2f}" if l4 is not None else "-", end="\t")
    print(f"{l1:.2f}" if l1 is not None else "-")