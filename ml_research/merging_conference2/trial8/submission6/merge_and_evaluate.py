import os
import argparse
import json
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from train import get_dataloader

def calculate_cosine_similarity(t1, t2):
    t1_flat = t1.view(-1)
    t2_flat = t2.view(-1)
    norm1 = t1_flat.norm(2)
    norm2 = t2_flat.norm(2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return torch.dot(t1_flat, t2_flat) / (norm1 * norm2)

def get_layer_updates(expert_state, base_state):
    updates = {}
    for key in expert_state.keys():
        if "fc" in key:  # Skip classification head
            continue
        # Only compute updates for parameters (weights, biases, BN weights, BN biases)
        if expert_state[key].dtype == torch.float32:
            updates[key] = expert_state[key] - base_state[key]
    return updates

def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100.0 * correct / total

def main():
    parser = argparse.ArgumentParser(description="Merge Experts and Evaluate")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bn_mode", type=str, default="uniform", choices=["uniform", "expert"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = False

    # Load base model progenitor
    weights = ResNet18_Weights.IMAGENET1K_V1
    base_model = resnet18(weights=weights)
    base_state = base_model.state_dict()

    # Load expert checkpoints
    expert_names = ["mnist", "fmnist", "cifar10"]
    expert_states = {}
    expert_accuracies = {}
    
    for name in expert_names:
        path = f"checkpoints/{name}_epochs{args.epochs}_wd{args.weight_decay}_seed{args.seed}.pth"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Expert checkpoint not found at {path}. Please train experts first.")
        ckpt = torch.load(path, map_location="cpu")
        expert_states[name] = ckpt["state_dict"]
        expert_accuracies[name] = ckpt["test_acc"]
        print(f"Loaded {name} expert (Individual test accuracy: {ckpt['test_acc']:.2f}%)")

    # Compute task updates
    print("\nComputing task updates relative to ImageNet base progenitor...")
    updates = {name: get_layer_updates(expert_states[name], base_state) for name in expert_names}

    # Measure cosine similarity between updates
    print("\nMeasuring task update pairwise cosine similarities across layers:")
    pairs = [("mnist", "fmnist"), ("mnist", "cifar10"), ("fmnist", "cifar10")]
    similarities = {f"{p1}_{p2}": [] for p1, p2 in pairs}
    
    for key in updates["mnist"].keys():
        # Only compute for weight parameters
        if "weight" in key:
            row_str = f"Layer {key:50s} -> "
            for p1, p2 in pairs:
                sim = calculate_cosine_similarity(updates[p1][key], updates[p2][key]).item()
                similarities[f"{p1}_{p2}"].append(sim)
                row_str += f"{p1}-{p2}: {sim:+.4f}  "
            # print(row_str)

    # Average similarities
    avg_similarities = {}
    print("\nAverage Cosine Similarity between Expert updates:")
    for p1, p2 in pairs:
        avg_sim = sum(similarities[f"{p1}_{p2}"]) / len(similarities[f"{p1}_{p2}"])
        avg_similarities[f"{p1}_{p2}"] = avg_sim
        print(f"  {p1} vs {p2}: {avg_sim:+.4f}")

    # Create dataloaders for evaluation
    print("\nLoading test datasets for evaluation...")
    dataloaders = {name: get_dataloader(name, batch_size=256, is_train=False) for name in expert_names}

    # Standard model for evaluation
    eval_model = resnet18()
    eval_model.fc = nn.Linear(512, 10)
    eval_model.to(device)

    # Dictionary to collect all results
    results = {}

    def run_eval(merged_state, title):
        accs = {}
        for name in expert_names:
            # Load the classification head of this specific expert
            eval_state = {k: v for k, v in merged_state.items()}
            eval_state["fc.weight"] = expert_states[name]["fc.weight"]
            eval_state["fc.bias"] = expert_states[name]["fc.bias"]
            
            # Apply BatchNorm statistics depending on bn_mode
            if args.bn_mode == "expert":
                # Inject this expert's original running statistics
                for k in expert_states[name].keys():
                    if "running_mean" in k or "running_var" in k or "num_batches_tracked" in k:
                        eval_state[k] = expert_states[name][k]

            eval_model.load_state_dict(eval_state)
            acc = evaluate_model(eval_model, dataloaders[name], device)
            accs[name] = acc
        avg_acc = sum(accs.values()) / len(accs)
        print(f"  {title:35s} -> MNIST: {accs['mnist']:.2f}% | FMNIST: {accs['fmnist']:.2f}% | CIFAR10: {accs['cifar10']:.2f}% | Avg: {avg_acc:.2f}%")
        results[title] = {**accs, "avg": avg_acc}

    # Set up BatchNorm running statistics merging
    # Uniform statistics merging: average running mean and variance
    uniform_bn_state = {}
    for key in base_state.keys():
        if "running_mean" in key or "running_var" in key:
            uniform_bn_state[key] = sum(expert_states[name][key] for name in expert_names) / len(expert_names)
        elif "num_batches_tracked" in key:
            uniform_bn_state[key] = expert_states["mnist"][key] # standard copy

    # Let's evaluate the baselines and merged models
    print(f"\nEvaluating with BatchNorm Mode: {args.bn_mode.upper()}")

    # 1. Weight Averaging (WA)
    wa_state = {}
    for key in base_state.keys():
        if "fc" in key:
            continue
        if base_state[key].dtype == torch.float32:
            wa_state[key] = sum(expert_states[name][key] for name in expert_names) / len(expert_names)
        else:
            wa_state[key] = expert_states["mnist"][key] # Copy non-float params (e.g. buffers)

    if args.bn_mode == "uniform":
        for k, v in uniform_bn_state.items():
            wa_state[k] = v
    run_eval(wa_state, "Weight Averaging (WA)")

    # 2. Task Arithmetic (TA) with sweeps
    for lmbda in [0.3, 0.5, 0.7, 1.0]:
        ta_state = {}
        for key in base_state.keys():
            if "fc" in key:
                continue
            if base_state[key].dtype == torch.float32:
                # W_merged = W_base + lmbda * sum(T_i)
                sum_updates = sum(updates[name][key] for name in expert_names)
                ta_state[key] = base_state[key] + lmbda * sum_updates
            else:
                ta_state[key] = expert_states["mnist"][key]

        if args.bn_mode == "uniform":
            for k, v in uniform_bn_state.items():
                ta_state[k] = v
        run_eval(ta_state, f"Task Arithmetic (TA, lambda={lmbda})")

    # 3. Update-level Isotropic Parameter Resonance (U-IPR)
    ipr_state = {}
    K = len(expert_names)
    for key in base_state.keys():
        if "fc" in key:
            continue
        if base_state[key].dtype == torch.float32:
            # Compute average norm of experts
            norms_experts = sum(updates[name][key].norm() for name in expert_names) / K
            # Compute norm of merged update (Weight Averaged)
            merged_update = sum(updates[name][key] for name in expert_names) / K
            norm_merged = merged_update.norm()
            
            # S_l scaling factor
            scale = norms_experts / (norm_merged + 1e-8)
            scale = torch.clamp(scale, min=0.1, max=10.0)
            
            # Rescale the merged update and add to base
            ipr_state[key] = base_state[key] + scale * merged_update
        else:
            ipr_state[key] = expert_states["mnist"][key]

    if args.bn_mode == "uniform":
        for k, v in uniform_bn_state.items():
            ipr_state[k] = v
    run_eval(ipr_state, "Update-level IPR (U-IPR)")

    # 4. Holographic Norm Scaling (HNS)
    # HNS creates task-specific scaled updates channel-by-channel
    # So we construct specialized states for each task and evaluate them individually
    hns_accs = {}
    for eval_task in expert_names:
        hns_task_state = {}
        for key in base_state.keys():
            if "fc" in key:
                continue
            if base_state[key].dtype == torch.float32 and "weight" in key and (base_state[key].dim() == 4 or base_state[key].dim() == 2):
                # Channel-wise scaling
                weight_base = base_state[key]
                merged_update = sum(updates[name][key] for name in expert_names) / K
                task_update = updates[eval_task][key]
                
                C_out = weight_base.size(0)
                reconstructed_update = torch.zeros_like(merged_update)
                
                for c in range(C_out):
                    norm_expert_c = task_update[c].norm()
                    norm_merged_c = merged_update[c].norm()
                    
                    gamma = norm_expert_c / (norm_merged_c + 1e-8)
                    gamma = torch.clamp(gamma, min=0.1, max=10.0)
                    
                    reconstructed_update[c] = gamma * merged_update[c]
                
                hns_task_state[key] = weight_base + reconstructed_update
            elif base_state[key].dtype == torch.float32:
                # Scalar scaling for 1D parameters (biases, BN weights/biases)
                norms_experts = sum(updates[name][key].norm() for name in expert_names) / K
                merged_update = sum(updates[name][key] for name in expert_names) / K
                scale = norms_experts / (merged_update.norm() + 1e-8)
                scale = torch.clamp(scale, min=0.1, max=10.0)
                hns_task_state[key] = base_state[key] + scale * merged_update
            else:
                hns_task_state[key] = expert_states["mnist"][key]

        if args.bn_mode == "uniform":
            for k, v in uniform_bn_state.items():
                hns_task_state[k] = v

        # Evaluate on the specific eval_task using its reconstructed weights
        eval_state = {k: v for k, v in hns_task_state.items()}
        eval_state["fc.weight"] = expert_states[eval_task]["fc.weight"]
        eval_state["fc.bias"] = expert_states[eval_task]["fc.bias"]
        
        if args.bn_mode == "expert":
            for k in expert_states[eval_task].keys():
                if "running_mean" in k or "running_var" in k or "num_batches_tracked" in k:
                    eval_state[k] = expert_states[eval_task][k]

        eval_model.load_state_dict(eval_state)
        acc = evaluate_model(eval_model, dataloaders[eval_task], device)
        hns_accs[eval_task] = acc

    hns_avg = sum(hns_accs.values()) / len(hns_accs)
    print(f"  {'Holographic Norm Scaling (HNS)':35s} -> MNIST: {hns_accs['mnist']:.2f}% | FMNIST: {hns_accs['fmnist']:.2f}% | CIFAR10: {hns_accs['cifar10']:.2f}% | Avg: {hns_avg:.2f}%")
    results["Holographic Norm Scaling (HNS)"] = {**hns_accs, "avg": hns_avg}

    # Save results to a file
    os.makedirs("results", exist_ok=True)
    results_path = f"results/merge_results_epochs{args.epochs}_wd{args.weight_decay}_seed{args.seed}_bn{args.bn_mode}.json"
    with open(results_path, "w") as f:
        json.dump({
            "avg_similarities": avg_similarities,
            "results": results,
            "bn_mode": args.bn_mode,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,
            "seed": args.seed
        }, f, indent=4)
    print(f"\nSaved merging results to {results_path}\n")

if __name__ == "__main__":
    main()
