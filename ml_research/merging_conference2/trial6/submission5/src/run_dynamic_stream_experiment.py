import os
import copy
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.models import resnet18

# Import utility functions from src/eval.py
from src.eval import (
    get_dataset,
    convert_to_ttbc,
    TestTimeBatchNorm2d
)

def run_dynamic_stream_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running dynamic multi-task streaming experiment on {device}...")

    tasks = ["mnist", "fashionmnist", "cifar10"]
    limit_per_task = 200  # 200 samples per task, total 600 steps

    # Load expert backbones and heads
    expert_backbones = []
    expert_heads = []
    for task in tasks:
        path = f"models/resnet18_{task}.pt"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Expert model checkpoint {path} not found.")
        model = resnet18()
        model.fc = nn.Linear(512, 10)
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device)
        expert_backbones.append(model)
        expert_heads.append(model.fc)

    # Merge models via Weight Averaging
    print("Merging models via Weight Averaging...")
    merged_backbone_wa = resnet18()
    merged_backbone_wa.fc = nn.Linear(512, 10)
    merged_backbone_wa = merged_backbone_wa.to(device)
    
    merged_state_dict = copy.deepcopy(expert_backbones[0].state_dict())
    for key in merged_state_dict.keys():
        weights = [exp.state_dict()[key].float() for exp in expert_backbones]
        merged_state_dict[key] = torch.stack(weights, dim=0).mean(dim=0).to(merged_state_dict[key].dtype)
    merged_backbone_wa.load_state_dict(merged_state_dict)

    # Save original merged running stats for resets
    original_running_stats = {}
    for name, module in merged_backbone_wa.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            original_running_stats[name] = {
                "running_mean": module.running_mean.data.clone() if module.running_mean is not None else None,
                "running_var": module.running_var.data.clone() if module.running_var is not None else None
            }

    # Construct the sequential dynamic stream of images (B=1)
    print("Constructing multi-task sequential stream...")
    stream_data = []
    for t_idx, task in enumerate(tasks):
        dataset = get_dataset(task, train=False)
        for i in range(limit_per_task):
            img, label = dataset[i]
            # Add to stream: (image_tensor, label, task_idx, image_idx)
            stream_data.append((img, label, t_idx, i))

    # Configurations to evaluate
    configs = {
        "Uncalibrated": {"alpha": 0.0, "stateful": False, "reset_on_switch": False, "color": "red", "style": ":"},
        "Stateless (Alpha=1.0)": {"alpha": 1.0, "stateful": False, "reset_on_switch": False, "color": "orange", "style": "--"},
        "Stateful Continuous (Alpha=0.05)": {"alpha": 0.05, "stateful": True, "reset_on_switch": False, "color": "purple", "style": "-."},
        "Stateful Task-Reset (Alpha=0.05)": {"alpha": 0.05, "stateful": True, "reset_on_switch": True, "color": "green", "style": "-"}
    }

    trajectories = {}
    rolling_trajectories = {}
    window_size = 40  # Rolling window size to compute localized accuracy

    for name, cfg in configs.items():
        print(f"\nEvaluating configuration: {name}...")
        # Instantiate fresh copy of merged model
        model = copy.deepcopy(merged_backbone_wa)
        if cfg['alpha'] > 0:
            model = convert_to_ttbc(model, alpha=cfg['alpha'], stateful=cfg['stateful'])
        model = model.to(device)
        model.eval()

        # Turn off gradients
        for param in model.parameters():
            param.requires_grad = False

        correct_history = []
        accuracies = []
        rolling_accuracies = []

        correct_count = 0
        total_count = 0

        # Run streaming
        for idx, (img, label, t_idx, i_idx) in enumerate(stream_data):
            img = img.unsqueeze(0).to(device) # B=1
            label = torch.tensor([label]).to(device)
            target_head = expert_heads[t_idx]
            target_head.eval()

            # Optional reset on task transition
            if cfg["reset_on_switch"] and idx > 0 and idx % limit_per_task == 0:
                # Reset the running mean and variance of TestTimeBatchNorm2d layers to pre-saved merged stats
                print(f"Task switch detected at step {idx}. Resetting BatchNorm running stats...")
                for bn_name, bn_module in model.named_modules():
                    if isinstance(bn_module, TestTimeBatchNorm2d):
                        if bn_module.running_mean is not None:
                            bn_module.running_mean.copy_(original_running_stats[bn_name]["running_mean"])
                        if bn_module.running_var is not None:
                            bn_module.running_var.copy_(original_running_stats[bn_name]["running_var"])

            # Inference
            # Extract features (by bypassing fc layer)
            orig_fc = model.fc
            model.fc = nn.Identity()
            feats = model(img)
            model.fc = orig_fc

            # Task classification head
            outputs = target_head(feats)
            _, predicted = outputs.max(1)
            
            is_correct = predicted.eq(label).item()
            correct_history.append(1 if is_correct else 0)
            
            if is_correct:
                correct_count += 1
            total_count += 1

            # Cumulative Accuracy
            accuracies.append((correct_count / total_count) * 100)

            # Rolling Accuracy over last `window_size` steps
            recent_history = correct_history[-window_size:]
            rolling_acc = (sum(recent_history) / len(recent_history)) * 100
            rolling_accuracies.append(rolling_acc)

            if (idx + 1) % 100 == 0:
                print(f"Step {idx+1}/{len(stream_data)} | Cumulative Acc: {accuracies[-1]:.2f}% | Rolling Acc: {rolling_accuracies[-1]:.2f}%")

        trajectories[name] = accuracies
        rolling_trajectories[name] = rolling_accuracies

    # 1. Plot the Rolling Accuracy Trajectory
    plt.figure(figsize=(11, 6))
    for name, roll_accs in rolling_trajectories.items():
        cfg = configs[name]
        plt.plot(roll_accs, label=name, color=cfg["color"], linestyle=cfg["style"], linewidth=2.5)

    # Vertical lines indicating task transitions
    plt.axvline(x=200, color="gray", linestyle="--", alpha=0.7)
    plt.text(100, 102, "MNIST Stream", fontsize=10, fontweight="bold", ha="center")
    plt.axvline(x=400, color="gray", linestyle="--", alpha=0.7)
    plt.text(300, 102, "Fashion-MNIST", fontsize=10, fontweight="bold", ha="center")
    plt.text(500, 102, "CIFAR-10", fontsize=10, fontweight="bold", ha="center")

    plt.title(f"Dynamic Multi-Task Sequential Streaming Accuracy (B=1, Rolling Window={window_size})", fontsize=13, fontweight="bold")
    plt.xlabel("Streaming Sequence Steps (Images Processed)", fontsize=11)
    plt.ylabel("Rolling Accuracy (%)", fontsize=11)
    plt.ylim(0, 108)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend(fontsize=10, loc="lower left")
    plt.tight_layout()

    # Save plots
    os.makedirs("models", exist_ok=True)
    plt.savefig("models/dynamic_streaming_comparison.png", dpi=300)
    plt.savefig("template/dynamic_streaming_comparison.png", dpi=300)
    plt.close()
    print("\nSaved dynamic streaming comparison plot to models/dynamic_streaming_comparison.png")

    # Save results to JSON
    results = {
        "cumulative": trajectories,
        "rolling": rolling_trajectories
    }
    with open("models/dynamic_streaming_results.json", "w") as f:
        json.dump(results, f)
    print("Saved dynamic streaming results data to models/dynamic_streaming_results.json")

if __name__ == "__main__":
    run_dynamic_stream_experiment()
