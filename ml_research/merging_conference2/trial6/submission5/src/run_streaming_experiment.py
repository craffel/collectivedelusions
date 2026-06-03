import os
import copy
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.models import resnet18

# Import utility functions from src/eval.py
from src.eval import (
    get_test_loader, 
    convert_to_ttbc, 
    evaluate_model
)

def run_streaming_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running streaming experiment on {device}...")

    tasks = ["mnist", "fashionmnist", "cifar10"]
    limit = 500 # use 500 samples for swiftness and stable convergence

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

    # Target task for streaming: Fashion-MNIST (a challenging but stable task)
    task_idx = 1
    target_task = tasks[task_idx]
    target_head = expert_heads[task_idx]
    
    # Batch size 1 loader for online streaming
    stream_loader = get_test_loader(target_task, batch_size=1, limit=limit)

    # Configurations to evaluate
    configs = {
        "Uncalibrated": {"alpha": 0.0, "stateful": False, "color": "red", "style": "-"},
        "Stateless (Alpha=1.0)": {"alpha": 1.0, "stateful": False, "color": "orange", "style": "--"},
        "Stateless (Alpha=0.05)": {"alpha": 0.05, "stateful": False, "color": "purple", "style": "-."},
        "Stateful (Alpha=0.05)": {"alpha": 0.05, "stateful": True, "color": "green", "style": "-"}
    }

    trajectories = {}

    for name, cfg in configs.items():
        print(f"\nEvaluating configuration: {name} (Alpha={cfg['alpha']}, Stateful={cfg['stateful']})...")
        # Instantiate model
        model = copy.deepcopy(merged_backbone_wa)
        if cfg['alpha'] > 0:
            model = convert_to_ttbc(model, alpha=cfg['alpha'], stateful=cfg['stateful'])
        model = model.to(device)
        model.eval()
        target_head.eval()

        correct = 0
        total = 0
        accuracies = []

        with torch.no_grad():
            for idx, (inputs, targets) in enumerate(stream_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                
                orig_fc = model.fc
                model.fc = nn.Identity()
                feats = model(inputs)
                model.fc = orig_fc
                
                outputs = target_head(feats)
                _, predicted = outputs.max(1)
                total += 1
                if predicted.eq(targets).item():
                    correct += 1
                
                # Append running accuracy percentage
                accuracies.append((correct / total) * 100)
                
                if (idx + 1) % 100 == 0:
                    print(f"Step {idx+1}/{limit} | Running Accuracy: {accuracies[-1]:.2f}%")

        trajectories[name] = accuracies

    # Plot the results
    plt.figure(figsize=(10, 6))
    for name, accs in trajectories.items():
        cfg = configs[name]
        plt.plot(accs, label=name, color=cfg["color"], linestyle=cfg["style"], linewidth=2)
    
    # Plot baseline expert upper bound
    expert_acc_pct = 93.50 # Expert Fashion-MNIST accuracy
    plt.axhline(y=expert_acc_pct, color="black", linestyle=":", linewidth=1.5, label="Expert Oracle (93.5%)")

    plt.title("Online Streaming Accuracy Trajectory (Fashion-MNIST, B=1)", fontsize=13, fontweight="bold")
    plt.xlabel("Streaming Sequence Steps (Images Processed)", fontsize=11)
    plt.ylabel("Cumulative Test Accuracy (%)", fontsize=11)
    plt.ylim(0, 105)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend(fontsize=10, loc="lower right")
    plt.tight_layout()

    plt.savefig("models/streaming_comparison.png", dpi=300)
    plt.savefig("template/streaming_comparison.png", dpi=300)
    plt.close()
    print("\nSaved streaming comparison plot to models/streaming_comparison.png")

    # Save data to json for reference
    with open("models/streaming_results.json", "w") as f:
        json.dump(trajectories, f)
    print("Saved streaming results data to models/streaming_results.json")

if __name__ == "__main__":
    run_streaming_experiment()
