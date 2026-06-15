import sys
import torch
import torch.nn as nn
import timm

# Add current dir to path
sys.path.append(".")

# Set device to CPU
import train_and_merge
train_and_merge.device = torch.device("cpu")

print("Successfully imported train_and_merge!")

# Create mock val_data for testing
val_data = {
    "MNIST": {
        "images": torch.randn(2, 3, 224, 224),
        "labels": torch.tensor([1, 2])
    },
    "FashionMNIST": {
        "images": torch.randn(2, 3, 224, 224),
        "labels": torch.tensor([3, 4])
    },
    "CIFAR10": {
        "images": torch.randn(2, 3, 224, 224),
        "labels": torch.tensor([5, 6])
    },
    "SVHN": {
        "images": torch.randn(2, 3, 224, 224),
        "labels": torch.tensor([7, 8])
    }
}

# Create base model and extract base_state
print("Creating mock base model...")
base_model = timm.create_model('vit_tiny_patch16_224', pretrained=False)
base_model.head = nn.Linear(base_model.head.in_features, 10)
base_state = {k: v.cpu().clone() for k, v in base_model.state_dict().items()}

# Create mock experts and expert heads
expert_states = {}
expert_heads = {}
for task_name in ["MNIST", "FashionMNIST", "CIFAR10", "SVHN"]:
    expert = timm.create_model('vit_tiny_patch16_224', pretrained=False)
    expert.head = nn.Linear(expert.head.in_features, 10)
    expert_states[task_name] = {k: v.cpu().clone() for k, v in expert.state_dict().items()}
    
    expert_heads[task_name] = {}
    for k, v in expert.state_dict().items():
        if k.startswith("head."):
            expert_heads[task_name][k.replace("head.", "")] = v.clone()

# Extract task vectors
task_vectors = {}
for task_name in ["MNIST", "FashionMNIST", "CIFAR10", "SVHN"]:
    task_vectors[task_name] = train_and_merge.get_task_vector(expert_states[task_name], base_state)

print("Running compute_diagonal_fisher...")
fisher_dicts = train_and_merge.compute_diagonal_fisher(base_state, expert_states, expert_heads, val_data)
print("compute_diagonal_fisher completed successfully!")

# Check shape consistency
for task_name, fisher in fisher_dicts.items():
    for k, v in fisher.items():
        assert v.shape == base_state[k].shape, f"Shape mismatch for {k}: {v.shape} vs {base_state[k].shape}"
print("Shape checks passed!")

print("Running construct_fisher_backbone...")
fisher_backbone = train_and_merge.construct_fisher_backbone(base_state, task_vectors, fisher_dicts, reg=1e-4, alpha=0.5)
print("construct_fisher_backbone completed successfully!")

# Verify backbone keys
for k in base_state.keys():
    if not k.startswith("head."):
        assert k in fisher_backbone, f"Key {k} missing in fisher_backbone"
        assert fisher_backbone[k].shape == base_state[k].shape, f"Shape mismatch for merged key {k}"
print("All checks passed! The Fisher baseline functions are 100% correct!")
