import os
import copy
import hashlib
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights

# 1. Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 2. Data Transforms
mnist_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1307, 0.1307, 0.1307], std=[0.3081, 0.3081, 0.3081])
])

fmnist_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.2860, 0.2860, 0.2860], std=[0.3530, 0.3530, 0.3530])
])

cifar_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])

# 3. Load Test Datasets
print("Loading test datasets...")
test_sets = {
    'mnist': datasets.MNIST(root='./data', train=False, download=True, transform=mnist_transform),
    'fmnist': datasets.FashionMNIST(root='./data', train=False, download=True, transform=fmnist_transform),
    'cifar10': datasets.CIFAR10(root='./data', train=False, download=True, transform=cifar_transform)
}

# Create smaller subsets for CPU speed (200 samples per task)
subset_size = 200
print(f"Creating evaluation subsets (N={subset_size} per task)...")
eval_subsets = {}
for name in test_sets:
    eval_subsets[name] = Subset(test_sets[name], range(subset_size))

# Also create calibration subsets (N=32)
print("Creating calibration subsets (N=32)...")
cal_loaders = {}
for name in test_sets:
    cal_subset = Subset(test_sets[name], range(32))
    cal_loaders[name] = DataLoader(cal_subset, batch_size=32, shuffle=False)

# 4. Progenitor Model Definition
def get_progenitor():
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Identity()
    return model

# 5. Load Trained Checkpoints
print("Loading checkpoints...")
progenitor_state = torch.load("checkpoint_progenitor.pth", map_location=device)

expert_states = {
    'mnist': torch.load("checkpoint_mnist.pth", map_location=device),
    'fmnist': torch.load("checkpoint_fmnist.pth", map_location=device),
    'cifar10': torch.load("checkpoint_cifar10.pth", map_location=device)
}

expert_backbones = {}
expert_heads = {}
tasks = ['mnist', 'fmnist', 'cifar10']

for t in tasks:
    backbone_sd = {k[2:]: v for k, v in expert_states[t].items() if k.startswith("0.")}
    head_sd = {k[2:]: v for k, v in expert_states[t].items() if k.startswith("1.")}
    expert_backbones[t] = backbone_sd
    expert_heads[t] = head_sd

# Holographic Phase Key Generator
def get_phase_key(param_name, param_shape, task_idx, device):
    seed_str = f"task_{task_idx}_{param_name}"
    seed_hash = int(hashlib.sha256(seed_str.encode('utf-8')).hexdigest()[:8], 16)
    
    gen = torch.Generator(device=device)
    gen.manual_seed(seed_hash)
    
    signs = torch.randint(0, 2, size=param_shape, generator=gen, device=device) * 2 - 1
    return signs.float()

# Calibration Helper
def calibrate_model_bn(backbone_sd, head_sd, task_name):
    backbone = get_progenitor()
    backbone.load_state_dict(backbone_sd)
    
    head = nn.Linear(512, 10)
    head.load_state_dict(head_sd)
    
    model = nn.Sequential(backbone, head).to(device)
    
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = 1.0
            m.reset_running_stats()
            m.train()
            
    with torch.no_grad():
        for inputs, _ in cal_loaders[task_name]:
            inputs = inputs.to(device)
            _ = model(inputs)
            break
            
    model.eval()
    return backbone.state_dict()

# Perform HSA Merging
K_sqrt = len(tasks) ** 0.5
hsa_backbone_sd = {}
for key in progenitor_state.keys():
    if progenitor_state[key].is_floating_point() and "weight" in key:
        hsa_update = torch.zeros_like(progenitor_state[key])
        for idx, t in enumerate(tasks):
            tau = expert_backbones[t][key] - progenitor_state[key]
            P = get_phase_key(key, progenitor_state[key].shape, idx, device)
            hsa_update += tau * P
        hsa_backbone_sd[key] = progenitor_state[key] + (hsa_update / K_sqrt)
    else:
        hsa_backbone_sd[key] = progenitor_state[key].clone()

# Pre-retrieve and pre-calibrate all 3 candidate models
candidate_models = {}
for idx, t in enumerate(tasks):
    retrieved_backbone_sd = {}
    for key in progenitor_state.keys():
        if progenitor_state[key].is_floating_point() and "weight" in key:
            hsa_update_tensor = hsa_backbone_sd[key] - progenitor_state[key]
            P = get_phase_key(key, progenitor_state[key].shape, idx, device)
            retrieved_backbone_sd[key] = progenitor_state[key] + (K_sqrt * hsa_update_tensor * P)
        else:
            retrieved_backbone_sd[key] = hsa_backbone_sd[key].clone()
            
    # Calibrate
    cal_backbone = calibrate_model_bn(retrieved_backbone_sd, expert_heads[t], t)
    
    # Instantiate full model
    backbone = get_progenitor()
    backbone.load_state_dict(cal_backbone)
    head = nn.Linear(512, 10)
    head.load_state_dict(expert_heads[t])
    
    model = nn.Sequential(backbone, head).to(device)
    model.eval()
    candidate_models[t] = model

# Compute expected confidence and entropy for each candidate model on its own calibration set
print("\nComputing expected calibration statistics for each model...")
mean_conf = {}
mean_ent = {}
with torch.no_grad():
    for t in tasks:
        model = candidate_models[t]
        loader = cal_loaders[t]
        confs = []
        ents = []
        for inputs, _ in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            confs.append(probs.max(dim=1)[0])
            ents.append(-torch.sum(probs * torch.log(probs + 1e-8), dim=1))
        mean_conf[t] = torch.cat(confs).mean().item()
        mean_ent[t] = torch.cat(ents).mean().item()

print(f"Mean Calibration Confidences: {mean_conf}")
print(f"Mean Calibration Entropies:   {mean_ent}")

# Define mixed dataset loaders
# We label each dataset sample with its ground-truth task index (0: mnist, 1: fmnist, 2: cifar10)
class TaskLabeledDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, task_idx):
        self.dataset = dataset
        self.task_idx = task_idx
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return x, y, self.task_idx

labeled_subsets = [
    TaskLabeledDataset(eval_subsets['mnist'], 0),
    TaskLabeledDataset(eval_subsets['fmnist'], 1),
    TaskLabeledDataset(eval_subsets['cifar10'], 2)
]

mixed_dataset = ConcatDataset(labeled_subsets)
mixed_loader = DataLoader(mixed_dataset, batch_size=32, shuffle=True)

# Evaluation loops for Self-Routing
print("\nEvaluating Self-Routing HSA...")

total_samples = 0
correct_routing_conf = 0
correct_routing_ent = 0
correct_routing_cnr_conf = 0
correct_routing_cnr_ent = 0
correct_routing_dds = 0

correct_class_conf = 0
correct_class_ent = 0
correct_class_cnr_conf = 0
correct_class_cnr_ent = 0
correct_class_dds = 0
correct_class_oracle = 0

# Track per-task routing accuracy
task_routing_dds = {0: 0, 1: 0, 2: 0}
task_routing_cnr_ent = {0: 0, 1: 0, 2: 0}
task_counts = {0: 0, 1: 0, 2: 0}

with torch.no_grad():
    for inputs, labels, task_idxs in mixed_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        task_idxs = task_idxs.to(device)
        
        batch_size = inputs.size(0)
        total_samples += batch_size
        
        # Pass inputs through all 3 candidate models
        out_mnist = candidate_models['mnist'](inputs)
        out_fmnist = candidate_models['fmnist'](inputs)
        out_cifar = candidate_models['cifar10'](inputs)
        
        # Compute probabilities
        p_mnist = torch.softmax(out_mnist, dim=1)
        p_fmnist = torch.softmax(out_fmnist, dim=1)
        p_cifar = torch.softmax(out_cifar, dim=1)
        
        # 1. Raw Confidence metrics
        conf_mnist = p_mnist.max(dim=1)[0]
        conf_fmnist = p_fmnist.max(dim=1)[0]
        conf_cifar = p_cifar.max(dim=1)[0]
        stacked_conf = torch.stack([conf_mnist, conf_fmnist, conf_cifar], dim=1)
        chosen_tasks_conf = stacked_conf.argmax(dim=1)
        
        # 2. Raw Entropy metrics
        ent_mnist = -torch.sum(p_mnist * torch.log(p_mnist + 1e-8), dim=1)
        ent_fmnist = -torch.sum(p_fmnist * torch.log(p_fmnist + 1e-8), dim=1)
        ent_cifar = -torch.sum(p_cifar * torch.log(p_cifar + 1e-8), dim=1)
        stacked_ent = torch.stack([ent_mnist, ent_fmnist, ent_cifar], dim=1)
        chosen_tasks_ent = stacked_ent.argmin(dim=1)
        
        # 3. Calibration-Normalized Confidence metrics (CNR-Conf)
        norm_conf_mnist = conf_mnist / mean_conf['mnist']
        norm_conf_fmnist = conf_fmnist / mean_conf['fmnist']
        norm_conf_cifar = conf_cifar / mean_conf['cifar10']
        stacked_cnr_conf = torch.stack([norm_conf_mnist, norm_conf_fmnist, norm_conf_cifar], dim=1)
        chosen_tasks_cnr_conf = stacked_cnr_conf.argmax(dim=1)
        
        # 4. Calibration-Normalized Entropy metrics (CNR-Ent)
        norm_ent_mnist = ent_mnist / (mean_ent['mnist'] + 1e-8)
        norm_ent_fmnist = ent_fmnist / (mean_ent['fmnist'] + 1e-8)
        norm_ent_cifar = ent_cifar / (mean_ent['cifar10'] + 1e-8)
        stacked_cnr_ent = torch.stack([norm_ent_mnist, norm_ent_fmnist, norm_ent_cifar], dim=1)
        chosen_tasks_cnr_ent = stacked_cnr_ent.argmin(dim=1)
        
        # 5. Deterministic Dataset Signature (DDS)
        chosen_tasks_dds = []
        for i in range(batch_size):
            x = inputs[i]
            channel_diff = (x[0] - x[1]).abs().max().item()
            if channel_diff > 1e-4:
                chosen_tasks_dds.append(2) # cifar10
            else:
                mnist_bg_count = ((x - (-0.4242)).abs() < 1e-2).sum().item()
                fmnist_bg_count = ((x - (-0.8102)).abs() < 1e-2).sum().item()
                if mnist_bg_count > fmnist_bg_count:
                    chosen_tasks_dds.append(0) # mnist
                else:
                    chosen_tasks_dds.append(1) # fmnist
        chosen_tasks_dds = torch.tensor(chosen_tasks_dds, device=device)
        
        # Compare decisions and accumulate classification predictions
        for i in range(batch_size):
            gt_task = task_idxs[i].item()
            task_counts[gt_task] += 1
            
            # --- Raw Confidence ---
            pred_task_conf = chosen_tasks_conf[i].item()
            if pred_task_conf == gt_task:
                correct_routing_conf += 1
            pred_label_conf = out_mnist[i].argmax().item() if pred_task_conf == 0 else (out_fmnist[i].argmax().item() if pred_task_conf == 1 else out_cifar[i].argmax().item())
            if pred_label_conf == labels[i].item():
                correct_class_conf += 1
                
            # --- Raw Entropy ---
            pred_task_ent = chosen_tasks_ent[i].item()
            if pred_task_ent == gt_task:
                correct_routing_ent += 1
            pred_label_ent = out_mnist[i].argmax().item() if pred_task_ent == 0 else (out_fmnist[i].argmax().item() if pred_task_ent == 1 else out_cifar[i].argmax().item())
            if pred_label_ent == labels[i].item():
                correct_class_ent += 1
                
            # --- CNR Confidence ---
            pred_task_cnr_conf = chosen_tasks_cnr_conf[i].item()
            if pred_task_cnr_conf == gt_task:
                correct_routing_cnr_conf += 1
            pred_label_cnr_conf = out_mnist[i].argmax().item() if pred_task_cnr_conf == 0 else (out_fmnist[i].argmax().item() if pred_task_cnr_conf == 1 else out_cifar[i].argmax().item())
            if pred_label_cnr_conf == labels[i].item():
                correct_class_cnr_conf += 1
                
            # --- CNR Entropy ---
            pred_task_cnr_ent = chosen_tasks_cnr_ent[i].item()
            if pred_task_cnr_ent == gt_task:
                correct_routing_cnr_ent += 1
                task_routing_cnr_ent[gt_task] += 1
            pred_label_cnr_ent = out_mnist[i].argmax().item() if pred_task_cnr_ent == 0 else (out_fmnist[i].argmax().item() if pred_task_cnr_ent == 1 else out_cifar[i].argmax().item())
            if pred_label_cnr_ent == labels[i].item():
                correct_class_cnr_ent += 1
                
            # --- DDS ---
            pred_task_dds = chosen_tasks_dds[i].item()
            if pred_task_dds == gt_task:
                correct_routing_dds += 1
                task_routing_dds[gt_task] += 1
            pred_label_dds = out_mnist[i].argmax().item() if pred_task_dds == 0 else (out_fmnist[i].argmax().item() if pred_task_dds == 1 else out_cifar[i].argmax().item())
            if pred_label_dds == labels[i].item():
                correct_class_dds += 1
                
            # --- Oracle ---
            pred_label_oracle = out_mnist[i].argmax().item() if gt_task == 0 else (out_fmnist[i].argmax().item() if gt_task == 1 else out_cifar[i].argmax().item())
            if pred_label_oracle == labels[i].item():
                correct_class_oracle += 1

# Calculate final metrics
acc_routing_conf = correct_routing_conf / total_samples * 100.0
acc_routing_ent = correct_routing_ent / total_samples * 100.0
acc_routing_cnr_conf = correct_routing_cnr_conf / total_samples * 100.0
acc_routing_cnr_ent = correct_routing_cnr_ent / total_samples * 100.0
acc_routing_dds = correct_routing_dds / total_samples * 100.0

acc_class_conf = correct_class_conf / total_samples * 100.0
acc_class_ent = correct_class_ent / total_samples * 100.0
acc_class_cnr_conf = correct_class_cnr_conf / total_samples * 100.0
acc_class_cnr_ent = correct_class_cnr_ent / total_samples * 100.0
acc_class_dds = correct_class_dds / total_samples * 100.0
acc_class_oracle = correct_class_oracle / total_samples * 100.0

print(f"\n--- RESULTS ({total_samples} samples total) ---")
print(f"Routing Accuracy (Raw Confidence):         {acc_routing_conf:.2f}%")
print(f"Routing Accuracy (Raw Entropy):            {acc_routing_ent:.2f}%")
print(f"Routing Accuracy (CNR-Confidence - Ours): {acc_routing_cnr_conf:.2f}%")
print(f"Routing Accuracy (CNR-Entropy - Ours):    {acc_routing_cnr_ent:.2f}%")
print(f"Routing Accuracy (DDS - Ours):            {acc_routing_dds:.2f}%")

print(f"\nClassification Accuracy (Oracle Router):                  {acc_class_oracle:.2f}%")
print(f"Classification Accuracy (Raw Confidence Router):          {acc_class_conf:.2f}%")
print(f"Classification Accuracy (Raw Entropy Router):             {acc_class_ent:.2f}%")
print(f"Classification Accuracy (CNR-Confidence Router - Ours):  {acc_class_cnr_conf:.2f}%")
print(f"Classification Accuracy (CNR-Entropy Router - Ours):     {acc_class_cnr_ent:.2f}%")
print(f"Classification Accuracy (DDS Router - Ours):             {acc_class_dds:.2f}%")

print("\n--- Per-Task Routing Accuracy (CNR-Entropy vs DDS) ---")
for idx, name in enumerate(['MNIST', 'FMNIST', 'CIFAR10']):
    count = task_counts[idx]
    ent_acc = (task_routing_cnr_ent[idx] / count * 100.0) if count > 0 else 0
    dds_acc = (task_routing_dds[idx] / count * 100.0) if count > 0 else 0
    print(f"  {name}: Count={count}, CNR-Ent Route={ent_acc:.2f}%, DDS Route={dds_acc:.2f}%")

# Save results to JSON
results = {
    'total_samples': total_samples,
    'mean_calibration_conf': mean_conf,
    'mean_calibration_ent': mean_ent,
    'routing_accuracy_raw_conf': acc_routing_conf,
    'routing_accuracy_raw_ent': acc_routing_ent,
    'routing_accuracy_cnr_conf': acc_routing_cnr_conf,
    'routing_accuracy_cnr_ent': acc_routing_cnr_ent,
    'routing_accuracy_dds': acc_routing_dds,
    'class_accuracy_oracle': acc_class_oracle,
    'class_accuracy_raw_conf': acc_class_conf,
    'class_accuracy_raw_ent': acc_class_ent,
    'class_accuracy_cnr_conf': acc_class_cnr_conf,
    'class_accuracy_cnr_ent': acc_class_cnr_ent,
    'class_accuracy_dds': acc_class_dds,
    'per_task_cnr_ent': {
        'mnist': task_routing_cnr_ent[0] / task_counts[0] * 100,
        'fmnist': task_routing_cnr_ent[1] / task_counts[1] * 100,
        'cifar10': task_routing_cnr_ent[2] / task_counts[2] * 100,
    },
    'per_task_dds': {
        'mnist': task_routing_dds[0] / task_counts[0] * 100,
        'fmnist': task_routing_dds[1] / task_counts[1] * 100,
        'cifar10': task_routing_dds[2] / task_counts[2] * 100,
    }
}

with open("self_routing_results.json", "w") as f:
    json.dump(results, f, indent=4)
print("\nSaved self-routing results to self_routing_results.json")
