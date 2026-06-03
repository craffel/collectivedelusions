import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from torch.func import functional_call
import copy

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Ablations using device: {device}")
if device.type == "cuda":
    torch.backends.cudnn.enabled = False
    print("Disabled cuDNN for stability on cluster nodes.")

# Define transforms
transform_rgb = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_gray = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define tasks and datasets
tasks_config = {
    "cifar10": {
        "dataset_class": datasets.CIFAR10,
        "transform": transform_rgb,
        "test_kwargs": {"download": True, "train": False}
    },
    "svhn": {
        "dataset_class": datasets.SVHN,
        "transform": transform_rgb,
        "test_kwargs": {"download": True, "split": "test"}
    },
    "mnist": {
        "dataset_class": datasets.MNIST,
        "transform": transform_gray,
        "test_kwargs": {"download": True, "train": False}
    }
}

# Load base model and experts
base_state_dict = torch.load("models/resnet18_pretrained.pt", map_location=device)

model_skeleton = models.resnet18()
model_skeleton.fc = nn.Linear(model_skeleton.fc.in_features, 10)
model_skeleton = model_skeleton.to(device)

expert_states = {}
task_vectors = {}
expert_heads = {}

# Separate backbone parameters from classification heads
for task_name in tasks_config.keys():
    state = torch.load(f"models/expert_{task_name}.pt", map_location=device)
    expert_states[task_name] = state
    
    # Extract classification head weights
    expert_heads[task_name] = {
        "fc.weight": state["fc.weight"].clone(),
        "fc.bias": state["fc.bias"].clone()
    }
    
    # Compute backbone task vectors
    task_vectors[task_name] = {}
    for name, param in state.items():
        if name != "fc.weight" and name != "fc.bias":
            task_vectors[task_name][name] = param - base_state_dict[name]

def get_merged_backbone_params(lambdas):
    merged = {}
    task_names = list(tasks_config.keys())
    for name, base_param in base_state_dict.items():
        if name != "fc.weight" and name != "fc.bias":
            val = base_param + \
                  lambdas[0] * task_vectors[task_names[0]][name] + \
                  lambdas[1] * task_vectors[task_names[1]][name] + \
                  lambdas[2] * task_vectors[task_names[2]][name]
            if "running_mean" in name or "running_var" in name or "num_batches_tracked" in name:
                val = val.detach()
            merged[name] = val
    return merged

def distillation_loss(student_logits, teacher_logits, temperature=2.0):
    soft_targets = torch.softmax(teacher_logits / temperature, dim=-1)
    log_probs = torch.log_softmax(student_logits / temperature, dim=-1)
    return -torch.sum(soft_targets * log_probs, dim=-1).mean()

def get_dataloaders(subset_size):
    loaders = {}
    for task_name, config in tasks_config.items():
        dataset = config["dataset_class"](
            root="./data", transform=config["transform"], **config["test_kwargs"]
        )
        torch.manual_seed(42)
        indices = torch.randperm(len(dataset))[:subset_size]
        subset = Subset(dataset, indices)
        loaders[task_name] = DataLoader(subset, batch_size=64, shuffle=False, num_workers=2)
    return loaders

def evaluate_model_with_loaders(lambdas, task_heads, loaders):
    merged_backbone = get_merged_backbone_params(lambdas)
    task_names = list(tasks_config.keys())
    accuracies = {}
    
    model_skeleton.eval()
    with torch.no_grad():
        for i, t_name in enumerate(task_names):
            eval_params = copy.copy(merged_backbone)
            eval_params["fc.weight"] = task_heads[t_name]["fc.weight"]
            eval_params["fc.bias"] = task_heads[t_name]["fc.bias"]
            
            correct = 0
            total = 0
            for images, labels in loaders[t_name]:
                images, labels = images.to(device), labels.to(device)
                outputs = functional_call(model_skeleton, eval_params, images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
            accuracies[t_name] = 100.0 * correct / total
            
    accuracies["average"] = sum(accuracies.values()) / len(accuracies)
    return accuracies


# --- PARAMETERIZED EXPERIMENT RUNNERS ---

def run_joint_tta(init_lambdas, loaders, lr_lambdas=0.01, lr_heads=0.01, temperature=2.0, clamp_val=0.3):
    lambdas = init_lambdas.clone().detach().requires_grad_(True)
    trainable_heads = {}
    optimizer_params = [lambdas]
    
    for t_name, head in expert_heads.items():
        t_head = {
            "fc.weight": head["fc.weight"].clone().detach().requires_grad_(True),
            "fc.bias": head["fc.bias"].clone().detach().requires_grad_(True)
        }
        trainable_heads[t_name] = t_head
        optimizer_params.extend([t_head["fc.weight"], t_head["fc.bias"]])
        
    optimizer = optim.AdamW([
        {"params": [lambdas], "lr": lr_lambdas},
        {"params": [t_head["fc.weight"] for t_head in trainable_heads.values()] + 
                   [t_head["fc.bias"] for t_head in trainable_heads.values()], "lr": lr_heads}
    ])
    
    task_names = list(tasks_config.keys())
    model_skeleton.eval()
    
    iters = {t_name: iter(loaders[t_name]) for t_name in task_names}
    max_batches = max(len(loaders[t_name]) for t_name in task_names)
    
    for b in range(max_batches):
        optimizer.zero_grad()
        total_loss = 0.0
        
        for t_name in task_names:
            try:
                images, _ = next(iters[t_name])
            except StopIteration:
                continue
            
            images = images.to(device)
            
            with torch.no_grad():
                expert_outputs = functional_call(model_skeleton, expert_states[t_name], images)
                
            merged_backbone = get_merged_backbone_params(lambdas)
            eval_params = copy.copy(merged_backbone)
            eval_params["fc.weight"] = trainable_heads[t_name]["fc.weight"]
            eval_params["fc.bias"] = trainable_heads[t_name]["fc.bias"]
            
            merged_outputs = functional_call(model_skeleton, eval_params, images)
            loss = distillation_loss(merged_outputs, expert_outputs, temperature)
            total_loss += loss
            
        if total_loss != 0.0:
            total_loss.backward()
            optimizer.step()
            with torch.no_grad():
                if clamp_val is not None:
                    lambdas.clamp_(0.0, clamp_val)
                    
    return lambdas.detach(), trainable_heads


def run_sequential_tta(init_lambdas, loaders, lr_lambdas=0.01, lr_heads=0.01, temperature=2.0, clamp_val=0.3):
    task_names = list(tasks_config.keys())
    
    # Phase A: Lambdas only
    lambdas = init_lambdas.clone().detach().requires_grad_(True)
    optimizer_lambdas = optim.AdamW([lambdas], lr=lr_lambdas)
    
    iters = {t_name: iter(loaders[t_name]) for t_name in task_names}
    max_batches = max(len(loaders[t_name]) for t_name in task_names)
    
    for b in range(max_batches):
        optimizer_lambdas.zero_grad()
        total_loss = 0.0
        
        for t_name in task_names:
            try:
                images, _ = next(iters[t_name])
            except StopIteration:
                continue
            
            images = images.to(device)
            
            with torch.no_grad():
                expert_outputs = functional_call(model_skeleton, expert_states[t_name], images)
            
            merged_backbone = get_merged_backbone_params(lambdas)
            eval_params = copy.copy(merged_backbone)
            eval_params["fc.weight"] = expert_heads[t_name]["fc.weight"]
            eval_params["fc.bias"] = expert_heads[t_name]["fc.bias"]
            
            merged_outputs = functional_call(model_skeleton, eval_params, images)
            loss = distillation_loss(merged_outputs, expert_outputs, temperature)
            total_loss += loss
            
        if total_loss != 0.0:
            total_loss.backward()
            optimizer_lambdas.step()
            with torch.no_grad():
                if clamp_val is not None:
                    lambdas.clamp_(0.0, clamp_val)
                
    opt_lambdas = lambdas.clone().detach()
    
    # Phase B: Heads only
    trainable_heads = {}
    head_params = []
    for t_name, head in expert_heads.items():
        t_head = {
            "fc.weight": head["fc.weight"].clone().detach().requires_grad_(True),
            "fc.bias": head["fc.bias"].clone().detach().requires_grad_(True)
        }
        trainable_heads[t_name] = t_head
        head_params.extend([t_head["fc.weight"], t_head["fc.bias"]])
        
    optimizer_heads = optim.AdamW(head_params, lr=lr_heads)
    
    iters = {t_name: iter(loaders[t_name]) for t_name in task_names}
    max_batches = max(len(loaders[t_name]) for t_name in task_names)
    
    for b in range(max_batches):
        optimizer_heads.zero_grad()
        total_loss = 0.0
        
        for t_name in task_names:
            try:
                images, _ = next(iters[t_name])
            except StopIteration:
                continue
            
            images = images.to(device)
            
            with torch.no_grad():
                expert_outputs = functional_call(model_skeleton, expert_states[t_name], images)
            
            merged_backbone = get_merged_backbone_params(opt_lambdas)
            eval_params = copy.copy(merged_backbone)
            eval_params["fc.weight"] = trainable_heads[t_name]["fc.weight"]
            eval_params["fc.bias"] = trainable_heads[t_name]["fc.bias"]
            
            merged_outputs = functional_call(model_skeleton, eval_params, images)
            loss = distillation_loss(merged_outputs, expert_outputs, temperature)
            total_loss += loss
            
        if total_loss != 0.0:
            total_loss.backward()
            optimizer_heads.step()
            
    return opt_lambdas, trainable_heads


def run_constant_coef_head_tta(init_lambdas, loaders, lr_heads=0.01, temperature=2.0):
    task_names = list(tasks_config.keys())
    const_lambdas = init_lambdas.clone().detach()
    
    trainable_heads = {}
    head_params = []
    for t_name, head in expert_heads.items():
        t_head = {
            "fc.weight": head["fc.weight"].clone().detach().requires_grad_(True),
            "fc.bias": head["fc.bias"].clone().detach().requires_grad_(True)
        }
        trainable_heads[t_name] = t_head
        head_params.extend([t_head["fc.weight"], t_head["fc.bias"]])
        
    optimizer_heads = optim.AdamW(head_params, lr=lr_heads)
    
    iters = {t_name: iter(loaders[t_name]) for t_name in task_names}
    max_batches = max(len(loaders[t_name]) for t_name in task_names)
    
    for b in range(max_batches):
        optimizer_heads.zero_grad()
        total_loss = 0.0
        
        for t_name in task_names:
            try:
                images, _ = next(iters[t_name])
            except StopIteration:
                continue
            
            images = images.to(device)
            
            with torch.no_grad():
                expert_outputs = functional_call(model_skeleton, expert_states[t_name], images)
            
            merged_backbone = get_merged_backbone_params(const_lambdas)
            eval_params = copy.copy(merged_backbone)
            eval_params["fc.weight"] = trainable_heads[t_name]["fc.weight"]
            eval_params["fc.bias"] = trainable_heads[t_name]["fc.bias"]
            
            merged_outputs = functional_call(model_skeleton, eval_params, images)
            loss = distillation_loss(merged_outputs, expert_outputs, temperature)
            total_loss += loss
            
        if total_loss != 0.0:
            total_loss.backward()
            optimizer_heads.step()
            
    return const_lambdas, trainable_heads


# --- EXECUTE ABLATIONS ---

# Default settings
default_size = 500
default_loaders = get_dataloaders(default_size)
poor_init = torch.tensor([0.0, 0.0, 0.0], device=device)
good_init = torch.tensor([0.3, 0.3, 0.3], device=device)

# --- ABLATION 1: Learning Rate for Lambdas (fixing lr_heads = 0.01, temperature = 2.0, size = 500) ---
print("\n" + "="*50)
print("ABLATION 1: LEARNING RATE FOR LAMBDAS")
print("="*50)
lr_lambdas_list = [0.0001, 0.001, 0.01, 0.1]

print("\nPOOR INITIALIZATION ([0.0, 0.0, 0.0]):")
print(f"{'lr_lambda':<10} | {'Method':<15} | {'Average Acc':<12} | {'Final Lambdas':<20}")
print("-"*65)
for lr in lr_lambdas_list:
    # Joint
    l_joint, h_joint = run_joint_tta(poor_init, default_loaders, lr_lambdas=lr, lr_heads=0.01, temperature=2.0)
    acc_joint = evaluate_model_with_loaders(l_joint, h_joint, default_loaders)
    # Seq
    l_seq, h_seq = run_sequential_tta(poor_init, default_loaders, lr_lambdas=lr, lr_heads=0.01, temperature=2.0)
    acc_seq = evaluate_model_with_loaders(l_seq, h_seq, default_loaders)
    
    print(f"{lr:<10} | {'Joint TTA':<15} | {acc_joint['average']:<12.2f}% | {[round(x, 4) for x in l_joint.cpu().tolist()]}")
    print(f"{lr:<10} | {'Sequential TTA':<15} | {acc_seq['average']:<12.2f}% | {[round(x, 4) for x in l_seq.cpu().tolist()]}")
    print("-" * 65)

print("\nGOOD INITIALIZATION ([0.3, 0.3, 0.3]):")
print(f"{'lr_lambda':<10} | {'Method':<15} | {'Average Acc':<12} | {'Final Lambdas':<20}")
print("-"*65)
for lr in lr_lambdas_list:
    # Joint
    l_joint, h_joint = run_joint_tta(good_init, default_loaders, lr_lambdas=lr, lr_heads=0.01, temperature=2.0)
    acc_joint = evaluate_model_with_loaders(l_joint, h_joint, default_loaders)
    # Seq
    l_seq, h_seq = run_sequential_tta(good_init, default_loaders, lr_lambdas=lr, lr_heads=0.01, temperature=2.0)
    acc_seq = evaluate_model_with_loaders(l_seq, h_seq, default_loaders)
    
    print(f"{lr:<10} | {'Joint TTA':<15} | {acc_joint['average']:<12.2f}% | {[round(x, 4) for x in l_joint.cpu().tolist()]}")
    print(f"{lr:<10} | {'Sequential TTA':<15} | {acc_seq['average']:<12.2f}% | {[round(x, 4) for x in l_seq.cpu().tolist()]}")
    print("-" * 65)


# --- ABLATION 2: Distillation Temperature (fixing lr_lambdas = 0.01, lr_heads = 0.01, size = 500) ---
print("\n" + "="*50)
print("ABLATION 2: DISTILLATION TEMPERATURE")
print("="*50)
temperatures = [0.5, 1.0, 2.0, 5.0]

print("\nPOOR INITIALIZATION ([0.0, 0.0, 0.0]):")
print(f"{'Temp':<6} | {'Method':<15} | {'Average Acc':<12} | {'Final Lambdas':<20}")
print("-"*65)
for temp in temperatures:
    # Joint
    l_joint, h_joint = run_joint_tta(poor_init, default_loaders, lr_lambdas=0.01, lr_heads=0.01, temperature=temp)
    acc_joint = evaluate_model_with_loaders(l_joint, h_joint, default_loaders)
    # Seq
    l_seq, h_seq = run_sequential_tta(poor_init, default_loaders, lr_lambdas=0.01, lr_heads=0.01, temperature=temp)
    acc_seq = evaluate_model_with_loaders(l_seq, h_seq, default_loaders)
    # HeadOnly
    l_head, h_head = run_constant_coef_head_tta(poor_init, default_loaders, lr_heads=0.01, temperature=temp)
    acc_head = evaluate_model_with_loaders(l_head, h_head, default_loaders)
    
    print(f"{temp:<6} | {'Joint TTA':<15} | {acc_joint['average']:<12.2f}% | {[round(x, 4) for x in l_joint.cpu().tolist()]}")
    print(f"{temp:<6} | {'Sequential TTA':<15} | {acc_seq['average']:<12.2f}% | {[round(x, 4) for x in l_seq.cpu().tolist()]}")
    print(f"{temp:<6} | {'Head-only TTA':<15} | {acc_head['average']:<12.2f}% | {[round(x, 4) for x in l_head.cpu().tolist()]}")
    print("-" * 65)


# --- ABLATION 3: Test-time Data Budget / Sample Complexity ---
print("\n" + "="*50)
print("ABLATION 3: SAMPLE COMPLEXITY (DATA BUDGET)")
print("="*50)
budgets = [50, 100, 250, 500]

print("\nPOOR INITIALIZATION ([0.0, 0.0, 0.0]):")
print(f"{'Budget':<6} | {'Method':<15} | {'Average Acc':<12} | {'Final Lambdas':<20}")
print("-"*65)
for budget in budgets:
    budget_loaders = get_dataloaders(budget)
    # Joint
    l_joint, h_joint = run_joint_tta(poor_init, budget_loaders, lr_lambdas=0.01, lr_heads=0.01, temperature=2.0)
    acc_joint = evaluate_model_with_loaders(l_joint, h_joint, budget_loaders)
    # Seq
    l_seq, h_seq = run_sequential_tta(poor_init, budget_loaders, lr_lambdas=0.01, lr_heads=0.01, temperature=2.0)
    acc_seq = evaluate_model_with_loaders(l_seq, h_seq, budget_loaders)
    # HeadOnly
    l_head, h_head = run_constant_coef_head_tta(poor_init, budget_loaders, lr_heads=0.01, temperature=2.0)
    acc_head = evaluate_model_with_loaders(l_head, h_head, budget_loaders)
    
    print(f"{budget:<6} | {'Joint TTA':<15} | {acc_joint['average']:<12.2f}% | {[round(x, 4) for x in l_joint.cpu().tolist()]}")
    print(f"{budget:<6} | {'Sequential TTA':<15} | {acc_seq['average']:<12.2f}% | {[round(x, 4) for x in l_seq.cpu().tolist()]}")
    print(f"{budget:<6} | {'Head-only TTA':<15} | {acc_head['average']:<12.2f}% | {[round(x, 4) for x in l_head.cpu().tolist()]}")
    print("-" * 65)


# --- ABLATION 4: Clamping Limit (fixing lr_lambdas = 0.01, lr_heads = 0.01, temperature = 2.0, size = 500) ---
print("\n" + "="*50)
print("ABLATION 4: CLAMPING BOUND SWEEP")
print("="*50)
clamping_limits = [0.1, 0.2, 0.3, 0.4, 0.5, None]

print("\nPOOR INITIALIZATION ([0.0, 0.0, 0.0]):")
print(f"{'Clamp':<6} | {'Method':<15} | {'Average Acc':<12} | {'Final Lambdas':<20}")
print("-"*65)
for limit in clamping_limits:
    # Joint
    l_joint, h_joint = run_joint_tta(poor_init, default_loaders, lr_lambdas=0.01, lr_heads=0.01, temperature=2.0, clamp_val=limit)
    acc_joint = evaluate_model_with_loaders(l_joint, h_joint, default_loaders)
    # Seq
    l_seq, h_seq = run_sequential_tta(poor_init, default_loaders, lr_lambdas=0.01, lr_heads=0.01, temperature=2.0, clamp_val=limit)
    acc_seq = evaluate_model_with_loaders(l_seq, h_seq, default_loaders)
    
    limit_str = str(limit) if limit is not None else "None"
    print(f"{limit_str:<6} | {'Joint TTA':<15} | {acc_joint['average']:<12.2f}% | {[round(x, 4) for x in l_joint.cpu().tolist()]}")
    print(f"{limit_str:<6} | {'Sequential TTA':<15} | {acc_seq['average']:<12.2f}% | {[round(x, 4) for x in l_seq.cpu().tolist()]}")
    print("-" * 65)

print("\nGOOD INITIALIZATION ([0.3, 0.3, 0.3]):")
print(f"{'Clamp':<6} | {'Method':<15} | {'Average Acc':<12} | {'Final Lambdas':<20}")
print("-"*65)
for limit in clamping_limits:
    # Joint
    l_joint, h_joint = run_joint_tta(good_init, default_loaders, lr_lambdas=0.01, lr_heads=0.01, temperature=2.0, clamp_val=limit)
    acc_joint = evaluate_model_with_loaders(l_joint, h_joint, default_loaders)
    # Seq
    l_seq, h_seq = run_sequential_tta(good_init, default_loaders, lr_lambdas=0.01, lr_heads=0.01, temperature=2.0, clamp_val=limit)
    acc_seq = evaluate_model_with_loaders(l_seq, h_seq, default_loaders)
    
    limit_str = str(limit) if limit is not None else "None"
    print(f"{limit_str:<6} | {'Joint TTA':<15} | {acc_joint['average']:<12.2f}% | {[round(x, 4) for x in l_joint.cpu().tolist()]}")
    print(f"{limit_str:<6} | {'Sequential TTA':<15} | {acc_seq['average']:<12.2f}% | {[round(x, 4) for x in l_seq.cpu().tolist()]}")
    print("-" * 65)


print("\n" + "="*50)
print("ABLATION STUDY COMPLETED SUCCESSFULLY")
print("="*50)
