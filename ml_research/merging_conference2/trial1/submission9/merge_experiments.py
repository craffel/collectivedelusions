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
print(f"Using device: {device}")

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

SUBSET_SIZE_TEST = 500
BATCH_SIZE = 64

# Load test datasets and create dataloaders
print("Loading datasets...")
dataloaders = {}
for task_name, config in tasks_config.items():
    dataset = config["dataset_class"](
        root="./data", transform=config["transform"], **config["test_kwargs"]
    )
    # Use deterministic indices for fair evaluation across methods
    torch.manual_seed(42)
    indices = torch.randperm(len(dataset))[:SUBSET_SIZE_TEST]
    subset = Subset(dataset, indices)
    dataloaders[task_name] = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# Load base model and experts
print("Loading model weights...")
base_state_dict = torch.load("models/resnet18_pretrained.pt", map_location=device)

# ResNet-18 model skeleton
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

# Helper function to compute merged backbone weights differentiably from lambdas
def get_merged_backbone_params(lambdas):
    merged = {}
    task_names = list(tasks_config.keys())
    for name, base_param in base_state_dict.items():
        if name != "fc.weight" and name != "fc.bias":
            # Apply task arithmetic merging
            val = base_param + \
                  lambdas[0] * task_vectors[task_names[0]][name] + \
                  lambdas[1] * task_vectors[task_names[1]][name] + \
                  lambdas[2] * task_vectors[task_names[2]][name]
            if "running_mean" in name or "running_var" in name or "num_batches_tracked" in name:
                val = val.detach()
            merged[name] = val
    return merged

# Soft Cross-Entropy (Distillation) Loss
def distillation_loss(student_logits, teacher_logits, temperature=2.0):
    soft_targets = torch.softmax(teacher_logits / temperature, dim=-1)
    log_probs = torch.log_softmax(student_logits / temperature, dim=-1)
    return -torch.sum(soft_targets * log_probs, dim=-1).mean()

# Standard evaluation function
def evaluate_model(lambdas, task_heads):
    merged_backbone = get_merged_backbone_params(lambdas)
    task_names = list(tasks_config.keys())
    accuracies = {}
    
    model_skeleton.eval()
    with torch.no_grad():
        for i, t_name in enumerate(task_names):
            # Construct dictionary of parameters for the functional call
            eval_params = copy.copy(merged_backbone)
            eval_params["fc.weight"] = task_heads[t_name]["fc.weight"]
            eval_params["fc.bias"] = task_heads[t_name]["fc.bias"]
            
            correct = 0
            total = 0
            for images, labels in dataloaders[t_name]:
                images, labels = images.to(device), labels.to(device)
                outputs = functional_call(model_skeleton, eval_params, images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
            accuracies[t_name] = 100.0 * correct / total
            
    accuracies["average"] = sum(accuracies.values()) / len(accuracies)
    return accuracies


# --- EXPERIMENT RUNNERS ---

def run_joint_tta(init_lambdas, epochs=1, lr_lambdas=0.01, lr_heads=0.01):
    """
    Joint Optimization (SyMerge style):
    Both merging coefficients (lambdas) and task heads are optimized jointly.
    """
    # Initialize trainable lambdas
    lambdas = init_lambdas.clone().detach().requires_grad_(True)
    
    # Initialize trainable heads from expert heads
    trainable_heads = {}
    optimizer_params = [lambdas]
    
    for t_name, head in expert_heads.items():
        t_head = {
            "fc.weight": head["fc.weight"].clone().detach().requires_grad_(True),
            "fc.bias": head["fc.bias"].clone().detach().requires_grad_(True)
        }
        trainable_heads[t_name] = t_head
        optimizer_params.extend([t_head["fc.weight"], t_head["fc.bias"]])
        
    optimizer = optim.AdamW(optimizer_params, lr=lr_heads) # lr for heads and lambdas
    
    # Adaptation loop on unlabeled test streams
    task_names = list(tasks_config.keys())
    model_skeleton.eval()
    
    for epoch in range(epochs):
        # We process batches sequentially from each task mimicking test-time adaptation
        # Get iterators
        iters = {t_name: iter(dataloaders[t_name]) for t_name in task_names}
        max_batches = max(len(dataloaders[t_name]) for t_name in task_names)
        
        for b in range(max_batches):
            optimizer.zero_grad()
            total_loss = 0.0
            
            for t_name in task_names:
                try:
                    images, _ = next(iters[t_name])
                except StopIteration:
                    continue  # task loader exhausted
                
                images = images.to(device)
                
                # 1. Get teacher (expert) frozen predictions
                with torch.no_grad():
                    expert_outputs = functional_call(model_skeleton, expert_states[t_name], images)
                    
                # 2. Get merged model outputs
                merged_backbone = get_merged_backbone_params(lambdas)
                eval_params = copy.copy(merged_backbone)
                eval_params["fc.weight"] = trainable_heads[t_name]["fc.weight"]
                eval_params["fc.bias"] = trainable_heads[t_name]["fc.bias"]
                
                merged_outputs = functional_call(model_skeleton, eval_params, images)
                
                # 3. Compute soft distillation loss
                loss = distillation_loss(merged_outputs, expert_outputs)
                total_loss += loss
                
            if total_loss != 0.0:
                total_loss.backward()
                optimizer.step()
                with torch.no_grad():
                    lambdas.clamp_(0.0, 0.3)
                
    return lambdas.detach(), trainable_heads


def run_sequential_tta(init_lambdas, epochs=1, lr_lambdas=0.01, lr_heads=0.01):
    """
    Sequential Optimization (Proposed Methodologist Baseline):
    Phase A: Optimize only lambdas using the self-labeling distillation loss.
    Phase B: Freeze optimized lambdas, and optimize only the task-specific heads.
    """
    task_names = list(tasks_config.keys())
    
    # === PHASE A: Optimize lambdas only ===
    lambdas = init_lambdas.clone().detach().requires_grad_(True)
    optimizer_lambdas = optim.AdamW([lambdas], lr=lr_lambdas)
    
    for epoch in range(epochs):
        iters = {t_name: iter(dataloaders[t_name]) for t_name in task_names}
        max_batches = max(len(dataloaders[t_name]) for t_name in task_names)
        
        for b in range(max_batches):
            optimizer_lambdas.zero_grad()
            total_loss = 0.0
            
            for t_name in task_names:
                try:
                    images, _ = next(iters[t_name])
                except StopIteration:
                    continue
                
                images = images.to(device)
                
                # Expert frozen prediction
                with torch.no_grad():
                    expert_outputs = functional_call(model_skeleton, expert_states[t_name], images)
                
                # Merged model outputs with frozen expert heads but trainable lambdas
                merged_backbone = get_merged_backbone_params(lambdas)
                eval_params = copy.copy(merged_backbone)
                eval_params["fc.weight"] = expert_heads[t_name]["fc.weight"]
                eval_params["fc.bias"] = expert_heads[t_name]["fc.bias"]
                
                merged_outputs = functional_call(model_skeleton, eval_params, images)
                
                loss = distillation_loss(merged_outputs, expert_outputs)
                total_loss += loss
                
            if total_loss != 0.0:
                total_loss.backward()
                optimizer_lambdas.step()
                with torch.no_grad():
                    lambdas.clamp_(0.0, 0.3)
                
    # Freeze optimized lambdas
    opt_lambdas = lambdas.clone().detach()
    
    # === PHASE B: Optimize Heads only ===
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
    
    for epoch in range(epochs):
        iters = {t_name: iter(dataloaders[t_name]) for t_name in task_names}
        max_batches = max(len(dataloaders[t_name]) for t_name in task_names)
        
        for b in range(max_batches):
            optimizer_heads.zero_grad()
            total_loss = 0.0
            
            for t_name in task_names:
                try:
                    images, _ = next(iters[t_name])
                except StopIteration:
                    continue
                
                images = images.to(device)
                
                # Expert frozen prediction
                with torch.no_grad():
                    expert_outputs = functional_call(model_skeleton, expert_states[t_name], images)
                
                # Merged model outputs with frozen optimized backbone and trainable heads
                merged_backbone = get_merged_backbone_params(opt_lambdas)
                eval_params = copy.copy(merged_backbone)
                eval_params["fc.weight"] = trainable_heads[t_name]["fc.weight"]
                eval_params["fc.bias"] = trainable_heads[t_name]["fc.bias"]
                
                merged_outputs = functional_call(model_skeleton, eval_params, images)
                
                loss = distillation_loss(merged_outputs, expert_outputs)
                total_loss += loss
                
            if total_loss != 0.0:
                total_loss.backward()
                optimizer_heads.step()
                
    return opt_lambdas, trainable_heads


def run_constant_coef_head_tta(init_lambdas, epochs=1, lr_heads=0.01):
    """
    Constant Coefficient + Head-Only TTA (No Coef optimization):
    Merging coefficients are kept strictly constant at initialization.
    Only the task-specific classification heads are adapted on the test stream.
    """
    task_names = list(tasks_config.keys())
    
    # Keep lambdas constant as initialized
    const_lambdas = init_lambdas.clone().detach()
    
    # Initialize trainable heads from expert heads
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
    
    for epoch in range(epochs):
        iters = {t_name: iter(dataloaders[t_name]) for t_name in task_names}
        max_batches = max(len(dataloaders[t_name]) for t_name in task_names)
        
        for b in range(max_batches):
            optimizer_heads.zero_grad()
            total_loss = 0.0
            
            for t_name in task_names:
                try:
                    images, _ = next(iters[t_name])
                except StopIteration:
                    continue
                
                images = images.to(device)
                
                # Expert frozen prediction
                with torch.no_grad():
                    expert_outputs = functional_call(model_skeleton, expert_states[t_name], images)
                
                # Merged model outputs with frozen backbone and trainable heads
                merged_backbone = get_merged_backbone_params(const_lambdas)
                eval_params = copy.copy(merged_backbone)
                eval_params["fc.weight"] = trainable_heads[t_name]["fc.weight"]
                eval_params["fc.bias"] = trainable_heads[t_name]["fc.bias"]
                
                merged_outputs = functional_call(model_skeleton, eval_params, images)
                
                loss = distillation_loss(merged_outputs, expert_outputs)
                total_loss += loss
                
            if total_loss != 0.0:
                total_loss.backward()
                optimizer_heads.step()
                
    return const_lambdas, trainable_heads


# --- EVALUATION RUNS ---

initializations = {
    "Good (0.3 / 0.3 / 0.3)": torch.tensor([0.3, 0.3, 0.3], device=device),
    "Poor (0.0 / 0.0 / 0.0)": torch.tensor([0.0, 0.0, 0.0], device=device), # base model weights
}

results_summary = {}

for init_name, init_val in initializations.items():
    print(f"\n=========================================")
    print(f"RUNNING EXPERIMENTS FOR INITIALIZATION: {init_name}")
    print(f"=========================================")
    
    # 1. No Adaptation (Task Arithmetic Baseline)
    print("\n--- Running No Adaptation (Task Arithmetic) ---")
    ta_acc = evaluate_model(init_val, expert_heads)
    print(f"No Adaptation ACC | CIFAR-10: {ta_acc['cifar10']:.2f}% | SVHN: {ta_acc['svhn']:.2f}% | MNIST: {ta_acc['mnist']:.2f}% | Average: {ta_acc['average']:.2f}%")
    
    # 2. Joint TTA (SyMerge Joint Optimization)
    print("\n--- Running Joint TTA (SyMerge Joint Optimization) ---")
    joint_lambdas, joint_heads = run_joint_tta(init_val, epochs=1, lr_lambdas=0.01, lr_heads=0.01)
    joint_acc = evaluate_model(joint_lambdas, joint_heads)
    print(f"Optimized Lambdas: {joint_lambdas.cpu().tolist()}")
    print(f"Joint TTA ACC     | CIFAR-10: {joint_acc['cifar10']:.2f}% | SVHN: {joint_acc['svhn']:.2f}% | MNIST: {joint_acc['mnist']:.2f}% | Average: {joint_acc['average']:.2f}%")
    
    # 3. Sequential TTA (Our Proposed Sequential Pipeline)
    print("\n--- Running Sequential TTA (Proposed) ---")
    seq_lambdas, seq_heads = run_sequential_tta(init_val, epochs=1, lr_lambdas=0.01, lr_heads=0.01)
    seq_acc = evaluate_model(seq_lambdas, seq_heads)
    print(f"Optimized Lambdas: {seq_lambdas.cpu().tolist()}")
    print(f"Sequential TTA ACC| CIFAR-10: {seq_acc['cifar10']:.2f}% | SVHN: {seq_acc['svhn']:.2f}% | MNIST: {seq_acc['mnist']:.2f}% | Average: {seq_acc['average']:.2f}%")
    
    # 4. Constant Coef + Head TTA (No Coef optimization, only head adaptation)
    print("\n--- Running Constant Coef + Head TTA (Head-only) ---")
    const_lambdas, const_heads = run_constant_coef_head_tta(init_val, epochs=1, lr_heads=0.01)
    const_acc = evaluate_model(const_lambdas, const_heads)
    print(f"Constant Lambdas: {const_lambdas.cpu().tolist()}")
    print(f"Head-only TTA ACC | CIFAR-10: {const_acc['cifar10']:.2f}% | SVHN: {const_acc['svhn']:.2f}% | MNIST: {const_acc['mnist']:.2f}% | Average: {const_acc['average']:.2f}%")
    
    results_summary[init_name] = {
        "TA": ta_acc,
        "Joint": joint_acc,
        "Seq": seq_acc,
        "HeadOnly": const_acc,
        "lambdas_joint": joint_lambdas.cpu().tolist(),
        "lambdas_seq": seq_lambdas.cpu().tolist()
    }

# Print final comprehensive results table
print("\n" + "="*80)
print("FINAL EMPIRICAL RESULTS AND SYNTHESIS")
print("="*80)
print(f"{'Method':<20} | {'CIFAR-10':<10} | {'SVHN':<10} | {'MNIST':<10} | {'Average':<10}")
print("-"*80)

for init_name, res in results_summary.items():
    print(f"INIT: {init_name}")
    print(f"{'  Task Arithmetic':<20} | {res['TA']['cifar10']:<10.2f}% | {res['TA']['svhn']:<10.2f}% | {res['TA']['mnist']:<10.2f}% | {res['TA']['average']:<10.2f}%")
    print(f"{'  Joint TTA':<20} | {res['Joint']['cifar10']:<10.2f}% | {res['Joint']['svhn']:<10.2f}% | {res['Joint']['mnist']:<10.2f}% | {res['Joint']['average']:<10.2f}%")
    print(f"{'  Sequential TTA':<20} | {res['Seq']['cifar10']:<10.2f}% | {res['Seq']['svhn']:<10.2f}% | {res['Seq']['mnist']:<10.2f}% | {res['Seq']['average']:<10.2f}%")
    print(f"{'  Head-only TTA':<20} | {res['HeadOnly']['cifar10']:<10.2f}% | {res['HeadOnly']['svhn']:<10.2f}% | {res['HeadOnly']['mnist']:<10.2f}% | {res['HeadOnly']['average']:<10.2f}%")
    print("-"*80)

print("\n--- Experiment Complete ---")
