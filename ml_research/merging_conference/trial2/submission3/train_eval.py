import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.func import functional_call
from torch.utils.data import DataLoader, Subset
import numpy as np

# Ensure results and checkpoints folders exist
os.makedirs("./checkpoints", exist_ok=True)
os.makedirs("./results", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.backends.cudnn.enabled = False
print("cuDNN has been disabled to prevent CUDNN_STATUS_NOT_INITIALIZED issues.")

# ----------------------------------------------------------------------
# 1. Dataset Transforms & Custom Corruptions
# ----------------------------------------------------------------------

# Custom transforms for corruptions
class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=0.4):
        self.std = std
        self.mean = mean
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size(), device=tensor.device if isinstance(tensor, torch.Tensor) else None) * self.std + self.mean

class ContrastReduction(object):
    def __init__(self, factor=0.25):
        self.factor = factor
    def __call__(self, img):
        return TF.adjust_contrast(img, self.factor)

class FixedRotation(object):
    def __init__(self, angle=30.0):
        self.angle = angle
    def __call__(self, img):
        return TF.rotate(img, self.angle)

# Base transform: Resize to 32x32, 3 channels, normalized
transform_base = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_dataset(task_idx, train=True, corruption="clean"):
    if task_idx == 0:
        dataset_class = torchvision.datasets.MNIST
    elif task_idx == 1:
        dataset_class = torchvision.datasets.FashionMNIST
    elif task_idx == 2:
        dataset_class = torchvision.datasets.KMNIST
    else:
        raise ValueError(f"Unknown task index: {task_idx}")

    # Build specific transform based on corruption
    if corruption == "clean":
        transform = transform_base
    elif corruption == "noise":
        transform = transforms.Compose([
            transform_base,
            AddGaussianNoise(0.0, 0.4)
        ])
    elif corruption == "blur":
        transform = transforms.Compose([
            transform_base,
            transforms.GaussianBlur(kernel_size=5, sigma=1.5)
        ])
    elif corruption == "contrast":
        transform = transforms.Compose([
            transform_base,
            ContrastReduction(0.25)
        ])
    elif corruption == "rotation":
        transform = transforms.Compose([
            transform_base,
            FixedRotation(30.0)
        ])
    else:
        raise ValueError(f"Unknown corruption: {corruption}")

    return dataset_class(root="./data", train=train, transform=transform, download=False)

# ----------------------------------------------------------------------
# 2. Experts Training Pipeline
# ----------------------------------------------------------------------

def train_experts():
    print("=== Training Expert Models ===")
    
    # Load pretrained resnet18 weights
    weights = torchvision.models.ResNet18_Weights.DEFAULT
    base_model = torchvision.models.resnet18(weights=weights).to(device)
    
    # Save the initial pre-trained encoder weights (excluding fc)
    pre_encoder_state = {k: v.cpu().clone() for k, v in base_model.state_dict().items() if not k.startswith("fc.")}
    pre_encoder_buffers = {k: v.cpu().clone() for k, v in base_model.state_dict().items() if not k.startswith("fc.") and any(b in k for b in ["running_mean", "running_var", "num_batches_tracked"])}
    
    torch.save(pre_encoder_state, "./checkpoints/pre_encoder.pt")
    torch.save(pre_encoder_buffers, "./checkpoints/pre_encoder_buffers.pt")
    
    for task_idx in range(3):
        task_name = ["MNIST", "FashionMNIST", "KMNIST"][task_idx]
        print(f"\nTraining Expert for {task_name}...")
        
        # Load a fresh pretrained ResNet18 model
        model = torchvision.models.resnet18(weights=weights).to(device)
        model.fc = nn.Linear(512, 10).to(device) # 10-class output
        
        train_set = get_dataset(task_idx, train=True, corruption="clean")
        train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=0)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        for epoch in range(3):
            total_loss = 0.0
            correct = 0
            total = 0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
            epoch_loss = total_loss / total
            epoch_acc = correct / total * 100.0
            print(f"Epoch {epoch+1}/3 - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
            
        # Save trained expert weights (encoder + head)
        expert_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        torch.save(expert_state, f"./checkpoints/expert_{task_idx}.pt")
        print(f"Saved expert_{task_idx}.pt")

# Check if experts are already trained
experts_exist = all(os.path.exists(f"./checkpoints/expert_{i}.pt") for i in range(3)) and os.path.exists("./checkpoints/pre_encoder.pt")
if not experts_exist:
    train_experts()
else:
    print("Expert checkpoints already exist. Skipping training.")

# ----------------------------------------------------------------------
# 3. Loading Models and Initial States
# ----------------------------------------------------------------------

# Load base model structure
base_model = torchvision.models.resnet18().to(device)
base_model.fc = nn.Linear(512, 10).to(device)
base_model.eval()

# Load state dicts
pre_encoder_state = torch.load("./checkpoints/pre_encoder.pt", map_location=device)
pre_encoder_buffers = torch.load("./checkpoints/pre_encoder_buffers.pt", map_location=device)

expert_encoder_states = []
expert_encoder_buffers = []
expert_head_states = []

for i in range(3):
    state = torch.load(f"./checkpoints/expert_{i}.pt", map_location=device)
    # Extract encoder params
    enc_state = {k: v for k, v in state.items() if not k.startswith("fc.")}
    # Extract encoder buffers
    enc_bufs = {k: v for k, v in state.items() if not k.startswith("fc.") and any(b in k for b in ["running_mean", "running_var", "num_batches_tracked"])}
    # Extract classifier head params
    head_state = {k: v for k, v in state.items() if k.startswith("fc.")}
    
    expert_encoder_states.append(enc_state)
    expert_encoder_buffers.append(enc_bufs)
    expert_head_states.append(head_state)

# Gather names of encoder parameters and buffers
encoder_param_names = [name for name, _ in base_model.named_parameters() if not name.startswith("fc.")]
encoder_buffer_names = [name for name, _ in base_model.named_buffers() if not name.startswith("fc.")]

# ----------------------------------------------------------------------
# 4. Helper Functions for Model Reconstruction
# ----------------------------------------------------------------------

def reconstruct_state_dict(lambda_list, adapted_heads, task_idx):
    """
    Differentiably reconstructs the state dictionary of the model for task task_idx.
    """
    state_dict_k = {}
    
    # 1. Merged encoder parameters: Θ_merged = Θ_pre + \sum_k λ_k * (Θ_k - Θ_pre)
    for name in encoder_param_names:
        pre_param = pre_encoder_state[name]
        diff = 0.0
        for k in range(3):
            diff = diff + lambda_list[k] * (expert_encoder_states[k][name] - pre_param)
        state_dict_k[name] = pre_param + diff
        
    # 2. Merged encoder buffers: We use pre-trained base model buffers (frozen at test-time)
    for name in encoder_buffer_names:
        state_dict_k[name] = pre_encoder_buffers[name]
        
    # 3. Task-specific classification head
    state_dict_k["fc.weight"] = adapted_heads[task_idx]["weight"]
    state_dict_k["fc.bias"] = adapted_heads[task_idx]["bias"]
    
    return state_dict_k

def compute_kl_loss(p_expert, logits_merged):
    log_q = F.log_softmax(logits_merged, dim=-1)
    return F.kl_div(log_q, p_expert, reduction="batchmean")

def compute_loss(lambda_list, adapted_heads, batches, isr_type=None, beta=0.0):
    total_kl_loss = 0.0
    
    for k in range(3):
        inputs, targets = batches[k]
        
        # 1. Expert prediction distribution (no grad)
        with torch.no_grad():
            expert_state = {}
            for name in encoder_param_names:
                expert_state[name] = expert_encoder_states[k][name]
            for name in encoder_buffer_names:
                expert_state[name] = expert_encoder_buffers[k][name]
            expert_state["fc.weight"] = expert_head_states[k]["fc.weight"]
            expert_state["fc.bias"] = expert_head_states[k]["fc.bias"]
            
            expert_logits = functional_call(base_model, expert_state, (inputs,))
            p_expert = F.softmax(expert_logits, dim=-1)
            
        # 2. Merged model prediction
        state_dict_k = reconstruct_state_dict(lambda_list, adapted_heads, k)
        merged_logits = functional_call(base_model, state_dict_k, (inputs,))
        
        # 3. KL Loss
        kl = compute_kl_loss(p_expert, merged_logits)
        total_kl_loss += kl
        
    loss = total_kl_loss
    
    # 4. Spectral Regularization (ISR or SOSR)
    if beta > 0.0 and isr_type is not None:
        reg_loss = 0.0
        for k in range(3):
            W = adapted_heads[k]["weight"] # Shape (10, 512)
            W_W_T = torch.matmul(W, W.t()) # Shape (10, 10)
            
            if isr_type == "isr-fo":
                I = torch.eye(10, device=W.device)
                reg_loss += torch.sum((W_W_T - I) ** 2)
            elif isr_type == "sosr":
                diag_W_W_T = torch.diag(torch.diag(W_W_T))
                reg_loss += torch.sum((W_W_T - diag_W_W_T) ** 2)
                
        loss += beta * reg_loss
        
    return loss

# ----------------------------------------------------------------------
# 5. Model Evaluation
# ----------------------------------------------------------------------

def evaluate_merged_model(lambda_list, adapted_heads, test_loaders):
    base_model.eval()
    task_accs = []
    
    with torch.no_grad():
        for k in range(3):
            state_dict_k = reconstruct_state_dict(lambda_list, adapted_heads, k)
            
            correct = 0
            total = 0
            for inputs, targets in test_loaders[k]:
                inputs, targets = inputs.to(device), targets.to(device)
                logits = functional_call(base_model, state_dict_k, (inputs,))
                _, predicted = logits.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
            task_accs.append(correct / total * 100.0)
            
    return task_accs

# ----------------------------------------------------------------------
# 6. Test-Time Adaptation Algorithms
# ----------------------------------------------------------------------

def run_tta(method_name, test_pools, test_loaders, isr_type=None, beta=0.0, use_sam=False, use_asam=False, rho=0.05, eta=1e-2):
    """
    Performs test-time adaptation for 10 steps on the test_pools of 512 images per task.
    """
    # 1. Initialize merging coefficients Lambda to [0.3, 0.3, 0.3]
    lambda_list = torch.tensor([0.3, 0.3, 0.3], device=device, requires_grad=True)
    
    # 2. Initialize adapted heads from expert heads
    adapted_heads = []
    for k in range(3):
        w = expert_head_states[k]["fc.weight"].clone().to(device).requires_grad_(True)
        b = expert_head_states[k]["fc.bias"].clone().to(device).requires_grad_(True)
        adapted_heads.append({"weight": w, "bias": b})
        
    # 3. Setup optimizers based on the method
    params_heads = []
    for h in adapted_heads:
        params_heads.extend([h["weight"], h["bias"]])
        
    if method_name == "TA":
        # Static Task Arithmetic, no optimization
        return evaluate_merged_model(lambda_list, adapted_heads, test_loaders)
        
    elif method_name == "AdaMerging":
        # Only optimize merging coefficients
        optimizer = torch.optim.Adam([lambda_list], lr=0.001)
    else:
        # Jointly optimize merging coefficients and task heads
        optimizer = torch.optim.Adam([
            {"params": [lambda_list], "lr": 0.001},
            {"params": params_heads, "lr": 0.01}
        ])
        
    # Prepare batch iterators for test pools
    batch_size = 32
    pool_loaders = [DataLoader(Subset(test_pools[k], list(range(512))), batch_size=batch_size, shuffle=False) for k in range(3)]
    pool_iters = [iter(l) for l in pool_loaders]
    
    # 10 optimization steps
    for step in range(10):
        # Retrieve next batch for each task
        batches = []
        for k in range(3):
            try:
                inputs, targets = next(pool_iters[k])
            except StopIteration:
                pool_iters[k] = iter(pool_loaders[k])
                inputs, targets = next(pool_iters[k])
            inputs, targets = inputs.to(device), targets.to(device)
            batches.append((inputs, targets))
            
        # Forward pass & optimization
        if not (use_sam or use_asam):
            # Standard optimization
            optimizer.zero_grad()
            loss = compute_loss(lambda_list, adapted_heads, batches, isr_type=isr_type, beta=beta)
            loss.backward()
            optimizer.step()
        else:
            # Sharpness-aware optimization (SAM or ASAM)
            optimizer.zero_grad()
            loss = compute_loss(lambda_list, adapted_heads, batches, isr_type=isr_type, beta=beta)
            loss.backward()
            
            # Save original gradients and clone parameters
            grad_lambda = lambda_list.grad.clone()
            grad_heads = [{"weight": h["weight"].grad.clone(), "bias": h["bias"].grad.clone()} for h in adapted_heads]
            
            optimizer.zero_grad()
            
            # Handle ASAM vs standard SAM
            if use_asam:
                # 1. Compute element-wise scaled gradients
                g_scaled_norm_sq = torch.sum(((torch.abs(lambda_list.detach()) + eta) * grad_lambda) ** 2)
                for k in range(3):
                    w_scaled = (torch.abs(adapted_heads[k]["weight"].detach()) + eta) * grad_heads[k]["weight"]
                    b_scaled = (torch.abs(adapted_heads[k]["bias"].detach()) + eta) * grad_heads[k]["bias"]
                    g_scaled_norm_sq += torch.sum(w_scaled ** 2) + torch.sum(b_scaled ** 2)
                g_scaled_norm = torch.sqrt(g_scaled_norm_sq) + 1e-8
                
                # 2. Perturb parameters
                with torch.no_grad():
                    eps_lambda = rho * ((torch.abs(lambda_list) + eta) ** 2) * grad_lambda / g_scaled_norm
                    perturbed_lambda = (lambda_list + eps_lambda).requires_grad_(True)
                    
                    perturbed_heads = []
                    for k in range(3):
                        eps_w = rho * ((torch.abs(adapted_heads[k]["weight"]) + eta) ** 2) * grad_heads[k]["weight"] / g_scaled_norm
                        eps_b = rho * ((torch.abs(adapted_heads[k]["bias"]) + eta) ** 2) * grad_heads[k]["bias"] / g_scaled_norm
                        p_w = (adapted_heads[k]["weight"] + eps_w).requires_grad_(True)
                        p_b = (adapted_heads[k]["bias"] + eps_b).requires_grad_(True)
                        perturbed_heads.append({"weight": p_w, "bias": p_b})
            else:
                # Standard SAM
                # 1. Compute global gradient norm
                g_norm_sq = torch.sum(grad_lambda ** 2)
                for k in range(3):
                    g_norm_sq += torch.sum(grad_heads[k]["weight"] ** 2) + torch.sum(grad_heads[k]["bias"] ** 2)
                g_norm = torch.sqrt(g_norm_sq) + 1e-8
                
                # 2. Perturb parameters
                with torch.no_grad():
                    perturbed_lambda = (lambda_list + rho * (grad_lambda / g_norm)).requires_grad_(True)
                    perturbed_heads = []
                    for k in range(3):
                        p_w = (adapted_heads[k]["weight"] + rho * (grad_heads[k]["weight"] / g_norm)).requires_grad_(True)
                        p_b = (adapted_heads[k]["bias"] + rho * (grad_heads[k]["bias"] / g_norm)).requires_grad_(True)
                        perturbed_heads.append({"weight": p_w, "bias": p_b})
            
            # Second pass: Compute gradients at the perturbed location
            loss_perturbed = compute_loss(perturbed_lambda, perturbed_heads, batches, isr_type=isr_type, beta=beta)
            loss_perturbed.backward()
            
            # Assign perturbed gradients to the original leaf variables and step
            lambda_list.grad = perturbed_lambda.grad.clone()
            for k in range(3):
                adapted_heads[k]["weight"].grad = perturbed_heads[k]["weight"].grad.clone()
                adapted_heads[k]["bias"].grad = perturbed_heads[k]["bias"].grad.clone()
                
            optimizer.step()
            
    # Return the evaluated performance of the adapted model
    return evaluate_merged_model(lambda_list, adapted_heads, test_loaders)

# ----------------------------------------------------------------------
# 7. Main Evaluation Sweep across Corruptions & Methods
# ----------------------------------------------------------------------

corruptions = ["clean", "noise", "blur", "contrast", "rotation"]
methods = [
    # Baselines
    {"name": "TA", "isr_type": None, "beta": 0.0, "use_sam": False, "use_asam": False},
    {"name": "AdaMerging", "isr_type": None, "beta": 0.0, "use_sam": False, "use_asam": False},
    {"name": "SyMerge", "isr_type": None, "beta": 0.0, "use_sam": False, "use_asam": False},
    {"name": "SAT-SyMerge", "isr_type": None, "beta": 0.0, "use_sam": True, "use_asam": False, "rho": 0.08},
    {"name": "ASAM-SyMerge", "isr_type": None, "beta": 0.0, "use_sam": False, "use_asam": True, "rho": 0.08},
    
    # Proposed Spectral-Regularized Test-Time Adaptation
    {"name": "ISR-TTA (Ours)", "isr_type": "isr-fo", "beta": 0.1, "use_sam": False, "use_asam": False},
    {"name": "SOSR-TTA (Ours)", "isr_type": "sosr", "beta": 0.1, "use_sam": False, "use_asam": False},
    
    # Unified Flatness and Spectral Regularization
    {"name": "SAT-SOSR-TTA (Ours)", "isr_type": "sosr", "beta": 0.1, "use_sam": True, "use_asam": False, "rho": 0.08},
    {"name": "ASAM-SOSR-TTA (Ours)", "isr_type": "sosr", "beta": 0.1, "use_sam": False, "use_asam": True, "rho": 0.08}
]

results_dict = {m["name"]: {c: [] for c in corruptions} for m in methods}

print("\n=== Running Test-Time Adaptation & Evaluation ===")

for corruption in corruptions:
    print(f"\n--- Environment: {corruption.upper()} ---")
    
    # For TTA, we need a pool of 512 images per task to perform TTA on.
    # We must seed the split/dataloaders so that ALL methods are evaluated on the exact same TTA pools.
    torch.manual_seed(42)
    np.random.seed(42)
    
    test_pools = [get_dataset(k, train=False, corruption=corruption) for k in range(3)]
    
    # Create evaluation test loaders (full test set)
    test_loaders = [DataLoader(get_dataset(k, train=False, corruption=corruption), batch_size=128, shuffle=False, num_workers=0) for k in range(3)]
    
    for m in methods:
        method_name = m["name"]
        isr_type = m["isr_type"]
        beta = m["beta"]
        use_sam = m["use_sam"]
        use_asam = m["use_asam"]
        rho = m.get("rho", 0.05)
        
        # Reset seed before each method to guarantee identical TTA stream
        torch.manual_seed(42)
        np.random.seed(42)
        
        print(f"Evaluating {method_name}...")
        task_accs = run_tta(
            method_name=method_name,
            test_pools=test_pools,
            test_loaders=test_loaders,
            isr_type=isr_type,
            beta=beta,
            use_sam=use_sam,
            use_asam=use_asam,
            rho=rho
        )
        
        avg_acc = sum(task_accs) / 3.0
        results_dict[method_name][corruption] = {
            "MNIST": task_accs[0],
            "FashionMNIST": task_accs[1],
            "KMNIST": task_accs[2],
            "Average": avg_acc
        }
        print(f"  MNIST: {task_accs[0]:.2f}%, FashionMNIST: {task_accs[1]:.2f}%, KMNIST: {task_accs[2]:.2f}% | Avg: {avg_acc:.2f}%")

# Save results to JSON
with open("./results/results.json", "w") as f:
    json.dump(results_dict, f, indent=4)

print("\n=== Experiments Completed Successfully! ===")

# ----------------------------------------------------------------------
# 8. Produce beautiful markdown table
# ----------------------------------------------------------------------
print("\n### Summary of Results (Multi-Task Average %)")
print("| Method | Clean | Noise | Blur | Contrast | Rotation | OOD Avg | Overall Avg |")
print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

for m in methods:
    name = m["name"]
    clean = results_dict[name]["clean"]["Average"]
    noise = results_dict[name]["noise"]["Average"]
    blur = results_dict[name]["blur"]["Average"]
    contrast = results_dict[name]["contrast"]["Average"]
    rotation = results_dict[name]["rotation"]["Average"]
    ood_avg = (noise + blur + contrast + rotation) / 4.0
    overall_avg = (clean + ood_avg) / 2.0
    print(f"| {name} | {clean:.2f}% | {noise:.2f}% | {blur:.2f}% | {contrast:.2f}% | {rotation:.2f}% | {ood_avg:.2f}% | {overall_avg:.2f}% |")
