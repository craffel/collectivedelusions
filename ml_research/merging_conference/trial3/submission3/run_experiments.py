import os
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import vit_b_16
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    # Disable cuDNN to bypass CUDNN_STATUS_NOT_INITIALIZED errors on this cluster
    torch.backends.cudnn.enabled = False

# Global variables
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# 1. Custom LoRA Linear Layer
class LoRALinear(nn.Module):
    def __init__(self, original_linear, r=8, alpha=16):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        
        # Original frozen weights
        self.weight = nn.Parameter(original_linear.weight.clone(), requires_grad=False)
        self.bias = nn.Parameter(original_linear.bias.clone(), requires_grad=False) if original_linear.bias is not None else None
        
        # LoRA parameters
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        self.lora_a = nn.Parameter(torch.zeros(r, self.in_features))
        self.lora_b = nn.Parameter(torch.zeros(self.out_features, r))
        
        # Initialization
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)
        
        # Activation tracking
        self.current_activations = None
        self.track_activations = False

        # References for differentiable dynamic merging
        self.w_a_cifar = None
        self.w_a_svhn = None
        self.w_b_cifar = None
        self.w_b_svhn = None
        self.coeffs_ref = None
        self.coeff_idx = None

    def forward(self, x):
        # x shape can be [B, S, F] or [B, F]
        if self.track_activations:
            self.current_activations = x.detach()
            
        result = F.linear(x, self.weight, self.bias)
        
        # Fully differentiable dynamic model merging path
        if self.coeffs_ref is not None and self.w_a_cifar is not None:
            lmbda = torch.sigmoid(self.coeffs_ref[self.coeff_idx])
            lora_a = lmbda * self.w_a_cifar + (1.0 - lmbda) * self.w_a_svhn
            lora_b = lmbda * self.w_b_cifar + (1.0 - lmbda) * self.w_b_svhn
        else:
            lora_a = self.lora_a
            lora_b = self.lora_b
            
        lora_update = (x @ lora_a.t()) @ lora_b.t()
        return result + lora_update * self.scaling

def inject_lora(model, r=8, alpha=16):
    """Recursively replaces linear layers in ViT (excluding heads) with LoRALinear."""
    def _inject(module):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear) and name != "head":
                # Check that parent is not the final classification heads
                setattr(module, name, LoRALinear(child, r, alpha))
            else:
                _inject(child)
    _inject(model)

def get_lora_params(model):
    """Returns lists of lora_a and lora_b parameters."""
    lora_a_list = []
    lora_b_list = []
    for m in model.modules():
        if isinstance(m, LoRALinear):
            lora_a_list.append(m.lora_a)
            lora_b_list.append(m.lora_b)
    return lora_a_list, lora_b_list

# 2. Dataset and Dataloader utilities
def get_dataloaders(batch_size=128):
    # ViT expects 224x224 images
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Datasets
    cifar_train = datasets.CIFAR10(root="./data", train=True, transform=transform_train)
    cifar_test = datasets.CIFAR10(root="./data", train=False, transform=transform_test)
    
    svhn_train = datasets.SVHN(root="./data", split="train", transform=transform_train)
    svhn_test = datasets.SVHN(root="./data", split="test", transform=transform_test)

    # Subsample for extremely fast training (e.g. 5000 train samples)
    cifar_train_sub = Subset(cifar_train, list(range(5000)))
    svhn_train_sub = Subset(svhn_train, list(range(5000)))
    
    # Subsample test sets for TTA evaluation streams (e.g. 1000 test samples)
    cifar_test_sub = Subset(cifar_test, list(range(1000)))
    svhn_test_sub = Subset(svhn_test, list(range(1000)))

    loaders = {
        "cifar_train": DataLoader(cifar_train_sub, batch_size=batch_size, shuffle=True, num_workers=2),
        "cifar_test": DataLoader(cifar_test_sub, batch_size=batch_size, shuffle=False, num_workers=2),
        "svhn_train": DataLoader(svhn_train_sub, batch_size=batch_size, shuffle=True, num_workers=2),
        "svhn_test": DataLoader(svhn_test_sub, batch_size=batch_size, shuffle=False, num_workers=2)
    }
    return loaders

# Corruptions for OOD testing
def add_gaussian_noise(x, severity=0.15):
    noise = torch.randn_like(x) * severity
    return torch.clamp(x + noise, -2.5, 2.5)

# 3. Model Initialization and Expert Training
def get_base_vit():
    os.environ["TORCH_HOME"] = "./torch_cache"
    # Load architecture
    model = vit_b_16(weights=None)
    # Load pre-trained weights locally
    checkpoint_path = "./torch_cache/hub/checkpoints/vit_b_16-c867db91.pth"
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    return model

def train_expert(dataset_name, dataloader, epochs=3):
    print(f"\n--- Training Expert for {dataset_name} ---")
    model = get_base_vit()
    inject_lora(model)
    
    # Replace final head with a 10-class linear layer
    model.heads.head = nn.Linear(768, 10)
    model = model.to(DEVICE)
    
    # Only train LoRA parameters and classification head
    trainable_params = []
    for n, p in model.named_parameters():
        if "lora" in n or "heads.head" in n:
            p.requires_grad = True
            trainable_params.append(p)
        else:
            p.requires_grad = False

    optimizer = torch.optim.AdamW(trainable_params, lr=5e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=-1)
            correct += preds.eq(y).sum().item()
            total += x.size(0)
            
        epoch_loss = total_loss / total
        epoch_acc = correct / total * 100
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")
        
    return model

# 4. Evaluation helper
def evaluate(model, dataloader, head, corrupt=False):
    model.eval()
    head.eval()
    correct = 0
    total = 0
    
    # Set heads
    model.heads.head = head
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            if corrupt:
                x = add_gaussian_noise(x)
            logits = model(x)
            preds = logits.argmax(dim=-1)
            correct += preds.eq(y).sum().item()
            total += x.size(0)
            
    return (correct / total) * 100

# 5. Model Merging and Test-Time Adaptation
class ModelMerger(nn.Module):
    def __init__(self, base_model, expert_cifar_lora, expert_svhn_lora, cifar_head, svhn_head):
        super().__init__()
        self.base_model = base_model
        self.expert_cifar_lora = expert_cifar_lora # dict of name: param
        self.expert_svhn_lora = expert_svhn_lora
        
        # Heads are task-specific
        self.cifar_head = cifar_head
        self.svhn_head = svhn_head
        
        # We define layer-wise merging coefficients. ViT-B has 12 layers, each has attention and MLP blocks.
        # Let's define layer-wise coefficients (one per transformer block, plus input conv_proj, etc.)
        # To keep it elegant, we use 12 block coefficients + 1 for heads
        self.coeffs = nn.Parameter(torch.ones(13) * 0.5) # initialized to 0.5 (even merge)

    def apply_merge(self):
        """Linearly interpolates LoRA parameters using the layer-wise coefficients."""
        # Layer 0 to 11 correspond to the 12 transformer blocks
        # We group LoRA layers by block index
        c_idx = 0
        for m_name, m in self.base_model.named_modules():
            if isinstance(m, LoRALinear):
                # Map module to a coefficient index based on its name (e.g., encoder.layers.encoder_layer_X)
                # If name has 'encoder_layer_X', we map to X, otherwise map to 12
                coeff_idx = 12
                for i in range(12):
                    if f"encoder_layer_{i}" in m_name:
                        coeff_idx = i
                        break
                
                # Get lambda
                lmbda = torch.sigmoid(self.coeffs[coeff_idx])
                
                # Retrieve matching weights from experts
                # We assume the layout is identical
                # Construct key
                key_a = m_name + ".lora_a"
                key_b = m_name + ".lora_b"
                
                w_a_cifar = self.expert_cifar_lora[key_a]
                w_a_svhn = self.expert_svhn_lora[key_a]
                w_b_cifar = self.expert_cifar_lora[key_b]
                w_b_svhn = self.expert_svhn_lora[key_b]
                
                # Set active on-device references for fully differentiable forward passes
                m.w_a_cifar = w_a_cifar
                m.w_a_svhn = w_a_svhn
                m.w_b_cifar = w_b_cifar
                m.w_b_svhn = w_b_svhn
                m.coeffs_ref = self.coeffs
                m.coeff_idx = coeff_idx
                
                # Direct interpolation of A and B (for static parameters if needed)
                m.lora_a.data.copy_(lmbda * w_a_cifar + (1.0 - lmbda) * w_a_svhn)
                m.lora_b.data.copy_(lmbda * w_b_cifar + (1.0 - lmbda) * w_b_svhn)

# 6. Test-Time Adaptation (TTA) Algorithms
def run_tta_evaluation(loaders, expert_cifar_state, expert_svhn_state, method="static", corrupt=False):
    print(f"\nEvaluating method: {method.upper()} (OOD Corrupt: {corrupt})")
    
    # Set random seed for reproducibility of noise across methods
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Initialize base model
    base_model = get_base_vit()
    inject_lora(base_model)
    
    # Extract expert LoRA weights
    cifar_lora = {k: v.to(DEVICE) for k, v in expert_cifar_state.items() if "lora" in k}
    svhn_lora = {k: v.to(DEVICE) for k, v in expert_svhn_state.items() if "lora" in k}
    
    # Heads
    cifar_head = nn.Linear(768, 10).to(DEVICE)
    cifar_head.load_state_dict({k.replace("heads.head.", ""): v for k, v in expert_cifar_state.items() if "heads.head" in k})
    
    svhn_head = nn.Linear(768, 10).to(DEVICE)
    svhn_head.load_state_dict({k.replace("heads.head.", ""): v for k, v in expert_svhn_state.items() if "heads.head" in k})
    
    # Initialize merger
    merger = ModelMerger(base_model, cifar_lora, svhn_lora, cifar_head, svhn_head).to(DEVICE)
    merger.apply_merge()
    
    # Decide what to optimize
    trainable_params = []
    if method == "static":
        pass
    elif method == "adamerging":
        # Optimize merging coefficients only
        merger.coeffs.requires_grad = True
        trainable_params = [merger.coeffs]
    elif method in ["symerge", "sat_tta", "cas_merge"]:
        # Optimize merging coefficients AND the classification heads
        merger.coeffs.requires_grad = True
        merger.cifar_head.weight.requires_grad = True
        merger.cifar_head.bias.requires_grad = True
        merger.svhn_head.weight.requires_grad = True
        merger.svhn_head.bias.requires_grad = True
        trainable_params = [merger.coeffs, merger.cifar_head.weight, merger.cifar_head.bias, merger.svhn_head.weight, merger.svhn_head.bias]
    
    optimizer = torch.optim.Adam(trainable_params, lr=1e-3) if trainable_params else None
    
    # Evaluation streams
    tasks = [
        {"name": "cifar10", "loader": loaders["cifar_test"], "head": merger.cifar_head, "expert_state": expert_cifar_state},
        {"name": "svhn", "loader": loaders["svhn_test"], "head": merger.svhn_head, "expert_state": expert_svhn_state}
    ]
    
    task_accuracies = {}
    
    for task in tasks:
        task_name = task["name"]
        loader = task["loader"]
        head = task["head"]
        
        # To guide TTA (expert-guided self-labeling), we use a frozen copy of the expert model
        expert_model = get_base_vit()
        inject_lora(expert_model)
        expert_model.heads.head = nn.Linear(768, 10)
        # Load weights
        expert_model.load_state_dict(task["expert_state"])
        expert_model = expert_model.to(DEVICE)
        expert_model.eval()
        
        # Reset merger parameters and coefficients to static starting point before each task stream
        with torch.no_grad():
            merger.coeffs.data.fill_(0.5)
            merger.apply_merge()
            
        correct = 0
        total = 0
        
        # Enable activation tracking for CAS-Merge
        if method == "cas_merge":
            for m in merger.base_model.modules():
                if isinstance(m, LoRALinear):
                    m.track_activations = True
                    
        for batch_idx, (x, y) in enumerate(loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            if corrupt:
                x = add_gaussian_noise(x)
                
            # 1. Evaluate current step model
            merger.base_model.eval()
            head.eval()
            merger.base_model.heads.head = head
            with torch.no_grad():
                logits_eval = merger.base_model(x)
                preds = logits_eval.argmax(dim=-1)
                correct += preds.eq(y).sum().item()
                total += x.size(0)
                
            # 2. Adaptation Step
            if method != "static" and optimizer is not None:
                merger.base_model.train()
                head.train()
                merger.base_model.heads.head = head
                
                # Get expert soft pseudo-labels (unsupervised distillation) to prevent collapse
                with torch.no_grad():
                    expert_logits = expert_model(x)
                    expert_probs = F.softmax(expert_logits, dim=-1)
                
                def compute_loss():
                    logits = merger.base_model(x)
                    # KL Divergence from Expert predictions (unsupervised soft distillation)
                    loss_distill = F.kl_div(F.log_softmax(logits, dim=-1), expert_probs, reduction="batchmean")
                    
                    # Add CAS-Merge Covariance-Aligned Low-Rank Projection (CALP) regularization
                    loss_calp = 0.0
                    if method == "cas_merge":
                        # Beta parameter (regularization weight)
                        beta = 1.0
                        for m_name, m in merger.base_model.named_modules():
                            if isinstance(m, LoRALinear) and m.current_activations is not None:
                                # Compute activation variance
                                act = m.current_activations
                                if act.dim() == 3:
                                    act_flat = act.view(-1, act.size(-1))
                                else:
                                    act_flat = act
                                var = act_flat.var(dim=0, unbiased=False) # [in_features]
                                
                                # Retrieve expert weights
                                key_a = m_name + ".lora_a"
                                key_b = m_name + ".lora_b"
                                w_a_cifar = merger.expert_cifar_lora[key_a]
                                w_a_svhn = merger.expert_svhn_lora[key_a]
                                w_b_cifar = merger.expert_cifar_lora[key_b]
                                w_b_svhn = merger.expert_svhn_lora[key_b]
                                
                                # Sigmoid coefficient
                                coeff_idx = 12
                                for i in range(12):
                                    if f"encoder_layer_{i}" in m_name:
                                        coeff_idx = i
                                        break
                                lmbda = torch.sigmoid(merger.coeffs[coeff_idx])
                                
                                # Current merged low-rank weight
                                dyn_lora_a = lmbda * w_a_cifar + (1.0 - lmbda) * w_a_svhn
                                dyn_lora_b = lmbda * w_b_cifar + (1.0 - lmbda) * w_b_svhn
                                merged_W = dyn_lora_b @ dyn_lora_a * m.scaling
                                # Ideal interpolated weight (target)
                                target_W = (lmbda * (w_b_cifar @ w_a_cifar) + (1.0 - lmbda) * (w_b_svhn @ w_a_svhn)) * m.scaling
                                
                                # Compute column-scaled Frobenius norm discrepancy
                                diff = merged_W - target_W
                                loss_calp += torch.sum((diff ** 2) * (var.unsqueeze(0) + 1e-5))
                                
                        loss_distill += beta * loss_calp
                        
                    return loss_distill
                
                # Apply optimizer steps
                if method in ["sat_tta", "cas_merge"]:
                    # Sharpness-Aware Minimization (SAM) 2-step update
                    rho = 0.05 # perturbation radius
                    
                    # First forward-backward pass
                    optimizer.zero_grad()
                    loss = compute_loss()
                    loss.backward()
                    
                    # Apply SAM perturbation
                    with torch.no_grad():
                        grad_norm = torch.norm(torch.stack([p.grad.norm(2) for p in trainable_params if p.grad is not None]), 2)
                        if grad_norm > 1e-5:
                            for p in trainable_params:
                                if p.grad is not None:
                                    eps = p.grad * (rho / (grad_norm + 1e-12))
                                    p.add_(eps)
                                    p.eps = eps
                                    
                    # Second forward-backward pass on perturbed parameters
                    optimizer.zero_grad()
                    loss_perturbed = compute_loss()
                    loss_perturbed.backward()
                    
                    # Restore and step
                    with torch.no_grad():
                        for p in trainable_params:
                            if hasattr(p, "eps"):
                                p.sub_(p.eps)
                                del p.eps
                    optimizer.step()
                    
                else:
                    # Standard gradient descent
                    optimizer.zero_grad()
                    loss = compute_loss()
                    loss.backward()
                    optimizer.step()
                
                # Update LoRA weights based on new coefficients
                with torch.no_grad():
                    merger.apply_merge()
                    
        acc = (correct / total) * 100
        print(f"-> {task_name.upper()} Test Accuracy: {acc:.2f}%")
        task_accuracies[task_name] = acc
        
    avg_acc = sum(task_accuracies.values()) / len(task_accuracies)
    print(f"-> MULTI-TASK AVERAGE ACCURACY: {avg_acc:.2f}%")
    task_accuracies["average"] = avg_acc
    return task_accuracies

# 7. Main Execution Flow
def main():
    print("=========================================")
    print("  CAS-Merge: Model Merging Experiments  ")
    print("=========================================")
    
    # Check assets directory
    if not os.path.exists("./data") or not os.path.exists("./torch_cache"):
        print("Assets not found. Please run download_assets.py first on a CPU node.")
        return
        
    loaders = get_dataloaders()
    
    # Train experts or load if already present
    expert_cifar_path = "expert_cifar10.pt"
    expert_svhn_path = "expert_svhn.pt"
    
    if os.path.exists(expert_cifar_path) and os.path.exists(expert_svhn_path):
        print("\nLoading pre-trained experts...")
        expert_cifar_state = torch.load(expert_cifar_path, map_location=DEVICE)
        expert_svhn_state = torch.load(expert_svhn_path, map_location=DEVICE)
    else:
        print("\nTraining experts from scratch...")
        expert_cifar = train_expert("CIFAR10", loaders["cifar_train"], epochs=3)
        expert_svhn = train_expert("SVHN", loaders["svhn_train"], epochs=3)
        
        # Save experts
        print("\nSaving expert weights...")
        torch.save(expert_cifar.state_dict(), expert_cifar_path)
        torch.save(expert_svhn.state_dict(), expert_svhn_path)
        
        expert_cifar_state = expert_cifar.state_dict()
        expert_svhn_state = expert_svhn.state_dict()
        
    # Standard single expert evaluation
    # Let's verify single expert performance on both tasks (interference analysis)
    base_model = get_base_vit()
    inject_lora(base_model)
    base_model.heads.head = nn.Linear(768, 10)
    
    cifar_head = nn.Linear(768, 10).to(DEVICE)
    cifar_head.load_state_dict({k.replace("heads.head.", ""): v for k, v in expert_cifar_state.items() if "heads.head" in k})
    svhn_head = nn.Linear(768, 10).to(DEVICE)
    svhn_head.load_state_dict({k.replace("heads.head.", ""): v for k, v in expert_svhn_state.items() if "heads.head" in k})
    
    base_model = base_model.to(DEVICE)
    
    # Eval CIFAR-10 expert on CIFAR-10 & SVHN
    base_model.load_state_dict(expert_cifar_state)
    acc_cifar_on_cifar = evaluate(base_model, loaders["cifar_test"], cifar_head)
    acc_cifar_on_svhn = evaluate(base_model, loaders["svhn_test"], svhn_head)
    
    # Eval SVHN expert on CIFAR-10 & SVHN
    base_model.load_state_dict(expert_svhn_state)
    acc_svhn_on_cifar = evaluate(base_model, loaders["cifar_test"], cifar_head)
    acc_svhn_on_svhn = evaluate(base_model, loaders["svhn_test"], svhn_head)
    
    print("\n=== Single Expert Baselines (Clean) ===")
    print(f"CIFAR-10 Expert -> CIFAR-10 Acc: {acc_cifar_on_cifar:.2f}%, SVHN Acc: {acc_cifar_on_svhn:.2f}%")
    print(f"SVHN Expert     -> CIFAR-10 Acc: {acc_svhn_on_cifar:.2f}%, SVHN Acc: {acc_svhn_on_svhn:.2f}%")
    
    # Run TTA evaluations
    methods = ["static", "adamerging", "symerge", "sat_tta", "cas_merge"]
    
    results_clean = {}
    results_corrupt = {}
    
    # 1. Clean test streams
    for m in methods:
        res = run_tta_evaluation(loaders, expert_cifar_state, expert_svhn_state, method=m, corrupt=False)
        results_clean[m] = res
        
    # 2. Corrupted test streams (OOD)
    for m in methods:
        res = run_tta_evaluation(loaders, expert_cifar_state, expert_svhn_state, method=m, corrupt=True)
        results_corrupt[m] = res
        
    # Summary of results
    print("\n=========================================")
    print("             SUMMARY OF RESULTS           ")
    print("=========================================")
    print("Method | Clean CIFAR10 | Clean SVHN | Clean Avg | OOD CIFAR10 | OOD SVHN | OOD Avg")
    print("-" * 90)
    for m in methods:
        rc = results_clean[m]
        ro = results_corrupt[m]
        print(f"{m:<10} | {rc['cifar10']:.2f}% | {rc['svhn']:.2f}% | {rc['average']:.2f}% | {ro['cifar10']:.2f}% | {ro['svhn']:.2f}% | {ro['average']:.2f}%")
        
    # Save results to JSON
    with open("results.json", "w") as f:
        json.dump({"clean": results_clean, "corrupt": results_corrupt}, f, indent=4)
        
    # Generate visualization
    labels = [m.upper() for m in methods]
    clean_avgs = [results_clean[m]["average"] for m in methods]
    corrupt_avgs = [results_corrupt[m]["average"] for m in methods]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, clean_avgs, width, label='Clean Stream', color='#1f77b4')
    rects2 = ax.bar(x + width/2, corrupt_avgs, width, label='Corrupted Stream (OOD)', color='#ff7f0e')
    
    ax.set_ylabel('Multi-Task Average Accuracy (%)')
    ax.set_title('Performance Comparison across Test-Time Adaptation Methods')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add values on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
            
    autolabel(rects1)
    autolabel(rects2)
    
    fig.tight_layout()
    plt.savefig("results_comparison.png", dpi=300)
    print("\nGenerated 'results_comparison.png' visualization successfully.")

if __name__ == "__main__":
    main()
