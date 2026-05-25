import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

# Force clean execution and disable symlinks warning
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
torch.backends.cudnn.enabled = False

# 1. SAM Optimizer Implementation
class SAM(optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = p.grad * scale
                p.add_(e_w)  # climb to local maximum
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # retrieve original point
        self.base_optimizer.step()  # step on original parameters with new gradient
        if zero_grad: self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

# 2. OrthoMerge Helper functions
def orthomerge_weights(w_a, w_b, w_0):
    """
    Perform Orthogonal-Residual Decoupling and Magnitude-Corrected Merging
    on weights w_a, w_b relative to base weight w_0.
    """
    # Store original shape
    orig_shape = w_a.shape
    
    # Flatten to 2D: [d_out, d_in]
    if len(orig_shape) > 2:
        d_out = orig_shape[0]
        d_in = torch.numel(w_a) // d_out
        W_a_2d = w_a.view(d_out, d_in)
        W_b_2d = w_b.view(d_out, d_in)
        W_0_2d = w_0.view(d_out, d_in)
    elif len(orig_shape) == 2:
        W_a_2d = w_a
        W_b_2d = w_b
        W_0_2d = w_0
    else:
        # 1D or scalar: fallback to standard linear interpolation
        return 0.5 * (w_a + w_b)

    device = w_a.device
    W_0_2d = W_0_2d.to(device)
    I = torch.eye(W_a_2d.shape[0], device=device)
    
    # Extract Orthogonal components via Procrustes: Ri = U_i V_i^T where U,S,V = SVD(W_i W_0^T)
    # Task A
    M_a = W_a_2d @ W_0_2d.T
    U_a, S_a, V_a_T = torch.linalg.svd(M_a)
    R_a = U_a @ V_a_T
    
    # Task B
    M_b = W_b_2d @ W_0_2d.T
    U_b, S_b, V_b_T = torch.linalg.svd(M_b)
    R_b = U_b @ V_b_T
    
    # Inverse Cayley Transform to Lie algebra: Q_i = (R_i - I)(R_i + I)^-1
    # Adding 1e-6 * I for perfect numerical stability
    R_a_stab = R_a + I + 1e-6 * I
    R_b_stab = R_b + I + 1e-6 * I
    Q_a = torch.linalg.solve(R_a_stab, R_a - I)
    Q_b = torch.linalg.solve(R_b_stab, R_b - I)
    
    # Magnitude-Corrected Merging of orthogonal components
    Q_sum = Q_a + Q_b
    norm_sum = torch.linalg.norm(Q_sum, ord='fro')
    sum_norm = torch.linalg.norm(Q_a, ord='fro') + torch.linalg.norm(Q_b, ord='fro')
    
    c = sum_norm / (norm_sum + 1e-8)
    Q_merged = c * (0.5 * Q_sum)
    
    # Map back to Orthogonal Group via Cayley: R_merged = (I + Q_merged)(I - Q_merged)^-1
    R_merged = torch.linalg.solve(I - Q_merged, I + Q_merged)
    
    # Residuals
    rho_a = W_a_2d - R_a @ W_0_2d
    rho_b = W_b_2d - R_b @ W_0_2d
    
    # Standard Euclidean merge of residuals
    rho_merged = 0.5 * (rho_a + rho_b)
    
    # Reconstruct final 2D weight
    W_final_2d = R_merged @ W_0_2d + rho_merged
    
    # Return to original shape
    return W_final_2d.view(orig_shape)


def perform_orthomerge(model_a, model_b, model_base):
    """
    Traverse through all parameters and apply OrthoMerge to linear/conv weights
    and linear averaging to biases and other params.
    """
    model_merged = copy.deepcopy(model_base)
    
    state_a = model_a.state_dict()
    state_b = model_b.state_dict()
    state_0 = model_base.state_dict()
    state_m = model_merged.state_dict()
    
    total_layers_merged = 0
    total_residual_norm_a = 0.0
    total_residual_norm_b = 0.0
    
    for name, p_0 in model_base.named_parameters():
        if not p_0.requires_grad:
            continue
            
        p_a = state_a[name]
        p_b = state_b[name]
        
        # We apply OrthoMerge to weights of Linear and Conv2d layers
        # Usually they contain 'weight' in their names and have dim >= 2
        if 'weight' in name and p_0.dim() >= 2:
            w_merged = orthomerge_weights(p_a, p_b, p_0)
            state_m[name].copy_(w_merged)
            
            # Record residual stats for our scientific analysis
            # Compute R_i for logging
            orig_shape = p_a.shape
            d_out = orig_shape[0]
            d_in = torch.numel(p_a) // d_out
            W_a_2d = p_a.view(d_out, d_in)
            W_b_2d = p_b.view(d_out, d_in)
            W_0_2d = p_0.view(d_out, d_in).to(p_a.device)
            
            with torch.no_grad():
                # Task A SVD & R
                M_a = W_a_2d @ W_0_2d.T
                U_a, _, V_a_T = torch.linalg.svd(M_a)
                R_a = U_a @ V_a_T
                res_a = W_a_2d - R_a @ W_0_2d
                
                # Task B SVD & R
                M_b = W_b_2d @ W_0_2d.T
                U_b, _, V_b_T = torch.linalg.svd(M_b)
                R_b = U_b @ V_b_T
                res_b = W_b_2d - R_b @ W_0_2d
                
                total_residual_norm_a += torch.linalg.norm(res_a).item()
                total_residual_norm_b += torch.linalg.norm(res_b).item()
                total_layers_merged += 1
        else:
            # Fallback to simple Task Arithmetic for biases/1D/batchnorm
            w_merged = 0.5 * (p_a + p_b)
            state_m[name].copy_(w_merged)
            
    # Also handle non-parameter buffers like Batch Normalization running statistics
    for name, buf_0 in model_base.named_buffers():
        buf_a = state_a[name]
        buf_b = state_b[name]
        # Average running mean/var, keep class/int variables as is
        if buf_0.is_floating_point():
            state_m[name].copy_(0.5 * (buf_a + buf_b))
        else:
            state_m[name].copy_(buf_a) # fallback to a
            
    model_merged.load_state_dict(state_m)
    avg_res_norm = (total_residual_norm_a + total_residual_norm_b) / (2 * total_layers_merged + 1e-8)
    return model_merged, avg_res_norm


def perform_task_arithmetic(model_a, model_b, model_base):
    """
    Standard linear model merging: W_merged = W_0 + 0.5 * (Delta W_a + Delta W_b) = 0.5 * (W_a + W_b)
    """
    model_merged = copy.deepcopy(model_base)
    state_a = model_a.state_dict()
    state_b = model_b.state_dict()
    state_m = model_merged.state_dict()
    
    for name in state_m.keys():
        val_a = state_a[name]
        val_b = state_b[name]
        if val_a.is_floating_point():
            state_m[name].copy_(0.5 * (val_a + val_b))
        else:
            state_m[name].copy_(val_a)
            
    model_merged.load_state_dict(state_m)
    return model_merged


# 3. Model Fine-Tuning Function
def train_model(model, dataloader, epochs, device, optimizer_type="sgd", lr=0.01, rho=0.05):
    model = model.to(device)
    model.train()
    
    # Loss criteria
    criterion = nn.CrossEntropyLoss()
    
    if optimizer_type == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    elif optimizer_type == "sam":
        base_optimizer = optim.SGD
        optimizer = SAM(model.parameters(), base_optimizer, rho=rho, lr=lr, momentum=0.9, weight_decay=5e-4)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer if optimizer_type == "sgd" else optimizer.base_optimizer, 
        T_max=epochs
    )
    
    print(f"Fine-tuning model using {optimizer_type.upper()} for {epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            if optimizer_type == "sgd":
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            else: # SAM
                # First step
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.first_step(zero_grad=True)
                
                # Second step
                criterion(model(images), labels).backward()
                optimizer.second_step(zero_grad=True)
                
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        scheduler.step()
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        print(f"  Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")
        
    print(f"Fine-tuning complete in {time.time() - start_time:.1f}s.")
    return model


# 4. Evaluation Function
def evaluate_model(model, dataloader, device, task_name=""):
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    acc = 100. * correct / total
    print(f"  Accuracy on {task_name}: {acc:.2f}%")
    return acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 5. Data Preparation
    print("Preparing CIFAR-10 Datasets...")
    # Resize to 128x128 for speed + ImageNet feature utilization
    transform_train = transforms.Compose([
        transforms.Resize(128),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize(128),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Load dataset
    full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    full_testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    # Split into Task A (labels 0-4) and Task B (labels 5-9)
    train_labels = full_trainset.targets
    test_labels = full_testset.targets
    
    task_a_train_indices = [i for i, label in enumerate(train_labels) if label < 5]
    task_b_train_indices = [i for i, label in enumerate(train_labels) if label >= 5]
    
    task_a_test_indices = [i for i, label in enumerate(test_labels) if label < 5]
    task_b_test_indices = [i for i, label in enumerate(test_labels) if label >= 5]
    
    # Subsets
    train_a_set = Subset(full_trainset, task_a_train_indices)
    train_b_set = Subset(full_trainset, task_b_train_indices)
    
    test_a_set = Subset(full_testset, task_a_test_indices)
    test_b_set = Subset(full_testset, task_b_test_indices)
    
    # Dataloaders
    batch_size = 128
    loader_train_a = DataLoader(train_a_set, batch_size=batch_size, shuffle=True, num_workers=4)
    loader_train_b = DataLoader(train_b_set, batch_size=batch_size, shuffle=True, num_workers=4)
    
    loader_test_a = DataLoader(test_a_set, batch_size=batch_size, shuffle=False, num_workers=4)
    loader_test_b = DataLoader(test_b_set, batch_size=batch_size, shuffle=False, num_workers=4)
    loader_test_full = DataLoader(full_testset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"Task A (classes 0-4) | Train: {len(train_a_set)} | Test: {len(test_a_set)}")
    print(f"Task B (classes 5-9) | Train: {len(train_b_set)} | Test: {len(test_b_set)}")
    
    # 6. Initialize Base Pre-trained Model
    print("\nInitializing Pretrained ResNet-18 Base Model...")
    # Ensure it outputs 10 classes
    base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    base_model.fc = nn.Linear(base_model.fc.in_features, 10)
    
    # Save the base model checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(base_model.state_dict(), "checkpoints/model_base.pt")
    
    # --- FINE-TUNING TASK A ---
    # Standard SGD
    file_a_std = "checkpoints/model_a_standard.pt"
    if os.path.exists(file_a_std):
        print(f"Loading existing checkpoint {file_a_std}")
        model_a_std = models.resnet18(weights=None)
        model_a_std.fc = nn.Linear(model_a_std.fc.in_features, 10)
        model_a_std.load_state_dict(torch.load(file_a_std))
    else:
        print("\n--- Training Task A (Standard SGD) ---")
        model = copy.deepcopy(base_model)
        model_a_std = train_model(model, loader_train_a, epochs=5, device=device, optimizer_type="sgd", lr=0.005)
        torch.save(model_a_std.state_dict(), file_a_std)
        
    # SAM
    file_a_sam = "checkpoints/model_a_sam.pt"
    if os.path.exists(file_a_sam):
        print(f"Loading existing checkpoint {file_a_sam}")
        model_a_sam = models.resnet18(weights=None)
        model_a_sam.fc = nn.Linear(model_a_sam.fc.in_features, 10)
        model_a_sam.load_state_dict(torch.load(file_a_sam))
    else:
        print("\n--- Training Task A (SAM) ---")
        model = copy.deepcopy(base_model)
        model_a_sam = train_model(model, loader_train_a, epochs=5, device=device, optimizer_type="sam", lr=0.005, rho=0.05)
        torch.save(model_a_sam.state_dict(), file_a_sam)
        
    # --- FINE-TUNING TASK B ---
    # Standard SGD
    file_b_std = "checkpoints/model_b_standard.pt"
    if os.path.exists(file_b_std):
        print(f"Loading existing checkpoint {file_b_std}")
        model_b_std = models.resnet18(weights=None)
        model_b_std.fc = nn.Linear(model_b_std.fc.in_features, 10)
        model_b_std.load_state_dict(torch.load(file_b_std))
    else:
        print("\n--- Training Task B (Standard SGD) ---")
        model = copy.deepcopy(base_model)
        model_b_std = train_model(model, loader_train_b, epochs=5, device=device, optimizer_type="sgd", lr=0.005)
        torch.save(model_b_std.state_dict(), file_b_std)
        
    # SAM
    file_b_sam = "checkpoints/model_b_sam.pt"
    if os.path.exists(file_b_sam):
        print(f"Loading existing checkpoint {file_b_sam}")
        model_b_sam = models.resnet18(weights=None)
        model_b_sam.fc = nn.Linear(model_b_sam.fc.in_features, 10)
        model_b_sam.load_state_dict(torch.load(file_b_sam))
    else:
        print("\n--- Training Task B (SAM) ---")
        model = copy.deepcopy(base_model)
        model_b_sam = train_model(model, loader_train_b, epochs=5, device=device, optimizer_type="sam", lr=0.005, rho=0.05)
        torch.save(model_b_sam.state_dict(), file_b_sam)
        
    print("\nAll fine-tuning completed. Evaluating expert models...")
    print("Standard Expert A:")
    acc_a_std_on_a = evaluate_model(model_a_std, loader_test_a, device, "Task A Test")
    acc_a_std_on_b = evaluate_model(model_a_std, loader_test_b, device, "Task B Test")
    
    print("SAM Expert A:")
    acc_a_sam_on_a = evaluate_model(model_a_sam, loader_test_a, device, "Task A Test")
    acc_a_sam_on_b = evaluate_model(model_a_sam, loader_test_b, device, "Task B Test")
    
    print("Standard Expert B:")
    acc_b_std_on_a = evaluate_model(model_b_std, loader_test_a, device, "Task A Test")
    acc_b_std_on_b = evaluate_model(model_b_std, loader_test_b, device, "Task B Test")
    
    print("SAM Expert B:")
    acc_b_sam_on_a = evaluate_model(model_b_sam, loader_test_a, device, "Task A Test")
    acc_b_sam_on_b = evaluate_model(model_b_sam, loader_test_b, device, "Task B Test")
    
    # 7. MODEL MERGING
    print("\n================== MODEL MERGING ==================")
    
    # Condition 1: Standard SGD Experts
    print("\n>>> Merging Standard SGD Experts...")
    # Task Arithmetic (Linear Average)
    print("Applying Task Arithmetic to Standard Experts...")
    model_std_ta = perform_task_arithmetic(model_a_std, model_b_std, base_model)
    acc_std_ta_a = evaluate_model(model_std_ta, loader_test_a, device, "Task A Test")
    acc_std_ta_b = evaluate_model(model_std_ta, loader_test_b, device, "Task B Test")
    acc_std_ta_full = evaluate_model(model_std_ta, loader_test_full, device, "Full Test")
    
    # OrthoMerge
    print("\nApplying OrthoMerge to Standard Experts...")
    model_std_ortho, res_norm_std = perform_orthomerge(model_a_std, model_b_std, base_model)
    acc_std_ortho_a = evaluate_model(model_std_ortho, loader_test_a, device, "Task A Test")
    acc_std_ortho_b = evaluate_model(model_std_ortho, loader_test_b, device, "Task B Test")
    acc_std_ortho_full = evaluate_model(model_std_ortho, loader_test_full, device, "Full Test")
    
    # Condition 2: SAM Experts
    print("\n>>> Merging SAM Experts...")
    # Task Arithmetic (Linear Average)
    print("Applying Task Arithmetic to SAM Experts...")
    model_sam_ta = perform_task_arithmetic(model_a_sam, model_b_sam, base_model)
    acc_sam_ta_a = evaluate_model(model_sam_ta, loader_test_a, device, "Task A Test")
    acc_sam_ta_b = evaluate_model(model_sam_ta, loader_test_b, device, "Task B Test")
    acc_sam_ta_full = evaluate_model(model_sam_ta, loader_test_full, device, "Full Test")
    
    # OrthoMerge
    print("\nApplying OrthoMerge to SAM Experts...")
    model_sam_ortho, res_norm_sam = perform_orthomerge(model_a_sam, model_b_sam, base_model)
    acc_sam_ortho_a = evaluate_model(model_sam_ortho, loader_test_a, device, "Task A Test")
    acc_sam_ortho_b = evaluate_model(model_sam_ortho, loader_test_b, device, "Task B Test")
    acc_sam_ortho_full = evaluate_model(model_sam_ortho, loader_test_full, device, "Full Test")
    
    # Print comparison
    print("\n" + "="*50)
    print("                     FINAL RESULTS COMPARISON")
    print("="*50)
    print(f"Standard Experts TA    | Task A: {acc_std_ta_a:.2f}% | Task B: {acc_std_ta_b:.2f}% | Full: {acc_std_ta_full:.2f}%")
    print(f"Standard Experts Ortho | Task A: {acc_std_ortho_a:.2f}% | Task B: {acc_std_ortho_b:.2f}% | Full: {acc_std_ortho_full:.2f}%")
    print(f"SAM Experts TA         | Task A: {acc_sam_ta_a:.2f}% | Task B: {acc_sam_ta_b:.2f}% | Full: {acc_sam_ta_full:.2f}%")
    print(f"SAM Experts Ortho      | Task A: {acc_sam_ortho_a:.2f}% | Task B: {acc_sam_ortho_b:.2f}% | Full: {acc_sam_ortho_full:.2f}%")
    print("-"*50)
    print(f"Standard Procrustes Residual Norm: {res_norm_std:.6f}")
    print(f"SAM Procrustes Residual Norm:        {res_norm_sam:.6f}")
    print(f"Relative Residual reduction:        {((res_norm_std - res_norm_sam)/res_norm_std)*100:.2f}%")
    print("="*50)
    
    # Check Hypothesis
    is_hypo_true = acc_sam_ortho_full > acc_std_ortho_full and res_norm_sam < res_norm_std
    print(f"Hypothesis Validated? {is_hypo_true}")
    
    # 8. Plot and save results
    methods = ['Std-TA', 'Std-Ortho', 'SAM-TA', 'SAM-Ortho']
    accuracies = [acc_std_ta_full, acc_std_ortho_full, acc_sam_ta_full, acc_sam_ortho_full]
    task_a_accs = [acc_std_ta_a, acc_std_ortho_a, acc_sam_ta_a, acc_sam_ortho_a]
    task_b_accs = [acc_std_ta_b, acc_std_ortho_b, acc_sam_ta_b, acc_sam_ortho_b]
    
    # Accuracy comparison plot
    plt.figure(figsize=(10, 6))
    x = range(len(methods))
    width = 0.25
    
    plt.bar([i - width for i in x], task_a_accs, width, label='Task A (Classes 0-4)', color='#1f77b4')
    plt.bar(x, task_b_accs, width, label='Task B (Classes 5-9)', color='#ff7f0e')
    plt.bar([i + width for i in x], accuracies, width, label='Full CIFAR-10 Test', color='#2ca02c')
    
    plt.xlabel('Merging Approach')
    plt.ylabel('Accuracy (%)')
    plt.title('Comparison of Model Merging Methods: Standard vs. SAM Experts')
    plt.xticks(x, methods)
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    os.makedirs("plots", exist_ok=True)
    plt.savefig('plots/merging_accuracy_comparison.png', dpi=300)
    plt.close()
    
    # Residual comparison plot
    plt.figure(figsize=(6, 5))
    res_norms = [res_norm_std, res_norm_sam]
    res_labels = ['Standard Experts', 'SAM Experts']
    plt.bar(res_labels, res_norms, color=['#d62728', '#9467bd'], width=0.5)
    plt.ylabel('Average Procrustes Residual Norm')
    plt.title('Orthogonal-Residual Decoupling: Standard vs. SAM')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('plots/procrustes_residual_comparison.png', dpi=300)
    plt.close()
    
    print("Diagnostic plots saved under plots/ directory.")
    
    # Save the best model as submission.pdf template placeholder later, and save the metrics
    with open("results.txt", "w") as f:
        f.write(f"Standard Experts TA | Task A: {acc_std_ta_a:.2f} | Task B: {acc_std_ta_b:.2f} | Full: {acc_std_ta_full:.2f}\n")
        f.write(f"Standard Experts Ortho | Task A: {acc_std_ortho_a:.2f} | Task B: {acc_std_ortho_b:.2f} | Full: {acc_std_ortho_full:.2f}\n")
        f.write(f"SAM Experts TA | Task A: {acc_sam_ta_a:.2f} | Task B: {acc_sam_ta_b:.2f} | Full: {acc_sam_ta_full:.2f}\n")
        f.write(f"SAM Experts Ortho | Task A: {acc_sam_ortho_a:.2f} | Task B: {acc_sam_ortho_b:.2f} | Full: {acc_sam_ortho_full:.2f}\n")
        f.write(f"Standard Residual Norm: {res_norm_std:.6f}\n")
        f.write(f"SAM Residual Norm: {res_norm_sam:.6f}\n")
        f.write(f"Relative Residual reduction: {((res_norm_std - res_norm_sam)/res_norm_std)*100:.2f}%\n")
        f.write(f"Hypothesis Validated: {is_hypo_true}\n")

if __name__ == "__main__":
    main()
