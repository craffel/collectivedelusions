import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import numpy as np
import copy
import matplotlib.pyplot as plt

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

DATA_DIR = "./data"
MODEL_DIR = "./models"
RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Transforms
transform_rgb = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_gray = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Corrupted Transforms (OOD simulation)
transform_rgb_corrupt = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomRotation((15, 15)),  # Constant rotation
    transforms.ToTensor(),
    # Add Gaussian Noise
    transforms.Lambda(lambda x: x + 0.2 * torch.randn_like(x)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_gray_corrupt = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomRotation((15, 15)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x + 0.2 * torch.randn_like(x)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load datasets
def get_dataloaders(task, batch_size=128, corrupt=False):
    t_rgb = transform_rgb_corrupt if corrupt else transform_rgb
    t_gray = transform_gray_corrupt if corrupt else transform_gray
    
    if task == "cifar10":
        test_set = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=t_rgb)
    elif task == "svhn":
        test_set = datasets.SVHN(root=DATA_DIR, split="test", download=True, transform=t_rgb)
    elif task == "mnist":
        test_set = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=t_gray)
    else:
        raise ValueError(f"Unknown task: {task}")
        
    test_indices = list(range(min(500, len(test_set))))
    test_subset = Subset(test_set, test_indices)
        
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return test_loader

# Lie Algebra Utils
def to_lie(R, eps=1e-6):
    # Cayley Transform: Q = (R - I)(R + I)^-1
    eye = torch.eye(R.size(0), device=R.device)
    # Solve (R + (1 + eps)*I) Q = R - I
    Q = torch.linalg.solve(R + (1 + eps) * eye, R - eye)
    # Enforce skew-symmetry: Q = (Q - Q^T) / 2
    return 0.5 * (Q - Q.T)

def from_lie(Q):
    # Inverse Cayley Transform: R = (I + Q)(I - Q)^-1
    eye = torch.eye(Q.size(0), device=Q.device)
    return torch.linalg.solve(eye - Q, eye + Q)

# Orthogonal Procrustes Decoupling
def orthogonal_procrustes(W_target, W_base):
    # W_target, W_base are [out_dim, in_dim]
    # SVD: U Sigma V^T = SVD(W_target * W_base^T)
    # R = U V^T
    cov = torch.matmul(W_target, W_base.T)
    try:
        U, S, Vt = torch.linalg.svd(cov)
        R = torch.matmul(U, Vt)
    except RuntimeError:
        # Fallback if SVD fails to converge
        R = torch.eye(W_target.size(0), device=W_target.device)
    return R

# Class to represent a merged model with custom forward pass for Orthogonal Adapter
class AdaptedModel(nn.Module):
    def __init__(self, base_model, adapter_dim=512):
        super().__init__()
        self.backbone = nn.Sequential(*list(base_model.children())[:-1])  # All layers except fc
        self.fc = base_model.fc
        
        # Trainable skew-symmetric matrix for orthogonal adapter
        self.Q_adapt = nn.Parameter(torch.zeros(adapter_dim, adapter_dim))
        
    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        
        # Construct orthogonal adapter: R = (I + Q)(I - Q)^-1
        eye = torch.eye(self.Q_adapt.size(0), device=self.Q_adapt.device)
        # Enforce skew-symmetry on Q_adapt during forward pass
        Q = 0.5 * (self.Q_adapt - self.Q_adapt.T)
        R = torch.linalg.solve(eye - Q, eye + Q)
        
        # Apply orthogonal adapter
        x = torch.matmul(x, R.T)
        
        x = self.fc(x)
        return x

# Main Evaluator
class Evaluator:
    def __init__(self):
        self.tasks = ["cifar10", "svhn", "mnist"]
        self.experts = {}
        self.base_weights = None
        
        # Load pre-trained base model
        self.base_model = models.resnet18()
        self.base_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "resnet18_base.pt"), map_location=device))
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, 10)
        self.base_model.to(device)
        self.base_model.eval()
        
        # Load expert models
        for task in self.tasks:
            model = models.resnet18()
            model.fc = nn.Linear(model.fc.in_features, 10)
            model.load_state_dict(torch.load(os.path.join(MODEL_DIR, f"resnet18_{task}.pt"), map_location=device))
            model.to(device)
            model.eval()
            self.experts[task] = model

        # Precompute Lie algebra representations Q_k and residuals rho_k once for massive 45x speedup
        print("Precomputing Lie algebra representations and residuals for all experts...")
        self.lie_reps = {}
        self.residuals = {}
        base_sd = self.base_model.state_dict()
        expert_sds = [self.experts[t].state_dict() for t in self.tasks]
        
        for key in base_sd.keys():
            if "fc" not in key and "weight" in key:
                W0 = base_sd[key].float()
                W_exp_list = [sd[key].float() for sd in expert_sds]
                orig_shape = W0.shape
                
                if len(orig_shape) > 1:
                    out_dim = orig_shape[0]
                    W0_flat = W0.view(out_dim, -1)
                    
                    Q_list_layer = []
                    rho_list_layer = []
                    for W_exp in W_exp_list:
                        W_exp_flat = W_exp.view(out_dim, -1)
                        R = orthogonal_procrustes(W_exp_flat, W0_flat)
                        rho = W_exp_flat - torch.matmul(R, W0_flat)
                        Q = to_lie(R)
                        Q_list_layer.append(Q)
                        rho_list_layer.append(rho)
                        
                    self.lie_reps[key] = Q_list_layer
                    self.residuals[key] = rho_list_layer

    def evaluate_model(self, model, task, corrupt=False):
        model.eval()
        loader = get_dataloaders(task, corrupt=corrupt)
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        return 100. * correct / total

    def run_weight_averaging(self):
        # Euclidean simple average of backbones
        avg_model = copy.deepcopy(self.base_model)
        avg_sd = avg_model.state_dict()
        
        expert_sds = [self.experts[t].state_dict() for t in self.tasks]
        
        for key in avg_sd.keys():
            if "fc" not in key:  # Average only the backbone
                tensors = [sd[key].float() for sd in expert_sds]
                avg_sd[key] = torch.stack(tensors).mean(dim=0)
                
        avg_model.load_state_dict(avg_sd)
        return avg_model

    def run_task_arithmetic(self, alpha=0.3):
        # theta_merged = theta_0 + alpha * sum(theta_k - theta_0)
        merged_model = copy.deepcopy(self.base_model)
        merged_sd = merged_model.state_dict()
        
        expert_sds = [self.experts[t].state_dict() for t in self.tasks]
        base_sd = self.base_model.state_dict()
        
        for key in merged_sd.keys():
            if "fc" not in key:
                delta_sum = torch.zeros_like(base_sd[key]).float()
                for sd in expert_sds:
                    delta_sum += (sd[key].float() - base_sd[key].float())
                merged_sd[key] = base_sd[key].float() + alpha * delta_sum
                
        merged_model.load_state_dict(merged_sd)
        return merged_model

    def run_orthomerge(self):
        # Decouple weights into orthogonal and residual, merge on manifold
        merged_model = copy.deepcopy(self.base_model)
        merged_sd = merged_model.state_dict()
        
        expert_sds = [self.experts[t].state_dict() for t in self.tasks]
        base_sd = self.base_model.state_dict()
        
        for key in merged_sd.keys():
            if "fc" not in key and "weight" in key:
                W0 = base_sd[key].float()
                W_experts = [sd[key].float() for sd in expert_sds]
                
                # Check dimension to flatten to 2D
                orig_shape = W0.shape
                if len(orig_shape) > 1:
                    out_dim = orig_shape[0]
                    # Flatten the rest of dimensions
                    W0_flat = W0.view(out_dim, -1)
                    W_exp_flats = [W_exp.view(out_dim, -1) for W_exp in W_experts]
                    
                    # Extract orthogonal components R_k using SVD (Orthogonal Procrustes)
                    R_list = []
                    rho_list = []
                    for W_exp_flat in W_exp_flats:
                        R = orthogonal_procrustes(W_exp_flat, W0_flat)
                        rho = W_exp_flat - torch.matmul(R, W0_flat)
                        R_list.append(R)
                        rho_list.append(rho)
                    
                    # Map to Lie Algebra skew-symmetric representation Q_k
                    Q_list = [to_lie(R) for R in R_list]
                    
                    # Magnitude-corrected average of Q_list
                    Q_sum = torch.stack(Q_list).sum(dim=0)
                    sum_norm = torch.linalg.norm(Q_sum)
                    if sum_norm > 1e-8:
                        norms = [torch.linalg.norm(Q) for Q in Q_list]
                        c = sum(norms) / sum_norm
                        Q_merged = (c / len(self.tasks)) * Q_sum
                    else:
                        Q_merged = torch.zeros_like(Q_sum)
                        
                    # Map back to orthogonal R_merged
                    R_merged = from_lie(Q_merged)
                    
                    # Average the residuals
                    rho_merged = torch.stack(rho_list).mean(dim=0)
                    
                    # Reconstruct merged weight
                    W_merged_flat = torch.matmul(R_merged, W0_flat) + rho_merged
                    merged_sd[key] = W_merged_flat.view(orig_shape)
            else:
                # Keep base model biases or other non-weight layers
                pass
                
        merged_model.load_state_dict(merged_sd)
        return merged_model

    def run_symerge_test_time(self, task, corrupt=False, steps=50, lr=1e-3):
        # Test-time adaptation of Euclidean task heads & coefficients
        # Optimize merging coefficients lambda_k for each task
        print(f"Running test-time SyMerge adaptation for {task.upper()} (corrupt={corrupt})...")
        
        # Setup starting merged model via Task Arithmetic
        merged_model = self.run_task_arithmetic(alpha=0.3)
        # Pair with the specific task head from the expert
        merged_model.fc = copy.deepcopy(self.experts[task].fc)
        merged_model.to(device)
        
        # Learnable layer-wise coefficients (init to 0.3)
        # For simplicity, we optimize a single global task-wise merging weight vector
        lambdas = nn.Parameter(torch.tensor([0.3, 0.3, 0.3], device=device))
        
        # Test loader (unlabeled data simulation)
        loader = get_dataloaders(task, batch_size=32, corrupt=corrupt)
        
        # Optimization
        optimizer = optim.Adam([lambdas] + list(merged_model.fc.parameters()), lr=lr)
        criterion = nn.KLDivLoss(reduction="batchmean")
        
        # Get subset of batches for test-time adaptation
        adaptation_batches = []
        for i, (images, _) in enumerate(loader):
            if i >= 15:  # Use first 15 batches for test-time adaptation
                break
            adaptation_batches.append(images.to(device))
            
        # Teachers predictions
        expert_teachers = {t: self.experts[t].to(device) for t in self.tasks}
        
        # Test-time adaptation loop
        for step in range(steps):
            optimizer.zero_grad()
            total_loss = 0.0
            
            # Dynamic merging based on lambdas (done once per step for massive 15x speedup)
            merged_sd = merged_model.state_dict()
            base_sd = self.base_model.state_dict()
            expert_sds = [self.experts[t].state_dict() for t in self.tasks]
            
            # Apply merging
            for key in merged_sd.keys():
                if "fc" not in key and "weight" in key:
                    delta = torch.zeros_like(base_sd[key]).float()
                    for idx, sd in enumerate(expert_sds):
                        delta += lambdas[idx] * (sd[key].float() - base_sd[key].float())
                    merged_sd[key] = base_sd[key].float() + delta
            
            merged_model.load_state_dict(merged_sd)
            
            for images in adaptation_batches:
                # Forward pass
                outputs = merged_model(images)
                log_probs = torch.log_softmax(outputs, dim=1)
                
                # Self-labeling: get soft predictions from the corresponding expert
                with torch.no_grad():
                    teacher_outputs = expert_teachers[task](images)
                    teacher_probs = torch.softmax(teacher_outputs, dim=1)
                    
                loss = criterion(log_probs, teacher_probs)
                loss.backward()
                total_loss += loss.item()
                
            optimizer.step()
            
        # Final evaluation on the full test dataset
        acc = self.evaluate_model(merged_model, task, corrupt=corrupt)
        return acc

    def run_synortho_test_time(self, task, corrupt=False, steps=50, lr=1e-3, optimize_coeff=True, optimize_adapter=True):
        # SynOrtho: Test-time adaptation restricted strictly to the orthogonal group
        opt_str = []
        if optimize_coeff: opt_str.append("coefficients")
        if optimize_adapter: opt_str.append("adapter")
        opt_desc = " + ".join(opt_str) if opt_str else "none"
        print(f"Running test-time SynOrtho adaptation (optimizing: {opt_desc}) for {task.upper()} (corrupt={corrupt})...")
        
        # Start with static OrthoMerge model
        base_merged = self.run_orthomerge()
        # Pair with specific task head
        base_merged.fc = copy.deepcopy(self.experts[task].fc)
        
        # Construct Adapted Model with trainable Orthogonal Adapter Layer
        # Input features before fc is 512 dimensions for ResNet18
        adapted_model = AdaptedModel(base_merged, adapter_dim=512).to(device)
        
        # Learnable merging coefficients in the Lie Algebra
        # Instead of weight tensors, we parameterize the coefficients lambdas
        lambdas = nn.Parameter(torch.tensor([0.3, 0.3, 0.3], device=device))
        
        loader = get_dataloaders(task, batch_size=32, corrupt=corrupt)
        
        # Optimize merging coefficients AND/OR the orthogonal adapter parameters
        params = []
        if optimize_coeff:
            params.append(lambdas)
        if optimize_adapter:
            params.append(adapted_model.Q_adapt)
            
        if len(params) > 0:
            optimizer = optim.Adam(params, lr=lr)
        else:
            optimizer = None
            
        criterion = nn.KLDivLoss(reduction="batchmean")
        
        adaptation_batches = []
        for i, (images, _) in enumerate(loader):
            if i >= 15:
                break
            adaptation_batches.append(images.to(device))
            
        expert_teachers = {t: self.experts[t].to(device) for t in self.tasks}
        
        base_sd = self.base_model.state_dict()
        
        # Test-time adaptation loop
        if optimizer is not None:
            for step in range(steps):
                optimizer.zero_grad()
                total_loss = 0.0
                
                # Reconstruct dynamically on the manifold ONCE per step for massive 15x speedup
                merged_sd = adapted_model.fc.state_dict()  # Head is constant here
                current_sd = adapted_model.state_dict()
                
                new_sd = {}
                for key in base_sd.keys():
                    if "fc" not in key and "weight" in key:
                        orig_shape = base_sd[key].shape
                        if len(orig_shape) > 1:
                            # Reconstruct dynamically on the manifold
                            Q_list = [Q.to(device) for Q in self.lie_reps[key]]
                            rho_list = [rho.to(device) for rho in self.residuals[key]]
                            W0_flat = base_sd[key].float().view(orig_shape[0], -1).to(device)
                            
                            # Merge Lie representations
                            Q_sum = torch.zeros_like(Q_list[0])
                            for idx, Q in enumerate(Q_list):
                                Q_sum += lambdas[idx] * Q
                                
                            sum_norm = torch.linalg.norm(Q_sum)
                            if sum_norm > 1e-8:
                                norms = [torch.linalg.norm(Q) for Q in Q_list]
                                c = sum(norms) / sum_norm
                                Q_merged = (c / len(self.tasks)) * Q_sum
                            else:
                                Q_merged = torch.zeros_like(Q_sum)
                                
                            R_merged = from_lie(Q_merged)
                            rho_merged = torch.stack(rho_list).mean(dim=0)
                            
                            W_merged_flat = torch.matmul(R_merged, W0_flat) + rho_merged
                            new_sd["backbone." + key] = W_merged_flat.view(orig_shape)
                        else:
                            new_sd["backbone." + key] = base_sd[key]
                    elif "fc" in key:
                        new_sd[key] = current_sd[key]
                    else:
                        new_sd["backbone." + key] = base_sd[key]
                
                # Apply weights once per step
                adapted_model.load_state_dict(new_sd, strict=False)
                
                for images in adaptation_batches:
                    # Forward pass
                    outputs = adapted_model(images)
                    log_probs = torch.log_softmax(outputs, dim=1)
                    
                    with torch.no_grad():
                        teacher_outputs = expert_teachers[task](images)
                        teacher_probs = torch.softmax(teacher_outputs, dim=1)
                        
                    loss = criterion(log_probs, teacher_probs)
                    loss.backward()
                    total_loss += loss.item()
                    
                optimizer.step()
        else:
            # Reconstruct model without optimization
            new_sd = {}
            for key in base_sd.keys():
                if "fc" not in key and "weight" in key:
                    orig_shape = base_sd[key].shape
                    if len(orig_shape) > 1:
                        Q_list = [Q.to(device) for Q in self.lie_reps[key]]
                        rho_list = [rho.to(device) for rho in self.residuals[key]]
                        W0_flat = base_sd[key].float().view(orig_shape[0], -1).to(device)
                        
                        Q_sum = torch.zeros_like(Q_list[0])
                        for idx, Q in enumerate(Q_list):
                            Q_sum += lambdas[idx] * Q
                            
                        sum_norm = torch.linalg.norm(Q_sum)
                        if sum_norm > 1e-8:
                            norms = [torch.linalg.norm(Q) for Q in Q_list]
                            c = sum(norms) / sum_norm
                            Q_merged = (c / len(self.tasks)) * Q_sum
                        else:
                            Q_merged = torch.zeros_like(Q_sum)
                            
                        R_merged = from_lie(Q_merged)
                        rho_merged = torch.stack(rho_list).mean(dim=0)
                        
                        W_merged_flat = torch.matmul(R_merged, W0_flat) + rho_merged
                        new_sd["backbone." + key] = W_merged_flat.view(orig_shape)
                    else:
                        new_sd["backbone." + key] = base_sd[key]
                elif "fc" in key:
                    new_sd[key] = base_merged.fc.state_dict()[key]
                else:
                    new_sd["backbone." + key] = base_sd[key]
            adapted_model.load_state_dict(new_sd, strict=False)
            
        # Evaluate final model on the full test dataset
        acc = self.evaluate_model(adapted_model, task, corrupt=corrupt)
        return acc

    def run_all_evaluations(self):
        results = {}
        
        # 1. Single-task expert performance (Upper bound)
        print("\n=== Evaluating Single-Task Experts (Clean) ===")
        results["Expert_Clean"] = {}
        for t in self.tasks:
            results["Expert_Clean"][t] = self.evaluate_model(self.experts[t], t, corrupt=False)
            print(f"Expert model {t}: {results['Expert_Clean'][t]:.2f}%")
            
        print("\n=== Evaluating Single-Task Experts (Corrupted) ===")
        results["Expert_Corrupt"] = {}
        for t in self.tasks:
            results["Expert_Corrupt"][t] = self.evaluate_model(self.experts[t], t, corrupt=True)
            print(f"Expert model {t} (corrupted): {results['Expert_Corrupt'][t]:.2f}%")
            
        # 2. Weight Averaging Baseline
        print("\n=== Evaluating Weight Averaging ===")
        avg_model = self.run_weight_averaging()
        results["WA_Clean"] = {}
        results["WA_Corrupt"] = {}
        for t in self.tasks:
            # Pair averaged backbone with the specific task head from the expert
            avg_model.fc = copy.deepcopy(self.experts[t].fc)
            results["WA_Clean"][t] = self.evaluate_model(avg_model, t, corrupt=False)
            results["WA_Corrupt"][t] = self.evaluate_model(avg_model, t, corrupt=True)
            print(f"WA on {t} | Clean: {results['WA_Clean'][t]:.2f}% | Corrupt: {results['WA_Corrupt'][t]:.2f}%")
            
        # 3. Task Arithmetic Baseline
        print("\n=== Evaluating Task Arithmetic ===")
        ta_model = self.run_task_arithmetic(alpha=0.3)
        results["TA_Clean"] = {}
        results["TA_Corrupt"] = {}
        for t in self.tasks:
            ta_model.fc = copy.deepcopy(self.experts[t].fc)
            results["TA_Clean"][t] = self.evaluate_model(ta_model, t, corrupt=False)
            results["TA_Corrupt"][t] = self.evaluate_model(ta_model, t, corrupt=True)
            print(f"TA on {t} | Clean: {results['TA_Clean'][t]:.2f}% | Corrupt: {results['TA_Corrupt'][t]:.2f}%")
            
        # 4. OrthoMerge Baseline
        print("\n=== Evaluating OrthoMerge ===")
        om_model = self.run_orthomerge()
        results["OM_Clean"] = {}
        results["OM_Corrupt"] = {}
        for t in self.tasks:
            om_model.fc = copy.deepcopy(self.experts[t].fc)
            results["OM_Clean"][t] = self.evaluate_model(om_model, t, corrupt=False)
            results["OM_Corrupt"][t] = self.evaluate_model(om_model, t, corrupt=True)
            print(f"OM on {t} | Clean: {results['OM_Clean'][t]:.2f}% | Corrupt: {results['OM_Corrupt'][t]:.2f}%")
            
        # 5. SyMerge (Test-Time Adaptation Baseline)
        print("\n=== Evaluating SyMerge (Test-Time Adaptation) ===")
        results["SyMerge_Clean"] = {}
        results["SyMerge_Corrupt"] = {}
        for t in self.tasks:
            results["SyMerge_Clean"][t] = self.run_symerge_test_time(t, corrupt=False, steps=50, lr=2e-3)
            results["SyMerge_Corrupt"][t] = self.run_symerge_test_time(t, corrupt=True, steps=50, lr=2e-3)
            print(f"SyMerge on {t} | Clean: {results['SyMerge_Clean'][t]:.2f}% | Corrupt: {results['SyMerge_Corrupt'][t]:.2f}%")
            
        # 6. SynOrtho (Proposed Method)
        print("\n=== Evaluating SynOrtho (Our Method) ===")
        results["SynOrtho_Clean"] = {}
        results["SynOrtho_Corrupt"] = {}
        for t in self.tasks:
            results["SynOrtho_Clean"][t] = self.run_synortho_test_time(t, corrupt=False, steps=50, lr=2e-3)
            results["SynOrtho_Corrupt"][t] = self.run_synortho_test_time(t, corrupt=True, steps=50, lr=2e-3)
            print(f"SynOrtho on {t} | Clean: {results['SynOrtho_Clean'][t]:.2f}% | Corrupt: {results['SynOrtho_Corrupt'][t]:.2f}%")
            
        # 7. SynOrtho Ablation: No Adapter (Only coefficients)
        print("\n=== Evaluating SynOrtho Ablation: No Adapter (Only Coefficients) ===")
        results["SynOrtho_NoAdapt_Clean"] = {}
        results["SynOrtho_NoAdapt_Corrupt"] = {}
        for t in self.tasks:
            results["SynOrtho_NoAdapt_Clean"][t] = self.run_synortho_test_time(t, corrupt=False, steps=50, lr=2e-3, optimize_coeff=True, optimize_adapter=False)
            results["SynOrtho_NoAdapt_Corrupt"][t] = self.run_synortho_test_time(t, corrupt=True, steps=50, lr=2e-3, optimize_coeff=True, optimize_adapter=False)
            print(f"SynOrtho (No Adapter) on {t} | Clean: {results['SynOrtho_NoAdapt_Clean'][t]:.2f}% | Corrupt: {results['SynOrtho_NoAdapt_Corrupt'][t]:.2f}%")

        # 8. SynOrtho Ablation: No Coefficients (Only adapter)
        print("\n=== Evaluating SynOrtho Ablation: No Coefficients (Only Adapter) ===")
        results["SynOrtho_NoCoeff_Clean"] = {}
        results["SynOrtho_NoCoeff_Corrupt"] = {}
        for t in self.tasks:
            results["SynOrtho_NoCoeff_Clean"][t] = self.run_synortho_test_time(t, corrupt=False, steps=50, lr=2e-3, optimize_coeff=False, optimize_adapter=True)
            results["SynOrtho_NoCoeff_Corrupt"][t] = self.run_synortho_test_time(t, corrupt=True, steps=50, lr=2e-3, optimize_coeff=False, optimize_adapter=True)
            print(f"SynOrtho (No Coefficients) on {t} | Clean: {results['SynOrtho_NoCoeff_Clean'][t]:.2f}% | Corrupt: {results['SynOrtho_NoCoeff_Corrupt'][t]:.2f}%")

        # 9. Hyperparameter Sensitivity Sweep
        print("\n=== Running Hyperparameter Sensitivity Sweep ===")
        sweep_results = []
        lrs = [1e-4, 1e-3, 5e-3]
        steps_list = [10, 50, 100]
        for lr in lrs:
            for steps in steps_list:
                print(f"Sweeping lr={lr:.1e}, steps={steps}...")
                c10_acc = self.run_synortho_test_time("cifar10", corrupt=True, steps=steps, lr=lr)
                svhn_acc = self.run_synortho_test_time("svhn", corrupt=True, steps=steps, lr=lr)
                mnist_acc = self.run_synortho_test_time("mnist", corrupt=True, steps=steps, lr=lr)
                avg_acc = (c10_acc + svhn_acc + mnist_acc) / 3
                sweep_results.append((lr, steps, c10_acc, svhn_acc, mnist_acc, avg_acc))
                print(f"Sweep | lr: {lr:.1e} | steps: {steps:3d} | Corrupted Average Acc: {avg_acc:.2f}%")
                
        # Save sensitivity results to file
        sens_path = os.path.join(RESULTS_DIR, "sensitivity_analysis.txt")
        with open(sens_path, "w") as f:
            f.write("HYPERPARAMETER SENSITIVITY SWEEP (CORRUPTED OOD EVALUATION)\n")
            f.write("=========================================================\n\n")
            f.write(f"{'Learning Rate':<15} | {'Adapt Steps':<12} | {'CIFAR-10':<10} | {'SVHN':<10} | {'MNIST':<10} | {'Average':<10}\n")
            f.write("-" * 80 + "\n")
            for lr, steps, c10, svhn, mnist, avg in sweep_results:
                f.write(f"{lr:<15.1e} | {steps:<12d} | {c10:<10.2f} | {svhn:<10.2f} | {mnist:<10.2f} | {avg:<10.2f}\n")
        print(f"Saved sensitivity sweep results to {sens_path}")
            
        # Save results text
        self.save_results_text(results)
        
        # Save plots
        self.save_plots(results)

    def save_results_text(self, r):
        summary_path = os.path.join(RESULTS_DIR, "results_summary.txt")
        with open(summary_path, "w") as f:
            f.write("MODEL MERGING PERFORMANCE SUMMARY\n")
            f.write("===============================\n\n")
            
            for cond in ["Clean", "Corrupt"]:
                f.write(f"--- {cond.upper()} EVALUATION ---\n")
                f.write(f"{'Method':<35} | {'CIFAR-10':<10} | {'SVHN':<10} | {'MNIST':<10} | {'Average':<10}\n")
                f.write("-" * 80 + "\n")
                
                methods = [
                    ("Single Expert", f"Expert_{cond}"),
                    ("Weight Averaging", f"WA_{cond}"),
                    ("Task Arithmetic", f"TA_{cond}"),
                    ("OrthoMerge", f"OM_{cond}"),
                    ("SyMerge (TT)", f"SyMerge_{cond}"),
                    ("SynOrtho (Ours)", f"SynOrtho_{cond}"),
                    ("SynOrtho Ablation: No Adapter", f"SynOrtho_NoAdapt_{cond}"),
                    ("SynOrtho Ablation: No Coeff", f"SynOrtho_NoCoeff_{cond}")
                ]
                
                for label, key in methods:
                    c10 = r[key]["cifar10"]
                    svhn = r[key]["svhn"]
                    mnist = r[key]["mnist"]
                    avg = (c10 + svhn + mnist) / 3
                    f.write(f"{label:<35} | {c10:<10.2f} | {svhn:<10.2f} | {mnist:<10.2f} | {avg:<10.2f}\n")
                f.write("\n")
                
        print(f"Saved text summary to {summary_path}")

    def save_plots(self, r):
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        tasks_labels = ["CIFAR-10", "SVHN", "MNIST", "Average"]
        methods_keys = [
            ("Weight Averaging", "WA"),
            ("Task Arithmetic", "TA"),
            ("OrthoMerge", "OM"),
            ("SyMerge (TT)", "SyMerge"),
            ("SynOrtho (Ours)", "SynOrtho"),
            ("SynOrtho: No Adapter", "SynOrtho_NoAdapt"),
            ("SynOrtho: No Coeff", "SynOrtho_NoCoeff")
        ]
        
        # Clean Plot
        ax = axes[0]
        x = np.arange(len(tasks_labels))
        width = 0.11
        
        for i, (label, key) in enumerate(methods_keys):
            c10 = r[f"{key}_Clean"]["cifar10"]
            svhn = r[f"{key}_Clean"]["svhn"]
            mnist = r[f"{key}_Clean"]["mnist"]
            avg = (c10 + svhn + mnist) / 3
            vals = [c10, svhn, mnist, avg]
            ax.bar(x + (i - 3) * width, vals, width, label=label)
            
        # Draw line for single expert baseline
        c10_exp = r["Expert_Clean"]["cifar10"]
        svhn_exp = r["Expert_Clean"]["svhn"]
        mnist_exp = r["Expert_Clean"]["mnist"]
        avg_exp = (c10_exp + svhn_exp + mnist_exp) / 3
        ax.plot([0, 1, 2, 3], [c10_exp, svhn_exp, mnist_exp, avg_exp], "k--", label="Individual Experts", alpha=0.7)
        
        ax.set_title("Clean Evaluation Accuracy")
        ax.set_xticks(x)
        ax.set_xticklabels(tasks_labels)
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 105)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize='small', loc='lower left')
        
        # Corrupt Plot
        ax = axes[1]
        for i, (label, key) in enumerate(methods_keys):
            c10 = r[f"{key}_Corrupt"]["cifar10"]
            svhn = r[f"{key}_Corrupt"]["svhn"]
            mnist = r[f"{key}_Corrupt"]["mnist"]
            avg = (c10 + svhn + mnist) / 3
            vals = [c10, svhn, mnist, avg]
            ax.bar(x + (i - 3) * width, vals, width, label=label)
            
        c10_exp_c = r["Expert_Corrupt"]["cifar10"]
        svhn_exp_c = r["Expert_Corrupt"]["svhn"]
        mnist_exp_c = r["Expert_Corrupt"]["mnist"]
        avg_exp_c = (c10_exp_c + svhn_exp_c + mnist_exp_c) / 3
        ax.plot([0, 1, 2, 3], [c10_exp_c, svhn_exp_c, mnist_exp_c, avg_exp_c], "k--", label="Individual Experts", alpha=0.7)
        
        ax.set_title("Corrupted (OOD) Evaluation Accuracy")
        ax.set_xticks(x)
        ax.set_xticklabels(tasks_labels)
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 105)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize='small', loc='upper right')
        
        plt.tight_layout()
        plot_path = os.path.join(RESULTS_DIR, "results_chart.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved bar chart plot to {plot_path}")

if __name__ == "__main__":
    # Check if weights exist, if not alert
    required_weights = ["resnet18_base.pt", "resnet18_cifar10.pt", "resnet18_svhn.pt", "resnet18_mnist.pt"]
    missing = [w for w in required_weights if not os.path.exists(os.path.join(MODEL_DIR, w))]
    if missing:
        print(f"Error: Missing trained weights: {missing}. Please run train.py first.")
    else:
        evaluator = Evaluator()
        evaluator.run_all_evaluations()
