import os
import copy
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.transforms.functional as F

# Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using evaluation device: {device}")

# Robust CUDA & cuDNN configuration to prevent initialization errors
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = False
    print("Disabled cuDNN to ensure stable CUDA execution during TTA.")

# Data Transformations (without random augmentations for evaluation)
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Load CIFAR-10 Test Set
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

# Task Splitting: Task A (classes 0-4), Task B (classes 5-9)
def get_task_subsets(dataset, classes):
    indices = [i for i, (_, label) in enumerate(dataset) if label in classes]
    return Subset(dataset, indices)

task_A_classes = list(range(5))
task_B_classes = list(range(5, 10))

test_subset_A = get_task_subsets(test_dataset, task_A_classes)
test_subset_B = get_task_subsets(test_dataset, task_B_classes)

print(f"Task A (0-4) Test size: {len(test_subset_A)}")
print(f"Task B (5-9) Test size: {len(test_subset_B)}")

# Helper to apply batch-wise OOD corruptions
def apply_corruption(images, corruption_type):
    if corruption_type == "none":
        return images
    elif corruption_type == "noise":
        # Gaussian Noise (sigma = 0.4)
        return images + torch.randn_like(images) * 0.4
    elif corruption_type == "blur":
        # Gaussian Blur (kernel 5x5, sigma 1.5)
        return F.gaussian_blur(images, kernel_size=[5, 5], sigma=[1.5, 1.5])
    elif corruption_type == "contrast":
        # Contrast Reduction (scale by 0.25)
        return F.adjust_contrast(images, 0.25)
    elif corruption_type == "rotation":
        # Image Rotation (30 degrees)
        return F.rotate(images, 30)
    else:
        raise ValueError(f"Unknown corruption: {corruption_type}")

# Custom self-contained LoRA layer for Conv2D
class LoRAConv2d(nn.Module):
    def __init__(self, original_conv, r=8, lora_alpha=16, lora_dropout=0.1):
        super().__init__()
        self.original = original_conv
        # Freeze original convolution weights
        for p in self.original.parameters():
            p.requires_grad = False
            
        in_channels = original_conv.in_channels
        out_channels = original_conv.out_channels
        kernel_size = original_conv.kernel_size
        stride = original_conv.stride
        padding = original_conv.padding
        
        # LoRA down-projection and up-projection
        self.lora_A = nn.Conv2d(in_channels, r, kernel_size, stride=stride, padding=padding, bias=False)
        self.lora_B = nn.Conv2d(r, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.scaling = lora_alpha / r
        self.dropout = nn.Dropout2d(lora_dropout)
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        original_output = self.original(x)
        lora_output = self.lora_B(self.lora_A(self.dropout(x)))
        return original_output + lora_output * self.scaling

def apply_lora_to_resnet(model, r=8, lora_alpha=16, lora_dropout=0.1):
    for layer_name in ["layer3", "layer4"]:
        layer = getattr(model, layer_name)
        for i in range(len(layer)):
            layer[i].conv1 = LoRAConv2d(layer[i].conv1, r, lora_alpha, lora_dropout)
            layer[i].conv2 = LoRAConv2d(layer[i].conv2, r, lora_alpha, lora_dropout)
    return model

# SOSR penalty computation for Custom LoRA adapters
def compute_sosr(model, eps=1e-8):
    loss_sosr = 0.0
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, LoRAConv2d):
            A = module.lora_A.weight  # shape [r, in_channels, K, K]
            B = module.lora_B.weight  # shape [out_channels, r, 1, 1]
            
            A_flat = A.view(A.size(0), -1)  # shape [r, in_channels * K * K]
            B_flat = B.view(B.size(0), -1)  # shape [out_channels, r]
            
            # B^T * B matrix
            BTB = torch.matmul(B_flat.t(), B_flat)
            diag_BTB = torch.diag(torch.diag(BTB))
            off_diag_BTB = BTB - diag_BTB
            loss_B = torch.sum(off_diag_BTB ** 2) / (torch.sum(diag_BTB ** 2) + eps)
            
            # A * A^T matrix
            AAT = torch.matmul(A_flat, A_flat.t())
            diag_AAT = torch.diag(torch.diag(AAT))
            off_diag_AAT = AAT - diag_AAT
            loss_A = torch.sum(off_diag_AAT ** 2) / (torch.sum(diag_AAT ** 2) + eps)
            
            loss_sosr += loss_B + loss_A
            count += 1
    if count > 0:
        return loss_sosr / count
    return torch.tensor(0.0, device=device)

# Load frozen individual expert models to serve as self-labeling teachers
def load_teacher(task_name, expert_type):
    model = models.resnet18(pretrained=True)
    model = apply_lora_to_resnet(model, r=8, lora_alpha=16, lora_dropout=0.1)
    model.fc = nn.Linear(512, 10)
    
    # Load LoRA weights
    sd = torch.load(f"./checkpoints/lora_{task_name}_{expert_type}.pt", map_location="cpu")
    model.load_state_dict(sd, strict=False)
    
    # Load task head
    head_state = torch.load(f"./checkpoints/head_{task_name}_{expert_type}.pt", map_location=device)
    model.fc.load_state_dict(head_state)
    
    model.to(device)
    model.eval()
    return model

# Load model, apply merged LoRA state dict
def build_merged_model(expert_type):
    model = models.resnet18(pretrained=True)
    model = apply_lora_to_resnet(model, r=8, lora_alpha=16, lora_dropout=0.1)
    model.fc = nn.Linear(512, 10)
    
    # Load individual expert checkpoints
    sd_A = torch.load(f"./checkpoints/lora_A_{expert_type}.pt", map_location="cpu")
    sd_B = torch.load(f"./checkpoints/lora_B_{expert_type}.pt", map_location="cpu")
    
    # Average the LoRA adapter parameters
    merged_sd = {}
    for k in sd_A.keys():
        if k in sd_B:
            merged_sd[k] = 0.5 * sd_A[k] + 0.5 * sd_B[k]
        else:
            merged_sd[k] = sd_A[k]
            
    # Load into model
    model.load_state_dict(merged_sd, strict=False)
    model.to(device)
    return model

# Run evaluation on standard test loader
def evaluate_model(model, test_loader, head_state, corruption_type):
    model.eval()
    # Temporarily swap in the task head
    model.fc.load_state_dict(head_state)
    
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            images_corrupted = apply_corruption(images, corruption_type)
            outputs = model(images_corrupted)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return 100.0 * correct / total

# Test-Time Adaptation Loop
def run_tta(merged_model, teacher_model, tta_data_pool, head_state, corruption_type, method="symerge", lr=1e-3, steps=10, rho=0.05, beta=0.1):
    # Set model to train mode for TTA parameter updates
    merged_model.train()
    # Swap in classification head to be adapted
    merged_model.fc.load_state_dict(copy.deepcopy(head_state))
    
    # Filter trainable parameters (LoRA weights and classification head)
    trainable_params = []
    for name, param in merged_model.named_parameters():
        if "lora" in name or "fc" in name:
            param.requires_grad = True
            trainable_params.append(param)
        else:
            param.requires_grad = False
            
    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    
    # Create dataloader for TTA pool (512 images)
    tta_loader = DataLoader(tta_data_pool, batch_size=32, shuffle=True)
    
    step_count = 0
    while step_count < steps:
        for images, _ in tta_loader:
            if step_count >= steps:
                break
                
            images = images.to(device)
            # Apply corruption
            images_corrupted = apply_corruption(images, corruption_type)
            
            # 1. Teacher Prediction (soft labels)
            with torch.no_grad():
                teacher_outputs = teacher_model(images_corrupted)
                soft_labels = torch.softmax(teacher_outputs / 1.0, dim=-1)
                
            if method == "symerge":
                optimizer.zero_grad()
                outputs = merged_model(images_corrupted)
                loss = kl_loss(torch.log_softmax(outputs, dim=-1), soft_labels)
                loss.backward()
                optimizer.step()
                
            elif method == "sat-symerge":
                # SAM First Pass (loss and gradient)
                optimizer.zero_grad()
                outputs = merged_model(images_corrupted)
                loss = kl_loss(torch.log_softmax(outputs, dim=-1), soft_labels)
                loss.backward()
                
                # Compute gradient norm and perturb
                grad_norm = torch.norm(torch.stack([p.grad.norm(2) for p in trainable_params if p.grad is not None]), p=2)
                scale = rho / (grad_norm + 1e-12)
                
                old_weights = {}
                for p in trainable_params:
                    if p.grad is not None:
                        old_weights[p] = p.data.clone()
                        p.data.add_(p.grad * scale)
                        
                # SAM Second Pass (perturbed gradient & update)
                optimizer.zero_grad()
                outputs_perturbed = merged_model(images_corrupted)
                loss_perturbed = kl_loss(torch.log_softmax(outputs_perturbed, dim=-1), soft_labels)
                loss_perturbed.backward()
                
                # Restore original weights
                for p in trainable_params:
                    if p in old_weights:
                        p.data.copy_(old_weights[p])
                        
                optimizer.step()
                
            elif method == "o-lorta":  # Our proposed O-LoRTA: SAM + SOSR
                # First Pass (Loss, SOSR and gradient)
                optimizer.zero_grad()
                outputs = merged_model(images_corrupted)
                loss_sl = kl_loss(torch.log_softmax(outputs, dim=-1), soft_labels)
                loss_sosr = compute_sosr(merged_model)
                loss = loss_sl + beta * loss_sosr
                loss.backward()
                
                # Compute gradient norm and perturb
                grad_norm = torch.norm(torch.stack([p.grad.norm(2) for p in trainable_params if p.grad is not None]), p=2)
                scale = rho / (grad_norm + 1e-12)
                
                old_weights = {}
                for p in trainable_params:
                    if p.grad is not None:
                        old_weights[p] = p.data.clone()
                        p.data.add_(p.grad * scale)
                        
                # Second Pass (perturbed loss and gradient)
                optimizer.zero_grad()
                outputs_perturbed = merged_model(images_corrupted)
                loss_perturbed_sl = kl_loss(torch.log_softmax(outputs_perturbed, dim=-1), soft_labels)
                loss_perturbed_sosr = compute_sosr(merged_model)
                loss_perturbed = loss_perturbed_sl + beta * loss_perturbed_sosr
                loss_perturbed.backward()
                
                # Restore original weights
                for p in trainable_params:
                    if p in old_weights:
                        p.data.copy_(old_weights[p])
                        
                optimizer.step()
                
            elif method == "sd-olorta":  # Scale-Decoupled O-LoRTA (SD-O-LoRTA)
                # First Pass (Loss, SOSR and gradient)
                optimizer.zero_grad()
                outputs = merged_model(images_corrupted)
                loss_sl = kl_loss(torch.log_softmax(outputs, dim=-1), soft_labels)
                loss_sosr = compute_sosr(merged_model)
                loss = loss_sl + beta * loss_sosr
                loss.backward()
                
                # We have separate groups: LoRA parameters vs FC (head) parameters
                lora_params = [p for name, p in merged_model.named_parameters() if "lora" in name and p.requires_grad]
                fc_params = [p for name, p in merged_model.named_parameters() if "fc" in name and p.requires_grad]
                
                # Separate perturbation scales (rho_fc = 0.01, rho_lora = rho)
                rho_lora = rho
                rho_fc = rho * 0.2  # Decoupled scale for classification head
                
                old_weights = {}
                
                # Perturb LoRA group
                if len(lora_params) > 0:
                    lora_grads = [p.grad for p in lora_params if p.grad is not None]
                    if len(lora_grads) > 0:
                        lora_grad_norm = torch.norm(torch.stack([g.norm(2) for g in lora_grads]), p=2)
                        lora_scale = rho_lora / (lora_grad_norm + 1e-12)
                        for p in lora_params:
                            if p.grad is not None:
                                old_weights[p] = p.data.clone()
                                p.data.add_(p.grad * lora_scale)
                                
                # Perturb FC (head) group
                if len(fc_params) > 0:
                    fc_grads = [p.grad for p in fc_params if p.grad is not None]
                    if len(fc_grads) > 0:
                        fc_grad_norm = torch.norm(torch.stack([g.norm(2) for g in fc_grads]), p=2)
                        fc_scale = rho_fc / (fc_grad_norm + 1e-12)
                        for p in fc_params:
                            if p.grad is not None:
                                old_weights[p] = p.data.clone()
                                p.data.add_(p.grad * fc_scale)
                        
                # Second Pass (perturbed loss and gradient)
                optimizer.zero_grad()
                outputs_perturbed = merged_model(images_corrupted)
                loss_perturbed_sl = kl_loss(torch.log_softmax(outputs_perturbed, dim=-1), soft_labels)
                loss_perturbed_sosr = compute_sosr(merged_model)
                loss_perturbed = loss_perturbed_sl + beta * loss_perturbed_sosr
                loss_perturbed.backward()
                
                # Restore original weights
                for p in trainable_params:
                    if p in old_weights:
                        p.data.copy_(old_weights[p])
                        
                optimizer.step()
                
            step_count += 1
            
    # Return adapted head state dict
    return copy.deepcopy(merged_model.fc.state_dict())

def run_full_evaluation_suite(expert_type):
    print(f"\n========================================================")
    print(f"Running Full Evaluation Suite for Experts: {expert_type.upper()}")
    print(f"========================================================")
    
    # Load teachers
    teacher_A = load_teacher("A", expert_type)
    teacher_B = load_teacher("B", expert_type)
    
    # Load classification heads
    head_state_A = torch.load(f"./checkpoints/head_A_{expert_type}.pt", map_location=device)
    head_state_B = torch.load(f"./checkpoints/head_B_{expert_type}.pt", map_location=device)
    
    # Setup test loaders
    test_loader_A = DataLoader(test_subset_A, batch_size=128, shuffle=False)
    test_loader_B = DataLoader(test_subset_B, batch_size=128, shuffle=False)
    
    # Setup 512-image TTA calibration pools
    torch.manual_seed(42)
    indices_A = torch.randperm(len(test_subset_A))[:512]
    indices_B = torch.randperm(len(test_subset_B))[:512]
    
    tta_pool_A = Subset(test_subset_A, indices_A)
    tta_pool_B = Subset(test_subset_B, indices_B)
    
    corruptions = ["none", "noise", "blur", "contrast", "rotation"]
    methods = ["baseline", "symerge", "sat-symerge", "o-lorta", "sd-olorta"]
    
    results = {m: {} for m in methods}
    
    for corruption in corruptions:
        print(f"\nEvaluating Corruption: {corruption.upper()}")
        
        # Method 1: Baseline (No TTA)
        model_baseline = build_merged_model(expert_type)
        acc_A = evaluate_model(model_baseline, test_loader_A, head_state_A, corruption)
        acc_B = evaluate_model(model_baseline, test_loader_B, head_state_B, corruption)
        avg_acc = (acc_A + acc_B) / 2
        results["baseline"][corruption] = {"A": acc_A, "B": acc_B, "avg": avg_acc}
        print(f"  Baseline (No TTA) - Task A: {acc_A:.2f}%, Task B: {acc_B:.2f}%, Avg: {avg_acc:.2f}%")
        
        # Method 2: SyMerge (Standard TTA)
        # Adapt Task A
        model_sym = build_merged_model(expert_type)
        adapted_head_A = run_tta(model_sym, teacher_A, tta_pool_A, head_state_A, corruption, method="symerge")
        acc_A = evaluate_model(model_sym, test_loader_A, adapted_head_A, corruption)
        
        # Adapt Task B
        model_sym = build_merged_model(expert_type)
        adapted_head_B = run_tta(model_sym, teacher_B, tta_pool_B, head_state_B, corruption, method="symerge")
        acc_B = evaluate_model(model_sym, test_loader_B, adapted_head_B, corruption)
        
        avg_acc = (acc_A + acc_B) / 2
        results["symerge"][corruption] = {"A": acc_A, "B": acc_B, "avg": avg_acc}
        print(f"  SyMerge (Std TTA) - Task A: {acc_A:.2f}%, Task B: {acc_B:.2f}%, Avg: {avg_acc:.2f}%")
        
        # Method 3: SAT-SyMerge (SAM TTA)
        # Adapt Task A
        model_sat = build_merged_model(expert_type)
        adapted_head_A = run_tta(model_sat, teacher_A, tta_pool_A, head_state_A, corruption, method="sat-symerge")
        acc_A = evaluate_model(model_sat, test_loader_A, adapted_head_A, corruption)
        
        # Adapt Task B
        model_sat = build_merged_model(expert_type)
        adapted_head_B = run_tta(model_sat, teacher_B, tta_pool_B, head_state_B, corruption, method="sat-symerge")
        acc_B = evaluate_model(model_sat, test_loader_B, adapted_head_B, corruption)
        
        avg_acc = (acc_A + acc_B) / 2
        results["sat-symerge"][corruption] = {"A": acc_A, "B": acc_B, "avg": avg_acc}
        print(f"  SAT-SyMerge (SAM) - Task A: {acc_A:.2f}%, Task B: {acc_B:.2f}%, Avg: {avg_acc:.2f}%")
        
        # Method 4: O-LoRTA (Ours - SAM + SOSR TTA)
        # Adapt Task A
        model_olorta = build_merged_model(expert_type)
        adapted_head_A = run_tta(model_olorta, teacher_A, tta_pool_A, head_state_A, corruption, method="o-lorta")
        acc_A = evaluate_model(model_olorta, test_loader_A, adapted_head_A, corruption)
        
        # Adapt Task B
        model_olorta = build_merged_model(expert_type)
        adapted_head_B = run_tta(model_olorta, teacher_B, tta_pool_B, head_state_B, corruption, method="o-lorta")
        acc_B = evaluate_model(model_olorta, test_loader_B, adapted_head_B, corruption)
        
        avg_acc = (acc_A + acc_B) / 2
        results["o-lorta"][corruption] = {"A": acc_A, "B": acc_B, "avg": avg_acc}
        print(f"  O-LoRTA (SAM+SOSR) - Task A: {acc_A:.2f}%, Task B: {acc_B:.2f}%, Avg: {avg_acc:.2f}%")

        # Method 5: SD-O-LoRTA (Ours - Scale-Decoupled SAM + SOSR TTA)
        # Adapt Task A
        model_sd = build_merged_model(expert_type)
        adapted_head_A = run_tta(model_sd, teacher_A, tta_pool_A, head_state_A, corruption, method="sd-olorta")
        acc_A = evaluate_model(model_sd, test_loader_A, adapted_head_A, corruption)
        
        # Adapt Task B
        model_sd = build_merged_model(expert_type)
        adapted_head_B = run_tta(model_sd, teacher_B, tta_pool_B, head_state_B, corruption, method="sd-olorta")
        acc_B = evaluate_model(model_sd, test_loader_B, adapted_head_B, corruption)
        
        avg_acc = (acc_A + acc_B) / 2
        results["sd-olorta"][corruption] = {"A": acc_A, "B": acc_B, "avg": avg_acc}
        print(f"  SD-O-LoRTA (Decoupled) - Task A: {acc_A:.2f}%, Task B: {acc_B:.2f}%, Avg: {avg_acc:.2f}%")

    return results

def run_ablation_beta(expert_type, betas=[0.001, 0.01, 0.1, 0.5, 1.0]):
    print(f"\n========================================================")
    print(f"Running Beta Ablation Study for Experts: {expert_type.upper()}")
    print(f"========================================================")
    
    # Load teachers
    teacher_A = load_teacher("A", expert_type)
    teacher_B = load_teacher("B", expert_type)
    
    # Load classification heads
    head_state_A = torch.load(f"./checkpoints/head_A_{expert_type}.pt", map_location=device)
    head_state_B = torch.load(f"./checkpoints/head_B_{expert_type}.pt", map_location=device)
    
    # Setup test loaders
    test_loader_A = DataLoader(test_subset_A, batch_size=128, shuffle=False)
    test_loader_B = DataLoader(test_subset_B, batch_size=128, shuffle=False)
    
    # Setup 512-image TTA calibration pools
    torch.manual_seed(42)
    indices_A = torch.randperm(len(test_subset_A))[:512]
    indices_B = torch.randperm(len(test_subset_B))[:512]
    
    tta_pool_A = Subset(test_subset_A, indices_A)
    tta_pool_B = Subset(test_subset_B, indices_B)
    
    corruptions = ["none", "blur"]
    results = {}
    
    for beta in betas:
        results[str(beta)] = {}
        for corruption in corruptions:
            # Adapt Task A
            model_olorta = build_merged_model(expert_type)
            adapted_head_A = run_tta(model_olorta, teacher_A, tta_pool_A, head_state_A, corruption, method="o-lorta", beta=beta)
            acc_A = evaluate_model(model_olorta, test_loader_A, adapted_head_A, corruption)
            
            # Adapt Task B
            model_olorta = build_merged_model(expert_type)
            adapted_head_B = run_tta(model_olorta, teacher_B, tta_pool_B, head_state_B, corruption, method="o-lorta", beta=beta)
            acc_B = evaluate_model(model_olorta, test_loader_B, adapted_head_B, corruption)
            
            avg_acc = (acc_A + acc_B) / 2
            results[str(beta)][corruption] = {"A": acc_A, "B": acc_B, "avg": avg_acc}
            print(f"  Beta: {beta} - Corruption: {corruption.upper()} - Task A: {acc_A:.2f}%, Task B: {acc_B:.2f}%, Avg: {avg_acc:.2f}%")
            
    return results

def run_ablation_rho(expert_type, rhos=[0.01, 0.05, 0.1, 0.2]):
    print(f"\n========================================================")
    print(f"Running Rho Ablation Study for Experts: {expert_type.upper()}")
    print(f"========================================================")
    
    # Load teachers
    teacher_A = load_teacher("A", expert_type)
    teacher_B = load_teacher("B", expert_type)
    
    # Load classification heads
    head_state_A = torch.load(f"./checkpoints/head_A_{expert_type}.pt", map_location=device)
    head_state_B = torch.load(f"./checkpoints/head_B_{expert_type}.pt", map_location=device)
    
    # Setup test loaders
    test_loader_A = DataLoader(test_subset_A, batch_size=128, shuffle=False)
    test_loader_B = DataLoader(test_subset_B, batch_size=128, shuffle=False)
    
    # Setup 512-image TTA calibration pools
    torch.manual_seed(42)
    indices_A = torch.randperm(len(test_subset_A))[:512]
    indices_B = torch.randperm(len(test_subset_B))[:512]
    
    tta_pool_A = Subset(test_subset_A, indices_A)
    tta_pool_B = Subset(test_subset_B, indices_B)
    
    corruptions = ["none", "blur"]
    results = {}
    
    for rho in rhos:
        results[str(rho)] = {}
        for corruption in corruptions:
            # Adapt Task A
            model_olorta = build_merged_model(expert_type)
            adapted_head_A = run_tta(model_olorta, teacher_A, tta_pool_A, head_state_A, corruption, method="o-lorta", beta=0.1, rho=rho)
            acc_A = evaluate_model(model_olorta, test_loader_A, adapted_head_A, corruption)
            
            # Adapt Task B
            model_olorta = build_merged_model(expert_type)
            adapted_head_B = run_tta(model_olorta, teacher_B, tta_pool_B, head_state_B, corruption, method="o-lorta", beta=0.1, rho=rho)
            acc_B = evaluate_model(model_olorta, test_loader_B, adapted_head_B, corruption)
            
            avg_acc = (acc_A + acc_B) / 2
            results[str(rho)][corruption] = {"A": acc_A, "B": acc_B, "avg": avg_acc}
            print(f"  Rho: {rho} - Corruption: {corruption.upper()} - Task A: {acc_A:.2f}%, Task B: {acc_B:.2f}%, Avg: {avg_acc:.2f}%")
            
    return results

if __name__ == "__main__":
    standard_res = run_full_evaluation_suite("standard")
    sam_res = run_full_evaluation_suite("sam")
    
    print("\nRunning O-LoRTA Beta Ablation Studies...")
    ablation_standard = run_ablation_beta("standard")
    ablation_sam = run_ablation_beta("sam")
    
    print("\nRunning O-LoRTA Rho Ablation Studies...")
    ablation_rho_standard = run_ablation_rho("standard")
    ablation_rho_sam = run_ablation_rho("sam")
    
    # Save results to json
    all_results = {
        "standard_experts": standard_res,
        "sam_experts": sam_res,
        "ablation_standard": ablation_standard,
        "ablation_sam": ablation_sam,
        "ablation_rho_standard": ablation_rho_standard,
        "ablation_rho_sam": ablation_rho_sam
    }
    with open("results.json", "w") as f:
        json.dump(all_results, f, indent=4)
    print("\nResults and Ablations successfully saved to results.json!")
