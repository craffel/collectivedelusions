import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18
import copy

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = False
    print("Disabled cuDNN to prevent CUDNN_STATUS_NOT_INITIALIZED error.")

# Define ResNet18 Expert and MLP structures exactly as in train_experts.py
class ResNet18Expert(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = resnet18(weights=None)
        self.backbone.fc = nn.Identity()
        self.head = nn.Linear(512, num_classes)
        
    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

class MLPBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3072, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 512)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(512, 512)
        self.relu3 = nn.ReLU()
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        return x

class MLPExpert(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = MLPBackbone()
        self.head = nn.Linear(512, num_classes)
        
    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

# Load data helper
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def get_dataloader(name, train=True, batch_size=128):
    if name == 'mnist':
        dataset = datasets.MNIST('data', train=train, download=True, transform=transform)
    elif name == 'fashion':
        dataset = datasets.FashionMNIST('data', train=train, download=True, transform=transform)
    elif name == 'cifar10':
        dataset = datasets.CIFAR10('data', train=train, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=2 if torch.cuda.is_available() else 0)

def evaluate_model(model, dataloader):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
    return 100.0 * correct / total

# 8-bit Uniform Quantization helper
def quantize_weight_8bit(w):
    max_val = torch.max(torch.abs(w))
    if max_val == 0:
        return w
    S = max_val / 127.0
    w_q = torch.clamp(torch.round(w / S), -128, 127) * S
    return w_q

def apply_8bit_quantization(model):
    quant_model = copy.deepcopy(model)
    for name, param in quant_model.named_parameters():
        # Only quantize backbone weights of dimension >= 2
        if 'backbone' in name and 'weight' in name and param.dim() >= 2:
            param.data = quantize_weight_8bit(param.data)
    return quant_model

# Data-Efficient BatchNorm Calibration helper
def calibrate_bn(model, task_name, num_samples=32):
    cal_model = copy.deepcopy(model)
    cal_model.to(device)
    
    orig_momentums = {}
    for name, m in cal_model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            m.train()
            m.reset_running_stats()
            orig_momentums[name] = m.momentum
            m.momentum = 1.0
            
    loader = get_dataloader(task_name, train=True, batch_size=num_samples)
    x, _ = next(iter(loader))
    x = x.to(device)
    with torch.no_grad():
        cal_model(x)
        
    for name, m in cal_model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = orig_momentums[name]
            
    cal_model.eval()
    return cal_model

# Bures-Wasserstein helpers for BWPA
def spd_sqrt(A, eps=1e-6):
    L, V = torch.linalg.eigh(A)
    L = torch.clamp(L, min=eps)
    return V @ torch.diag(L.sqrt()) @ V.T

def spd_inv_sqrt(A, eps=1e-6):
    L, V = torch.linalg.eigh(A)
    L = torch.clamp(L, min=eps)
    return V @ torch.diag(1.0 / L.sqrt()) @ V.T

def compute_bures_barycenter(sigmas, max_iter=5, eps=1e-6):
    Sigma_c = torch.stack(sigmas, dim=0).mean(dim=0)
    for _ in range(max_iter):
        Sigma_c_sqrt = spd_sqrt(Sigma_c, eps)
        Sigma_c_inv_sqrt = spd_inv_sqrt(Sigma_c, eps)
        
        sum_terms = torch.zeros_like(Sigma_c)
        for k in range(len(sigmas)):
            term = Sigma_c_sqrt @ sigmas[k] @ Sigma_c_sqrt
            sum_terms += spd_sqrt(term, eps)
            
        sum_terms = sum_terms / len(sigmas)
        Sigma_c = Sigma_c_inv_sqrt @ sum_terms @ sum_terms @ Sigma_c_inv_sqrt
    return Sigma_c

def bw_align_layer(w_merged, w_experts, eps=1e-8):
    if w_merged.dim() < 2:
        return torch.stack(w_experts, dim=0).mean(dim=0)
        
    orig_shape = w_merged.shape
    d1 = orig_shape[0]
    
    wm_2d = w_merged.view(d1, -1)
    we_2ds = [we.view(d1, -1) for we in w_experts]
    
    # Compute standard deviations across the input features (dim=1)
    std_merged = wm_2d.std(dim=1, keepdim=True)
    std_experts = [we.std(dim=1, keepdim=True) for we in we_2ds]
    
    # Target standard deviation is the average of experts' standard deviations (Wasserstein barycenter of 1D Gaussians)
    std_target = torch.stack(std_experts, dim=0).mean(dim=0)
    
    # Scaling ratio (positive diagonal element)
    ratio = std_target / (std_merged + eps)
    
    # Align the mean (first-order moment)
    mu_merged = wm_2d.mean(dim=1, keepdim=True)
    mu_experts = [we.mean(dim=1, keepdim=True) for we in we_2ds]
    mu_target = torch.stack(mu_experts, dim=0).mean(dim=0)
    
    # Scale and shift (Diagonal Bures-Wasserstein transport map)
    w_transported = mu_target + ratio * (wm_2d - mu_merged)
    return w_transported.view(orig_shape)

def full_bw_align_layer(w_merged, w_experts, eps=1e-6):
    if w_merged.dim() < 2:
        return torch.stack(w_experts, dim=0).mean(dim=0)
        
    orig_shape = w_merged.shape
    d1 = orig_shape[0]
    
    wm_2d = w_merged.view(d1, -1)
    we_2ds = [we.view(d1, -1) for we in w_experts]
    
    # Align the mean (first-order moment)
    mu_merged = wm_2d.mean(dim=1, keepdim=True)
    mu_experts = [we.mean(dim=1, keepdim=True) for we in we_2ds]
    mu_target = torch.stack(mu_experts, dim=0).mean(dim=0)
    
    # Compute d1 x d1 covariance matrices across the input features (dim=1)
    N = wm_2d.shape[1]
    
    wm_centered = wm_2d - mu_merged
    we_centereds = [we - mue for we, mue in zip(we_2ds, mu_experts)]
    
    sigma_merged = (wm_centered @ wm_centered.T) / N + eps * torch.eye(d1, device=w_merged.device)
    sigmas_experts = [(we @ we.T) / N + eps * torch.eye(d1, device=w_merged.device) for we in we_centereds]
    
    # Compute Bures-Wasserstein barycenter
    sigma_c = compute_bures_barycenter(sigmas_experts, max_iter=5, eps=eps)
    
    # Transport matrix M = Sigma_merged^{-1/2} (Sigma_merged^{1/2} Sigma_c Sigma_merged^{1/2})^{1/2} Sigma_merged^{-1/2}
    sigma_m_sqrt = spd_sqrt(sigma_merged, eps)
    sigma_m_inv_sqrt = spd_inv_sqrt(sigma_merged, eps)
    
    inner_term = sigma_m_sqrt @ sigma_c @ sigma_m_sqrt
    inner_sqrt = spd_sqrt(inner_term, eps)
    
    M = sigma_m_inv_sqrt @ inner_sqrt @ sigma_m_inv_sqrt
    
    # Apply full-covariance transport: T(W) = mu_target + M @ (W_merged - mu_merged)
    w_transported = mu_target + M @ wm_centered
    return w_transported.view(orig_shape)

# WCPR Calibration
def wcpr_calibrate_layer(w_merged, w_experts):
    if w_merged.dim() < 2:
        return torch.stack(w_experts, dim=0).mean(dim=0)
        
    C_out = w_merged.shape[0]
    w_calibrated = w_merged.clone()
    for c in range(C_out):
        mc = w_merged[c].flatten()
        sorted_indices = torch.argsort(mc)
        
        sorted_experts = []
        for exp in w_experts:
            ec = exp[c].flatten()
            sorted_experts.append(torch.sort(ec)[0])
        target_sorted = torch.stack(sorted_experts, dim=0).mean(dim=0)
        
        mc_calibrated = mc.clone()
        mc_calibrated[sorted_indices] = target_sorted
        w_calibrated[c] = mc_calibrated.view(w_merged[c].shape)
    return w_calibrated

def merge_and_eval(model_type='resnet18'):
    print(f"\n========================================\nEvaluating Model Merging on {model_type.upper()}\n========================================")
    tasks = ['mnist', 'fashion', 'cifar10']
    
    # Load progenitor
    if model_type == 'resnet18':
        progenitor = ResNet18Expert()
        progenitor.load_state_dict(torch.load('checkpoints/resnet18/progenitor.pt', map_location=device))
    else:
        progenitor = MLPExpert()
        progenitor.load_state_dict(torch.load('checkpoints/mlp/progenitor.pt', map_location=device))
    progenitor = progenitor.to(device)
        
    # Load experts
    experts = {}
    for task in tasks:
        if model_type == 'resnet18':
            expert = ResNet18Expert()
            expert.load_state_dict(torch.load(f'checkpoints/resnet18/{task}_expert.pt', map_location=device))
        else:
            expert = MLPExpert()
            expert.load_state_dict(torch.load(f'checkpoints/mlp/{task}_expert.pt', map_location=device))
        experts[task] = expert.to(device)
        
    test_loaders = {task: get_dataloader(task, train=False) for task in tasks}
    
    # 1. Oracle Performance
    print("Evaluating Oracle Experts...")
    oracle_accs = {}
    for task in tasks:
        oracle_accs[task] = evaluate_model(experts[task], test_loaders[task])
    print(f"Oracle: MNIST={oracle_accs['mnist']:.2f}%, Fashion={oracle_accs['fashion']:.2f}%, CIFAR10={oracle_accs['cifar10']:.2f}%, Avg={sum(oracle_accs.values())/3:.2f}%")
    
    # Helper to construct merged model state dict
    def get_merged_base(model_type):
        if model_type == 'resnet18':
            m = ResNet18Expert()
        else:
            m = MLPExpert()
        # Set task specific heads from the trained experts
        m_sd = m.state_dict()
        for task in tasks:
            for k, v in experts[task].state_dict().items():
                if 'head' in k:
                    # Save task specific head with a unique prefix
                    m_sd[f"{task}_{k}"] = v.clone()
        return m, m_sd
    
    def evaluate_merged_backbone(backbone_sd, name, with_bn_cal=False, quantize=False):
        model, m_sd = get_merged_base(model_type)
        
        # Load the merged backbone and copy task-specific heads
        for task in tasks:
            # First load backbone
            for k, v in backbone_sd.items():
                m_sd[k] = v.clone()
            # Then load the specific head for this task
            for k, v in m_sd.items():
                if f"{task}_head" in k:
                    orig_key = k.replace(f"{task}_", "")
                    m_sd[orig_key] = v.clone()
            
            model.load_state_dict(m_sd, strict=False)
            
            if quantize:
                eval_model = apply_8bit_quantization(model)
            else:
                eval_model = model
                
            if with_bn_cal and model_type == 'resnet18':
                eval_model = calibrate_bn(eval_model, task, num_samples=32)
                
            accs[task] = evaluate_model(eval_model, test_loaders[task])
        avg_acc = sum(accs.values()) / 3
        print(f"{name}: MNIST={accs['mnist']:.2f}%, Fashion={accs['fashion']:.2f}%, CIFAR10={accs['cifar10']:.2f}%, Avg={avg_acc:.2f}%")
        return avg_acc, copy.deepcopy(accs)

    # Dictionary to keep results
    results = {}
    accs = {}
    
    # Extract backbone state dicts
    prog_backbone = {k: v for k, v in progenitor.state_dict().items() if 'backbone' in k}
    expert_backbones = {task: {k: v for k, v in experts[task].state_dict().items() if 'backbone' in k} for task in tasks}
    
    # ---- 2. Weight Averaging (WA) ----
    wa_backbone = {}
    for k in prog_backbone.keys():
        if torch.is_floating_point(prog_backbone[k]):
            wa_backbone[k] = torch.stack([expert_backbones[task][k] for task in tasks], dim=0).mean(dim=0)
        else:
            wa_backbone[k] = prog_backbone[k].clone()
        
    results['WA'] = evaluate_merged_backbone(wa_backbone, "Weight Averaging (WA)")
    results['WA_Q'] = evaluate_merged_backbone(wa_backbone, "Weight Averaging (WA) + 8bit Quant", quantize=True)
    
    if model_type == 'resnet18':
        results['WA_DEBN'] = evaluate_merged_backbone(wa_backbone, "WA + DE-BN", with_bn_cal=True)
        results['WA_DEBN_Q'] = evaluate_merged_backbone(wa_backbone, "WA + DE-BN + 8bit Quant", with_bn_cal=True, quantize=True)

    # ---- 3. Tuned Task Arithmetic (TA) ----
    best_ta_acc = 0
    best_ta_sd = None
    best_lambda = 0.5
    for lam in [0.3, 0.4, 0.5, 0.6]:
        ta_backbone = {}
        for k in prog_backbone.keys():
            if torch.is_floating_point(prog_backbone[k]):
                updates = [expert_backbones[task][k] - prog_backbone[k] for task in tasks]
                ta_backbone[k] = prog_backbone[k] + lam * torch.stack(updates, dim=0).sum(dim=0)
            else:
                ta_backbone[k] = prog_backbone[k].clone()
        acc, _ = evaluate_merged_backbone(ta_backbone, f"Task Arithmetic (TA, lambda={lam})")
        if acc > best_ta_acc:
            best_ta_acc = acc
            best_ta_sd = ta_backbone
            best_lambda = lam
            
    print(f"Best Tuned TA (lambda={best_lambda}): {best_ta_acc:.2f}%")
    results['TA'] = evaluate_merged_backbone(best_ta_sd, f"Tuned TA (lambda={best_lambda})")
    results['TA_Q'] = evaluate_merged_backbone(best_ta_sd, f"Tuned TA + 8bit Quant", quantize=True)
    
    if model_type == 'resnet18':
        results['TA_DEBN'] = evaluate_merged_backbone(best_ta_sd, "Tuned TA + DE-BN", with_bn_cal=True)
        results['TA_DEBN_Q'] = evaluate_merged_backbone(best_ta_sd, "Tuned TA + DE-BN + 8bit Quant", with_bn_cal=True, quantize=True)

    # ---- 4. Isotropic Parameter Resonance (U-IPR) ----
    u_ipr_backbone = {}
    for k in prog_backbone.keys():
        if expert_backbones['mnist'][k].dtype != torch.float32:
            u_ipr_backbone[k] = wa_backbone[k]
            continue
        updates = [expert_backbones[task][k] - prog_backbone[k] for task in tasks]
        merged_update = wa_backbone[k] - prog_backbone[k]
        
        norm_merged = torch.norm(merged_update) + 1e-8
        avg_norm_experts = torch.stack([torch.norm(u) for u in updates]).mean()
        
        cal_update = merged_update * (avg_norm_experts / norm_merged)
        u_ipr_backbone[k] = prog_backbone[k] + cal_update
        
    results['U-IPR'] = evaluate_merged_backbone(u_ipr_backbone, "U-IPR")
    results['U-IPR_Q'] = evaluate_merged_backbone(u_ipr_backbone, "U-IPR + 8bit Quant", quantize=True)

    # ---- 5. Spectral Parameter Resonance (S-IPR) ----
    s_ipr_backbone = {}
    for k in prog_backbone.keys():
        param_merged = wa_backbone[k]
        if param_merged.dim() < 2 or param_merged.dtype != torch.float32:
            s_ipr_backbone[k] = param_merged
            continue
            
        # S-IPR operates on updates
        updates = [expert_backbones[task][k] - prog_backbone[k] for task in tasks]
        merged_update = param_merged - prog_backbone[k]
        
        orig_shape = merged_update.shape
        d1 = orig_shape[0]
        
        m_up_2d = merged_update.view(d1, -1)
        u_m, s_m, v_m = torch.svd(m_up_2d)
        
        # Get expert updates singular values
        s_experts = []
        for u in updates:
            u_2d = u.view(d1, -1)
            _, s_e, _ = torch.svd(u_2d)
            s_experts.append(s_e)
        avg_s = torch.stack(s_experts, dim=0).mean(dim=0)
        
        # S-IPR aligns singular values of the merged update to the average expert singular values
        # We match size in case dimensions differ
        s_target = s_m.clone()
        min_len = min(len(s_target), len(avg_s))
        s_target[:min_len] = avg_s[:min_len]
        
        cal_up_2d = u_m @ torch.diag_embed(s_target) @ v_m.T
        s_ipr_backbone[k] = prog_backbone[k] + cal_up_2d.view(orig_shape)
        
    results['S-IPR'] = evaluate_merged_backbone(s_ipr_backbone, "S-IPR")
    results['S-IPR_Q'] = evaluate_merged_backbone(s_ipr_backbone, "S-IPR + 8bit Quant", quantize=True)

    # ---- 6. Wasserstein-Calibrated Parameter Resonance (WCPR) ----
    wcpr_backbone = {}
    for k in prog_backbone.keys():
        p_merged = wa_backbone[k]
        p_experts = [expert_backbones[task][k] for task in tasks]
        if p_merged.dim() < 2 or p_merged.dtype != torch.float32:
            wcpr_backbone[k] = p_merged
        else:
            wcpr_backbone[k] = wcpr_calibrate_layer(p_merged, p_experts)
            
    results['WCPR'] = evaluate_merged_backbone(wcpr_backbone, "WCPR")
    results['WCPR_Q'] = evaluate_merged_backbone(wcpr_backbone, "WCPR + 8bit Quant", quantize=True)

    # ---- 7. Bures-Wasserstein Parameter Alignment (BWPA - Ours) ----
    bwpa_backbone = {}
    for k in prog_backbone.keys():
        p_merged = wa_backbone[k]
        p_experts = [expert_backbones[task][k] for task in tasks]
        if p_merged.dim() < 2 or p_merged.dtype != torch.float32:
            bwpa_backbone[k] = p_merged
        else:
            bwpa_backbone[k] = bw_align_layer(p_merged, p_experts, eps=1e-6)
            
    results['BWPA'] = evaluate_merged_backbone(bwpa_backbone, "BWPA (Ours)")
    results['BWPA_Q'] = evaluate_merged_backbone(bwpa_backbone, "BWPA (Ours) + 8bit Quant", quantize=True)
    
    if model_type == 'resnet18':
        results['BWPA_DEBN'] = evaluate_merged_backbone(bwpa_backbone, "BWPA + DE-BN", with_bn_cal=True)
        results['BWPA_DEBN_Q'] = evaluate_merged_backbone(bwpa_backbone, "BWPA + DE-BN + 8bit Quant", with_bn_cal=True, quantize=True)
        
    # ---- 8. Full-Covariance Bures-Wasserstein Parameter Alignment (Full-BWPA Ablation) ----
    full_bwpa_backbone = {}
    for k in prog_backbone.keys():
        p_merged = wa_backbone[k]
        p_experts = [expert_backbones[task][k] for task in tasks]
        if p_merged.dim() < 2 or p_merged.dtype != torch.float32:
            full_bwpa_backbone[k] = p_merged
        else:
            full_bwpa_backbone[k] = full_bw_align_layer(p_merged, p_experts, eps=1e-6)
            
    results['Full-BWPA'] = evaluate_merged_backbone(full_bwpa_backbone, "Full-BWPA (Ablation)")
    results['Full-BWPA_Q'] = evaluate_merged_backbone(full_bwpa_backbone, "Full-BWPA (Ablation) + 8bit Quant", quantize=True)
    
    if model_type == 'resnet18':
        results['Full-BWPA_DEBN'] = evaluate_merged_backbone(full_bwpa_backbone, "Full-BWPA + DE-BN", with_bn_cal=True)
        results['Full-BWPA_DEBN_Q'] = evaluate_merged_backbone(full_bwpa_backbone, "Full-BWPA + DE-BN + 8bit Quant", with_bn_cal=True, quantize=True)

    return results, oracle_accs

if __name__ == '__main__':
    resnet_results, resnet_oracle = merge_and_eval('resnet18')
    mlp_results, mlp_oracle = merge_and_eval('mlp')
    
    # Save results to a file for later latex writing and logging
    import pickle
    with open('experiment_results.pkl', 'wb') as f:
        pickle.dump({
            'resnet_results': resnet_results,
            'resnet_oracle': resnet_oracle,
            'mlp_results': mlp_results,
            'mlp_oracle': mlp_oracle
        }, f)
        
    print("\nEvaluation successfully completed! Results pickled in experiment_results.pkl.")
