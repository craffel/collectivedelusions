import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from contextlib import nullcontext

try:
    # PyTorch 2.2+
    from torch.nn.attention import SDPBackend, sdpa_kernel
    def get_attention_context():
        return sdpa_kernel(SDPBackend.MATH)
except ImportError:
    try:
        # PyTorch 2.0 and 2.1
        from torch.backends.cuda import sdp_kernel
        def get_attention_context():
            return sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)
    except ImportError:
        def get_attention_context():
            return nullcontext()
import torchvision
from torchvision import transforms
import timm
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Setup directories
os.makedirs("results", exist_ok=True)

# Devices
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Layer index mapping for vit_tiny_patch16_224 (14 layer groups)
def get_layer_idx(name):
    if name.startswith("blocks."):
        parts = name.split(".")
        block_idx = int(parts[1])
        return block_idx + 1  # blocks.0 -> 1, ..., blocks.11 -> 12
    elif name.startswith("norm."):
        return 13
    elif name.startswith("head."):
        return -1  # Skip classification head from backbone merging
    else:
        return 0  # cls_token, pos_embed, patch_embed

# Quantization schemas
def quantize_symmetric(w, bits=8, per_channel=False):
    q_max = (1 << (bits - 1)) - 1
    q_min = -q_max
    
    if w.dim() <= 1 or not per_channel:
        v_max = torch.max(torch.abs(w))
        if v_max == 0:
            return w.clone()
        scale = v_max / q_max
        w_q = torch.clamp(torch.round(w / scale), q_min, q_max)
        return w_q * scale
    else:
        w_deq = w.clone()
        for i in range(w.size(0)):
            v_max = torch.max(torch.abs(w[i]))
            if v_max == 0:
                continue
            scale = v_max / q_max
            w_q = torch.clamp(torch.round(w[i] / scale), q_min, q_max)
            w_deq[i] = w_q * scale
        return w_deq

def quantize_asymmetric(w, bits=8, per_channel=False):
    q_max = (1 << (bits - 1)) - 1
    q_min = -(1 << (bits - 1))
    
    if w.dim() <= 1 or not per_channel:
        v_min = torch.min(w)
        v_max = torch.max(w)
        if v_max == v_min:
            return w.clone()
        scale = (v_max - v_min) / (q_max - q_min)
        zero_point = torch.round(q_min - v_min / scale)
        w_q = torch.clamp(torch.round(w / scale) + zero_point, q_min, q_max)
        return (w_q - zero_point) * scale
    else:
        w_deq = w.clone()
        for i in range(w.size(0)):
            v_min = torch.min(w[i])
            v_max = torch.max(w[i])
            if v_max == v_min:
                continue
            scale = (v_max - v_min) / (q_max - q_min)
            zero_point = torch.round(q_min - v_min / scale)
            w_q = torch.clamp(torch.round(w[i] / scale) + zero_point, q_min, q_max)
            w_deq[i] = (w_q - zero_point) * scale
        return w_deq

def quantize_weight(w, schema):
    if schema == 0:
        return w
    elif schema == 1:
        return quantize_symmetric(w, bits=8, per_channel=False)
    elif schema == 2:
        return quantize_symmetric(w, bits=8, per_channel=True)
    elif schema == 3:
        return quantize_asymmetric(w, bits=8, per_channel=False)
    elif schema == 4:
        return quantize_asymmetric(w, bits=8, per_channel=True)
    elif schema == 5:
        return quantize_symmetric(w, bits=4, per_channel=True)
    else:
        raise ValueError(f"Unknown quantization schema: {schema}")

class STEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, schema):
        return quantize_weight(w, schema)
        
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

def ste_quantize(w, schema):
    return STEQuantize.apply(w, schema)

# Loading data
def get_transforms(is_grayscale=False):
    t_list = []
    if is_grayscale:
        t_list.append(transforms.Grayscale(num_output_channels=3))
    t_list.extend([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transforms.Compose(t_list)

datasets_config = {
    "mnist": (torchvision.datasets.MNIST, True),
    "fashionmnist": (torchvision.datasets.FashionMNIST, True),
    "cifar10": (torchvision.datasets.CIFAR10, False),
    "svhn": (torchvision.datasets.SVHN, False)
}

def load_eval_data():
    loaders = {}
    calib_batches = {}
    for name, (dataset_cls, is_grayscale) in datasets_config.items():
        transform = get_transforms(is_grayscale)
        if name == "svhn":
            test_set = dataset_cls(root="./data", split="test", download=False, transform=transform)
        else:
            test_set = dataset_cls(root="./data", train=False, download=False, transform=transform)
            
        eval_indices = list(range(min(1000, len(test_set))))
        eval_subset = Subset(test_set, eval_indices)
        loaders[name] = DataLoader(eval_subset, batch_size=64, shuffle=False)
        
        calib_indices = list(range(min(16, len(test_set))))
        calib_subset = Subset(test_set, calib_indices)
        calib_loader = DataLoader(calib_subset, batch_size=16, shuffle=False)
        calib_batches[name] = next(iter(calib_loader))
        
    return loaders, calib_batches

def evaluate_model(merged_params, loaders, model, K, q_schema=0):
    quantized_params = {}
    for name, param in merged_params.items():
        if name.startswith("head."):
            quantized_params[name] = param
        else:
            quantized_params[name] = quantize_weight(param, q_schema)
            
    accuracies = {}
    task_keys = list(datasets_config.keys())
    model.eval()
    with torch.no_grad():
        with get_attention_context():
            for k, name in enumerate(task_keys):
                loader = loaders[name]
                task_params = quantized_params.copy()
                task_params["head.weight"] = expert_heads[k]["head.weight"]
                task_params["head.bias"] = expert_heads[k]["head.bias"]
                
                correct = 0
                total = 0
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    logits = torch.func.functional_call(model, task_params, x)
                    preds = logits.argmax(dim=-1)
                    correct += preds.eq(y).sum().item()
                    total += x.size(0)
                accuracies[name] = correct / total
    accuracies["mean"] = np.mean(list(accuracies.values()))
    return accuracies

def compute_entropy_loss(lambdas, params_pre, task_vectors, model, calib_batches, K, q_schema=0, use_ste=False):
    total_entropy = 0.0
    task_keys = list(datasets_config.keys())
    
    merged_params = {}
    for name in params_pre.keys():
        l_idx = get_layer_idx(name)
        if l_idx == -1:
            continue
            
        param_val = params_pre[name]
        for k in range(K):
            param_val = param_val + lambdas[l_idx, k] * task_vectors[k][name]
            
        if use_ste:
            merged_params[name] = ste_quantize(param_val, q_schema)
        else:
            merged_params[name] = param_val
            
    with get_attention_context():
        for k, name in enumerate(task_keys):
            x, _ = calib_batches[name]
            x = x.to(device)
            
            task_params = merged_params.copy()
            task_params["head.weight"] = expert_heads[k]["head.weight"]
            task_params["head.bias"] = expert_heads[k]["head.bias"]
            
            logits = torch.func.functional_call(model, task_params, x)
            probs = torch.softmax(logits, dim=-1)
            log_probs = torch.log_softmax(logits, dim=-1)
            entropy = -torch.sum(probs * log_probs, dim=-1).mean()
            total_entropy = total_entropy + entropy
            
    return total_entropy / K

if __name__ == "__main__":
    print("Loading datasets...")
    loaders, calib_batches = load_eval_data()
    
    print("Loading checkpoints...")
    base_state = torch.load("checkpoints/vit_tiny_pretrained.pt", map_location=device)
    
    expert_states = []
    for d in datasets_config.keys():
        expert_states.append(torch.load(f"checkpoints/vit_tiny_{d}.pt", map_location=device))
        
    K = len(expert_states)
    L = 14
    
    params_pre = {k: v for k, v in base_state.items() if not k.startswith("head.")}
    expert_heads = []
    task_vectors = []
    
    for k in range(K):
        expert_heads.append({
            "head.weight": expert_states[k]["head.weight"],
            "head.bias": expert_states[k]["head.bias"]
        })
        tv = {}
        for name, p in expert_states[k].items():
            if not name.startswith("head."):
                tv[name] = p - base_state[name]
        task_vectors.append(tv)
        
    model = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=10)
    model = model.to(device)
    
    # Compute task-vector L2 norms per layer group
    N = torch.zeros(L, K, device=device)
    for l in range(L):
        for k in range(K):
            sq_norm = 0.0
            for name, tv_p in task_vectors[k].items():
                if get_layer_idx(name) == l:
                    sq_norm += torch.sum(tv_p ** 2).item()
            N[l, k] = np.sqrt(sq_norm)
            
    print("Computed Task-Vector Norms:")
    for l in range(L):
        print(f"Layer {l:2d}: " + " ".join([f"{N[l, k].item():.4f}" for k in range(K)]))
        
    # --- Helper Optimization Loop ---
    def optimize_lambdas(objective_type="adamerging", q_schema=0, epochs=40, lr=1e-2, clip_val=0.1):
        if objective_type in ["poly", "polysacm", "polycr_sacm"]:
            poly_coeffs = torch.zeros(3, K, requires_grad=True, device=device)
            optimizer = torch.optim.Adam([poly_coeffs], lr=lr)
        else:
            theta = torch.ones(L, K, device=device) * -0.847  # sigmoid(-0.847) = 0.3
            theta.requires_grad_(True)
            optimizer = torch.optim.Adam([theta], lr=lr)
            
        for epoch in range(epochs):
            if objective_type in ["poly", "polysacm", "polycr_sacm"]:
                lambdas = torch.zeros(L, K, device=device)
                for l in range(L):
                    depth = l / L
                    for k in range(K):
                        logits = poly_coeffs[0, k] + poly_coeffs[1, k] * depth + poly_coeffs[2, k] * (depth ** 2)
                        lambdas[l, k] = torch.sigmoid(logits)
            else:
                lambdas = torch.sigmoid(theta)
                
            loss = compute_entropy_loss(lambdas, params_pre, task_vectors, model, calib_batches, K, q_schema, use_ste=(objective_type=="qmerge"))
            
            # Regularizations
            if objective_type == "regcal":
                tv_penalty = 0.0
                for k in range(K):
                    for l in range(L - 1):
                        tv_penalty += (lambdas[l+1, k] - lambdas[l, k]) ** 2
                loss += 0.5 * tv_penalty
            elif objective_type == "hessmerge":
                # Standard unregularized/unnormalized SACM
                grads = torch.autograd.grad(loss, lambdas, retain_graph=True)[0]
                rho = 0.15
                grad_norm = torch.norm(grads) + 1e-12
                epsilon = rho * grads / grad_norm
                perturbed_lambdas = torch.clamp(lambdas + epsilon, 0.0, 1.0)
                loss = compute_entropy_loss(perturbed_lambdas, params_pre, task_vectors, model, calib_batches, K, q_schema, use_ste=False)
            elif objective_type == "hessmerge_cr":
                # Clipping-Regularized Normalized SACM (CR-SACM, Ours)
                grads = torch.autograd.grad(loss, lambdas, retain_graph=True)[0]
                N_clipped = torch.clamp(N, min=clip_val)
                grads_hat = grads / N_clipped
                grad_hat_norm = torch.norm(grads_hat) + 1e-12
                rho = 0.15
                epsilon = rho * grads_hat / (grad_hat_norm * N_clipped)
                perturbed_lambdas = torch.clamp(lambdas + epsilon, 0.0, 1.0)
                loss = compute_entropy_loss(perturbed_lambdas, params_pre, task_vectors, model, calib_batches, K, q_schema, use_ste=False)
            elif objective_type == "polysacm":
                # Standard PolySACM
                grads = torch.autograd.grad(loss, poly_coeffs, retain_graph=True)[0]
                rho = 0.15
                grad_norm = torch.norm(grads) + 1e-12
                epsilon = rho * grads / grad_norm
                perturbed_poly_coeffs = poly_coeffs + epsilon
                perturbed_lambdas = torch.zeros(L, K, device=device)
                for l in range(L):
                    depth = l / L
                    for k in range(K):
                        logits = perturbed_poly_coeffs[0, k] + perturbed_poly_coeffs[1, k] * depth + perturbed_poly_coeffs[2, k] * (depth ** 2)
                        perturbed_lambdas[l, k] = torch.sigmoid(logits)
                loss = compute_entropy_loss(perturbed_lambdas, params_pre, task_vectors, model, calib_batches, K, q_schema, use_ste=False)
            elif objective_type == "polycr_sacm":
                # Clipping-Regularized Normalized PolySACM (CR-PolySACM, Ours)
                grads = torch.autograd.grad(loss, lambdas, retain_graph=True)[0]
                N_clipped = torch.clamp(N, min=clip_val)
                grads_hat = grads / N_clipped
                grad_hat_norm = torch.norm(grads_hat) + 1e-12
                rho = 0.15
                epsilon = rho * grads_hat / (grad_hat_norm * N_clipped)
                perturbed_lambdas = torch.clamp(lambdas + epsilon, 0.0, 1.0)
                loss = compute_entropy_loss(perturbed_lambdas, params_pre, task_vectors, model, calib_batches, K, q_schema, use_ste=False)
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        with torch.no_grad():
            if objective_type in ["poly", "polysacm", "polycr_sacm"]:
                lambdas = torch.zeros(L, K, device=device)
                for l in range(L):
                    depth = l / L
                    for k in range(K):
                        logits = poly_coeffs[0, k] + poly_coeffs[1, k] * depth + poly_coeffs[2, k] * (depth ** 2)
                        lambdas[l, k] = torch.sigmoid(logits)
            else:
                lambdas = torch.sigmoid(theta)
                
        merged_params = {}
        for name, p in params_pre.items():
            l_idx = get_layer_idx(name)
            if l_idx == -1:
                continue
            val = p.clone()
            for k in range(K):
                val += lambdas[l_idx, k] * task_vectors[k][name]
            merged_params[name] = val
        return merged_params

    # Run sweep over clip_val
    clip_vals = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]
    sweep_results = {}
    
    # We want to run evaluation for PolyMerge as baseline
    print("\nOptimizing Baseline: PolyMerge...")
    merged_poly = optimize_lambdas("poly")
    
    print("\nEvaluating PolyMerge Baseline...")
    poly_fp32 = evaluate_model(merged_poly, loaders, model, K, q_schema=0)["mean"] * 100
    poly_int8 = evaluate_model(merged_poly, loaders, model, K, q_schema=2)["mean"] * 100
    poly_int4 = evaluate_model(merged_poly, loaders, model, K, q_schema=5)["mean"] * 100
    print(f"PolyMerge: FP32 = {poly_fp32:.4f}%, INT8 Sym Channel = {poly_int8:.4f}%, INT4 Sym Channel = {poly_int4:.4f}%")
    
    for cv in clip_vals:
        print(f"\n--- Sweeping CR-PolySACM with clip_val = {cv} ---")
        merged_polycr = optimize_lambdas("polycr_sacm", clip_val=cv)
        
        fp32_acc = evaluate_model(merged_polycr, loaders, model, K, q_schema=0)["mean"] * 100
        int8_acc = evaluate_model(merged_polycr, loaders, model, K, q_schema=2)["mean"] * 100
        int4_acc = evaluate_model(merged_polycr, loaders, model, K, q_schema=5)["mean"] * 100
        
        sweep_results[cv] = (fp32_acc, int8_acc, int4_acc)
        print(f"CR-PolySACM (clip={cv:.2f}): FP32 = {fp32_acc:.4f}%, INT8 = {int8_acc:.4f}%, INT4 = {int4_acc:.4f}%")
        
    print("\n--- Final Sweep Summary ---")
    print(f"PolyMerge Baseline: FP32 = {poly_fp32:.4f}%, INT8 = {poly_int8:.4f}%, INT4 = {poly_int4:.4f}%")
    for cv, (fp32, int8, int4) in sweep_results.items():
        improved_fp32 = fp32 > poly_fp32
        improved_int8 = int8 > poly_int8
        improved_int4 = int4 > poly_int4
        print(f"clip={cv:.2f}: FP32={fp32:.4f}% ({'+' if improved_fp32 else ''}{fp32-poly_fp32:+.4f}%), "
              f"INT8={int8:.4f}% ({'+' if improved_int8 else ''}{int8-poly_int8:+.4f}%), "
              f"INT4={int4:.4f}% ({'+' if improved_int4 else ''}{int4-poly_int4:+.4f}%)")
