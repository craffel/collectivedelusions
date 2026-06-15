import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import timm
import os
import json

# Set random seeds for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Programmatic image generators for 4 distinct visual domains
def gen_checkerboard(seed=None):
    if seed is not None:
        np.random.seed(seed)
    img = np.zeros((224, 224, 3), dtype=np.float32)
    s = 28
    phase = np.random.randint(0, 10) if seed is not None else 0
    for i in range(8):
        for j in range(8):
            if (i + j + phase) % 2 == 0:
                img[i*s:(i+1)*s, j*s:(j+1)*s, :] = 1.0
    if seed is not None:
        tint = np.random.rand(3).astype(np.float32) * 0.2
        img = img * (1.0 - tint) + tint
    return torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

def gen_sinusoid(seed=None):
    if seed is not None:
        np.random.seed(seed)
    freq = np.random.uniform(8.0, 12.0) if seed is not None else 10.0
    x, y = np.meshgrid(np.linspace(0, freq * np.pi, 224), np.linspace(0, freq * np.pi, 224))
    z = np.sin(x) * np.cos(y)
    img = np.stack([(z + 1.0) / 2.0] * 3, axis=-1).astype(np.float32)
    return torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

def gen_noise(seed=None):
    if seed is not None:
        np.random.seed(seed)
    img = np.random.rand(224, 224, 3).astype(np.float32)
    img = (img + np.roll(img, 1, axis=0) + np.roll(img, 1, axis=1)) / 3.0
    return torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

def gen_gradient(seed=None):
    if seed is not None:
        np.random.seed(seed)
    angle = np.random.uniform(0, 2*np.pi) if seed is not None else 0.0
    x, y = np.meshgrid(np.linspace(0, 1, 224), np.linspace(0, 1, 224))
    r = x * np.cos(angle) + y * np.sin(angle)
    r = (r - r.min()) / (r.max() - r.min() + 1e-6)
    img = np.stack([r, 1.0 - r, (r + 0.5) % 1.0], axis=-1).astype(np.float32)
    return torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

# Generate single sample for a specific task
def get_task_sample(task, seed=None, pixel_noise=0.05):
    if task == 0:
        img = gen_checkerboard(seed)
    elif task == 1:
        img = gen_sinusoid(seed)
    elif task == 2:
        img = gen_noise(seed)
    else:
        img = gen_gradient(seed)
        
    if pixel_noise > 0.0:
        img = img + torch.randn_like(img) * pixel_noise
        img = torch.clamp(img, 0.0, 1.0)
        
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    img_normalized = (img - mean) / std
    return img_normalized.float()

# Set physical simulation parameters
D = 192               # Representation dimension for ViT-Tiny
K = 4                 # 4 tasks
L = 12                # 12 blocks in vit_tiny
L_frozen = 3          # First 3 layers are frozen
T = 200               # Sequential stream length
g_scale = 0.35         # Adapter scale
tau = 0.10            # Softmax temperature
sigma_layer_noise = 0.01

# Trace ViT-Tiny layer-wise representations
def get_layer_activations(model, img_batch):
    activations = []
    with torch.no_grad():
        x = model.patch_embed(img_batch)
        x = model._pos_embed(x)
        x = model.patch_drop(x)
        x = model.norm_pre(x)
        for block in model.blocks:
            x = block(x)
            activations.append(x[:, 0].clone()) # CLS token (B, D)
    return activations

# Offline centroid calibration (Unnormalized space)
def calibrate_centroids(model, num_samples=16):
    print(f"Starting offline calibration with N_cal = {num_samples} samples per task...")
    model.eval()
    centroids = torch.zeros(L + 1, K, D)
    
    for k in range(K):
        task_imgs = []
        for s in range(num_samples):
            img = get_task_sample(k, seed=s, pixel_noise=0.0)
            task_imgs.append(img)
        task_imgs = torch.cat(task_imgs, dim=0)
        
        layer_acts = get_layer_activations(model, task_imgs)
        
        for l in range(1, L + 1):
            act_l = layer_acts[l-1]
            centroids[l, k] = torch.mean(act_l, dim=0)
            
    print("Calibration completed successfully!")
    return centroids

# Analytical PAC-Kinetics transition/input matrices
def get_pac_kinetics_matrices(homogeneous):
    a = 0.85
    if homogeneous:
        A = torch.eye(K) * a
        W = torch.eye(K) * (1.0 - a)
    else:
        A = torch.ones(K, K) * (a / K) + torch.eye(K) * (a * 0.2)
        A = A / (torch.sum(A, dim=1, keepdim=True) + 1e-9) * a
        W = torch.eye(K) * (1.0 - a)
    return A, W

# Main serving simulation
def run_real_serving(model, centroids, seed, homogeneous=True):
    set_seed(seed)
    
    if homogeneous:
        y = []
        for k in range(K):
            y.extend([k] * (T // K))
        y = np.array(y)
    else:
        np.random.seed(12345)
        y = np.random.randint(0, K, size=T)
        
    methods = [
        "Oracle", "Uniform", "SABLE", "Momentum-Merge", 
        "ChemMerge (Proxy)", "ChemMerge (Dynamic)", "PAC-Kinetics", 
        "2D-STEM"
    ]
    
    metrics = {m: {"accuracies": [], "coefficients": []} for m in methods}
    
    pk_A, pk_W = get_pac_kinetics_matrices(homogeneous)
    pk_s = torch.zeros(K)
    
    stem_alpha = {l: torch.ones(K) / K for l in range(L_frozen, L + 1)}
    cm_alpha = {l: torch.ones(K) / K for l in range(L_frozen, L + 1)}
    cmd_alpha = {l: torch.ones(K) / K for l in range(L_frozen, L + 1)}
    
    for t in range(T):
        target_k = y[t]
        
        img = get_task_sample(target_k, seed=seed+t, pixel_noise=0.05)
        layer_acts = get_layer_activations(model, img)
        
        target_v_last = centroids[L, target_k]
        h_3 = layer_acts[L_frozen - 1][0]
        
        # Compute coordinate projection at layer 3
        e_t = torch.zeros(K)
        for k in range(K):
            e_t[k] = torch.max(torch.tensor(0.0), torch.dot(h_3, centroids[L_frozen, k]) / (torch.norm(h_3) * torch.norm(centroids[L_frozen, k]) + 1e-9))
            
        if t == 0:
            Sim_t = torch.tensor(1.0)
            e_prev = e_t.clone()
        else:
            Sim_t = torch.dot(e_t, e_prev) / (torch.norm(e_t) * torch.norm(e_prev) + 1e-9)
            Sim_t = torch.clamp(Sim_t, 0.0, 1.0)
            e_prev = e_t.clone()
            
        # 1. Oracle alignment simulation to find the reference ceiling
        h_oracle = h_3.clone()
        for l in range(L_frozen + 1, L + 1):
            h_oracle = h_oracle + g_scale * (centroids[l, target_k] - h_oracle) + torch.randn(D) * sigma_layer_noise
        oracle_sim = torch.dot(h_oracle, target_v_last) / (torch.norm(h_oracle) * torch.norm(target_v_last) + 1e-9)
        oracle_sim = oracle_sim.item()
        
        # 2. Run other baselines
        for m in methods:
            h_l = h_3.clone()
            m_coeffs = []
            
            if m == "PAC-Kinetics":
                if t == 0:
                    pk_s = e_t.clone()
                else:
                    pk_s = torch.matmul(pk_A, pk_s) + torch.matmul(pk_W, e_t)
                alpha_pk = torch.softmax(pk_s / tau, dim=0)
                
            for l in range(L_frozen + 1, L + 1):
                S = torch.zeros(K)
                for k in range(K):
                    S[k] = torch.dot(h_l, centroids[l, k]) / (torch.norm(h_l) * torch.norm(centroids[l, k]) + 1e-9)
                S_noise = torch.randn(K) * 0.04
                w_l_t = torch.softmax((S + S_noise) / tau, dim=0)
                
                if m == "Oracle":
                    alpha = torch.zeros(K)
                    alpha[target_k] = 1.0
                elif m == "Uniform":
                    alpha = torch.ones(K) / K
                elif m == "SABLE":
                    alpha = w_l_t.clone()
                elif m == "Momentum-Merge":
                    if l == L_frozen + 1:
                        alpha_prev_depth = w_l_t.clone()
                    beta_depth = 0.40
                    alpha = beta_depth * alpha_prev_depth + (1.0 - beta_depth) * w_l_t
                    alpha_prev_depth = alpha.clone()
                elif m == "ChemMerge (Proxy)":
                    beta_depth = 0.60
                    beta_temp = 0.30
                    if l == L_frozen + 1:
                        alpha_prev_depth = w_l_t.clone()
                    alpha_prev_temp = cm_alpha[l]
                    alpha = beta_depth * alpha_prev_depth + beta_temp * alpha_prev_temp + (1.0 - beta_depth - beta_temp) * w_l_t
                    alpha_prev_depth = alpha.clone()
                    cm_alpha[l] = alpha.clone()
                elif m == "ChemMerge (Dynamic)":
                    if l == L_frozen + 1:
                        alpha_prev_depth = w_l_t.clone()
                    beta_depth = 0.60
                    mismatch = torch.norm(w_l_t - cmd_alpha[l], p=2)
                    T_0 = 0.40
                    lam = 2.0
                    temp = T_0 * (1.0 + lam * mismatch.item())
                    E_a, R, A_f, A_b = 0.80, 1.0, 12.0, 6.0
                    k_f = A_f * np.exp(-E_a / (R * temp))
                    k_b = A_b * np.exp(-E_a / (R * temp))
                    
                    alpha_tau = cmd_alpha[l].clone()
                    N_steps, dtau = 5, 0.20
                    for _ in range(N_steps):
                        dalpha = dtau * (k_f * w_l_t * (1.0 - alpha_tau) - k_b * alpha_tau)
                        alpha_tau = alpha_tau + dalpha
                    alpha_tau = torch.clamp(alpha_tau, min=1e-6)
                    alpha_tau = alpha_tau / (torch.sum(alpha_tau) + 1e-12)
                    
                    alpha = beta_depth * alpha_prev_depth + (1.0 - beta_depth) * alpha_tau
                    alpha_prev_depth = alpha.clone()
                    cmd_alpha[l] = alpha.clone()
                elif m == "PAC-Kinetics":
                    alpha = alpha_pk.clone()
                elif m == "2D-STEM":
                    beta_depth, beta_temp_0 = 0.40, 0.40
                    beta_temp_t = beta_temp_0 * (Sim_t.item() ** 3) if t > 0 else 0.0
                    if l == L_frozen + 1:
                        alpha_prev_depth = e_t / (torch.sum(e_t) + 1e-9)
                    alpha_prev_temp = stem_alpha[l]
                    alpha = beta_depth * alpha_prev_depth + beta_temp_t * alpha_prev_temp + (1.0 - beta_depth - beta_temp_t) * w_l_t
                    alpha_prev_depth = alpha.clone()
                    stem_alpha[l] = alpha.clone()
                    
                m_coeffs.append(alpha.clone())
                h_l = h_l + g_scale * torch.matmul(alpha, centroids[l] - h_l) + torch.randn(D) * sigma_layer_noise
                
            sim_m = torch.dot(h_l, target_v_last) / (torch.norm(h_l) * torch.norm(target_v_last) + 1e-9)
            # Define relative accuracy normalized by Oracle similarity
            acc = max(0.0, float(sim_m.item() / (oracle_sim + 1e-9))) * 100.0
            
            metrics[m]["accuracies"].append(acc)
            metrics[m]["coefficients"].append(m_coeffs[-1].numpy())
            
    results = {}
    for m in methods:
        accs = np.array(metrics[m]["accuracies"])
        coeffs = np.array(metrics[m]["coefficients"])
        jitters = np.sum(np.abs(coeffs[1:] - coeffs[:-1]), axis=1)
        results[m] = {
            "accuracy": np.mean(accs),
            "jitter": np.mean(jitters)
        }
    return results

if __name__ == "__main__":
    print("="*60)
    print("PHYSICAL REPRESENTATION VALIDATION OF 2D-STEM ON PRE-TRAINED VIT")
    print("="*60)
    
    print("Loading pre-trained ViT-Tiny model from timm...")
    model = timm.create_model('vit_tiny_patch16_224', pretrained=True)
    model.eval()
    
    centroids = calibrate_centroids(model, num_samples=16)
    
    seeds = [101, 102, 103, 104, 105]
    streams = [("Homogeneous", True), ("Heterogeneous", False)]
    
    all_metrics = {}
    
    for stream_name, homogeneous in streams:
        print(f"\nEvaluating over 5 random seeds under {stream_name} Serving Stream...")
        stream_results = {}
        for seed in seeds:
            res = run_real_serving(model, centroids, seed, homogeneous=homogeneous)
            for m in res:
                if m not in stream_results:
                    stream_results[m] = {"accuracies": [], "jitters": []}
                stream_results[m]["accuracies"].append(res[m]["accuracy"])
                stream_results[m]["jitters"].append(res[m]["jitter"])
                
        all_metrics[stream_name] = {}
        for m in stream_results:
            all_metrics[stream_name][m] = {
                "mean_acc": np.mean(stream_results[m]["accuracies"]),
                "std_acc": np.std(stream_results[m]["accuracies"]),
                "mean_jitter": np.mean(stream_results[m]["jitters"]),
                "std_jitter": np.std(stream_results[m]["jitters"])
            }
            
    print("\n" + "="*50)
    print("PHYSICAL REPRESENTATION VALIDATION RESULTS")
    print("="*50)
    
    for stream_name in ["Homogeneous", "Heterogeneous"]:
        print(f"\n### {stream_name} stream results:")
        print(f"| Method | Alignment Accuracy (%) | Jitter |")
        print(f"| :--- | :---: | :---: |")
        for m in ["Oracle", "Uniform", "SABLE", "Momentum-Merge", "ChemMerge (Proxy)", "ChemMerge (Dynamic)", "PAC-Kinetics", "2D-STEM"]:
            metrics = all_metrics[stream_name][m]
            print(f"| {m} | {metrics['mean_acc']:.2f}% ± {metrics['std_acc']:.2f}% | {metrics['mean_jitter']:.4f} ± {metrics['std_jitter']:.4f} |")
            
    json_results = {
        "Homogeneous": {m: {"acc": float(all_metrics["Homogeneous"][m]["mean_acc"]), "jitter": float(all_metrics["Homogeneous"][m]["mean_jitter"])} for m in all_metrics["Homogeneous"]},
        "Heterogeneous": {m: {"acc": float(all_metrics["Heterogeneous"][m]["mean_acc"]), "jitter": float(all_metrics["Heterogeneous"][m]["mean_jitter"])} for m in all_metrics["Heterogeneous"]}
    }
    os.makedirs("submission/results", exist_ok=True)
    with open("submission/results/physical_vit_metrics.json", "w") as f:
        json.dump(json_results, f, indent=2)
    print("\nSaved physical metrics to submission/results/physical_vit_metrics.json")
