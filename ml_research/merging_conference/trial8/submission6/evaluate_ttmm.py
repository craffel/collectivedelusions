import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import os
import time

# Set deterministic seeds
torch.manual_seed(42)
np.random.seed(42)

def get_resnet18_1channel():
    model = models.resnet18(weights=None)
    # Modify conv1 to accept 1 channel
    new_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.conv1 = new_conv
    model.fc = nn.Linear(512, 10)
    return model

def project_simplex(v):
    if v.shape[-1] == 2:
        v0, v1 = v[..., 0], v[..., 1]
        d = (v0 - v1) / 2.0
        y0 = torch.clamp(0.5 + d, 0.0, 1.0)
        y1 = 1.0 - y0
        return torch.stack([y0, y1], dim=-1)
    else:
        shape = v.shape
        v = v.view(-1, shape[-1])
        n_features = v.shape[1]
        u, _ = torch.sort(v, descending=True, dim=-1)
        cssv = torch.cumsum(u, dim=-1) - 1.0
        ind = torch.arange(1, n_features + 1, device=v.device, dtype=v.dtype)
        cond = u - cssv / ind > 0
        rho = cond.sum(dim=-1, keepdim=True) - 1
        theta = torch.gather(cssv, 1, rho) / (rho + 1)
        w = torch.clamp(v - theta, min=0.0)
        return w.view(shape)

def get_merged_state_dict(sd0, sd1, coefs, w_bn=None):
    # coefs is a dict mapping layer/parameter prefixes to a tensor of shape [2]
    # w_bn is a tensor of shape [2] for BN running statistics
    merged_sd = {}
    
    # If w_bn is not provided, we default it to the average of coefs across layers
    if w_bn is None:
        all_coefs = torch.stack(list(coefs.values())) # [num_layers, 2]
        w_bn = all_coefs.mean(dim=0)
        
    for key in sd0.keys():
        if key in sd1:
            # Check if this is a BN buffer
            if any(buf in key for buf in ["running_mean", "running_var", "num_batches_tracked"]):
                if "running_mean" in key:
                    var_key = key.replace("running_mean", "running_var")
                    mu0 = sd0[key]
                    mu1 = sd1[key]
                    var0 = sd0[var_key]
                    var1 = sd1[var_key]
                    
                    # Soft BN Buffer Fusion (Mixture of Gaussians moment matching)
                    mu_fused = w_bn[0] * mu0 + w_bn[1] * mu1
                    var_fused = w_bn[0] * (var0 + (mu0 - mu_fused)**2) + w_bn[1] * (var1 + (mu1 - mu_fused)**2)
                    
                    merged_sd[key] = mu_fused
                    merged_sd[var_key] = var_fused
                elif "num_batches_tracked" in key:
                    merged_sd[key] = sd0[key] # keep track of batches
            else:
                # Weight/bias parameter merging
                # Find matching coefficient prefix
                matched_prefix = None
                for prefix in coefs.keys():
                    if key.startswith(prefix):
                        matched_prefix = prefix
                        break
                if matched_prefix is not None:
                    c = coefs[matched_prefix]
                    merged_sd[key] = c[0] * sd0[key] + c[1] * sd1[key]
                else:
                    merged_sd[key] = 0.5 * sd0[key] + 0.5 * sd1[key]
        else:
            merged_sd[key] = sd0[key]
    return merged_sd

def extract_features(model, images):
    x = model.conv1(images)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    x = model.avgpool(x)
    features = torch.flatten(x, 1)
    return features

# Precompute offline prototypes for baselines
def precompute_offline_prototypes(sd0, sd1, device="cuda"):
    print("Precomputing offline prototypes for baseline...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    mnist_set = torchvision.datasets.MNIST(root=".", train=True, download=False, transform=transform)
    kmnist_set = torchvision.datasets.KMNIST(root=".", train=True, download=False, transform=transform)
    
    # Take 500 samples
    indices = list(range(500))
    mnist_subset = Subset(mnist_set, indices)
    kmnist_subset = Subset(kmnist_set, indices)
    
    mnist_loader = DataLoader(mnist_subset, batch_size=250, shuffle=False)
    kmnist_loader = DataLoader(kmnist_subset, batch_size=250, shuffle=False)
    
    # Load expert models
    model0 = get_resnet18_1channel().to(device)
    model0.load_state_dict(sd0)
    model0.eval()
    
    model1 = get_resnet18_1channel().to(device)
    model1.load_state_dict(sd1)
    model1.eval()
    
    # Static model (average)
    model_static = get_resnet18_1channel().to(device)
    static_coef = {prefix: torch.tensor([0.5, 0.5], device=device) for prefix in ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "fc"]}
    model_static.load_state_dict(get_merged_state_dict(sd0, sd1, static_coef))
    model_static.eval()
    
    # Extract features and compute means and class-wise prototypes
    # Domain 0 (MNIST)
    all_feats0 = []
    all_labels0 = []
    with torch.no_grad():
        for imgs, lbls in mnist_loader:
            imgs = imgs.to(device)
            feats = extract_features(model_static, imgs)
            all_feats0.append(feats)
            all_labels0.append(lbls)
    all_feats0 = torch.cat(all_feats0, dim=0) # [500, 512]
    all_labels0 = torch.cat(all_labels0, dim=0).to(device) # [500]
    
    mu0 = all_feats0.mean(dim=0) # [512]
    
    # Domain 1 (KMNIST)
    all_feats1 = []
    all_labels1 = []
    with torch.no_grad():
        for imgs, lbls in kmnist_loader:
            imgs = imgs.to(device)
            feats = extract_features(model_static, imgs)
            all_feats1.append(feats)
            all_labels1.append(lbls)
    all_feats1 = torch.cat(all_feats1, dim=0) # [500, 512]
    all_labels1 = torch.cat(all_labels1, dim=0).to(device) # [500]
    
    mu1 = all_feats1.mean(dim=0) # [512]
    
    mu_static = 0.5 * mu0 + 0.5 * mu1
    
    # Centered class prototypes
    class_prototypes0 = {}
    class_prototypes1 = {}
    
    for c in range(10):
        mask0 = (all_labels0 == c)
        if mask0.sum() > 0:
            class_prototypes0[c] = (all_feats0[mask0] - mu_static).mean(dim=0)
        else:
            class_prototypes0[c] = torch.zeros(512, device=device)
            
        mask1 = (all_labels1 == c)
        if mask1.sum() > 0:
            class_prototypes1[c] = (all_feats1[mask1] - mu_static).mean(dim=0)
        else:
            class_prototypes1[c] = torch.zeros(512, device=device)
            
    print("Offline prototypes precomputed successfully!")
    return mu_static, class_prototypes0, class_prototypes1

def get_cohesion_score(features, mu_static, class_prototypes, device="cuda"):
    # features shape: [B, 512]
    # mu_static shape: [512]
    # class_prototypes: dict mapping class c to [512] tensor
    centered = features - mu_static
    centered_norm = centered / (centered.norm(dim=1, keepdim=True) + 1e-8)
    
    # class_prototypes matrix
    proto_mat = torch.stack([class_prototypes[c] for c in range(10)]) # [10, 512]
    proto_mat_norm = proto_mat / (proto_mat.norm(dim=1, keepdim=True) + 1e-8)
    
    # Cosine similarities
    sims = torch.mm(centered_norm, proto_mat_norm.t()) # [B, 10]
    max_sims, _ = sims.max(dim=1) # [B]
    return max_sims.mean().item()

def evaluate_method(method_name, sd0, sd1, test_stream, mu_static, class_prototypes0, class_prototypes1, device="cuda",
                    tau_entropy=0.70, alpha_ema=0.1, lr=0.005, beta_damping=0.5, gamma=5.0, anchor_layers=None):
    print(f"\nEvaluating method: {method_name}")
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Define model keys we want to adapt layer-wise
    adapt_layers = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "fc"]
    
    # Initialize coefficients
    coefs = {prefix: torch.tensor([0.5, 0.5], device=device, requires_grad=True) for prefix in adapt_layers}
    
    # Build models
    model0 = get_resnet18_1channel().to(device)
    model0.load_state_dict(sd0)
    model0.eval()
    
    model1 = get_resnet18_1channel().to(device)
    model1.load_state_dict(sd1)
    model1.eval()
    
    merged_model = get_resnet18_1channel().to(device)
    
    accuracies = []
    mnist_accs = []
    kmnist_accs = []
    fashion_accs = []
    
    # For FDF-DPA (Our Method) Class Prototypes (init empty)
    dynamic_prototypes = {0: {}, 1: {}}
    dynamic_mu0_sum = torch.zeros(512, device=device)
    dynamic_mu1_sum = torch.zeros(512, device=device)
    num_samples0 = 0
    num_samples1 = 0
    calibration_entropies = []
    
    # Kronecker Trace registration helpers
    activations = {}
    gradients = {}
    forward_hooks = []
    backward_hooks = []
    
    def register_hooks(model):
        nonlocal forward_hooks, backward_hooks
        # Remove any existing hooks
        for h in forward_hooks: h.remove()
        for h in backward_hooks: h.remove()
        forward_hooks.clear()
        backward_hooks.clear()
        
        # We only register hooks on the layers that have weights to adapt
        for name, module in model.named_modules():
            if name in adapt_layers or any(name == prefix for prefix in adapt_layers):
                def get_f_hook(n):
                    def hook(m, inp, out):
                        x = inp[0].detach()
                        activations[n] = (x ** 2).sum().item() / (x.numel() / x.shape[1] if len(x.shape) > 1 else 1)
                    return hook
                def get_b_hook(n):
                    def hook(m, g_inp, g_out):
                        g = g_out[0].detach()
                        gradients[n] = (g ** 2).sum().item() / (g.numel() / g.shape[1] if len(g.shape) > 1 else 1)
                    return hook
                forward_hooks.append(module.register_forward_hook(get_f_hook(name)))
                backward_hooks.append(module.register_backward_hook(get_b_hook(name)))

    for batch_idx, (images, labels, source_task) in enumerate(test_stream):
        images, labels = images.to(device), labels.to(device)
        
        # 1. EVALUATION (Predict before update)
        with torch.no_grad():
            # Update merged model weights using current coefs
            # For our method and DF-Bayes-TTMM, we use Soft BN Buffer Fusion using soft Bayesian posterior weights computed on predictions
            if method_name in ["FDF-DPA", "FDF-DPA (Auto)", "DF-Bayes-TTMM"]:
                # Compute expert predictive entropies on this batch
                outputs0 = model0(images)
                outputs1 = model1(images)
                
                probs0 = torch.softmax(outputs0, dim=-1)
                entropy0 = -torch.sum(probs0 * torch.log(probs0 + 1e-8), dim=-1).mean().item()
                
                probs1 = torch.softmax(outputs1, dim=-1)
                entropy1 = -torch.sum(probs1 * torch.log(probs1 + 1e-8), dim=-1).mean().item()
                
                avg_entropy = 0.5 * (entropy0 + entropy1)
                
                if method_name == "DF-Bayes-TTMM" and avg_entropy > tau_entropy:
                    # Novel domain for DF-Bayes-TTMM: uniform BN weights as in DF-Bayes-TTMM paper
                    w_bn = torch.tensor([0.5, 0.5], device=device)
                else:
                    # Soft Bayesian posterior
                    w0 = np.exp(-gamma * entropy0)
                    w1 = np.exp(-gamma * entropy1)
                    w_bn = torch.tensor([w0 / (w0 + w1), w1 / (w0 + w1)], device=device)
                
                # Merge model state dict with Soft BN Buffer Fusion
                merged_sd = get_merged_state_dict(sd0, sd1, {k: coefs[k].detach() for k in coefs}, w_bn=w_bn)
            else:
                # Default merging
                merged_sd = get_merged_state_dict(sd0, sd1, {k: coefs[k].detach() for k in coefs})
                
            merged_model.load_state_dict(merged_sd)
            merged_model.eval()
            
            # Predict
            outputs = merged_model(images)
            _, preds = outputs.max(1)
            correct = preds.eq(labels).sum().item()
            acc = 100.0 * correct / images.size(0)
            
            accuracies.append(acc)
            if source_task == "MNIST":
                mnist_accs.append(acc)
            elif source_task == "KMNIST":
                kmnist_accs.append(acc)
            elif source_task == "FashionMNIST":
                fashion_accs.append(acc)
                
        # 2. ADAPTATION (Compute update for next step)
        if method_name == "Static":
            # No update
            pass
            
        elif method_name == "PROTO-TTMM" or method_name == "KT-Fisher":
            # 2a. Novelty routing
            with torch.no_grad():
                feats = extract_features(merged_model, images)
                c0 = get_cohesion_score(feats, mu_static, class_prototypes0, device)
                c1 = get_cohesion_score(feats, mu_static, class_prototypes1, device)
                max_cohesion = max(c0, c1)
                
            tau_N = 0.58 # novelty threshold
            is_novel = (max_cohesion < tau_N)
            
            if not is_novel:
                # Routed to known expert
                k_star = 0 if c0 > c1 else 1
                target_vector = torch.zeros(2, device=device)
                target_vector[k_star] = 1.0
                
                # EMA update
                alpha_ema = 0.9
                for k in coefs:
                    coefs[k] = alpha_ema * coefs[k].detach() + (1.0 - alpha_ema) * target_vector
            else:
                # Novel domain adaptation
                # Find expert with lowest entropy
                with torch.no_grad():
                    out0 = model0(images)
                    p0 = torch.softmax(out0, dim=-1)
                    ent0 = -torch.sum(p0 * torch.log(p0 + 1e-8), dim=-1).mean().item()
                    
                    out1 = model1(images)
                    p1 = torch.softmax(out1, dim=-1)
                    ent1 = -torch.sum(p1 * torch.log(p1 + 1e-8), dim=-1).mean().item()
                    
                target_k = 0 if ent0 < ent1 else 1
                Y_t = torch.zeros(2, device=device)
                Y_t[target_k] = 1.0
                
                if method_name == "PROTO-TTMM":
                    # Unpreconditioned adaptation: uniform lr = 0.005
                    lr = 0.005
                    for k in coefs:
                        new_val = coefs[k].detach() - lr * (coefs[k].detach() - Y_t)
                        coefs[k] = project_simplex(new_val)
                else:
                    # KT-Fisher preconditioned adaptation
                    # Register hooks and run a single forward/backward with entropy loss through merged model
                    register_hooks(merged_model)
                    
                    # Single forward with entropy loss
                    optimizer = torch.optim.SGD(merged_model.parameters(), lr=0.0) # dummy optimizer just to clear grads
                    optimizer.zero_grad()
                    
                    # Compute prediction entropy of merged model
                    merged_out = merged_model(images)
                    probs = torch.softmax(merged_out, dim=-1)
                    entropy_loss = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()
                    entropy_loss.backward()
                    
                    # Update coefficients using Kronecker Trace preconditioning
                    lr = 0.005
                    beta_damping = 0.5
                    eps_scale = 1e-5
                    
                    for k in coefs:
                        # Find matching layer in activations/gradients
                        name = k
                        tr_A = activations.get(name, 1.0)
                        tr_G = gradients.get(name, 1.0)
                        
                        # Param size
                        param_size = 1.0
                        for name_p, p in merged_model.named_parameters():
                            if name_p.startswith(k) and "weight" in name_p:
                                param_size = p.numel()
                                break
                                
                        # Sensitivity estimate
                        F_w = (tr_G * tr_A) / param_size
                        lr_w = lr * ((F_w + eps_scale) ** (-beta_damping))
                        
                        # Direct Riemannian update
                        new_val = coefs[k].detach() - lr_w * (coefs[k].detach() - Y_t)
                        coefs[k] = project_simplex(new_val)
                        
        elif method_name == "DF-Bayes-TTMM":
            # DF-Bayes-TTMM Baseline (Data-Free Bayesian TTMM)
            with torch.no_grad():
                out0 = model0(images)
                p0 = torch.softmax(out0, dim=-1)
                ent0 = -torch.sum(p0 * torch.log(p0 + 1e-8), dim=-1).mean().item()
                
                out1 = model1(images)
                p1 = torch.softmax(out1, dim=-1)
                ent1 = -torch.sum(p1 * torch.log(p1 + 1e-8), dim=-1).mean().item()
                
            avg_entropy = 0.5 * (ent0 + ent1)
            is_novel = (avg_entropy > tau_entropy)
            
            if not is_novel:
                # Known domain: use soft posterior weights directly as merging coefficients
                w0 = np.exp(-gamma * ent0)
                w1 = np.exp(-gamma * ent1)
                w = torch.tensor([w0 / (w0 + w1), w1 / (w0 + w1)], device=device)
                for k in coefs:
                    coefs[k] = w.clone()
            else:
                # Novel domain adaptation
                # We target the expert with lowest prediction entropy
                target_k = 0 if ent0 < ent1 else 1
                Y_t = torch.zeros(2, device=device)
                Y_t[target_k] = 1.0
                
                # KT-Fisher sensitivity preconditioning but NO layer anchoring (adapts all layers)
                register_hooks(merged_model)
                optimizer = torch.optim.SGD(merged_model.parameters(), lr=0.0)
                optimizer.zero_grad()
                
                merged_out = merged_model(images)
                probs = torch.softmax(merged_out, dim=-1)
                entropy_loss = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()
                entropy_loss.backward()
                
                eps_scale = 1e-5
                for k in coefs:
                    tr_A = activations.get(k, 1.0)
                    tr_G = gradients.get(k, 1.0)
                    
                    param_size = 1.0
                    for name_p, p in merged_model.named_parameters():
                        if name_p.startswith(k) and "weight" in name_p:
                            param_size = p.numel()
                            break
                    F_w = (tr_G * tr_A) / param_size
                    lr_w = lr * ((F_w + eps_scale) ** (-beta_damping))
                    
                    # Update coefficients with NO anchoring
                    new_val = coefs[k].detach() - lr_w * (coefs[k].detach() - Y_t)
                    coefs[k] = project_simplex(new_val)
                    
        elif method_name in ["FDF-DPA", "FDF-DPA (Auto)"]:
            # Our Method: Fully Data-Free Dynamic Prototype Adaptation with Kronecker-Trace Feature Anchoring
            # 1. Compute expert entropies and average expert entropy
            with torch.no_grad():
                out0 = model0(images)
                p0 = torch.softmax(out0, dim=-1)
                ent0 = -torch.sum(p0 * torch.log(p0 + 1e-8), dim=-1).mean().item()
                
                out1 = model1(images)
                p1 = torch.softmax(out1, dim=-1)
                ent1 = -torch.sum(p1 * torch.log(p1 + 1e-8), dim=-1).mean().item()
                
            avg_entropy = 0.5 * (ent0 + ent1)
            
            if method_name == "FDF-DPA (Auto)":
                if len(calibration_entropies) < 10:
                    calibration_entropies.append(avg_entropy)
                    is_novel = False
                else:
                    if len(calibration_entropies) == 10:
                        mu_H = np.mean(calibration_entropies)
                        sigma_H = np.std(calibration_entropies)
                        # Set threshold as mu_H + 8 * sigma_H, capped at [0.55, 0.75]
                        dynamic_tau = mu_H + 8.0 * sigma_H
                        dynamic_tau = max(0.55, min(0.75, dynamic_tau))
                        print(f"[Auto-Threshold] Dynamically calibrated threshold: {dynamic_tau:.4f} (mu_H: {mu_H:.4f}, sigma_H: {sigma_H:.4f})")
                        calibration_entropies.append(-1.0) # mark as complete
                        tau_entropy = dynamic_tau
                    is_novel = (avg_entropy > tau_entropy)
            else:
                is_novel = (avg_entropy > tau_entropy)
            
            if not is_novel:
                # Routed to known domain
                k_star = 0 if ent0 < ent1 else 1
                target_vector = torch.zeros(2, device=device)
                target_vector[k_star] = 1.0
                
                # EMA update for coefficients
                for k in coefs:
                    coefs[k] = alpha_ema * coefs[k].detach() + (1.0 - alpha_ema) * target_vector
                    
                # Dynamically update known-domain prototypes using high-confidence predictions!
                with torch.no_grad():
                    feats = extract_features(merged_model, images)
                    # Use expert model outputs for class prediction
                    active_outputs = out0 if k_star == 0 else out1
                    active_probs = torch.softmax(active_outputs, dim=-1)
                    max_probs, pred_classes = active_probs.max(dim=1)
                    
                    # Accumulate sample features for mean estimation
                    for i in range(images.size(0)):
                        if max_probs[i] > 0.95:
                            c = pred_classes[i].item()
                            feat = feats[i].detach()
                            
                            # Initialize or update class prototype
                            if c not in dynamic_prototypes[k_star]:
                                dynamic_prototypes[k_star][c] = feat.clone()
                            else:
                                alpha_proto = 0.95
                                dynamic_prototypes[k_star][c] = alpha_proto * dynamic_prototypes[k_star][c] + (1.0 - alpha_proto) * feat
                                
            else:
                # Novel domain adaptation
                # We target the expert with lowest prediction entropy
                target_k = 0 if ent0 < ent1 else 1
                Y_t = torch.zeros(2, device=device)
                Y_t[target_k] = 1.0
                
                # Kronecker Trace sensitivity estimation for anchoring
                register_hooks(merged_model)
                optimizer = torch.optim.SGD(merged_model.parameters(), lr=0.0)
                optimizer.zero_grad()
                
                merged_out = merged_model(images)
                probs = torch.softmax(merged_out, dim=-1)
                entropy_loss = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()
                entropy_loss.backward()
                
                eps_scale = 1e-5
                if anchor_layers is None:
                    anchor_layers = ["conv1", "bn1", "layer1", "layer2"]
                
                for k in coefs:
                    # For our Kronecker-Trace Guided Feature Anchoring:
                    # Anchor layers have learning rate = 0
                    if k in anchor_layers:
                        lr_w = 0.0 # Anchor features!
                    else:
                        tr_A = activations.get(k, 1.0)
                        tr_G = gradients.get(k, 1.0)
                        
                        param_size = 1.0
                        for name_p, p in merged_model.named_parameters():
                            if name_p.startswith(k) and "weight" in name_p:
                                param_size = p.numel()
                                break
                        F_w = (tr_G * tr_A) / param_size
                        lr_w = lr * ((F_w + eps_scale) ** (-beta_damping))
                        
                    # Update
                    new_val = coefs[k].detach() - lr_w * (coefs[k].detach() - Y_t)
                    coefs[k] = project_simplex(new_val)
                    
    # Remove any lingering hooks
    for h in forward_hooks: h.remove()
    for h in backward_hooks: h.remove()
    
    # Print results
    mnist_m = np.mean(mnist_accs) if mnist_accs else 0.0
    kmnist_m = np.mean(kmnist_accs) if kmnist_accs else 0.0
    fashion_m = np.mean(fashion_accs) if fashion_accs else 0.0
    overall_m = np.mean(accuracies)
    
    print(f"MNIST Segment Accuracy: {mnist_m:.2f}%")
    print(f"KMNIST Segment Accuracy: {kmnist_m:.2f}%")
    print(f"FashionMNIST Segment Accuracy: {fashion_m:.2f}%")
    print(f"Overall Accuracy: {overall_m:.2f}%")
    
    return {
        "mnist": mnist_m,
        "kmnist": kmnist_m,
        "fashion": fashion_m,
        "overall": overall_m
    }

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Disable cuDNN to avoid CUDNN_STATUS_NOT_INITIALIZED errors on this cluster
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = False
        print("Disabled cuDNN for stability")
    
    # Load expert model weights
    mnist_path = "expert_mnist.pth"
    kmnist_path = "expert_kmnist.pth"
    
    if not os.path.exists(mnist_path) or not os.path.exists(kmnist_path):
        print("Error: Expert models not found! Please make sure train_experts.py runs first.")
        exit(1)
        
    sd_mnist = torch.load(mnist_path, map_location=device)
    sd_kmnist = torch.load(kmnist_path, map_location=device)
    
    # Prepare test stream dataloaders
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    mnist_test = torchvision.datasets.MNIST(root=".", train=False, download=False, transform=transform)
    kmnist_test = torchvision.datasets.KMNIST(root=".", train=False, download=False, transform=transform)
    fashion_test = torchvision.datasets.FashionMNIST(root=".", train=False, download=False, transform=transform)
    
    # Create non-stationary test stream:
    # 90 sequential batches of size 64
    # Batches 1-30: MNIST
    # Batches 31-60: KMNIST
    # Batches 61-90: FashionMNIST
    mnist_loader = DataLoader(mnist_test, batch_size=64, shuffle=True)
    kmnist_loader = DataLoader(kmnist_test, batch_size=64, shuffle=True)
    fashion_loader = DataLoader(fashion_test, batch_size=64, shuffle=True)
    
    stream_batches = []
    
    # Get 30 batches of each
    mnist_iter = iter(mnist_loader)
    for _ in range(30):
        imgs, lbls = next(mnist_iter)
        stream_batches.append((imgs, lbls, "MNIST"))
        
    kmnist_iter = iter(kmnist_loader)
    for _ in range(30):
        imgs, lbls = next(kmnist_iter)
        stream_batches.append((imgs, lbls, "KMNIST"))
        
    fashion_iter = iter(fashion_loader)
    for _ in range(30):
        imgs, lbls = next(fashion_iter)
        stream_batches.append((imgs, lbls, "FashionMNIST"))
        
    print(f"Built non-stationary test stream with {len(stream_batches)} batches.")
    
    # Precompute offline prototypes (used by KT-Fisher and PROTO-TTMM)
    mu_static, class_prototypes0, class_prototypes1 = precompute_offline_prototypes(sd_mnist, sd_kmnist, device)
    
    results = {}
    
    # Run evaluations
    results["Static"] = evaluate_method("Static", sd_mnist, sd_kmnist, stream_batches, mu_static, class_prototypes0, class_prototypes1, device)
    results["PROTO-TTMM"] = evaluate_method("PROTO-TTMM", sd_mnist, sd_kmnist, stream_batches, mu_static, class_prototypes0, class_prototypes1, device)
    results["DF-Bayes-TTMM"] = evaluate_method("DF-Bayes-TTMM", sd_mnist, sd_kmnist, stream_batches, mu_static, class_prototypes0, class_prototypes1, device)
    results["KT-Fisher"] = evaluate_method("KT-Fisher", sd_mnist, sd_kmnist, stream_batches, mu_static, class_prototypes0, class_prototypes1, device)
    results["FDF-DPA"] = evaluate_method("FDF-DPA", sd_mnist, sd_kmnist, stream_batches, mu_static, class_prototypes0, class_prototypes1, device)
    results["FDF-DPA (Auto)"] = evaluate_method("FDF-DPA (Auto)", sd_mnist, sd_kmnist, stream_batches, mu_static, class_prototypes0, class_prototypes1, device)
    
    print("\n" + "="*50)
    print("FINAL RESULTS COMPARISON")
    print("="*50)
    print(f"{'Method':<20} | {'MNIST':<8} | {'KMNIST':<8} | {'Fashion':<8} | {'Overall':<8}")
    print("-"*65)
    for method in ["Static", "PROTO-TTMM", "DF-Bayes-TTMM", "KT-Fisher", "FDF-DPA", "FDF-DPA (Auto)"]:
        res = results[method]
        print(f"{method:<20} | {res['mnist']:<8.2f}% | {res['kmnist']:<8.2f}% | {res['fashion']:<8.2f}% | {res['overall']:<8.2f}%")
    print("="*50)
    
    # Save results to a file for later plotting
    import json
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
