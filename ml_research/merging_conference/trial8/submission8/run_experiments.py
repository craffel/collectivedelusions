import os
import argparse
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.func import functional_call

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.enabled = False

# -------------------------------------------------------------------------
# 1. Models and Expert Setup
# -------------------------------------------------------------------------

def get_resnet18_expert():
    # Use weights=None for scratch, or load weights and sum first conv channel
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    old_conv = model.conv1
    new_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        new_conv.weight.copy_(old_conv.weight.sum(dim=1, keepdim=True))
    model.conv1 = new_conv
    model.fc = nn.Linear(512, 10)
    return model

def train_expert(name, dataset, device, epochs=3, batch_size=256):
    print(f"\n--- Training Expert: {name} ---")
    model = get_resnet18_expert().to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        t0 = time.time()
        running_loss = 0.0
        correct = 0
        total = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * x.size(0)
            _, predicted = out.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            
        epoch_loss = running_loss / len(dataset)
        epoch_acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}% | Time: {time.time()-t0:.2f}s")
        
    return model

def evaluate_model(model, dataset, device, batch_size=256):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            _, predicted = out.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
    return 100.0 * correct / total

# -------------------------------------------------------------------------
# 2. Kronecker Trace & Joint Fisher Computation Helpers
# -------------------------------------------------------------------------

class KroneckerFisherTracker:
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.activations = {}
        self.gradients = {}
        self.register_hooks()
        
    def register_hooks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # Forward hook to save input activation
                def fw_hook(mod, inp, out, name=name):
                    self.activations[name] = inp[0].detach()
                # Backward hook to save output gradient of the loss
                def bw_hook(mod, grad_inp, grad_out, name=name):
                    self.gradients[name] = grad_out[0].detach()
                
                h_fw = module.register_forward_hook(fw_hook)
                h_bw = module.register_full_backward_hook(bw_hook)
                self.hooks.extend([h_fw, h_bw])
                
    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
        
    def compute_sensitivities(self):
        sensitivities = {}
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if name not in self.activations or name not in self.gradients:
                    sensitivities[name + ".weight"] = 1.0
                    if module.bias is not None:
                        sensitivities[name + ".bias"] = 1.0
                    continue
                
                act = self.activations[name]
                grad = self.gradients[name]
                b = act.shape[0]
                
                if isinstance(module, nn.Conv2d):
                    # act shape: (b, cin, hin, win)
                    # grad shape: (b, cout, hout, wout)
                    hin, win = act.shape[2], act.shape[3]
                    hout, wout = grad.shape[2], grad.shape[3]
                    
                    act_trace = (act ** 2).sum() / (b * hin * win)
                    grad_trace = (grad ** 2).sum() / (b * hout * wout)
                    
                    cout, cin, kh, kw = module.weight.shape
                    num_params = cout * cin * kh * kw
                    
                    sens = (grad_trace * act_trace) / num_params
                    
                elif isinstance(module, nn.Linear):
                    if len(act.shape) == 3:
                        act_flat = act.reshape(-1, act.shape[-1])
                        grad_flat = grad.reshape(-1, grad.shape[-1])
                    else:
                        act_flat = act
                        grad_flat = grad
                    
                    act_trace = (act_flat ** 2).sum() / act_flat.shape[0]
                    grad_trace = (grad_flat ** 2).sum() / grad_flat.shape[0]
                    
                    cout, cin = module.weight.shape
                    num_params = cout * cin
                    
                    sens = (grad_trace * act_trace) / num_params
                
                # Safeguard sensitivities from being exactly zero or infinite
                sens_val = sens.item()
                if np.isnan(sens_val) or np.isinf(sens_val) or sens_val <= 0:
                    sens_val = 1.0
                sensitivities[name + ".weight"] = sens_val
                
                if module.bias is not None:
                    bias_sens = (grad ** 2).sum().item() / b
                    if np.isnan(bias_sens) or np.isinf(bias_sens) or bias_sens <= 0:
                        bias_sens = 1.0
                    sensitivities[name + ".bias"] = bias_sens
                    
        # Global normalization of sensitivities
        if len(sensitivities) > 0:
            max_val = max(sensitivities.values())
            if max_val > 0:
                for k in sensitivities:
                    sensitivities[k] = sensitivities[k] / max_val
                    if np.isnan(sensitivities[k]) or sensitivities[k] <= 0:
                        sensitivities[k] = 1.0
                    
        return sensitivities

def compute_offline_joint_fisher(expert1, expert2, cal1_dataset, cal2_dataset, device, num_samples=500):
    print("Computing offline Joint Fisher Information...")
    expert1.eval()
    expert2.eval()
    
    # We will compute diagonal empirical Fisher on both experts
    def compute_single_fisher(model, dataset):
        fisher = {}
        for name, param in model.named_parameters():
            fisher[name] = torch.zeros_like(param)
        
        loader = DataLoader(Subset(dataset, list(range(min(num_samples, len(dataset))))), batch_size=1, shuffle=False)
        for x, _ in loader:
            x = x.to(device)
            model.zero_grad()
            out = model(x)
            # Use pseudo-labeling / entropy as objective or standard log-likelihood
            prob = F.softmax(out, dim=1)
            pseudo_y = torch.multiclass_sample = torch.argmax(prob, dim=1)
            loss = F.cross_entropy(out, pseudo_y)
            loss.backward()
            
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        fisher[name] += (param.grad ** 2)
                        
        with torch.no_grad():
            for name in fisher:
                fisher[name] = fisher[name] / min(num_samples, len(dataset))
        return fisher

    f1 = compute_single_fisher(expert1, cal1_dataset)
    f2 = compute_single_fisher(expert2, cal2_dataset)
    
    # Joint Fisher and normalize globally
    joint_fisher = {}
    max_val = 1e-8
    for name in f1:
        joint_fisher[name] = f1[name] + f2[name]
        max_val = max(max_val, joint_fisher[name].max().item())
        
    # Scale sensitivities
    scaled_fisher = {}
    for name in joint_fisher:
        scaled_fisher[name] = (joint_fisher[name] / max_val).mean().item()
        if np.isnan(scaled_fisher[name]) or scaled_fisher[name] <= 0:
            scaled_fisher[name] = 1.0
            
    print("Offline Joint Fisher computed.")
    return scaled_fisher

# -------------------------------------------------------------------------
# 3. Prototype Extraction & SCTS Prior Calculation
# -------------------------------------------------------------------------

def extract_prototypes(expert1, expert2, cal1_dataset, cal2_dataset, device, num_samples=500, num_classes=10):
    print("Extracting class prototypes in the Unified Static Space...")
    expert1.eval()
    expert2.eval()
    
    def extract_features(model, x):
        # Forward pass up to avgpool
        with torch.no_grad():
            x = model.conv1(x)
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

    def compute_class_centroids(model, dataset):
        loader = DataLoader(Subset(dataset, list(range(min(num_samples, len(dataset))))), batch_size=64, shuffle=False)
        class_feats = {c: [] for c in range(num_classes)}
        for x, y in loader:
            x = x.to(device)
            feats = extract_features(model, x)
            for i in range(x.size(0)):
                label = y[i].item()
                if label in class_feats:
                    class_feats[label].append(feats[i].cpu())
                    
        centroids = {}
        for c in range(num_classes):
            if len(class_feats[c]) > 0:
                centroids[c] = torch.stack(class_feats[c]).mean(dim=0).to(device)
            else:
                centroids[c] = torch.zeros(512).to(device)
        return centroids

    p1 = compute_class_centroids(expert1, cal1_dataset)
    p2 = compute_class_centroids(expert2, cal2_dataset)
    return p1, p2, extract_features

def compute_scts_prior(batch_x, p1, p2, feat_fn, expert1, expert2, device, target_scale=50.0, eps_stab=10.0):
    # Extract features using both experts
    f1 = feat_fn(expert1, batch_x)
    f2 = feat_fn(expert2, batch_x)
    
    # Compute average minimum Euclidean distance for expert 1
    # p1 is class -> centroid
    p1_tensor = torch.stack([p1[c] for c in range(10)]) # (10, 512)
    p2_tensor = torch.stack([p2[c] for c in range(10)]) # (10, 512)
    
    b = batch_x.size(0)
    
    # Expert 1 distance
    dist1 = torch.cdist(f1, p1_tensor.unsqueeze(0)).squeeze(0) # (b, 10)
    min_dist1 = dist1.min(dim=1)[0].mean().item()
    
    # Expert 2 distance
    dist2 = torch.cdist(f2, p2_tensor.unsqueeze(0)).squeeze(0) # (b, 10)
    min_dist2 = dist2.min(dim=1)[0].mean().item()
    
    # Calculate SCTS prior
    # S = [-D1, -D2]
    S1 = -min_dist1
    S2 = -min_dist2
    
    delta = abs(min_dist1 - min_dist2)
    tau = delta / target_scale + eps_stab
    
    # Softmax
    exp1 = np.exp(S1 / tau)
    exp2 = np.exp(S2 / tau)
    p_prior = exp1 / (exp1 + exp2)
    
    return p_prior, min_dist1, min_dist2

# -------------------------------------------------------------------------
# 4. Differentiable Parameter Merging and Soft BN Buffer Fusion
# -------------------------------------------------------------------------

def get_merged_params_and_buffers(w_global, deltas, expert1_state, expert2_state, model, use_vp_bn=True):
    merged_dict = {}
    
    # Parameters
    lambdas = {}
    for name, param in model.named_parameters():
        if name in deltas:
            lambdas[name] = torch.sigmoid(w_global + deltas[name])
        else:
            lambdas[name] = torch.sigmoid(w_global)
        merged_dict[name] = lambdas[name] * expert1_state[name] + (1.0 - lambdas[name]) * expert2_state[name]
        
    # Calculate average lambda for buffers
    all_lambdas = list(lambdas.values())
    if len(all_lambdas) > 0:
        lambda_bar = torch.mean(torch.stack([l.mean() for l in all_lambdas]))
    else:
        lambda_bar = torch.sigmoid(w_global)
        
    w1 = lambda_bar.detach()
    w2 = 1.0 - w1
    
    # Buffers
    for name, buf in model.named_buffers():
        if "running_mean" in name:
            mean1 = expert1_state[name]
            mean2 = expert2_state[name]
            merged_dict[name] = w1 * mean1 + w2 * mean2
        elif "running_var" in name:
            var1 = expert1_state[name]
            var2 = expert2_state[name]
            if use_vp_bn:
                mean1 = expert1_state[name.replace("running_var", "running_mean")]
                mean2 = expert2_state[name.replace("running_var", "running_mean")]
                mean_fused = w1 * mean1 + w2 * mean2
                # Variance-Preserving Soft BN Fusion
                var_fused = w1 * var1 + w2 * var2 + w1 * (mean1 - mean_fused)**2 + w2 * (mean2 - mean_fused)**2
                merged_dict[name] = var_fused
            else:
                # Standard linear interpolation
                merged_dict[name] = w1 * var1 + w2 * var2
        else:
            merged_dict[name] = expert1_state[name]
            
    return merged_dict

# -------------------------------------------------------------------------
# 5. Stream Evaluator
# -------------------------------------------------------------------------

def evaluate_stream(method_name, stream_batches, expert1, expert2, p1, p2, feat_fn, offline_fisher, device, lr=1e-3, beta=1.0, gamma=0.02, steps=1):
    print(f"\nEvaluating Method: {method_name}")
    set_seed(42) # Ensure evaluation order and inputs are identical
    
    # Expert state dicts
    expert1_state = expert1.state_dict()
    expert2_state = expert2.state_dict()
    
    # Unified evaluation model
    eval_model = get_resnet18_expert().to(device)
    eval_model.eval()
    
    # Track segment performance
    segment_accuracies = []
    total_correct = 0
    total_samples = 0
    
    segment_stats = {}
    
    # Initialize global logit
    # Default to uniform merging
    w_global_val = 0.0
    
    # Tracking latency
    latencies = []
    
    # Initialize EMA sensitivities for momentum-augmented tracking
    ema_sens = {}
    alpha_ema = 0.8
    
    # Robustly parse custom hyperparameters from method_name
    curr_lr = lr
    if "lr = " in method_name:
        try:
            curr_lr = float(method_name.split("lr = ")[1].split(")")[0].split(",")[0].strip())
        except Exception:
            pass
            
    curr_beta = beta
    if "beta = " in method_name:
        try:
            curr_beta = float(method_name.split("beta = ")[1].split(")")[0].split(",")[0].strip())
        except Exception:
            pass
            
    curr_gamma = gamma
    if "gamma = " in method_name:
        try:
            curr_gamma = float(method_name.split("gamma = ")[1].split(")")[0].split(",")[0].strip())
        except Exception:
            pass
    
    for b_idx, (batch_x, batch_y, segment_name) in enumerate(stream_batches):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        t0 = time.time()
        
        # 1. Prior-Guided Routing and Initialization
        p_prior, d1, d2 = compute_scts_prior(batch_x, p1, p2, feat_fn, expert1, expert2, device)
        p_prior = max(1e-4, min(1.0 - 1e-4, p_prior))
        
        is_ours_variant = "CL-KT-Fisher" in method_name or any(x in method_name for x in ["No VP-BN", "gamma =", "EMA"])
        if method_name == "CL W-Fisher (Paper 8)" or is_ours_variant:
            w_global_init = np.log(p_prior / (1.0 - p_prior))
        else:
            w_global_init = 0.0 # Standard uniform start
            
        # Initialize optimization variables for this batch
        w_global = torch.tensor(w_global_init, requires_grad=True, device=device)
        deltas = {}
        for name, param in eval_model.named_parameters():
            deltas[name] = torch.zeros_like(param, requires_grad=True, device=device)
            
        # Determine sensitivities / preconditioning
        sensitivities = {}
        if method_name == "CL W-Fisher (Paper 8)":
            sensitivities = {name: offline_fisher.get(name, 1.0) for name in eval_model.state_dict()}
        elif method_name == "KT-Fisher (Paper 9)" or is_ours_variant:
            # We estimate on-the-fly Kronecker Fisher trace
            # First, run a forward/backward of entropy to get gradients
            tracker = KroneckerFisherTracker(eval_model)
            temp_params = get_merged_params_and_buffers(w_global, deltas, expert1_state, expert2_state, eval_model)
            out = functional_call(eval_model, temp_params, batch_x)
            entropy = - (F.softmax(out, dim=1) * F.log_softmax(out, dim=1)).sum(dim=1).mean()
            entropy.backward()
            sensitivities = tracker.compute_sensitivities()
            tracker.remove_hooks()
            eval_model.zero_grad()
            
            # Momentum-augmented Kronecker-Fisher preconditioning (EMA tracking across batches)
            if "EMA" in method_name:
                if len(ema_sens) == 0:
                    ema_sens = sensitivities.copy()
                else:
                    for name in sensitivities:
                        ema_sens[name] = alpha_ema * ema_sens.get(name, sensitivities[name]) + (1.0 - alpha_ema) * sensitivities[name]
                sensitivities = ema_sens.copy()
        else:
            sensitivities = {name: 1.0 for name in eval_model.state_dict()}
            
        # 2. Optimization Loop (Test-Time Adaptation)
        actual_steps = 0 if method_name == "Static Uniform" else steps
        for step in range(actual_steps):
            # Compute merged params
            use_vp = method_name == "DF-Bayes-TTMM (Paper 10)" or ("CL-KT-Fisher" in method_name and "No VP-BN" not in method_name)
            merged_params = get_merged_params_and_buffers(w_global, deltas, expert1_state, expert2_state, eval_model, use_vp_bn=use_vp)
            
            # Forward pass
            out = functional_call(eval_model, merged_params, batch_x)
            
            # Loss formulation
            # Entropy
            loss_ent = - (F.softmax(out, dim=1) * F.log_softmax(out, dim=1)).sum(dim=1).mean()
            
            # KL Regularization to prior
            lambda_bar = torch.sigmoid(w_global)
            prior_t = torch.tensor([p_prior, 1.0 - p_prior], device=device)
            post_t = torch.stack([lambda_bar, 1.0 - lambda_bar])
            loss_kl = (prior_t * torch.log(prior_t / (post_t + 1e-8))).sum()
            
            # Coherence penalty
            loss_coherence = 0.0
            if method_name == "CL W-Fisher (Paper 8)" or is_ours_variant:
                delta_norms = [d.pow(2).sum() for d in deltas.values()]
                loss_coherence = curr_gamma * torch.stack(delta_norms).sum()
                
            loss = loss_ent
            if method_name == "CL W-Fisher (Paper 8)" or is_ours_variant:
                loss += curr_beta * loss_kl + loss_coherence
            elif method_name == "DF-Bayes-TTMM (Paper 10)":
                loss += curr_beta * loss_kl # Bayesian posterior update
                
            # Backward pass
            # Extract gradients manually to update
            if method_name in ["AdaMerging (Standard)", "KT-Fisher (Paper 9)", "DF-Bayes-TTMM (Paper 10)"]:
                # Only update global coefficient
                grads = torch.autograd.grad(loss, [w_global], allow_unused=True)
                g_wg = grads[0]
                if g_wg is not None:
                    with torch.no_grad():
                        w_global -= curr_lr * g_wg
            else:
                # Update global and layer-wise deltas
                vars_to_opt = [w_global] + list(deltas.values())
                grads = torch.autograd.grad(loss, vars_to_opt, allow_unused=True)
                
                g_wg = grads[0]
                g_deltas = grads[1:]
                
                with torch.no_grad():
                    if g_wg is not None:
                        w_global -= curr_lr * g_wg
                    for j, (name, delta) in enumerate(deltas.items()):
                        g_d = g_deltas[j]
                        if g_d is not None:
                            sens_val = sensitivities.get(name, 1.0)
                            # Preconditioned gradient update
                            delta -= curr_lr / (sens_val + 1e-2) * g_d


                            
        # 3. Final Evaluation Pass
        use_vp = method_name == "DF-Bayes-TTMM (Paper 10)" or ("CL-KT-Fisher" in method_name and "No VP-BN" not in method_name)
        final_params = get_merged_params_and_buffers(w_global, deltas, expert1_state, expert2_state, eval_model, use_vp_bn=use_vp)
        with torch.no_grad():
            out = functional_call(eval_model, final_params, batch_x)
            _, preds = out.max(1)
            correct = preds.eq(batch_y).sum().item()
            
        t_elapsed = time.time() - t0
        latencies.append(t_elapsed)
        
        # Accumulate metrics
        total_correct += correct
        total_samples += batch_y.size(0)
        
        if segment_name not in segment_stats:
            segment_stats[segment_name] = {"correct": 0, "total": 0}
        segment_stats[segment_name]["correct"] += correct
        segment_stats[segment_name]["total"] += batch_y.size(0)
        
    avg_acc = 100.0 * total_correct / total_samples
    print(f"Overall stream accuracy: {avg_acc:.2f}% | Mean latency: {np.mean(latencies)*1000:.1f}ms")
    
    results = {"overall": avg_acc, "latency": np.mean(latencies)*1000}
    for seg in segment_stats:
        acc = 100.0 * segment_stats[seg]["correct"] / segment_stats[seg]["total"]
        print(f"  Segment: {seg:15s} | Accuracy: {acc:.2f}%")
        results[seg] = acc
        
    return results

# -------------------------------------------------------------------------
# 6. Stream Construction
# -------------------------------------------------------------------------

def construct_stream(profile_idx, mnist_test, kmnist_test, fashion_test, batch_size=64):
    set_seed(42)
    # MNIST loader
    mnist_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True)
    kmnist_loader = DataLoader(kmnist_test, batch_size=batch_size, shuffle=True)
    fashion_loader = DataLoader(fashion_test, batch_size=batch_size, shuffle=True)
    
    mnist_iter = iter(mnist_loader)
    kmnist_iter = iter(kmnist_loader)
    fashion_iter = iter(fashion_loader)
    
    stream_batches = []
    
    if profile_idx == 1:
        # Stream profile 1 (Closed Sequential): 15 MNIST, 15 KMNIST
        print("\nConstructing Profile 1 (Closed Sequential)...")
        for _ in range(15):
            x, y = next(mnist_iter)
            stream_batches.append((x, y, "MNIST (Known)"))
        for _ in range(15):
            x, y = next(kmnist_iter)
            stream_batches.append((x, y, "KMNIST (Known)"))
            
    elif profile_idx == 2:
        # Stream profile 2 (Closed Alternating): 30 alternating
        print("\nConstructing Profile 2 (Closed Alternating)...")
        for i in range(15):
            x, y = next(mnist_iter)
            stream_batches.append((x, y, "MNIST (Known)"))
            x, y = next(kmnist_iter)
            stream_batches.append((x, y, "KMNIST (Known)"))
            
    elif profile_idx == 3:
        # Stream profile 3 (Open-World): 10 MNIST, 10 KMNIST, 10 Fashion (novel)
        print("\nConstructing Profile 3 (Open-World Short)...")
        for _ in range(10):
            x, y = next(mnist_iter)
            stream_batches.append((x, y, "MNIST (Known)"))
        for _ in range(10):
            x, y = next(kmnist_iter)
            stream_batches.append((x, y, "KMNIST (Known)"))
        for _ in range(10):
            x, y = next(fashion_iter)
            stream_batches.append((x, y, "Fashion (Novel)"))
            
    elif profile_idx == 4:
        # Stream profile 4 (Open-World Standard): 30 MNIST, 30 KMNIST, 30 Fashion
        print("\nConstructing Profile 4 (Open-World Standard)...")
        for _ in range(30):
            x, y = next(mnist_iter)
            stream_batches.append((x, y, "MNIST (Known)"))
        for _ in range(30):
            x, y = next(kmnist_iter)
            stream_batches.append((x, y, "KMNIST (Known)"))
        for _ in range(30):
            x, y = next(fashion_iter)
            stream_batches.append((x, y, "Fashion (Novel)"))
            
    return stream_batches

# -------------------------------------------------------------------------
# 7. Main Loop Orchestrator
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Test-Time Model Merging Experimentation")
    parser.add_argument("--mode", type=str, choices=["train", "eval"], default="eval", help="Execution mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs for experts")
    parser.add_argument("--batch_size", type=int, default=256, help="Training batch size")
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load datasets
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    print("Loading Datasets...")
    mnist_train = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    mnist_test = torchvision.datasets.MNIST(root="./data", train=False, transform=transform, download=True)
    
    kmnist_train = torchvision.datasets.KMNIST(root="./data", train=True, transform=transform, download=True)
    kmnist_test = torchvision.datasets.KMNIST(root="./data", train=False, transform=transform, download=True)
    
    fashion_train = torchvision.datasets.FashionMNIST(root="./data", train=True, transform=transform, download=True)
    fashion_test = torchvision.datasets.FashionMNIST(root="./data", train=False, transform=transform, download=True)
    
    # Create experts folder if not exists
    os.makedirs("experts", exist_ok=True)
    
    # 2. Train or Load Experts
    mnist_path = "experts/mnist_expert.pth"
    kmnist_path = "experts/kmnist_expert.pth"
    fashion_path = "experts/fashion_expert.pth"
    
    if args.mode == "train" or not (os.path.exists(mnist_path) and os.path.exists(kmnist_path) and os.path.exists(fashion_path)):
        print("Training expert models...")
        mnist_expert = train_expert("MNIST Expert", mnist_train, device, epochs=args.epochs, batch_size=args.batch_size)
        torch.save(mnist_expert.state_dict(), mnist_path)
        
        kmnist_expert = train_expert("KMNIST Expert", kmnist_train, device, epochs=args.epochs, batch_size=args.batch_size)
        torch.save(kmnist_expert.state_dict(), kmnist_path)
        
        fashion_expert = train_expert("FashionMNIST Expert", fashion_train, device, epochs=args.epochs, batch_size=args.batch_size)
        torch.save(fashion_expert.state_dict(), fashion_path)
        print("Expert models saved successfully!")
    else:
        print("Loading pre-trained expert models...")
        
    mnist_expert = get_resnet18_expert().to(device)
    mnist_expert.load_state_dict(torch.load(mnist_path, map_location=device, weights_only=True))
    mnist_expert.eval()
    
    kmnist_expert = get_resnet18_expert().to(device)
    kmnist_expert.load_state_dict(torch.load(kmnist_path, map_location=device, weights_only=True))
    kmnist_expert.eval()
    
    fashion_expert = get_resnet18_expert().to(device)
    fashion_expert.load_state_dict(torch.load(fashion_path, map_location=device, weights_only=True))
    fashion_expert.eval()
    
    # Standalone Evaluation
    print("\n--- Standalone Accuracies ---")
    mnist_acc = evaluate_model(mnist_expert, mnist_test, device)
    kmnist_acc = evaluate_model(kmnist_expert, kmnist_test, device)
    fashion_acc = evaluate_model(fashion_expert, fashion_test, device)
    print(f"MNIST Expert on MNIST Test: {mnist_acc:.2f}%")
    print(f"KMNIST Expert on KMNIST Test: {kmnist_acc:.2f}%")
    print(f"FashionMNIST Expert on FashionMNIST Test: {fashion_acc:.2f}%")
    
    # 3. Precompute Offline Calibration Components
    p1, p2, feat_fn = extract_prototypes(mnist_expert, kmnist_expert, mnist_train, kmnist_train, device)
    offline_fisher = compute_offline_joint_fisher(mnist_expert, kmnist_expert, mnist_train, kmnist_train, device)
    
    # 4. Stream Evaluation loop across Profiles
    all_results = {}
    
    for profile_idx in [1, 2, 3, 4]:
        profile_name = f"Profile_{profile_idx}"
        print(f"\n=========================================================================")
        print(f"EVALUATING ON STREAM PROFILE {profile_idx}")
        print(f"=========================================================================")
        
        stream_batches = construct_stream(profile_idx, mnist_test, kmnist_test, fashion_test, batch_size=64)
        
        profile_results = {}
        
        # Method List
        methods = [
            "Static Uniform",
            "AdaMerging (Standard)",
            "KT-Fisher (Paper 9)",
            "CL W-Fisher (Paper 8)",
            "DF-Bayes-TTMM (Paper 10)",
            "CL-KT-Fisher (Ours)",
            "CL-KT-Fisher (No VP-BN)",
            "CL-KT-Fisher (EMA)",
            "CL-KT-Fisher (EMA, No VP-BN)",
            "CL-KT-Fisher (gamma = 0)",
            "CL-KT-Fisher (gamma = 0.1)",
            "CL-KT-Fisher (lr = 0.05)",
            "CL-KT-Fisher (lr = 0.005)",
            "CL-KT-Fisher (beta = 0.1)",
            "CL-KT-Fisher (beta = 5.0)",
            "CL-KT-Fisher (gamma = 0.002)",
            "CL-KT-Fisher (gamma = 0.5)"
        ]
        
        for method in methods:
            res = evaluate_stream(
                method, 
                stream_batches, 
                mnist_expert, 
                kmnist_expert, 
                p1, 
                p2, 
                feat_fn, 
                offline_fisher, 
                device, 
                lr=0.01, 
                beta=1.0, 
                gamma=0.02, 
                steps=1
            )
            profile_results[method] = res
            
        all_results[profile_name] = profile_results
        
    # Write experimental findings summary to progress.md and stdout
    print("\n=========================================================================")
    print("EXPERIMENTAL SUMMARY")
    print("=========================================================================")
    
    summary_text = "\n## Experimental Results Summary\n\n"
    for profile in all_results:
        summary_text += f"### {profile}\n"
        print(f"\n--- {profile} ---")
        headers = ["Method", "Overall Acc (%)", "Latency (ms)"]
        print(f"{headers[0]:30s} | {headers[1]:15s} | {headers[2]:12s}")
        print("-" * 65)
        
        summary_text += "| Method | Overall Accuracy (%) | Latency (ms) |\n|---|---|---|\n"
        
        for method in all_results[profile]:
            acc = all_results[profile][method]["overall"]
            lat = all_results[profile][method]["latency"]
            print(f"{method:30s} | {acc:15.2f}% | {lat:12.1f}ms")
            summary_text += f"| {method} | {acc:.2f}% | {lat:.1f}ms |\n"
            
    # Append to progress.md
    with open("progress.md", "a") as f:
        f.write(summary_text)
        
    print("\nAll experiments completed and logged to progress.md.")

if __name__ == "__main__":
    main()
