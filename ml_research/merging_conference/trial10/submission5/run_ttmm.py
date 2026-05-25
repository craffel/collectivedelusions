import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
from torch.func import functional_call

# 1. Define SimpleCNN Architecture
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.25)
        self.classifier = nn.Linear(128, 10)

    def forward(self, x, return_features=True):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = torch.flatten(x, 1)
        features = self.fc1(x)
        features_relu = F.relu(features)
        x = self.dropout(features_relu)
        logits = self.classifier(x)
        if return_features:
            return logits, features_relu
        return logits

# 2. Define Hoyer Sparsity Gating
def compute_hoyers_sparsity(x):
    # Map to positive intensity
    x_pos = (x + 1.0) / 2.0
    # Threshold denoising
    x_denoised = (x_pos > 0.35).float() * x_pos
    # Flatten
    flat = x_denoised.view(x.size(0), -1)
    d = flat.size(1)
    l1 = torch.norm(flat, p=1, dim=1)
    l2 = torch.norm(flat, p=2, dim=1) + 1e-8
    sparsity = (np.sqrt(d) - (l1 / l2)) / (np.sqrt(d) - 1.0)
    return sparsity.mean().item()

# 3. Define Model Merging function
def merge_models(model_0, model_1, merged_model, lambdas):
    state_dict_0 = model_0.state_dict()
    state_dict_1 = model_1.state_dict()
    merged_state_dict = {}
    
    for name, param in state_dict_0.items():
        if "running_mean" in name:
            # Moment-matched BN mean fusion
            var_name = name.replace("running_mean", "running_var")
            mu0 = state_dict_0[name]
            mu1 = state_dict_1[name]
            bn_key = name.split(".")[0] + ".weight"
            l = lambdas.get(bn_key, 0.5)
            merged_state_dict[name] = l * mu0 + (1 - l) * mu1
        elif "running_var" in name:
            # Moment-matched BN var fusion
            mean_name = name.replace("running_var", "running_mean")
            mu0 = state_dict_0[mean_name]
            mu1 = state_dict_1[mean_name]
            var0 = state_dict_0[name]
            var1 = state_dict_1[name]
            bn_key = name.split(".")[0] + ".weight"
            l = lambdas.get(bn_key, 0.5)
            merged_state_dict[name] = l * var0 + (1 - l) * var1 + l * (1 - l) * (mu0 - mu1) ** 2
        elif "num_batches_tracked" in name:
            merged_state_dict[name] = state_dict_0[name]
        else:
            # Weight or bias
            l = lambdas.get(name, 0.5)
            merged_state_dict[name] = l * state_dict_0[name] + (1 - l) * state_dict_1[name]
            
    merged_model.load_state_dict(merged_state_dict)

# Map param names to their layer groups
# Group 0: conv1, Group 1: conv2, Group 2: fc1, Group 3: classifier
def get_layer_group(param_name):
    if "conv1" in param_name or "bn1" in param_name:
        return 0
    elif "conv2" in param_name or "bn2" in param_name:
        return 1
    elif "fc1" in param_name:
        return 2
    elif "classifier" in param_name:
        return 3
    return 2 # Default fallback

# Helper to compute prediction entropy
def compute_entropy(logits):
    probs = F.softmax(logits, dim=1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
    return entropy.mean()

# Helper to compute merged parameters differentiably
def get_merged_state_dict_differentiable(expert_0_state, expert_1_state, w_global, delta):
    merged_params_and_buffers = {}
    for name in expert_0_state.keys():
        if "running_mean" in name:
            mu0 = expert_0_state[name]
            mu1 = expert_1_state[name]
            bn_key = name.split(".")[0] + ".weight"
            group = get_layer_group(bn_key)
            l = torch.sigmoid(w_global + delta[group]).detach()
            merged_params_and_buffers[name] = l * mu0 + (1 - l) * mu1
        elif "running_var" in name:
            mean_name = name.replace("running_var", "running_mean")
            mu0 = expert_0_state[mean_name]
            mu1 = expert_1_state[mean_name]
            var0 = expert_0_state[name]
            var1 = expert_1_state[name]
            bn_key = name.split(".")[0] + ".weight"
            group = get_layer_group(bn_key)
            l = torch.sigmoid(w_global + delta[group]).detach()
            merged_params_and_buffers[name] = l * var0 + (1 - l) * var1 + l * (1 - l) * (mu0 - mu1) ** 2
        elif "num_batches_tracked" in name:
            merged_params_and_buffers[name] = expert_0_state[name]
        else:
            group = get_layer_group(name)
            l = torch.sigmoid(w_global + delta[group])
            merged_params_and_buffers[name] = l * expert_0_state[name] + (1 - l) * expert_1_state[name]
    return merged_params_and_buffers

# 4. Evaluation Loop
def evaluate_stream(expert_0, expert_1, stream_batches, regime="static", lr_base=0.01, rho=0.05, damping_base=0.1, alpha=8.0, beta=15.0):
    device = torch.device("cpu")
    expert_0.eval()
    expert_1.eval()
    
    # Initialize a merged model template
    merged_model = SimpleCNN().to(device)
    merged_model.eval() # Ensure template is in eval mode!
    
    phase_accuracies = []
    all_preds = []
    all_targets = []
    
    # Track metrics over time
    curvature_values = []
    adaptive_lrs = []
    adaptive_damps = []
    
    for b_idx, (x, y) in enumerate(stream_batches):
        x, y = x.to(device), y.to(device)
        
        # A. Hoyer sparsity-based routing
        S = compute_hoyers_sparsity(x)
        # Initialize global coefficient prior
        if S >= 0.50:
            w_global = 2.0  # Favor MNIST Expert 0
        else:
            w_global = -2.0 # Favor FashionMNIST Expert 1
            
        # B. Initialize layer-wise merging offsets delta
        # We have 4 layer groups: conv1, conv2, fc1, classifier
        # delta is a tensor of shape 4, representing offsets for each group
        delta = torch.zeros(4, requires_grad=True)
        
        # Optimize delta using the specified regime
        if regime == "static":
            # No test-time optimization
            with torch.no_grad():
                merged_params = get_merged_state_dict_differentiable(
                    expert_0.state_dict(), expert_1.state_dict(), w_global, delta
                )
                logits, _ = functional_call(merged_model, merged_params, (x,))
                
        elif regime == "tta":
            # Standard entropy minimization on delta
            optimizer = torch.optim.SGD([delta], lr=lr_base)
            for step in range(1):
                optimizer.zero_grad()
                merged_params = get_merged_state_dict_differentiable(
                    expert_0.state_dict(), expert_1.state_dict(), w_global, delta
                )
                logits, _ = functional_call(merged_model, merged_params, (x,))
                loss = compute_entropy(logits)
                loss.backward()
                optimizer.step()
                
            # Final forward pass
            with torch.no_grad():
                merged_params = get_merged_state_dict_differentiable(
                    expert_0.state_dict(), expert_1.state_dict(), w_global, delta
                )
                logits, _ = functional_call(merged_model, merged_params, (x,))
                
        elif regime == "sam_ttmm":
            # Sharpness-Aware Test-Time Model Merging with Static Preconditioning
            # Step 1: Compute original loss
            merged_params = get_merged_state_dict_differentiable(
                expert_0.state_dict(), expert_1.state_dict(), w_global, delta
            )
            logits, _ = functional_call(merged_model, merged_params, (x,))
            loss_orig = compute_entropy(logits)
            
            # Compute gradient of original loss
            loss_orig.backward()
            g = delta.grad.clone()
            
            # Step 2: Sharpness-aware perturbation
            epsilon = rho * g / (torch.norm(g) + 1e-12)
            delta.grad.zero_()
            
            # Step 3: Compute perturbed loss
            merged_params_pert = get_merged_state_dict_differentiable(
                expert_0.state_dict(), expert_1.state_dict(), w_global, delta + epsilon
            )
            logits_pert, _ = functional_call(merged_model, merged_params_pert, (x,))
            loss_pert = compute_entropy(logits_pert)
            
            # Compute gradient of perturbed loss
            loss_pert.backward()
            g_pert = delta.grad.clone()
            
            # Preconditioning using trace (normalized squared gradient)
            F_j = g_pert ** 2
            F_tilde = F_j / (F_j.sum() + 1e-8)
            
            # Update delta
            eta_j = lr_base / (F_tilde + damping_base)
            with torch.no_grad():
                delta -= eta_j * g_pert
            delta.grad.zero_()
                
            # Final forward pass
            with torch.no_grad():
                merged_params_final = get_merged_state_dict_differentiable(
                    expert_0.state_dict(), expert_1.state_dict(), w_global, delta
                )
                logits, _ = functional_call(merged_model, merged_params_final, (x,))
                
        elif regime == "cg_mttmm":
            # Our proposed Curvature-Guided Meta-Learning for TTMM
            # Step 1: Compute original loss
            merged_params = get_merged_state_dict_differentiable(
                expert_0.state_dict(), expert_1.state_dict(), w_global, delta
            )
            logits, _ = functional_call(merged_model, merged_params, (x,))
            loss_orig = compute_entropy(logits)
            
            # Compute gradient of original loss
            loss_orig.backward()
            g = delta.grad.clone()
            
            # Step 2: Sharpness-aware perturbation
            epsilon = rho * g / (torch.norm(g) + 1e-12)
            delta.grad.zero_()
            
            # Step 3: Compute perturbed loss
            merged_params_pert = get_merged_state_dict_differentiable(
                expert_0.state_dict(), expert_1.state_dict(), w_global, delta + epsilon
            )
            logits_pert, _ = functional_call(merged_model, merged_params_pert, (x,))
            loss_pert = compute_entropy(logits_pert)
            
            # Loss gap (curvature)
            loss_gap = max(0.0, (loss_pert - loss_orig).item())
            curvature_values.append(loss_gap)
            
            # CURVATURE-GUIDED ADAPTIVE LR AND DAMPING!
            lr_adaptive = lr_base * np.exp(-alpha * loss_gap)
            damping_adaptive = damping_base * (1.0 + beta * loss_gap)
            
            adaptive_lrs.append(lr_adaptive)
            adaptive_damps.append(damping_adaptive)
            
            # Compute gradient of perturbed loss
            loss_pert.backward()
            g_pert = delta.grad.clone()
            
            # Preconditioning using trace
            F_j = g_pert ** 2
            F_tilde = F_j / (F_j.sum() + 1e-8)
            
            # Update delta using adaptive parameters!
            eta_j = lr_adaptive / (F_tilde + damping_adaptive)
            with torch.no_grad():
                delta -= eta_j * g_pert
            delta.grad.zero_()
                
            # Final forward pass
            with torch.no_grad():
                merged_params_final = get_merged_state_dict_differentiable(
                    expert_0.state_dict(), expert_1.state_dict(), w_global, delta
                )
                logits, _ = functional_call(merged_model, merged_params_final, (x,))
        
        preds = logits.argmax(dim=1).cpu().numpy()
        targets = y.cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets)
        
    # Calculate phase-by-phase accuracy
    # Each phase is 10 batches of size 64
    phase_accs = []
    for p in range(5):
        p_preds = all_preds[p*640 : (p+1)*640]
        p_targets = all_targets[p*640 : (p+1)*640]
        acc = np.mean(np.array(p_preds) == np.array(p_targets))
        phase_accs.append(acc)
        
    overall_acc = np.mean(np.array(all_preds) == np.array(all_targets))
    
    return phase_accs, overall_acc, curvature_values, adaptive_lrs, adaptive_damps

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device("cpu")
    print("Loading experts...")
    expert_0 = SimpleCNN().to(device)
    expert_1 = SimpleCNN().to(device)
    expert_0.load_state_dict(torch.load("models/expert_0.pt", map_location=device, weights_only=True))
    expert_1.load_state_dict(torch.load("models/expert_1.pt", map_location=device, weights_only=True))
    
    # 2. Setup transforms and load streaming test sets
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    mnist_test = datasets.MNIST(root="data", train=False, download=True, transform=transform)
    fmnist_test = datasets.FashionMNIST(root="data", train=False, download=True, transform=transform)
    kmnist_test = datasets.KMNIST(root="data", train=False, download=True, transform=transform)
    
    # We need 10 batches (size 64) per phase.
    # Total samples per phase: 640
    # Let's create the streaming batches
    mnist_clean_loader = DataLoader(Subset(mnist_test, list(range(640))), batch_size=64, shuffle=False)
    fmnist_clean_loader = DataLoader(Subset(fmnist_test, list(range(640))), batch_size=64, shuffle=False)
    kmnist_loader = DataLoader(Subset(kmnist_test, list(range(640))), batch_size=64, shuffle=False)
    
    # Let's construct the noisy stream loader by applying Gaussian noise
    stream_batches = []
    
    # Phase 1: Clean MNIST (batches 0-9)
    for x, y in mnist_clean_loader:
        stream_batches.append((x, y))
        
    # Phase 2: Noisy MNIST (batches 10-19)
    for x, y in mnist_clean_loader:
        # Add Gaussian noise with std=0.6, clamped to valid range
        noisy_x = x + 0.6 * torch.randn_like(x)
        # Re-normalize to roughly preserve range
        noisy_x = torch.clamp(noisy_x, -1.0, 1.0)
        stream_batches.append((noisy_x, y))
        
    # Phase 3: Clean FashionMNIST (batches 20-29)
    for x, y in fmnist_clean_loader:
        stream_batches.append((x, y))
        
    # Phase 4: Noisy FashionMNIST (batches 30-39)
    for x, y in fmnist_clean_loader:
        noisy_x = x + 0.6 * torch.randn_like(x)
        noisy_x = torch.clamp(noisy_x, -1.0, 1.0)
        stream_batches.append((noisy_x, y))
        
    # Phase 5: Novel KMNIST (batches 40-49)
    for x, y in kmnist_loader:
        stream_batches.append((x, y))
        
    print(f"Stream loaded. Total batches: {len(stream_batches)} (each of size 64)")
    
    # Evaluate Regimes
    regimes = ["static", "tta", "sam_ttmm", "cg_mttmm", "sam_ttmm_tuned", "cg_mttmm_tuned"]
    results = {}
    
    for regime in regimes:
        print(f"\n--- Evaluating Regime: {regime.upper()} ---")
        if regime == "static":
            phase_accs, overall_acc, curv, lrs, damps = evaluate_stream(
                expert_0, expert_1, stream_batches, regime=regime
            )
        elif regime == "tta":
            phase_accs, overall_acc, curv, lrs, damps = evaluate_stream(
                expert_0, expert_1, stream_batches, regime=regime, lr_base=20.0
            )
        elif regime == "sam_ttmm":
            phase_accs, overall_acc, curv, lrs, damps = evaluate_stream(
                expert_0, expert_1, stream_batches, regime=regime, lr_base=150.0
            )
        elif regime == "cg_mttmm":
            phase_accs, overall_acc, curv, lrs, damps = evaluate_stream(
                expert_0, expert_1, stream_batches, regime=regime, lr_base=150.0, alpha=0.0, beta=500.0
            )
        elif regime == "sam_ttmm_tuned":
            phase_accs, overall_acc, curv, lrs, damps = evaluate_stream(
                expert_0, expert_1, stream_batches, regime="sam_ttmm", lr_base=150.0, rho=0.03, damping_base=0.05
            )
        elif regime == "cg_mttmm_tuned":
            phase_accs, overall_acc, curv, lrs, damps = evaluate_stream(
                expert_0, expert_1, stream_batches, regime="cg_mttmm", lr_base=150.0, alpha=10.0, beta=50.0, rho=0.03, damping_base=0.05
            )
        results[regime] = {
            "phase_accs": phase_accs,
            "overall_acc": overall_acc,
            "curv": curv,
            "lrs": lrs,
            "damps": damps
        }
        print(f"  Phase 1 (Clean MNIST):     {phase_accs[0]*100:.2f}%")
        print(f"  Phase 2 (Noisy MNIST):     {phase_accs[1]*100:.2f}%")
        print(f"  Phase 3 (Clean Fashion):   {phase_accs[2]*100:.2f}%")
        print(f"  Phase 4 (Noisy Fashion):   {phase_accs[3]*100:.2f}%")
        print(f"  Phase 5 (Novel KMNIST):    {phase_accs[4]*100:.2f}%")
        print(f"  OVERALL ACCURACY:          {overall_acc*100:.2f}%")
        
    # Print comparison table
    print("\n" + "="*95)
    print(f"{'Method':<18} | {'Clean MNIST':<11} | {'Noisy MNIST':<11} | {'Clean Fash':<10} | {'Noisy Fash':<10} | {'Novel KMN':<9} | {'Overall':<7}")
    print("-"*95)
    for r in regimes:
        pa = results[r]["phase_accs"]
        oa = results[r]["overall_acc"]
        print(f"{r:<18} | {pa[0]*100:>10.2f}% | {pa[1]*100:>10.2f}% | {pa[2]*100:>9.2f}% | {pa[3]*100:>9.2f}% | {pa[4]*100:>8.2f}% | {oa*100:>6.2f}%")
    print("="*95)
    
    # Save results to numpy file for plotting later
    np.save("ttmm_results.npy", results)
    print("Results saved to ttmm_results.npy")

if __name__ == "__main__":
    main()
