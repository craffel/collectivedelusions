import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import copy
from model import SimpleCNN

# Disable cuDNN to avoid CUDNN_STATUS_NOT_INITIALIZED on cluster nodes
torch.backends.cudnn.enabled = False

# Set random seeds for exact scientific reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
ADAPT_STEPS = 3
ADAPT_LR = 0.05

class MergedCNN(nn.Module):
    def __init__(self, expert_state_dicts):
        super().__init__()
        self.expert_state_dicts = [copy.deepcopy(sd) for sd in expert_state_dicts]
        # Initialize unconstrained scores to zeros (uniform softmax [1/3, 1/3, 1/3])
        self.scores = nn.ParameterDict({
            name.replace('.', '_'): nn.Parameter(torch.zeros(3))
            for name in expert_state_dicts[0].keys()
        })
        
    def get_merged_param(self, name, temp=1.0):
        score_name = name.replace('.', '_')
        coef = F.softmax(self.scores[score_name] / temp, dim=0)
        merged = torch.zeros_like(self.expert_state_dicts[0][name], device=DEVICE)
        for k, sd in enumerate(self.expert_state_dicts):
            merged += coef[k] * sd[name].to(DEVICE)
        return merged
        
    def forward(self, x, temp=1.0):
        w_conv1 = self.get_merged_param('conv1.weight', temp)
        b_conv1 = self.get_merged_param('conv1.bias', temp)
        w_conv2 = self.get_merged_param('conv2.weight', temp)
        b_conv2 = self.get_merged_param('conv2.bias', temp)
        w_conv3 = self.get_merged_param('conv3.weight', temp)
        b_conv3 = self.get_merged_param('conv3.bias', temp)
        w_fc1 = self.get_merged_param('fc1.weight', temp)
        b_fc1 = self.get_merged_param('fc1.bias', temp)
        w_fc2 = self.get_merged_param('fc2.weight', temp)
        b_fc2 = self.get_merged_param('fc2.bias', temp)
        
        # Differentiable forward pass
        x = F.relu(F.conv2d(x, w_conv1, b_conv1, padding=1))
        x = F.max_pool2d(F.relu(F.conv2d(x, w_conv2, b_conv2, padding=1)), 2, 2)
        x = F.max_pool2d(F.relu(F.conv2d(x, w_conv3, b_conv3, padding=1)), 2, 2)
        x = x.view(x.size(0), -1)
        feat = F.relu(F.linear(x, w_fc1, b_fc1))
        out = F.linear(feat, w_fc2, b_fc2)
        return out, feat

    def estimate_online_fisher(self, x, temp=1.0):
        w_conv1 = self.get_merged_param('conv1.weight', temp)
        b_conv1 = self.get_merged_param('conv1.bias', temp)
        w_conv2 = self.get_merged_param('conv2.weight', temp)
        b_conv2 = self.get_merged_param('conv2.bias', temp)
        w_conv3 = self.get_merged_param('conv3.weight', temp)
        b_conv3 = self.get_merged_param('conv3.bias', temp)
        w_fc1 = self.get_merged_param('fc1.weight', temp)
        b_fc1 = self.get_merged_param('fc1.bias', temp)
        w_fc2 = self.get_merged_param('fc2.weight', temp)
        b_fc2 = self.get_merged_param('fc2.bias', temp)
        
        # Forward pass to pool2
        x = F.relu(F.conv2d(x, w_conv1, b_conv1, padding=1))
        x = F.max_pool2d(F.relu(F.conv2d(x, w_conv2, b_conv2, padding=1)), 2, 2)
        x = F.max_pool2d(F.relu(F.conv2d(x, w_conv3, b_conv3, padding=1)), 2, 2)
        x_pool2 = x.view(x.size(0), -1)
        
        # Clone and require grad for fc1.bias
        b_fc1_leaf = b_fc1.clone().detach().requires_grad_(True)
        
        # Forward pass from fc1
        feat = F.relu(F.linear(x_pool2, w_fc1, b_fc1_leaf))
        out = F.linear(feat, w_fc2, b_fc2)
        
        # Pseudo-labels
        probs = F.softmax(out, dim=1)
        pseudo_y = probs.argmax(dim=1)
        
        # Calculate loss (sum log p(y_i|x_i))
        loss = F.cross_entropy(out, pseudo_y, reduction='sum')
        
        # Backward to get gradients w.r.t b_fc1_leaf
        grads = torch.autograd.grad(loss, b_fc1_leaf)[0]
        
        # The Fisher Information estimate per sample is:
        # (grads / x.size(0)) ** 2
        fisher_est = (grads / x.size(0)) ** 2
        return fisher_est

def apply_corruption(images, corruption_type, severity=2):
    if corruption_type == 'none':
        return images
    elif corruption_type == 'gaussian_noise':
        noise = torch.randn_like(images) * (0.15 * severity)
        return torch.clamp(images + noise, -1.0, 1.0)
    elif corruption_type == 'blur':
        kernel_size = 2 * severity + 1
        padding = severity
        pooled = F.avg_pool2d(images, kernel_size, stride=1, padding=padding)
        return pooled
    elif corruption_type == 'contrast':
        mean = images.mean(dim=(2, 3), keepdim=True)
        scaled = mean + (images - mean) * (1.0 / (severity + 1))
        return torch.clamp(scaled, -1.0, 1.0)
    else:
        return images

def build_test_stream():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    fmnist_test = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    kmnist_test = torchvision.datasets.KMNIST(root='./data', train=False, download=True, transform=transform)
    
    # Let's take 20 batches of size 64 for each task
    num_batches = 20
    indices_mnist = torch.randperm(len(mnist_test))[:num_batches * BATCH_SIZE]
    indices_fmnist = torch.randperm(len(fmnist_test))[:num_batches * BATCH_SIZE]
    indices_kmnist = torch.randperm(len(kmnist_test))[:num_batches * BATCH_SIZE]
    
    stream = []
    
    # MNIST with severe Gaussian Noise
    for i in range(num_batches):
        subset = Subset(mnist_test, indices_mnist[i*BATCH_SIZE : (i+1)*BATCH_SIZE])
        loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False)
        x, y = next(iter(loader))
        x_corr = apply_corruption(x, 'gaussian_noise', severity=3)
        stream.append((x_corr, y, 'mnist'))
        
    # FashionMNIST with severe Blur
    for i in range(num_batches):
        subset = Subset(fmnist_test, indices_fmnist[i*BATCH_SIZE : (i+1)*BATCH_SIZE])
        loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False)
        x, y = next(iter(loader))
        x_corr = apply_corruption(x, 'blur', severity=2)
        stream.append((x_corr, y, 'fmnist'))
        
    # KMNIST with severe Contrast reduction
    for i in range(num_batches):
        subset = Subset(kmnist_test, indices_kmnist[i*BATCH_SIZE : (i+1)*BATCH_SIZE])
        loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False)
        x, y = next(iter(loader))
        x_corr = apply_corruption(x, 'contrast', severity=3)
        stream.append((x_corr, y, 'kmnist'))
        
    return stream

def run_evaluation(method_name, stream, expert_sds, prototypes, fisher_infos, beta=1.0, tau=0.5):
    print(f"\nRunning online evaluation for method: {method_name} (tau={tau})...")
    
    # Initialize merged model
    merged_model = MergedCNN(expert_sds).to(DEVICE)
    optimizer = optim.Adam(merged_model.parameters(), lr=ADAPT_LR)
    
    # Compute joint Fisher weight vector for feature dimensions
    fisher_fc1_bias = torch.zeros(128, device=DEVICE)
    for exp_name in ['mnist', 'fmnist', 'kmnist']:
        fisher_fc1_bias += fisher_infos[exp_name]['fc1.bias'].to(DEVICE) / 3.0
    
    # Normalize Fisher
    fisher_fc1_bias = fisher_fc1_bias / (fisher_fc1_bias.mean() + 1e-8)
    
    # Compute weights
    epsilon = 1e-5
    if method_name.startswith("FWPA"):
        if "exp" in method_name:
            W_feat_static = torch.exp(-beta * fisher_fc1_bias)
        else:
            W_feat_static = 1.0 / (fisher_fc1_bias + epsilon) ** beta
        W_feat_static = W_feat_static / W_feat_static.mean()
    else:
        W_feat_static = torch.ones(128, device=DEVICE)
        
    batch_accuracies = []
    domain_accuracies = {'mnist': [], 'fmnist': [], 'kmnist': []}
    
    prev_entropy = None
    fisher_online = torch.zeros(128, device=DEVICE)
    
    for step, (x, y, domain) in enumerate(stream):
        x, y = x.to(DEVICE), y.to(DEVICE)
        
        # Calculate initial prediction entropy to detect spikes
        with torch.no_grad():
            init_out, _ = merged_model(x, temp=1.0)
            init_probs = F.softmax(init_out, dim=1)
            init_entropy = -torch.sum(init_probs * torch.log(init_probs + 1e-8), dim=1).mean().item()
            
        if prev_entropy is None:
            entropy_spike = 0.0
        else:
            entropy_spike = max(0.0, init_entropy - prev_entropy)
            
        prev_entropy = init_entropy
        
        # Determine temperature for this batch's adaptation
        if "MASA" in method_name:
            # Scale temperature based on the spike: T_t = 1.0 + alpha * spike
            # Let's say alpha = 2.0
            temp_t = 1.0 + 2.0 * entropy_spike
        else:
            temp_t = 1.0
            
        # Estimate online Fisher
        if method_name.startswith("O-FWPA"):
            fisher_batch = merged_model.estimate_online_fisher(x, temp=temp_t)
            if step == 0:
                fisher_online = fisher_batch.detach()
            else:
                fisher_online = 0.8 * fisher_online + 0.2 * fisher_batch.detach()
            
        # Test-Time Adaptation steps on the current batch
        if method_name != "Static":
            for _ in range(ADAPT_STEPS):
                optimizer.zero_grad()
                out, feat = merged_model(x, temp=temp_t)
                
                if method_name == "AdaMerging":
                    # Entropy minimization loss
                    probs = F.softmax(out, dim=1)
                    loss = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
                elif method_name in ["ProtoAlign"] or method_name.startswith("FWPA") or method_name.startswith("A-FWPA") or method_name.startswith("O-FWPA"):
                    # Prototype-driven self-supervised loss
                    # Extract current average coefficient across all layers to blend prototypes
                    scores_list = [F.softmax(merged_model.scores[k] / temp_t, dim=0) for k in merged_model.scores]
                    lambda_avg = torch.stack(scores_list).mean(dim=0)
                    
                    # Compute dynamic merged prototypes
                    pi_merged = torch.zeros(10, 128, device=DEVICE)
                    for k, exp_name in enumerate(['mnist', 'fmnist', 'kmnist']):
                        pi_merged += lambda_avg[k] * prototypes[exp_name].to(DEVICE)
                        
                    # Center representations and prototypes
                    mu_merged = pi_merged.mean(dim=0, keepdim=True)
                    z = feat - mu_merged
                    pi_centered = pi_merged - mu_merged
                    
                    # Normalize centered prototypes
                    pi_centered_norm = pi_centered / (pi_centered.norm(p=2, dim=1, keepdim=True) + 1e-8)
                    
                    # Pseudo-labels and confidence masking
                    probs = F.softmax(out, dim=1)
                    conf, pseudo_y = probs.max(dim=1)
                    mask = conf >= tau
                    
                    if mask.sum() > 0:
                        z_conf = z[mask]
                        y_conf = pseudo_y[mask]
                        proto_conf = pi_centered_norm[y_conf]
                        
                        if method_name == "ProtoAlign":
                            # Unweighted Cosine Similarity
                            z_norm = z_conf / (z_conf.norm(p=2, dim=1, keepdim=True) + 1e-8)
                            sims = torch.sum(z_norm * proto_conf, dim=1)
                            loss = -sims.mean()
                        else:
                            # FWPA, A-FWPA, or O-FWPA
                            if method_name.startswith("A-FWPA"):
                                # Adaptive dynamic Fisher weight vector based on current lambda_avg
                                fisher_dyn = torch.zeros(128, device=DEVICE)
                                for k, exp_name in enumerate(['mnist', 'fmnist', 'kmnist']):
                                    fisher_dyn += lambda_avg[k] * fisher_infos[exp_name]['fc1.bias'].to(DEVICE)
                                fisher_dyn_norm = fisher_dyn / (fisher_dyn.mean() + 1e-8)
                                
                                if "exp" in method_name:
                                    W_feat = torch.exp(-beta * fisher_dyn_norm)
                                else:
                                    W_feat = 1.0 / (fisher_dyn_norm + epsilon) ** beta
                                W_feat = W_feat / W_feat.mean()
                            elif method_name.startswith("O-FWPA"):
                                # Online Fisher weight vector based on running online Fisher estimate
                                fisher_online_norm = fisher_online / (fisher_online.mean() + 1e-8)
                                W_feat = 1.0 / (fisher_online_norm + epsilon) ** beta
                                W_feat = W_feat / W_feat.mean()
                            else:
                                # Static Fisher weight vector
                                W_feat = W_feat_static
                                
                            # Fisher-weighted Cosine Similarity
                            dot = torch.sum(W_feat * z_conf * proto_conf, dim=1)
                            norm_z = torch.sqrt(torch.sum(W_feat * (z_conf ** 2), dim=1) + 1e-8)
                            norm_proto = torch.sqrt(torch.sum(W_feat * (proto_conf ** 2), dim=1) + 1e-8)
                            sims = dot / (norm_z * norm_proto)
                            loss = -sims.mean()
                    else:
                        # Fallback to entropy minimization if no sample is confident
                        loss = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
                        
                loss.backward()
                optimizer.step()
                
        # Final evaluation pass on the adapted model for this batch (evaluated at temp=1.0)
        with torch.no_grad():
            out, _ = merged_model(x, temp=1.0)
            _, predicted = out.max(1)
            correct = predicted.eq(y).sum().item()
            acc = 100. * correct / y.size(0)
            
            batch_accuracies.append(acc)
            domain_accuracies[domain].append(acc)
            
    print(f"Finished evaluation. Avg Accuracy: {np.mean(batch_accuracies):.2f}%")
    return batch_accuracies, domain_accuracies

def main():
    print("Loading expert checkpoints and metadata...")
    # Load expert models
    expert_names = ['mnist', 'fmnist', 'kmnist']
    expert_sds = []
    for name in expert_names:
        sd = torch.load(f"checkpoints/expert_{name}.pt", map_location=DEVICE)
        expert_sds.append(sd)
        
    prototypes = torch.load("checkpoints/prototypes.pt", map_location=DEVICE)
    fisher_infos = torch.load("checkpoints/fisher_infos.pt", map_location=DEVICE)
    
    print("Building non-stationary, corrupted test-time stream...")
    stream = build_test_stream()
    
    # We will evaluate different methods
    results = {}
    domain_results = {}
    
    # 1. Oracle: evaluate single dedicated experts on their respective domains (as an upper bound reference)
    oracle_accuracies = []
    print("\nRunning Oracle specialized expert evaluation...")
    with torch.no_grad():
        for x, y, domain in stream:
            x, y = x.to(DEVICE), y.to(DEVICE)
            # Use the correct expert model
            exp_model = SimpleCNN().to(DEVICE)
            exp_model.load_state_dict(torch.load(f"checkpoints/expert_{domain}.pt", map_location=DEVICE))
            exp_model.eval()
            out, _ = exp_model(x)
            _, predicted = out.max(1)
            acc = 100. * predicted.eq(y).sum().item() / y.size(0)
            oracle_accuracies.append(acc)
    results['Oracle'] = oracle_accuracies
    print(f"Oracle Expert Avg Accuracy: {np.mean(oracle_accuracies):.2f}%")
    
    # Run other adaptation methods
    methods = [
        ("Static", 0.0),
        ("AdaMerging", 0.0),
        ("ProtoAlign", 0.0),
        ("FWPA (beta=-0.5)", -0.5),
        ("FWPA (beta=0.5)", 0.5),
        ("FWPA (beta=1.0)", 1.0),
        ("FWPA (beta=2.0)", 2.0),
        ("FWPA_exp (beta=1.0)", 1.0),
        ("FWPA_exp (beta=2.0)", 2.0),
        ("A-FWPA (beta=1.0)", 1.0),
        ("A-FWPA (beta=1.5)", 1.5),
        ("A-FWPA (beta=2.0)", 2.0),
        ("A-FWPA (beta=2.5)", 2.5),
        ("A-FWPA (beta=3.0)", 3.0),
        ("A-FWPA_exp (beta=1.0)", 1.0),
        ("A-FWPA_exp (beta=2.0)", 2.0),
        ("A-FWPA_exp (beta=5.0)", 5.0),
        ("A-FWPA + MASA (beta=1.5)", 1.5),
        ("A-FWPA + MASA (beta=2.0)", 2.0),
        ("O-FWPA (beta=1.5)", 1.5),
        ("O-FWPA (beta=2.0)", 2.0),
        ("O-FWPA (beta=2.5)", 2.5)
    ]
    
    for method_name, b_val in methods:
        batch_accs, dom_accs = run_evaluation(method_name, stream, expert_sds, prototypes, fisher_infos, beta=b_val)
        results[method_name] = batch_accs
        domain_results[method_name] = dom_accs
        
    # Summarize and print results in a beautiful Markdown table
    print("\n" + "="*90)
    print("                      SUMMARY EXPERIMENTAL RESULTS TABLE")
    print("="*90)
    print(f"| {'Method':<25} | {'MNIST Acc':<11} | {'F-MNIST Acc':<11} | {'K-MNIST Acc':<11} | {'Average Acc':<11} |")
    print("|" + "-"*27 + "|" + "-"*13 + "|" + "-"*13 + "|" + "-"*13 + "|" + "-"*13 + "|")
    
    # Add Oracle to domain results
    domain_results['Oracle'] = {'mnist': oracle_accuracies[:20], 'fmnist': oracle_accuracies[20:40], 'kmnist': oracle_accuracies[40:]}
    
    all_methods_to_print = ['Oracle', 'Static', 'AdaMerging', 'ProtoAlign'] + [m[0] for m in methods if m[0] not in ['Static', 'AdaMerging', 'ProtoAlign']]
    
    for m in all_methods_to_print:
        mnist_acc = np.mean(domain_results[m]['mnist'])
        fmnist_acc = np.mean(domain_results[m]['fmnist'])
        kmnist_acc = np.mean(domain_results[m]['kmnist'])
        avg_acc = np.mean(results[m])
        print(f"| {m:<25} | {mnist_acc:10.2f}% | {fmnist_acc:10.2f}% | {kmnist_acc:10.2f}% | {avg_acc:10.2f}% |")
    print("="*90 + "\n")
    
    # Run confidence threshold sweep for A-FWPA (beta=2.0)
    print("\n" + "="*90)
    print("                CONFIDENCE THRESHOLD (TAU) SWEEP ON A-FWPA (beta=2.0)")
    print("="*90)
    print(f"| {'Tau (Threshold)':<15} | {'MNIST Acc':<11} | {'F-MNIST Acc':<11} | {'K-MNIST Acc':<11} | {'Average Acc':<11} |")
    print("|" + "-"*17 + "|" + "-"*13 + "|" + "-"*13 + "|" + "-"*13 + "|" + "-"*13 + "|")
    
    tau_sweep_results = {}
    for tau_val in [0.3, 0.5, 0.7, 0.9]:
        sweep_name = f"A-FWPA (beta=2.0, tau={tau_val})"
        batch_accs, dom_accs = run_evaluation("A-FWPA (beta=2.0)", stream, expert_sds, prototypes, fisher_infos, beta=2.0, tau=tau_val)
        tau_sweep_results[tau_val] = {
            'mnist': np.mean(dom_accs['mnist']),
            'fmnist': np.mean(dom_accs['fmnist']),
            'kmnist': np.mean(dom_accs['kmnist']),
            'avg': np.mean(batch_accs)
        }
    
    print("\n" + "="*90)
    print("                      TAU SWEEP RESULTS TABLE")
    print("="*90)
    print(f"| {'Tau (Threshold)':<15} | {'MNIST Acc':<11} | {'F-MNIST Acc':<11} | {'K-MNIST Acc':<11} | {'Average Acc':<11} |")
    print("|" + "-"*17 + "|" + "-"*13 + "|" + "-"*13 + "|" + "-"*13 + "|" + "-"*13 + "|")
    for tau_val in [0.3, 0.5, 0.7, 0.9]:
        res = tau_sweep_results[tau_val]
        print(f"| {tau_val:<15.1f} | {res['mnist']:10.2f}% | {res['fmnist']:10.2f}% | {res['kmnist']:10.2f}% | {res['avg']:10.2f}% |")
    print("="*90 + "\n")
    
    # Save results dictionary for paper writing
    torch.save({
        'results': results,
        'domain_results': domain_results,
        'tau_sweep_results': tau_sweep_results
    }, "checkpoints/results_data.pt")
    
    # Plot results
    plt.figure(figsize=(12, 6))
    epochs_range = range(1, len(stream) + 1)
    
    plt.plot(epochs_range, results['Static'], label='Static (No Adaptation)', color='gray', linestyle='--')
    plt.plot(epochs_range, results['AdaMerging'], label='AdaMerging (Entropy)', color='red', alpha=0.7)
    plt.plot(epochs_range, results['ProtoAlign'], label='ProtoAlign (Unweighted)', color='blue', alpha=0.7)
    plt.plot(epochs_range, results['FWPA (beta=2.0)'], label='FWPA (Proposed, beta=2.0)', color='darkgreen', linewidth=1.5, alpha=0.8)
    if 'A-FWPA (beta=2.0)' in results:
        plt.plot(epochs_range, results['A-FWPA (beta=2.0)'], label='A-FWPA (Proposed, beta=2.0)', color='darkorange', linewidth=2.0, alpha=0.8)
    if 'A-FWPA + MASA (beta=2.0)' in results:
        plt.plot(epochs_range, results['A-FWPA + MASA (beta=2.0)'], label='A-FWPA + MASA (Proposed, beta=2.0)', color='purple', linewidth=2.5)
    if 'O-FWPA (beta=2.0)' in results:
        plt.plot(epochs_range, results['O-FWPA (beta=2.0)'], label='O-FWPA (Proposed Online, beta=2.0)', color='crimson', linewidth=2.0, linestyle='-.', alpha=0.9)
    if 'A-FWPA_exp (beta=2.0)' in results:
        plt.plot(epochs_range, results['A-FWPA_exp (beta=2.0)'], label='A-FWPA_exp (Proposed, beta=2.0)', color='cyan', linewidth=1.5, alpha=0.6)
    if 'A-FWPA_exp (beta=5.0)' in results:
        plt.plot(epochs_range, results['A-FWPA_exp (beta=5.0)'], label='A-FWPA_exp (Proposed, beta=5.0)', color='magenta', linewidth=1.5, alpha=0.6)
    plt.plot(epochs_range, results['Oracle'], label='Oracle specialized expert', color='black', linestyle=':', alpha=0.5)
    
    # Draw vertical lines for task shifts
    plt.axvline(x=20.5, color='purple', linestyle=':', alpha=0.5)
    plt.text(10, 95, "MNIST\n(Noise)", color='purple', ha='center')
    plt.axvline(x=40.5, color='purple', linestyle=':', alpha=0.5)
    plt.text(30, 95, "FashionMNIST\n(Blur)", color='purple', ha='center')
    plt.text(50, 95, "KMNIST\n(Contrast)", color='purple', ha='center')
    
    plt.title("Online Accuracy on Non-Stationary Corrupted Vision Stream")
    plt.xlabel("Streaming Step (Batch)")
    plt.ylabel("Accuracy (%)")
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 105)
    plt.tight_layout()
    plt.savefig("results_plot.png", dpi=150)
    print("Saved results visualization plot to 'results_plot.png'.")

if __name__ == "__main__":
    main()
