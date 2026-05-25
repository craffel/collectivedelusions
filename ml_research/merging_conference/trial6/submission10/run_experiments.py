import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Simplex projection
def project_simplex(v, z=1.0):
    device = v.device
    n_features = v.shape[0]
    u, _ = torch.sort(v, descending=True)
    css = torch.cumsum(u, dim=0) - z
    ind = torch.arange(1, n_features + 1, device=device)
    cond = u - css / ind > 0
    rho = ind[cond][-1]
    theta = css[rho - 1] / rho
    return torch.clamp(v - theta, min=0)

# Base model definition (must match train_experts.py)
def get_base_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model

# Define corruptions on raw [0, 1] images
class CorruptTransform:
    def __init__(self, corruption_type):
        self.corruption_type = corruption_type
        self.normalize = transforms.Normalize((0.5,), (0.5,))
        
    def __call__(self, img):
        # img is a PIL image
        x = transforms.ToTensor()(img) # [0, 1]
        
        if self.corruption_type == "clean":
            pass
        elif self.corruption_type == "gaussian_noise":
            noise = torch.randn_like(x) * 0.2
            x = torch.clamp(x + noise, 0.0, 1.0)
        elif self.corruption_type == "contrast_shift":
            mean = x.mean()
            x = torch.clamp((x - mean) * 0.3 + mean, 0.0, 1.0)
            
        return self.normalize(x)

# Load test datasets and return standard subsets
def get_test_datasets(corruption_type):
    transform = CorruptTransform(corruption_type)
    
    # We load datasets from ./data
    mnist_test = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    fashion_test = torchvision.datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
    kmnist_test = torchvision.datasets.KMNIST(root="./data", train=False, download=True, transform=transform)
    
    # Take first 1600 samples of each test set
    mnist_sub = Subset(mnist_test, list(range(1600)))
    fashion_sub = Subset(fashion_test, list(range(1600)))
    kmnist_sub = Subset(kmnist_test, list(range(1600)))
    
    return mnist_sub, fashion_sub, kmnist_sub

# Create clean training datasets for S-Fisher calibration
def get_clean_train_datasets():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    mnist_train = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    fashion_train = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
    kmnist_train = torchvision.datasets.KMNIST(root="./data", train=True, download=True, transform=transform)
    
    return mnist_train, fashion_train, kmnist_train

# Compute parameter-level diagonal Fisher with batch size 1 for precision
def compute_expert_fisher(model, dataset, device, num_samples=500, use_labels=True):
    model.eval()
    fisher = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher[name] = torch.zeros_like(param)
            
    # Use batch size 1 for true empirical Fisher
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    count = 0
    criterion = nn.CrossEntropyLoss()
    
    for x, y in loader:
        x = x.to(device)
        outputs = model(x)
        
        if use_labels:
            y = y.to(device)
            loss = criterion(outputs, y)
        else:
            with torch.no_grad():
                pseudo_y = torch.argmax(outputs, dim=-1)
            loss = criterion(outputs, pseudo_y)
            
        model.zero_grad()
        loss.backward()
        
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher[name] += param.grad.data ** 2
                
        count += 1
        if count >= num_samples:
            break
            
    # Average and convert to tensor-level sensitivity prior
    layer_fisher = {}
    for name in fisher:
        # Average over samples
        avg_f = fisher[name] / count
        # Average over all parameters within this tensor
        layer_fisher[name] = avg_f.mean().item()
        
    return layer_fisher

# Reconstruct merged model weights and buffers in-place
def update_merged_weights(merged_model, experts, lambdas, param_names, buffer_names):
    # Merge parameters (requires_grad)
    for name in param_names:
        expert_params = [dict(expert.named_parameters())[name] for expert in experts]
        coeff = lambdas[name]
        merged_weight = sum(coeff[k] * expert_params[k] for k in range(len(experts)))
        
        parts = name.split('.')
        submodule = merged_model
        for part in parts[:-1]:
            submodule = getattr(submodule, part)
        
        attr_name = parts[-1]
        if hasattr(submodule, attr_name):
            delattr(submodule, attr_name)
        setattr(submodule, attr_name, merged_weight)

    # Compute mean_weights once to avoid massive redundant stack operations
    mean_weights = torch.stack([lambdas[p] for p in param_names]).mean(dim=0).detach()

    # Merge buffers (running_mean, running_var, etc.)
    for name in buffer_names:
        expert_bufs = [dict(expert.named_buffers())[name] for expert in experts]
        buf = dict(experts[0].named_buffers())[name]
        
        if buf.dtype in [torch.float16, torch.float32, torch.float64]:
            merged_buf = sum(mean_weights[k] * expert_bufs[k] for k in range(len(experts)))
        else:
            merged_buf = expert_bufs[0]
            
        parts = name.split('.')
        submodule = merged_model
        for part in parts[:-1]:
            submodule = getattr(submodule, part)
        attr_name = parts[-1]
        submodule._buffers[attr_name] = merged_buf.to(expert_bufs[0].device)

def run_experiment_stream(stream_type, corruption_type, base_model, experts, s_fisher_prior, device, param_names, buffer_names):
    print(f"\n--- Running: Stream={stream_type}, Corruption={corruption_type} ---")
    
    # Get test datasets
    mnist_sub, fashion_sub, kmnist_sub = get_test_datasets(corruption_type)
    
    # We construct batches of size 32
    batch_size = 32
    num_batches_per_task = 1600 // batch_size # 50 batches per task
    
    # Create dataloaders
    mnist_loader = DataLoader(mnist_sub, batch_size=batch_size, shuffle=False)
    fashion_loader = DataLoader(fashion_sub, batch_size=batch_size, shuffle=False)
    kmnist_loader = DataLoader(kmnist_sub, batch_size=batch_size, shuffle=False)
    
    mnist_batches = list(mnist_loader)
    fashion_batches = list(fashion_loader)
    kmnist_batches = list(kmnist_loader)
    
    # Construct the stream of batches
    # Each element: (images, labels, task_id)
    stream = []
    if stream_type == "alternating":
        for b in range(num_batches_per_task):
            stream.append((mnist_batches[b][0], mnist_batches[b][1], 0))
            stream.append((fashion_batches[b][0], fashion_batches[b][1], 1))
            stream.append((kmnist_batches[b][0], kmnist_batches[b][1], 2))
    elif stream_type == "sequential":
        for b in range(num_batches_per_task):
            stream.append((mnist_batches[b][0], mnist_batches[b][1], 0))
        for b in range(num_batches_per_task):
            stream.append((fashion_batches[b][0], fashion_batches[b][1], 1))
        for b in range(num_batches_per_task):
            stream.append((kmnist_batches[b][0], kmnist_batches[b][1], 2))
            
    # For fair evaluation of TT-Fisher, we use the first 15 batches (approx 480 samples, i.e., 5 batches per task)
    # as the unlabeled calibration set to estimate TT-Fisher sensitivities on-the-fly!
    # The remaining batches (150 - 15 = 135 batches, total 4320 samples) are used for evaluation and adaptation!
    calibration_batches = stream[:15]
    eval_batches = stream[15:]
    
    # Let's compute TT-Fisher sensitivity prior on calibration_batches
    print("Computing Test-Time Fisher Information on calibration stream...")
    tt_fisher = {}
    for name in param_names:
        tt_fisher[name] = [torch.zeros_like(dict(experts[0].named_parameters())[name]) for _ in range(len(experts))]
            
    # For each calibration batch, we feed it to each expert and compute gradients on pseudo-labels
    criterion = nn.CrossEntropyLoss()
    for img, _, _ in calibration_batches:
        img = img.to(device)
        for k, expert in enumerate(experts):
            expert.eval()
            outputs = expert(img)
            with torch.no_grad():
                pseudo_y = torch.argmax(outputs, dim=-1)
            loss = criterion(outputs, pseudo_y)
            expert.zero_grad()
            loss.backward()
            
            for name in param_names:
                param = dict(expert.named_parameters())[name]
                if param.grad is not None:
                    # Accumulate squared gradients
                    tt_fisher[name][k] += (param.grad.data ** 2) * img.size(0)
                    
    # Average TT-Fisher sensitivities
    tt_fisher_prior = {}
    num_cal_samples = len(calibration_batches) * batch_size
    for name in param_names:
        joint_tensor_f = 0.0
        for k in range(len(experts)):
            avg_f = tt_fisher[name][k] / num_cal_samples
            joint_tensor_f += avg_f.mean().item()
        tt_fisher_prior[name] = joint_tensor_f / len(experts)
        
    # Normalize the priors to have a mean of 1.0 across all layers.
    s_prior_clean = {}
    mean_s = np.mean(list(s_fisher_prior.values()))
    for name in param_names:
        s_prior_clean[name] = s_fisher_prior[name] / (mean_s + 1e-12)

    tt_prior_clean = {}
    mean_tt = np.mean(list(tt_fisher_prior.values()))
    for name in param_names:
        tt_prior_clean[name] = tt_fisher_prior[name] / (mean_tt + 1e-12)

    # Evaluate methods
    methods = ["static", "no_fisher", "s_fisher", "tt_fisher", "dr_fisher"]
    results = {m: [] for m in methods}
    
    for m in methods:
        # Initialize coefficients as uniform simplex
        lambdas = {}
        for name in param_names:
            lambdas[name] = torch.tensor([1/3, 1/3, 1/3], device=device, requires_grad=True)
                
        # Define learning rates for this method
        lr_global = 1e-3
        eps = 1.0
        alpha = 1.0
        
        lrs = {}
        for name in param_names:
            if m == "static":
                lrs[name] = 0.0
            elif m == "no_fisher":
                # Uniform sensitivity
                lrs[name] = lr_global
            elif m == "s_fisher":
                # Pre-computed Source Fisher (normalized)
                mult = (s_prior_clean[name] + eps) ** (-alpha)
                lrs[name] = lr_global * np.clip(mult, 0.01, 10.0)
            elif m == "tt_fisher":
                # Test-time Fisher (normalized)
                mult = (tt_prior_clean[name] + eps) ** (-alpha)
                lrs[name] = lr_global * np.clip(mult, 0.01, 10.0)
            elif m == "dr_fisher":
                # TT-Fisher learning rates for DR-Fisher fine-tuning
                mult = (tt_prior_clean[name] + eps) ** (-alpha)
                lrs[name] = lr_global * np.clip(mult, 0.01, 10.0)
                
        # Evaluation and Adaptation Loop
        correct_count = 0
        total_count = 0
        acc_over_time = []
        
        # Create a clean model to represent the merged weights
        merged_model = get_base_model().to(device)
        merged_model.eval()
        
        for batch_idx, (img, label, task_id) in enumerate(eval_batches):
            img, label = img.to(device), label.to(device)
            
            if m == "dr_fisher":
                # Unsupervised Entropy-Based Expert Routing (EBER) on-the-fly
                entropies = []
                for k, expert in enumerate(experts):
                    expert.eval()
                    with torch.no_grad():
                        outputs = expert(img)
                        probs = torch.softmax(outputs, dim=-1)
                        entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1).mean().item()
                        entropies.append(entropy)
                
                # Identify expert with lowest prediction entropy
                routed_task = torch.argmin(torch.tensor(entropies)).item()
                
                # Dynamic routing initialization (0.99 on routed expert, 0.005 on others)
                init_w = [0.005, 0.005, 0.005]
                init_w[routed_task] = 0.99
                for name in param_names:
                    lambdas[name] = torch.tensor(init_w, device=device, requires_grad=True)
            
            # Reconstruct merged model weights and buffers using current coefficients
            update_merged_weights(merged_model, experts, lambdas, param_names, buffer_names)
            
            # Forward pass
            outputs = merged_model(img)
            
            # Predict & Record accuracy
            with torch.no_grad():
                _, preds = outputs.max(1)
                correct = preds.eq(label).sum().item()
                correct_count += correct
                total_count += img.size(0)
                acc_over_time.append(correct / img.size(0))
                
            # Perform Adaptation update (if not static)
            if m != "static":
                # Dynamic update of weight parameters as dependent tensors
                update_merged_weights(merged_model, experts, lambdas, param_names, buffer_names)
                
                # Dynamic forward pass
                outputs_diff = merged_model(img)
                
                # Compute entropy loss on pseudo-labels
                probs = torch.softmax(outputs_diff, dim=-1)
                entropy_loss = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1).mean()
                
                # Backward pass to get gradients with respect to lambdas
                for name in param_names:
                    lambdas[name].grad = None
                        
                entropy_loss.backward()
                
                # Update lambdas using their layer-wise learning rates and project to simplex
                with torch.no_grad():
                    for name in param_names:
                        if lambdas[name].grad is not None:
                            # Gradient descent step
                            lambdas[name] -= lrs[name] * lambdas[name].grad
                            # Project back to simplex
                            lambdas[name].copy_(project_simplex(lambdas[name]))
                            
        final_acc = 100.0 * correct_count / total_count
        print(f"Method {m:10s} - Final Accuracy: {final_acc:.2f}%")
        results[m] = {
            "final_acc": final_acc,
            "acc_over_time": acc_over_time
        }
        
    return results

def main():
    torch.backends.cudnn.enabled = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Check if expert checkpoints exist
    expert_mnist_path = "./models/expert_mnist.pt"
    expert_fashion_path = "./models/expert_fashion.pt"
    expert_kmnist_path = "./models/expert_kmnist.pt"
    base_model_path = "./models/base_model.pt"
    
    if not (os.path.exists(expert_mnist_path) and os.path.exists(expert_fashion_path) and os.path.exists(expert_kmnist_path)):
        print("Error: Expert models not found! Please ensure they are trained and saved in ./models")
        return
        
    # Load base model
    base_model = get_base_model().to(device)
    base_model.load_state_dict(torch.load(base_model_path, map_location=device, weights_only=True))
    
    # Load experts
    expert_mnist = get_base_model().to(device)
    expert_mnist.load_state_dict(torch.load(expert_mnist_path, map_location=device, weights_only=True))
    expert_fashion = get_base_model().to(device)
    expert_fashion.load_state_dict(torch.load(expert_fashion_path, map_location=device, weights_only=True))
    expert_kmnist = get_base_model().to(device)
    expert_kmnist.load_state_dict(torch.load(expert_kmnist_path, map_location=device, weights_only=True))
    
    experts = [expert_mnist, expert_fashion, expert_kmnist]
    
    # Precompute parameter and buffer names from experts[0]
    param_names = [name for name, param in experts[0].named_parameters() if param.requires_grad]
    buffer_names = [name for name, _ in experts[0].named_buffers()]

    # Compute Source Fisher prior (S-Fisher) using clean training datasets
    print("\n--- Pre-computing Source Fisher sensitivity prior (S-Fisher) ---")
    mnist_train, fashion_train, kmnist_train = get_clean_train_datasets()
    
    s_fisher_prior = {}
    for name in param_names:
        s_fisher_prior[name] = 0.0
            
    print("Computing Fisher for MNIST expert...")
    mnist_f = compute_expert_fisher(expert_mnist, mnist_train, device, num_samples=500)
    print("Computing Fisher for FashionMNIST expert...")
    fashion_f = compute_expert_fisher(expert_fashion, fashion_train, device, num_samples=500)
    print("Computing Fisher for KMNIST expert...")
    kmnist_f = compute_expert_fisher(expert_kmnist, kmnist_train, device, num_samples=500)
    
    # Compute joint sensitivity prior
    for name in param_names:
        s_fisher_prior[name] = (mnist_f[name] + fashion_f[name] + kmnist_f[name]) / 3.0

    # Run all stream and corruption combinations
    streams = ["alternating", "sequential"]
    corruptions = ["clean", "gaussian_noise", "contrast_shift"]
    
    all_results = {}
    for s in streams:
        all_results[s] = {}
        for c in corruptions:
            res = run_experiment_stream(s, c, base_model, experts, s_fisher_prior, device, param_names, buffer_names)
            all_results[s][c] = res
            
    # Print comparison table
    print("\n" + "="*50)
    print("                     FINAL RESULTS SUMMARY")
    print("="*50)
    for m in ["static", "no_fisher", "s_fisher", "tt_fisher", "dr_fisher"]:
        print(f"\nMethod: {m.upper()}")
        print("-" * 50)
        for s in streams:
            for c in corruptions:
                acc = all_results[s][c][m]["final_acc"]
                print(f"  Stream: {s:12s} | Corruption: {c:14s} | Accuracy: {acc:.2f}%")
                
    # Generate Plots
    os.makedirs("./plots", exist_ok=True)
    for s in streams:
        for c in corruptions:
            plt.figure(figsize=(10, 5))
            for m in ["static", "no_fisher", "s_fisher", "tt_fisher", "dr_fisher"]:
                accs = all_results[s][c][m]["acc_over_time"]
                # Compute running average for smoother curves
                run_avg = np.cumsum(accs) / (np.arange(len(accs)) + 1)
                plt.plot(run_avg * 100.0, label=m.upper())
            plt.title(f"Running Average Accuracy - Stream: {s.capitalize()}, Corruption: {c.capitalize()}")
            plt.xlabel("Batch Index")
            plt.ylabel("Accuracy (%)")
            plt.legend()
            plt.grid(True)
            plt.savefig(f"./plots/acc_{s}_{c}.png")
            plt.close()
            
    # Save the results dictionary as a python script/pickle or print LaTeX table
    print("\n" + "="*50)
    print("                   LATEX TABLE COMPILATION")
    print("="*50)
    
    latex_code = """
\\begin{table*}[t]
\\caption{Comparison of test-time model merging methods across different streams and corruptions. Accuracy (\\%) is evaluated on 4,320 stream samples following 480 calibration steps.}
\\label{tab:main_results}
\\vskip 0.15in
\\begin{center}
\\begin{small}
\\begin{sc}
\\begin{tabular}{llccccc}
\\toprule
Stream & Corruption & Static & AdaMerging & S-Fisher & TT-Fisher & DR-Fisher (Ours) \\\\
\\midrule
"""
    for s in streams:
        for c in corruptions:
            row_s = s.capitalize()
            row_c = c.replace("_", " ").capitalize()
            acc_static = all_results[s][c]["static"]["final_acc"]
            acc_nofish = all_results[s][c]["no_fisher"]["final_acc"]
            acc_sfish  = all_results[s][c]["s_fisher"]["final_acc"]
            acc_ttfish = all_results[s][c]["tt_fisher"]["final_acc"]
            acc_drfish = all_results[s][c]["dr_fisher"]["final_acc"]
            
            # Bold the best accuracy
            accs = [acc_static, acc_nofish, acc_sfish, acc_ttfish, acc_drfish]
            best_idx = np.argmax(accs)
            acc_strs = []
            for i, acc in enumerate(accs):
                if i == best_idx:
                    acc_strs.append(f"\\textbf{{{acc:.2f}}}")
                else:
                    acc_strs.append(f"{acc:.2f}")
                    
            latex_code += f"{row_s} & {row_c} & {acc_strs[0]} & {acc_strs[1]} & {acc_strs[2]} & {acc_strs[3]} & {acc_strs[4]} \\\\\n"
            
    latex_code += """\\bottomrule
\\end{tabular}
\\end{sc}
\\end{small}
\\end{center}
\\vskip -0.1in
\\end{table*}
"""
    print(latex_code)
    
    # Save LaTeX table to a text file for paper compilation
    with open("results_table.tex", "w") as f:
        f.write(latex_code)
    print("Saved LaTeX table to results_table.tex")

if __name__ == "__main__":
    main()
