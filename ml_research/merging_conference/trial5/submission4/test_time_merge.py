import os
import copy
import torch
# Disable cuDNN to avoid CUDNN_STATUS_NOT_INITIALIZED errors on this cluster
torch.backends.cudnn.enabled = False
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import pandas as pd
from torch.func import functional_call

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Custom corruptions for evaluation
class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=0.4):
        self.std = std
        self.mean = mean
    def __call__(self, tensor):
        noise = torch.randn(tensor.size(), device=tensor.device) * self.std + self.mean
        return torch.clamp(tensor + noise, 0., 1.)

class ContrastShift(object):
    def __init__(self, alpha=0.3):
        self.alpha = alpha
    def __call__(self, tensor):
        return torch.clamp(0.5 + self.alpha * (tensor - 0.5), 0., 1.)

def get_transforms(corruption_type):
    base_transform = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
    
    if corruption_type == "clean":
        pass
    elif corruption_type == "noise":
        base_transform.append(AddGaussianNoise(std=0.4))
    elif corruption_type == "blur":
        base_transform.append(transforms.GaussianBlur(kernel_size=5, sigma=1.5))
    elif corruption_type == "contrast":
        base_transform.append(ContrastShift(alpha=0.3))
    else:
        raise ValueError(f"Unknown corruption type: {corruption_type}")
        
    base_transform.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return transforms.Compose(base_transform)

def load_calibration_data():
    # Load small subsets for calibration (200 images each)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    cifar_train = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=False)
    cifar_subset = Subset(cifar_train, list(range(200)))
    cifar_loader = DataLoader(cifar_subset, batch_size=32, shuffle=False)
    
    svhn_train = torchvision.datasets.SVHN(root='./data', split='train', transform=transform, download=False)
    svhn_subset = Subset(svhn_train, list(range(200)))
    svhn_loader = DataLoader(svhn_subset, batch_size=32, shuffle=False)
    
    return cifar_loader, svhn_loader

def compute_fisher_sensitivity(device):
    print("Computing Fisher sensitivity priors...")
    cifar_loader, svhn_loader = load_calibration_data()
    criterion = nn.CrossEntropyLoss()
    
    # 1. CIFAR-10 Fisher
    cifar_model = models.resnet18().to(device)
    cifar_model.fc = nn.Linear(512, 10).to(device)
    cifar_model.load_state_dict(torch.load("models/cifar10_expert.pt", map_location=device))
    cifar_model.eval()
    
    fisher_cifar = {}
    encoder_params = [name for name, _ in cifar_model.named_parameters() if not name.startswith('fc')]
    
    for name in encoder_params:
        p = dict(cifar_model.named_parameters())[name]
        fisher_cifar[name] = torch.zeros_like(p.data)
        
    for images, labels in cifar_loader:
        images, labels = images.to(device), labels.to(device)
        for i in range(len(images)):
            img = images[i:i+1]
            lbl = labels[i:i+1]
            cifar_model.zero_grad()
            outputs = cifar_model(img)
            loss = criterion(outputs, lbl)
            loss.backward()
            
            for name in encoder_params:
                p = dict(cifar_model.named_parameters())[name]
                if p.grad is not None:
                    fisher_cifar[name] += (p.grad.data ** 2)
                    
    for name in encoder_params:
        fisher_cifar[name] /= 200.0
        
    # 2. SVHN Fisher
    svhn_model = models.resnet18().to(device)
    svhn_model.fc = nn.Linear(512, 10).to(device)
    svhn_model.load_state_dict(torch.load("models/svhn_expert.pt", map_location=device))
    svhn_model.eval()
    
    fisher_svhn = {}
    for name in encoder_params:
        p = dict(svhn_model.named_parameters())[name]
        fisher_svhn[name] = torch.zeros_like(p.data)
        
    for images, labels in svhn_loader:
        images, labels = images.to(device), labels.to(device)
        for i in range(len(images)):
            img = images[i:i+1]
            lbl = labels[i:i+1]
            svhn_model.zero_grad()
            outputs = svhn_model(img)
            loss = criterion(outputs, lbl)
            loss.backward()
            
            for name in encoder_params:
                p = dict(svhn_model.named_parameters())[name]
                if p.grad is not None:
                    fisher_svhn[name] += (p.grad.data ** 2)
                    
    for name in encoder_params:
        fisher_svhn[name] /= 200.0
        
    # 3. Joint Layer-wise Fisher Sensitivity
    joint_fisher = {}
    for name in encoder_params:
        mean_cifar = fisher_cifar[name].mean().item()
        mean_svhn = fisher_svhn[name].mean().item()
        joint_fisher[name] = 0.5 * (mean_cifar + mean_svhn)
        
    print("Fisher sensitivity calculation completed.")
    return joint_fisher

def run_evaluation(
    preloaded_batches,
    base_weights,
    cifar_weights,
    svhn_weights,
    cifar_head_weights,
    svhn_head_weights,
    joint_fisher,
    method,
    sparsity_p,
    eta,
    alpha,
    opt_type,
    device
):
    # Re-initialize coefficients
    encoder_param_names = list(joint_fisher.keys())
    
    lambdas_cifar = {name: torch.tensor(0.5, requires_grad=True, device=device) for name in encoder_param_names}
    lambdas_svhn = {name: torch.tensor(0.5, requires_grad=True, device=device) for name in encoder_param_names}
    
    # 1. Determine which layers are frozen under FGS-Merge
    frozen_layers = set()
    if "fgs" in method and sparsity_p > 0:
        # Sort layers by sensitivity in descending order
        sorted_layers = sorted(joint_fisher.items(), key=lambda x: x[1], reverse=True)
        num_to_freeze = int(len(sorted_layers) * (sparsity_p / 100.0))
        frozen_layers = {layer[0] for layer in sorted_layers[:num_to_freeze]}
        
    # 2. Configure learning rates (LFWA or Uniform)
    eta_w = {}
    epsilon_scale = 1e-8
    for name in encoder_param_names:
        if "lfwa" in method:
            # Scale learning rate inversely to sensitivity
            eta_w[name] = eta / ((joint_fisher[name] + epsilon_scale) ** alpha)
        else:
            eta_w[name] = eta

    # Create PyTorch parameter groups for the optimizer
    param_groups = []
    for name in encoder_param_names:
        if name in frozen_layers or method == "static":
            # Set learning rate to 0 to keep the coefficients frozen at 0.5
            lr_c = 0.0
            lr_s = 0.0
            lambdas_cifar[name].requires_grad = False
            lambdas_svhn[name].requires_grad = False
        else:
            lr_c = eta_w[name]
            lr_s = eta_w[name]
            
        param_groups.append({'params': [lambdas_cifar[name]], 'lr': lr_c})
        param_groups.append({'params': [lambdas_svhn[name]], 'lr': lr_s})
        
    if method != "static" and len(param_groups) > 0:
        if opt_type == "adam":
            optimizer = torch.optim.Adam(param_groups)
        else:
            optimizer = torch.optim.SGD(param_groups, momentum=0.9)
            
    # Instantiate clean ResNet18 model to use with functional_call
    model = models.resnet18().to(device)
    model.load_state_dict(torch.load("models/base_pretrained.pt", map_location=device))
    model.fc = nn.Linear(512, 10).to(device) # Placeholder head
    model.eval()
    
    total_correct = 0
    total_samples = 0
    
    for images, labels, task_type in preloaded_batches:
        # Get active head parameters
        active_head = cifar_head_weights if task_type == "cifar" else svhn_head_weights
        
        if method != "static":
            # Step 1: Compute test-time adaptation loss (Entropy Minimization)
            # Create the merged parameters for the forward pass
            merged_params = {}
            for name in base_weights:
                coeff_name = name
                is_buffer = False
                if name.endswith(".running_mean") or name.endswith(".running_var") or name.endswith(".num_batches_tracked"):
                    prefix = name.rsplit(".", 1)[0]
                    coeff_name = f"{prefix}.weight"
                    is_buffer = True
                    
                if coeff_name in lambdas_cifar:
                    l_c = lambdas_cifar[coeff_name].detach() if is_buffer else lambdas_cifar[coeff_name]
                    l_s = lambdas_svhn[coeff_name].detach() if is_buffer else lambdas_svhn[coeff_name]
                    merged_params[name] = (
                        base_weights[name] 
                        + l_c * (cifar_weights[name] - base_weights[name]) 
                        + l_s * (svhn_weights[name] - base_weights[name])
                    )
                else:
                    merged_params[name] = base_weights[name]
                    
            for name in active_head:
                merged_params[f"fc.{name}"] = active_head[name]
                
            # Forward pass
            outputs = functional_call(model, merged_params, images)
            probs = F.softmax(outputs, dim=1)
            
            # Unsupervised Entropy Loss
            entropy_loss = -torch.mean(torch.sum(probs * torch.log(probs + 1e-5), dim=1))
            
            # Backward pass & update coefficients
            optimizer.zero_grad()
            entropy_loss.backward()
            optimizer.step()
            
            # Clamp merging coefficients to [0.0, 1.0] as in the papers
            with torch.no_grad():
                for name in encoder_param_names:
                    lambdas_cifar[name].clamp_(0.0, 1.0)
                    lambdas_svhn[name].clamp_(0.0, 1.0)
                    
        # Step 2: Final inference and evaluation on the same batch (post-adaptation)
        with torch.no_grad():
            merged_params = {}
            for name in base_weights:
                coeff_name = name
                is_buffer = False
                if name.endswith(".running_mean") or name.endswith(".running_var") or name.endswith(".num_batches_tracked"):
                    prefix = name.rsplit(".", 1)[0]
                    coeff_name = f"{prefix}.weight"
                    is_buffer = True
                    
                if coeff_name in lambdas_cifar:
                    l_c = lambdas_cifar[coeff_name].detach() if is_buffer else lambdas_cifar[coeff_name]
                    l_s = lambdas_svhn[coeff_name].detach() if is_buffer else lambdas_svhn[coeff_name]
                    merged_params[name] = (
                        base_weights[name] 
                        + l_c * (cifar_weights[name] - base_weights[name]) 
                        + l_s * (svhn_weights[name] - base_weights[name])
                    )
                else:
                    merged_params[name] = base_weights[name]
                    
            for name in active_head:
                merged_params[f"fc.{name}"] = active_head[name]
                
            outputs = functional_call(model, merged_params, images)
            _, predicted = outputs.max(1)
            correct = predicted.eq(labels).sum().item()
            
            total_correct += correct
            total_samples += labels.size(0)
            
    avg_accuracy = 100.0 * total_correct / total_samples
    return avg_accuracy

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--corruption", type=str, default="all", choices=["all", "clean", "noise", "blur", "contrast"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Fisher sensitivities
    joint_fisher = compute_fisher_sensitivity(device)
    
    # Load expert models and save their weights
    print("Loading model weights...")
    base_model = models.resnet18().to(device)
    base_model.load_state_dict(torch.load("models/base_pretrained.pt", map_location=device))
    base_weights = {name: param.clone().detach().to(device) for name, param in base_model.state_dict().items() if not name.startswith('fc')}
    
    cifar_model = models.resnet18().to(device)
    cifar_model.fc = nn.Linear(512, 10).to(device)
    cifar_model.load_state_dict(torch.load("models/cifar10_expert.pt", map_location=device))
    cifar_weights = {name: param.clone().detach().to(device) for name, param in cifar_model.state_dict().items() if not name.startswith('fc')}
    cifar_head_weights = {name.replace("fc.", ""): param.clone().detach().to(device) for name, param in cifar_model.named_parameters() if name.startswith('fc.')}
    
    svhn_model = models.resnet18().to(device)
    svhn_model.fc = nn.Linear(512, 10).to(device)
    svhn_model.load_state_dict(torch.load("models/svhn_expert.pt", map_location=device))
    svhn_weights = {name: param.clone().detach().to(device) for name, param in svhn_model.state_dict().items() if not name.startswith('fc')}
    svhn_head_weights = {name.replace("fc.", ""): param.clone().detach().to(device) for name, param in svhn_model.named_parameters() if name.startswith('fc.')}
    
    # Define test streams & corruptions
    stream_types = ["alternating", "sequential"]
    if args.corruption == "all":
        corruptions = ["clean", "noise", "blur", "contrast"]
    else:
        corruptions = [args.corruption]
    
    results = []
    
    # Run evaluations
    for corruption in corruptions:
        print(f"\nEvaluating under corruption: {corruption}")
        
        # Load datasets with the specific corruption
        transform = get_transforms(corruption)
        
        # Pre-load and transform test data
        print("  Pre-loading and transforming test subsets...")
        cifar_test = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=False)
        cifar_subset = Subset(cifar_test, list(range(1024)))
        cifar_loader_preload = DataLoader(cifar_subset, batch_size=64, shuffle=False)
        cifar_batches = []
        for imgs, lbls in cifar_loader_preload:
            cifar_batches.append((imgs.to(device), lbls.to(device)))
            
        svhn_test = torchvision.datasets.SVHN(root='./data', split='test', transform=transform, download=False)
        svhn_subset = Subset(svhn_test, list(range(1024)))
        svhn_loader_preload = DataLoader(svhn_subset, batch_size=64, shuffle=False)
        svhn_batches = []
        for imgs, lbls in svhn_loader_preload:
            svhn_batches.append((imgs.to(device), lbls.to(device)))
            
        # Construct preloaded streams
        preloaded_streams = {}
        for stream_type in stream_types:
            preloaded_batches = []
            if stream_type == "alternating":
                for i in range(16):
                    preloaded_batches.append((cifar_batches[i][0], cifar_batches[i][1], "cifar"))
                    preloaded_batches.append((svhn_batches[i][0], svhn_batches[i][1], "svhn"))
            elif stream_type == "sequential":
                for i in range(16):
                    preloaded_batches.append((cifar_batches[i][0], cifar_batches[i][1], "cifar"))
                for i in range(16):
                    preloaded_batches.append((svhn_batches[i][0], svhn_batches[i][1], "svhn"))
            preloaded_streams[stream_type] = preloaded_batches
            
        for stream_type in stream_types:
            print(f"    Stream type: {stream_type}")
            preloaded_batches = preloaded_streams[stream_type]
            
            # 1. Static Merging
            acc = run_evaluation(
                preloaded_batches, base_weights, cifar_weights, svhn_weights,
                cifar_head_weights, svhn_head_weights, joint_fisher,
                method="static", sparsity_p=0, eta=0.0, alpha=0.0, opt_type="sgd", device=device
            )
            results.append({
                "corruption": corruption, "stream": stream_type, "method": "Static Merging",
                "sparsity_p": 0, "eta": 0.0, "alpha": 0.0, "opt": "none", "accuracy": acc
            })
            
            # Sweeping hyperparameters for TTA and FGS-TTA
            for eta in [0.001, 0.01, 0.1, 1.0]:
                for opt in ["sgd", "adam"]:
                    # 2. Standard TTA
                    acc = run_evaluation(
                        preloaded_batches, base_weights, cifar_weights, svhn_weights,
                        cifar_head_weights, svhn_head_weights, joint_fisher,
                        method="tta", sparsity_p=0, eta=eta, alpha=0.0, opt_type=opt, device=device
                    )
                    results.append({
                        "corruption": corruption, "stream": stream_type, "method": "Standard TTA",
                        "sparsity_p": 0, "eta": eta, "alpha": 0.0, "opt": opt, "accuracy": acc
                    })
                    
                    # 3. FGS-TTA (Sparsity sweeps)
                    for p in [20, 50, 80]:
                        acc = run_evaluation(
                            preloaded_batches, base_weights, cifar_weights, svhn_weights,
                            cifar_head_weights, svhn_head_weights, joint_fisher,
                            method="fgs_tta", sparsity_p=p, eta=eta, alpha=0.0, opt_type=opt, device=device
                        )
                        results.append({
                            "corruption": corruption, "stream": stream_type, "method": f"FGS-TTA",
                            "sparsity_p": p, "eta": eta, "alpha": 0.0, "opt": opt, "accuracy": acc
                        })
            
            # Sweeping hyperparameters for LFWA and FGS-LFWA
            for eta in [0.001, 0.01, 0.1, 1.0]:
                for alpha in [0.2, 0.5, 1.0]:
                    for opt in ["sgd", "adam"]:
                        # 4. LFWA
                        acc = run_evaluation(
                            preloaded_batches, base_weights, cifar_weights, svhn_weights,
                            cifar_head_weights, svhn_head_weights, joint_fisher,
                            method="lfwa", sparsity_p=0, eta=eta, alpha=alpha, opt_type=opt, device=device
                        )
                        results.append({
                            "corruption": corruption, "stream": stream_type, "method": "LFWA",
                            "sparsity_p": 0, "eta": eta, "alpha": alpha, "opt": opt, "accuracy": acc
                        })
                        
                        # 5. FGS-LFWA (Sparsity sweeps)
                        for p in [20, 50, 80]:
                            acc = run_evaluation(
                                preloaded_batches, base_weights, cifar_weights, svhn_weights,
                                cifar_head_weights, svhn_head_weights, joint_fisher,
                                method="fgs_lfwa", sparsity_p=p, eta=eta, alpha=alpha, opt_type=opt, device=device
                            )
                            results.append({
                                "corruption": corruption, "stream": stream_type, "method": "FGS-LFWA",
                                "sparsity_p": p, "eta": eta, "alpha": alpha, "opt": opt, "accuracy": acc
                            })
                            
    # Save results to CSV
    df = pd.DataFrame(results)
    if args.corruption == "all":
        csv_name = "experiment_results.csv"
    else:
        csv_name = f"experiment_results_{args.corruption}.csv"
    df.to_csv(csv_name, index=False)
    print(f"\nAll experiments completed! Results saved to {csv_name}.")
    
    # Print a summary of best results per method
    print("\n--- Summary of Best Results per Method ---")
    summary = []
    for method_name in ["Static Merging", "Standard TTA", "FGS-TTA", "LFWA", "FGS-LFWA"]:
        method_df = df[df["method"].str.startswith(method_name)]
        if len(method_df) == 0:
            continue
        avg_acc = method_df["accuracy"].mean()
        print(f"{method_name}: Average Accuracy across all environments = {avg_acc:.2f}%")

if __name__ == "__main__":
    main()
