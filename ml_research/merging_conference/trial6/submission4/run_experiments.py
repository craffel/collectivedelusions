import os
import time
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Disable cuDNN to avoid CUDNN_STATUS_NOT_INITIALIZED issues on this node
    torch.backends.cudnn.enabled = False

set_seed(42)

# --- 1. Dataset & Model Setup ---

def get_modified_resnet18(num_classes=10):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Modify conv1 to accept 1-channel grayscale images
    old_conv = model.conv1
    model.conv1 = nn.Conv2d(1, old_conv.out_channels, kernel_size=old_conv.kernel_size, 
                            stride=old_conv.stride, padding=old_conv.padding, bias=old_conv.bias)
    with torch.no_grad():
        # Sum pre-trained weights across the input channel dimension to fit 1-channel inputs
        model.conv1.weight.copy_(old_conv.weight.sum(dim=1, keepdim=True))
    
    # Modify fc to output num_classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# Load datasets
def get_datasets():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    os.makedirs("data", exist_ok=True)
    mnist_train = torchvision.datasets.MNIST(root="data", train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.MNIST(root="data", train=False, download=True, transform=transform)
    
    fmnist_train = torchvision.datasets.FashionMNIST(root="data", train=True, download=True, transform=transform)
    fmnist_test = torchvision.datasets.FashionMNIST(root="data", train=False, download=True, transform=transform)
    
    kmnist_train = torchvision.datasets.KMNIST(root="data", train=True, download=True, transform=transform)
    kmnist_test = torchvision.datasets.KMNIST(root="data", train=False, download=True, transform=transform)
    
    return (mnist_train, mnist_test), (fmnist_train, fmnist_test), (kmnist_train, kmnist_test)

# Train specialized experts
def train_expert(name, train_dataset, device, epochs=4, lr=1e-4, weight_decay=1e-2):
    print(f"--- Training Expert {name} ---")
    model = get_modified_resnet18(num_classes=10).to(device)
    
    # Use first 10,000 samples for training
    indices = list(range(10000))
    train_subset = Subset(train_dataset, indices)
    loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        total_loss, correct, total = 0.0, 0, 0
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/total:.4f} | Acc: {acc:.2f}%")
        
    return model

# --- 2. Fisher and Prototype Computation ---

def compute_layer_wise_fisher(model, train_dataset, device, num_samples=500):
    model.eval()
    loader = DataLoader(Subset(train_dataset, list(range(num_samples))), batch_size=1, shuffle=False)
    
    # We group ResNet-18 layers into 6 layer groups
    # 1: conv1/bn1, 2: layer1, 3: layer2, 4: layer3, 5: layer4, 6: fc
    group_names = ["early", "layer1", "layer2", "layer3", "layer4", "fc"]
    group_fisher = {name: [] for name in group_names}
    
    def get_group_name(param_name):
        if "conv1" in param_name or "bn1" in param_name:
            return "early"
        elif "layer1" in param_name:
            return "layer1"
        elif "layer2" in param_name:
            return "layer2"
        elif "layer3" in param_name:
            return "layer3"
        elif "layer4" in param_name:
            return "layer4"
        elif "fc" in param_name:
            return "fc"
        return "early"
    
    criterion = nn.CrossEntropyLoss()
    
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                g_name = get_group_name(name)
                # Compute squared gradients
                g2 = param.grad.data.clone().pow(2).mean().item()
                group_fisher[g_name].append(g2)
                
    # Average across parameters in each group
    mean_fisher = {}
    for g_name in group_names:
        if len(group_fisher[g_name]) > 0:
            mean_fisher[g_name] = np.mean(group_fisher[g_name])
        else:
            mean_fisher[g_name] = 1e-5
            
    return mean_fisher

def compute_prototypes(model, train_dataset, device, num_samples=1000):
    model.eval()
    loader = DataLoader(Subset(train_dataset, list(range(num_samples))), batch_size=32, shuffle=False)
    
    # We want features before the fc layer
    # We can register a forward hook to capture the pooled features
    features_list = []
    targets_list = []
    
    def hook_fn(module, input, output):
        # output is of shape (batch_size, 512, 1, 1) or (batch_size, 512)
        pooled = torch.flatten(output, 1)
        features_list.append(pooled.detach().cpu())
        
    hook = model.avgpool.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            _ = model(inputs)
            targets_list.append(targets)
            
    hook.remove()
    
    all_features = torch.cat(features_list, dim=0)
    all_targets = torch.cat(targets_list, dim=0)
    
    prototypes = {}
    for c in range(10):
        mask = (all_targets == c)
        if mask.sum() > 0:
            prototypes[c] = all_features[mask].mean(dim=0).to(device)
        else:
            prototypes[c] = torch.zeros(512).to(device)
            
    return prototypes

# --- 3. Stream Construction & Corruption ---

def apply_corruption(images, corruption_type):
    # images are tensors in [0, 1]
    if corruption_type == "gaussian_noise":
        noise = torch.randn_like(images) * 0.2
        return torch.clamp(images + noise, -1.0, 1.0) # Normalized range is [-1, 1] due toNormalize((0.5,),(0.5,))
    elif corruption_type == "contrast_shift":
        # Raw image space of standard dataset before Normalize is [0, 1].
        # Our normalized images can be un-normalized to [0,1], shifted, and re-normalized.
        raw = images * 0.5 + 0.5
        shifted = (raw - 0.5) * 0.3 + 0.5
        clamped = torch.clamp(shifted, 0.0, 1.0)
        return (clamped - 0.5) / 0.5
    return images # Clean

def build_test_streams(mnist_test, fmnist_test, kmnist_test):
    # 1,600 samples per task, processed in batches of size 32 -> 50 batches per task
    # Total = 150 batches (4,800 samples)
    mnist_subset = Subset(mnist_test, list(range(1600)))
    fmnist_subset = Subset(fmnist_test, list(range(1600)))
    kmnist_subset = Subset(kmnist_test, list(range(1600)))
    
    mnist_batches = [Subset(mnist_subset, list(range(i*32, (i+1)*32))) for i in range(50)]
    fmnist_batches = [Subset(fmnist_subset, list(range(i*32, (i+1)*32))) for i in range(50)]
    kmnist_batches = [Subset(kmnist_subset, list(range(i*32, (i+1)*32))) for i in range(50)]
    
    # 1. Alternating Stream: T1, T2, T3, T1, T2, T3...
    alt_batches = []
    for i in range(50):
        alt_batches.append((mnist_batches[i], 0))  # (batch, task_id)
        alt_batches.append((fmnist_batches[i], 1))
        alt_batches.append((kmnist_batches[i], 2))
        
    # 2. Sequential Stream: 50T1 -> 50T2 -> 50T3
    seq_batches = []
    for i in range(50):
        seq_batches.append((mnist_batches[i], 0))
    for i in range(50):
        seq_batches.append((fmnist_batches[i], 1))
    for i in range(50):
        seq_batches.append((kmnist_batches[i], 2))
        
    return alt_batches, seq_batches

# --- 4. Test-Time Adaptation Methods ---

class MergedModel(nn.Module):
    def __init__(self, base_model, experts, device):
        super().__init__()
        self.base_model = base_model
        self.experts = experts # list of 3 experts
        self.device = device
        
        # 6 layer groups
        self.group_names = ["early", "layer1", "layer2", "layer3", "layer4", "fc"]
        
        # Logits of coefficients: shape (6, 3), initialized to equal weights (zeros)
        # So softmax gives [1/3, 1/3, 1/3]
        self.logits = nn.Parameter(torch.zeros(6, 3, device=device))
        
    def get_coefficients(self):
        return torch.softmax(self.logits, dim=1) # (6, 3)
        
    def get_group_idx(self, param_name):
        if "conv1" in param_name or "bn1" in param_name:
            return 0
        elif "layer1" in param_name:
            return 1
        elif "layer2" in param_name:
            return 2
        elif "layer3" in param_name:
            return 3
        elif "layer4" in param_name:
            return 4
        elif "fc" in param_name:
            return 5
        return 0
        
    def forward(self, x, return_features=False):
        coeffs = self.get_coefficients()
        
        merged_params = {}
        for name, param in self.base_model.named_parameters():
            g_idx = self.get_group_idx(name)
            w1 = self.experts[0].state_dict()[name]
            w2 = self.experts[1].state_dict()[name]
            w3 = self.experts[2].state_dict()[name]
            
            merged = coeffs[g_idx, 0] * w1 + coeffs[g_idx, 1] * w2 + coeffs[g_idx, 2] * w3
            merged_params[name] = merged
            
        for name, buf in self.base_model.named_buffers():
            merged_params[name] = buf
            
        if return_features:
            features = []
            def hook_fn(module, input, output):
                features.append(torch.flatten(output, 1))
            hook = self.base_model.avgpool.register_forward_hook(hook_fn)
            
            outputs = torch.func.functional_call(self.base_model, merged_params, x)
            hook.remove()
            return outputs, features[0]
        else:
            return torch.func.functional_call(self.base_model, merged_params, x)

# Feature extraction function (avgpool output) - non-differentiable helper for setup
def extract_features(model, x):
    features = []
    def hook_fn(module, input, output):
        features.append(torch.flatten(output, 1))
    
    target_model = model.base_model if hasattr(model, "base_model") else model
    hook = target_model.avgpool.register_forward_hook(hook_fn)
    with torch.no_grad():
        if hasattr(model, "base_model"):
            # If it's MergedModel, execute forward pass functional call
            _ = model(x)
        else:
            _ = model(x)
    hook.remove()
    return features[0]

# Unsupervised task detection via prototype similarity with Isotropic Feature Centering (IFC)
def detect_active_task(x, experts, prototypes, device):
    scores = []
    with torch.no_grad():
        for k in range(3):
            # Extract features of x using expert k
            feats = extract_features(experts[k], x) # (batch_size, 512)
            
            # Apply Isotropic Feature Centering (IFC) to handle contrast and covariate shifts
            feats_mean = feats.mean(dim=0, keepdim=True)
            feats_centered = feats - feats_mean
            feats_norm = nn.functional.normalize(feats_centered, p=2, dim=1)
            
            # Stack prototypes of expert k and center them
            protos_tensor = torch.stack([prototypes[k][c] for c in range(10)]) # (10, 512)
            protos_mean = protos_tensor.mean(dim=0, keepdim=True)
            protos_centered = protos_tensor - protos_mean
            protos_norm = nn.functional.normalize(protos_centered, p=2, dim=1)
            
            # Similarity matrix: (batch_size, 10)
            sim = torch.matmul(feats_norm, protos_norm.T)
            max_sim, _ = sim.max(dim=1)
            scores.append(max_sim.mean().item())
            
    return np.argmax(scores)

# Evaluation run for a specific method, stream, and corruption
def run_evaluation(base_model, experts, stream_batches, prototypes, group_fishers, corruption, method, device):
    adapted_base = get_modified_resnet18(num_classes=10).to(device)
    adapted_base.load_state_dict(base_model.state_dict())
    
    merged_model = MergedModel(adapted_base, experts, device)
    
    lr = 0.05
    if method == "ours":
        lr = 0.1
        
    coeff_history = []
    correct = 0
    total = 0
    
    active_task_predictions = []
    actual_tasks = []
    
    optimizer = optim.SGD([merged_model.logits], lr=lr)
    
    for b_idx, (batch_subset, actual_task) in enumerate(stream_batches):
        loader = DataLoader(batch_subset, batch_size=32, shuffle=False)
        inputs, targets = next(iter(loader))
        inputs, targets = inputs.to(device), targets.to(device)
        
        inputs_corrupted = apply_corruption(inputs, corruption)
        
        pred_task = detect_active_task(inputs_corrupted, experts, prototypes, device)
        active_task_predictions.append(pred_task)
        actual_tasks.append(actual_task)
        
        if method in ["cpa", "fp-ca", "ours"]:
            with torch.no_grad():
                merged_model.logits.copy_(torch.full_like(merged_model.logits, -10.0))
                merged_model.logits[:, pred_task].copy_(torch.full_like(merged_model.logits[:, pred_task], 10.0))
                
        coeffs = merged_model.get_coefficients().detach().cpu().numpy().copy()
        coeff_history.append(coeffs)
        
        # We need gradients with respect to logits for adaptation methods
        if method == "static":
            with torch.no_grad():
                outputs = merged_model(inputs_corrupted)
        else:
            merged_model.logits.requires_grad = True
            
            if method in ["adamerge", "lfwa", "iggs-merge"]:
                outputs = merged_model(inputs_corrupted)
            else: # cpa, fp-ca, ours
                outputs, feats = merged_model(inputs_corrupted, return_features=True)
                
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)
        
        if method == "static":
            continue
            
        optimizer.zero_grad()
        
        if method in ["adamerge", "lfwa", "iggs-merge"]:
            probs = torch.softmax(outputs, dim=1)
            loss = -torch.sum(probs * torch.log(probs + 1e-6), dim=1).mean()
            loss.backward()
            
            with torch.no_grad():
                if merged_model.logits.grad is not None:
                    grad = merged_model.logits.grad.clone()
                    
                    if method == "lfwa":
                        for g_idx, g_name in enumerate(merged_model.group_names):
                            joint_fish = sum([group_fishers[k][g_name] for k in range(3)]) / 3.0
                            grad[g_idx] /= (joint_fish + 1e-4)
                            
                    elif method == "iggs-merge":
                        for g_idx, g_name in enumerate(merged_model.group_names):
                            joint_fish = sum([group_fishers[k][g_name] for k in range(3)]) / 3.0
                            g_metric = (joint_fish + 1e-4) ** 0.5
                            grad[g_idx] *= (1.0 / g_metric)
                            
                    merged_model.logits.data -= lr * grad
                    
        elif method in ["cpa", "fp-ca", "ours"]:
            # Differentiable InfoNCE contrastive alignment loss
            # Get class prototypes of active expert as a stacked tensor
            protos_tensor = torch.stack([prototypes[pred_task][c] for c in range(10)]) # (10, 512)
            
            # Apply Isotropic Feature Centering (IFC) to both features and prototypes
            feats_mean = feats.mean(dim=0, keepdim=True)
            feats_centered = feats - feats_mean
            feats_norm = nn.functional.normalize(feats_centered, p=2, dim=1) # (batch_size, 512)
            
            protos_mean = protos_tensor.mean(dim=0, keepdim=True)
            protos_centered = protos_tensor - protos_mean
            protos_norm = nn.functional.normalize(protos_centered, p=2, dim=1) # (10, 512)
            
            # Similarity matrix: (batch_size, 10)
            sim_matrix = torch.matmul(feats_norm, protos_norm.T) / 0.1
            
            pseudo_labels = outputs.argmax(dim=1)
            criterion_contrastive = nn.CrossEntropyLoss(reduction='none')
            
            if method in ["fp-ca", "ours"]:
                probs = torch.softmax(outputs, dim=1)
                max_probs, _ = probs.max(dim=1)
                confidence_mask = (max_probs >= 0.7)
                
                if confidence_mask.sum() > 0:
                    loss = criterion_contrastive(sim_matrix[confidence_mask], pseudo_labels[confidence_mask]).mean()
                else:
                    loss = None
            else:
                loss = criterion_contrastive(sim_matrix, pseudo_labels).mean()
                
            if loss is not None:
                loss.backward()
                
                with torch.no_grad():
                    if merged_model.logits.grad is not None:
                        grad = merged_model.logits.grad.clone()
                        
                        if method == "fp-ca":
                            for g_idx, g_name in enumerate(merged_model.group_names):
                                joint_fish = sum([group_fishers[k][g_name] for k in range(3)]) / 3.0
                                grad[g_idx] /= (joint_fish + 1e-4)
                                
                        elif method == "ours":
                            # RGS-COP!
                            for g_idx, g_name in enumerate(merged_model.group_names):
                                f_active = group_fishers[pred_task][g_name]
                                f_inactive = sum([group_fishers[k][g_name] for k in range(3) if k != pred_task]) / 2.0
                                
                                cop_factor = f_active / (f_inactive + 1e-3)
                                cop_factor = np.clip(cop_factor, 0.05, 5.0)
                                grad[g_idx] *= cop_factor
                                
                        merged_model.logits.data -= lr * grad
                        
        merged_model.logits.grad = None
        
    accuracy = 100.0 * correct / total
    routing_accuracy = 100.0 * np.mean(np.array(active_task_predictions) == np.array(actual_tasks))
    
    return accuracy, routing_accuracy, np.array(coeff_history)

# --- 5. Main Execution Suite ---

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load datasets
    (mnist_train, mnist_test), (fmnist_train, fmnist_test), (kmnist_train, kmnist_test) = get_datasets()
    
    # 2. Train experts or check if they exist
    experts = []
    expert_paths = ["expert_mnist.pt", "expert_fmnist.pt", "expert_kmnist.pt"]
    expert_datasets = [mnist_train, fmnist_train, kmnist_train]
    expert_names = ["MNIST", "FashionMNIST", "KMNIST"]
    
    for path, dataset, name in zip(expert_paths, expert_datasets, expert_names):
        if os.path.exists(path):
            print(f"Loading pre-trained expert {name} from {path}...")
            model = get_modified_resnet18(num_classes=10).to(device)
            model.load_state_dict(torch.load(path, map_location=device))
            experts.append(model)
        else:
            model = train_expert(name, dataset, device, epochs=4)
            torch.save(model.state_dict(), path)
            experts.append(model)
            
    # Base model (modified ResNet-18 with default weights)
    base_model = get_modified_resnet18(num_classes=10).to(device)
    
    # 3. Compute Fisher Information and Prototypes for each expert
    print("--- Computing Fisher Information and Prototypes ---")
    group_fishers = []
    prototypes = []
    for k in range(3):
        print(f"Computing for Expert {expert_names[k]}...")
        fish = compute_layer_wise_fisher(experts[k], expert_datasets[k], device, num_samples=500)
        group_fishers.append(fish)
        print(f"Mean joint Fisher: {fish}")
        
        protos = compute_prototypes(experts[k], expert_datasets[k], device, num_samples=1000)
        prototypes.append(protos)
        
    # 4. Build streams
    alt_batches, seq_batches = build_test_streams(mnist_test, fmnist_test, kmnist_test)
    
    # 5. Run evaluations across Corruptions and Methods
    corruptions = ["clean", "gaussian_noise", "contrast_shift"]
    methods = ["static", "adamerge", "lfwa", "iggs-merge", "cpa", "fp-ca", "ours"]
    streams = {"alternating": alt_batches, "sequential": seq_batches}
    
    results = {stream_name: {corr: {} for corr in corruptions} for stream_name in streams}
    
    # Track trajectory for the Ours vs CPA-Merge plotting
    # We will save trajectory for Clean Sequential Stream
    trajectories = {}
    
    for stream_name, stream_batches in streams.items():
        print(f"\n==================== Evaluating Stream: {stream_name} ====================")
        for corr in corruptions:
            print(f"\n--- Corruption: {corr} ---")
            for method in methods:
                acc, r_acc, coeff_hist = run_evaluation(
                    base_model, experts, stream_batches, prototypes, group_fishers, corr, method, device
                )
                print(f"Method: {method:<10} | Test Accuracy: {acc:.2f}% | Routing Accuracy: {r_acc:.2f}%")
                results[stream_name][corr][method] = {
                    "accuracy": acc,
                    "routing_accuracy": r_acc
                }
                
                # Store trajectories for clean sequential stream
                if stream_name == "sequential" and corr == "clean" and method in ["cpa", "ours"]:
                    trajectories[method] = coeff_hist
                    
    # Save results to JSON
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\nSuccessfully saved results to results.json!")
    
    # 6. Generate Coefficient Trajectory Plots
    if "ours" in trajectories and "cpa" in trajectories:
        print("\n--- Generating Trajectory Plots ---")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        for idx, method in enumerate(["cpa", "ours"]):
            # trajectories[method] is of shape (150, 6, 3)
            # We average coefficients across the 6 layer groups to get a global trend for plotting
            global_coeffs = trajectories[method].mean(axis=1) # shape (150, 3)
            
            axes[idx].plot(global_coeffs[:, 0], label="$\lambda_1$ (MNIST)", color="red", linewidth=2)
            axes[idx].plot(global_coeffs[:, 1], label="$\lambda_2$ (FashionMNIST)", color="green", linewidth=2)
            axes[idx].plot(global_coeffs[:, 2], label="$\lambda_3$ (KMNIST)", color="blue", linewidth=2)
            
            axes[idx].axvline(x=50, color="gray", linestyle="--")
            axes[idx].axvline(x=100, color="gray", linestyle="--")
            
            axes[idx].set_title(f"Coefficient Trajectory - {'CPA-Merge' if method == 'cpa' else 'RGS-COP (Ours)'}", fontsize=14)
            axes[idx].set_xlabel("Adaptation Steps (Batches)", fontsize=12)
            axes[idx].set_ylabel("Merging Coefficient Value", fontsize=12)
            axes[idx].grid(True, alpha=0.3)
            axes[idx].legend(fontsize=11)
            axes[idx].set_ylim(-0.05, 1.05)
            
        plt.tight_layout()
        plt.savefig("coefficient_trajectories.png", dpi=300)
        print("Successfully saved trajectory plot to coefficient_trajectories.png!")

if __name__ == "__main__":
    main()
