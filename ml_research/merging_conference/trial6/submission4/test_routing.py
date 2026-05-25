import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np

# Set seed
def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

def get_modified_resnet18(num_classes=10):
    model = models.resnet18(weights=None)
    old_conv = model.conv1
    model.conv1 = nn.Conv2d(1, old_conv.out_channels, kernel_size=old_conv.kernel_size, 
                            stride=old_conv.stride, padding=old_conv.padding, bias=old_conv.bias)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def apply_corruption(images, corruption_type):
    if corruption_type == "gaussian_noise":
        noise = torch.randn_like(images) * 0.2
        return torch.clamp(images + noise, -1.0, 1.0)
    elif corruption_type == "contrast_shift":
        raw = images * 0.5 + 0.5
        shifted = (raw - 0.5) * 0.3 + 0.5
        clamped = torch.clamp(shifted, 0.0, 1.0)
        return (clamped - 0.5) / 0.5
    return images

def extract_features(model, x):
    features = []
    def hook_fn(module, input, output):
        features.append(torch.flatten(output, 1))
    
    hook = model.avgpool.register_forward_hook(hook_fn)
    with torch.no_grad():
        _ = model(x)
    hook.remove()
    return features[0]

def build_test_streams(mnist_test, fmnist_test, kmnist_test):
    mnist_subset = Subset(mnist_test, list(range(1600)))
    fmnist_subset = Subset(fmnist_test, list(range(1600)))
    kmnist_subset = Subset(kmnist_test, list(range(1600)))
    
    mnist_batches = [Subset(mnist_subset, list(range(i*32, (i+1)*32))) for i in range(50)]
    fmnist_batches = [Subset(fmnist_subset, list(range(i*32, (i+1)*32))) for i in range(50)]
    kmnist_batches = [Subset(kmnist_subset, list(range(i*32, (i+1)*32))) for i in range(50)]
    
    alt_batches = []
    for i in range(50):
        alt_batches.append((mnist_batches[i], 0))
        alt_batches.append((fmnist_batches[i], 1))
        alt_batches.append((kmnist_batches[i], 2))
        
    seq_batches = []
    for i in range(50):
        seq_batches.append((mnist_batches[i], 0))
    for i in range(50):
        seq_batches.append((fmnist_batches[i], 1))
    for i in range(50):
        seq_batches.append((kmnist_batches[i], 2))
        
    return alt_batches, seq_batches

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    mnist_test = torchvision.datasets.MNIST(root="data", train=False, download=True, transform=transform)
    fmnist_test = torchvision.datasets.FashionMNIST(root="data", train=False, download=True, transform=transform)
    kmnist_test = torchvision.datasets.KMNIST(root="data", train=False, download=True, transform=transform)
    
    mnist_train = torchvision.datasets.MNIST(root="data", train=True, download=True, transform=transform)
    fmnist_train = torchvision.datasets.FashionMNIST(root="data", train=True, download=True, transform=transform)
    kmnist_train = torchvision.datasets.KMNIST(root="data", train=True, download=True, transform=transform)
    
    # Load experts
    experts = []
    expert_paths = ["expert_mnist.pt", "expert_fmnist.pt", "expert_kmnist.pt"]
    for path in expert_paths:
        model = get_modified_resnet18(num_classes=10).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        experts.append(model)
        
    # Precompute prototypes and their means
    prototypes = []
    proto_means = []
    for k in range(3):
        # We compute prototypes using first 1000 train samples of that expert
        dataset = [mnist_train, fmnist_train, kmnist_train][k]
        loader = DataLoader(Subset(dataset, list(range(1000))), batch_size=32, shuffle=False)
        features_list = []
        targets_list = []
        
        def hook_fn(module, input, output):
            features_list.append(torch.flatten(output, 1).detach().cpu())
        hook = experts[k].avgpool.register_forward_hook(hook_fn)
        with torch.no_grad():
            for inputs, targets in loader:
                _ = experts[k](inputs.to(device))
                targets_list.append(targets)
        hook.remove()
        
        all_features = torch.cat(features_list, dim=0)
        all_targets = torch.cat(targets_list, dim=0)
        
        protos = {}
        for c in range(10):
            mask = (all_targets == c)
            if mask.sum() > 0:
                protos[c] = all_features[mask].mean(dim=0).to(device)
            else:
                protos[c] = torch.zeros(512).to(device)
        prototypes.append(protos)
        
        # Mean of all class prototypes for expert k
        stacked_protos = torch.stack([protos[c] for c in range(10)]) # (10, 512)
        proto_means.append(stacked_protos.mean(dim=0))
        
    alt_batches, seq_batches = build_test_streams(mnist_test, fmnist_test, kmnist_test)
    
    corruptions = ["clean", "gaussian_noise", "contrast_shift"]
    
    for corr in corruptions:
        print(f"\n--- Evaluating Routing under Corruption: {corr} ---")
        
        # Let's collect actual tasks and predictions for different methods
        actual_tasks = []
        preds_baseline = []
        preds_entropy = []
        preds_max_prob = []
        preds_ifc = [] # Isotropic Feature Centered prototypes similarity
        
        for b_idx, (batch_subset, actual_task) in enumerate(alt_batches):
            loader = DataLoader(batch_subset, batch_size=32, shuffle=False)
            inputs, targets = next(iter(loader))
            inputs = inputs.to(device)
            inputs_corrupted = apply_corruption(inputs, corr)
            
            actual_tasks.append(actual_task)
            
            # 1. Baseline: Cosine similarity to raw prototypes
            scores_base = []
            # 2. Entropy routing: entropy of prediction
            entropies = []
            # 3. Max probability routing: average of max probability
            max_probs = []
            # 4. IFC: Cosine similarity of centered features and centered prototypes
            scores_ifc = []
            
            with torch.no_grad():
                for k in range(3):
                    # Features
                    feats = extract_features(experts[k], inputs_corrupted) # (32, 512)
                    
                    # 1. Baseline
                    feats_norm = nn.functional.normalize(feats, p=2, dim=1)
                    protos_tensor = torch.stack([prototypes[k][c] for c in range(10)]) # (10, 512)
                    protos_norm = nn.functional.normalize(protos_tensor, p=2, dim=1)
                    sim = torch.matmul(feats_norm, protos_norm.T)
                    max_sim, _ = sim.max(dim=1)
                    scores_base.append(max_sim.mean().item())
                    
                    # 2 & 3. Outputs
                    outputs = experts[k](inputs_corrupted)
                    probs = torch.softmax(outputs, dim=1)
                    # Entropy
                    ent = -torch.sum(probs * torch.log(probs + 1e-6), dim=1).mean().item()
                    entropies.append(ent)
                    # Max Probability
                    max_p = probs.max(dim=1)[0].mean().item()
                    max_probs.append(max_p)
                    
                    # 4. IFC: Center features and center prototypes
                    feats_mean = feats.mean(dim=0, keepdim=True)
                    feats_centered = feats - feats_mean
                    feats_centered_norm = nn.functional.normalize(feats_centered, p=2, dim=1)
                    
                    protos_centered = protos_tensor - proto_means[k].unsqueeze(0)
                    protos_centered_norm = nn.functional.normalize(protos_centered, p=2, dim=1)
                    
                    sim_centered = torch.matmul(feats_centered_norm, protos_centered_norm.T)
                    max_sim_c, _ = sim_centered.max(dim=1)
                    scores_ifc.append(max_sim_c.mean().item())
                    
            preds_baseline.append(np.argmax(scores_base))
            preds_entropy.append(np.argmin(entropies)) # lower is better
            preds_max_prob.append(np.argmax(max_probs)) # higher is better
            preds_ifc.append(np.argmax(scores_ifc))
            
        acc_base = np.mean(np.array(preds_baseline) == np.array(actual_tasks)) * 100
        acc_ent = np.mean(np.array(preds_entropy) == np.array(actual_tasks)) * 100
        acc_max_p = np.mean(np.array(preds_max_prob) == np.array(actual_tasks)) * 100
        acc_ifc = np.mean(np.array(preds_ifc) == np.array(actual_tasks)) * 100
        
        print(f"  Baseline Routing Accuracy: {acc_base:.2f}%")
        print(f"  Entropy Routing Accuracy:  {acc_ent:.2f}%")
        print(f"  Max Prob Routing Accuracy: {acc_max_p:.2f}%")
        print(f"  IFC Routing Accuracy:      {acc_ifc:.2f}%")

if __name__ == "__main__":
    main()
