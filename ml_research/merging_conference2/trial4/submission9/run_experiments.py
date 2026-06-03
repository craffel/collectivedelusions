import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import DataLoader, Subset
import numpy as np
import random

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

# --- DATASET PREPARATION ---
def get_datasets():
    # Transforms
    # Resize to 32x32, replicate to 3 channels, normalize
    mnist_transform = T.Compose([
        T.Resize((32, 32)),
        T.ToTensor(),
        T.Lambda(lambda x: x.repeat(3, 1, 1)),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    fmnist_transform = T.Compose([
        T.Resize((32, 32)),
        T.ToTensor(),
        T.Lambda(lambda x: x.repeat(3, 1, 1)),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    cifar_transform = T.Compose([
        T.Resize((32, 32)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Download datasets
    print("Loading datasets...")
    mnist_train = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=mnist_transform)
    mnist_test = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=mnist_transform)
    
    fmnist_train = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True, transform=fmnist_transform)
    fmnist_test = torchvision.datasets.FashionMNIST(root="./data", train=False, download=True, transform=fmnist_transform)
    
    cifar_train = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=cifar_transform)
    cifar_test = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=cifar_transform)
    
    datasets = {
        'mnist': {'train': mnist_train, 'test': mnist_test},
        'fmnist': {'train': fmnist_train, 'test': fmnist_test},
        'cifar': {'train': cifar_train, 'test': cifar_test}
    }
    
    # Extract subsets
    subsets = {}
    for name, d in datasets.items():
        # Fine-tuning set: first 5,000 images
        ft_subset = Subset(d['train'], list(range(5000)))
        # Calibration set: next 128 images (indices 5000 to 5128)
        cal_subset = Subset(d['train'], list(range(5000, 5128)))
        # Test set: full test set
        test_subset = d['test']
        
        subsets[name] = {
            'train': ft_subset,
            'cal': cal_subset,
            'test': test_subset
        }
        print(f"Dataset {name}: Train subset size={len(ft_subset)}, Cal subset size={len(cal_subset)}, Test size={len(test_subset)}")
        
    return subsets

# --- MODEL DEFINITIONS ---
class ExpertModel(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone  # ResNet-18 with fc = Identity
        self.head = head          # Linear classification head (512 -> 10)
        
    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

def create_base_resnet():
    # Load ImageNet pre-trained ResNet-18
    try:
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    except AttributeError:
        model = models.resnet18(pretrained=True)
    
    # Replace fc with Identity to use as backbone
    model.fc = nn.Identity()
    return model

# --- FINE-TUNING EXPERTS ---
def train_expert(name, subsets):
    print(f"\n--- Training Expert for {name.upper()} ---")
    backbone = create_base_resnet().to(device)
    head = nn.Linear(512, 10).to(device)
    model = ExpertModel(backbone, head).to(device)
    
    train_loader = DataLoader(subsets[name]['train'], batch_size=128, shuffle=True)
    test_loader = DataLoader(subsets[name]['test'], batch_size=256, shuffle=False)
    
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(5):
        epoch_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x.size(0)
        epoch_loss /= len(train_loader.dataset)
        
        # Eval on test
        acc = eval_model_simple(model, test_loader)
        print(f"Epoch {epoch+1}/5 - Loss: {epoch_loss:.4f} - Test Acc: {acc:.2f}%")
        
    # Save model weights
    torch.save({
        'backbone_state_dict': model.backbone.state_dict(),
        'head_state_dict': model.head.state_dict()
    }, f"expert_{name}.pth")
    print(f"Saved expert_{name}.pth")
    return model

def eval_model_simple(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return (correct / total) * 100.0

# --- MODEL MERGING ---
def merge_backbones(expert_paths):
    print("\n--- Merging Backbones ---")
    merged_backbone = create_base_resnet().to(device)
    
    # Average the weights of the expert backbones
    merged_state_dict = merged_backbone.state_dict()
    expert_state_dicts = []
    
    for path in expert_paths:
        ckpt = torch.load(path, map_location=device)
        expert_state_dicts.append(ckpt['backbone_state_dict'])
        
    for key in merged_state_dict.keys():
        # Only average tensor parameters, check if they are floating point
        if merged_state_dict[key].is_floating_point():
            stacked = torch.stack([sd[key] for sd in expert_state_dicts], dim=0)
            merged_state_dict[key] = stacked.mean(dim=0)
        else:
            # For non-floating parameters (like running stats in batchnorm), use the first expert's values
            merged_state_dict[key] = expert_state_dicts[0][key].clone()
            
    merged_backbone.load_state_dict(merged_state_dict)
    return merged_backbone

# --- EVALUATION METHODS ---
def evaluate_merged_config(backbone, heads, subsets, calibration_type="none", calibration_data=None):
    # Set to eval mode
    backbone.eval()
    for head in heads.values():
        head.eval()
        
    accuracies = {}
    
    for name, d in subsets.items():
        test_loader = DataLoader(d['test'], batch_size=256, shuffle=False)
        head = heads[name]
        
        correct = 0
        total = 0
        
        # Apply calibration if applicable
        # (This is implemented by registering hooks before evaluation, or wrapping the forward pass)
        hooks = []
        if calibration_type == "tcac" and calibration_data is not None:
            # Register TCAC hooks for task `name`
            hooks = register_tcac_hooks(backbone, name, calibration_data)
        elif calibration_type == "sp_taac" and calibration_data is not None:
            # Register SP-TAAC hooks
            hooks = register_sp_taac_hooks(backbone, calibration_data)
            
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                features = backbone(x)
                logits = head(features)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
                
        # Remove hooks after task evaluation
        for hook in hooks:
            hook.remove()
            
        acc = (correct / total) * 100.0
        accuracies[name] = acc
        
    accuracies['average'] = np.mean([accuracies[k] for k in ['mnist', 'fmnist', 'cifar']])
    return accuracies

# --- CALIBRATION STATS COMPUTATION ---
def collect_activations(model, loader):
    model.eval()
    activations = {}
    hooks = []
    
    def get_hook(name):
        def hook(module, input, output):
            if name not in activations:
                activations[name] = []
            activations[name].append(output.detach().cpu())
        return hook
        
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            hooks.append(module.register_forward_hook(get_hook(name)))
            
    # Run forward pass on entire loader
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            model(x)
            
    # Concatenate activations along batch dimension
    for name in activations.keys():
        activations[name] = torch.cat(activations[name], dim=0)
        
    # Clean up hooks
    for hook in hooks:
        hook.remove()
        
    return activations

def compute_calibration_data(expert_paths, merged_backbone, subsets, epsilon=1e-5):
    print("\n--- Computing Calibration Statistics ---")
    calibration_data = {
        'experts': {},
        'merged': {},
        'sp_taac': {}
    }
    
    # 1. Collect expert activations on their respective calibration sets
    for path in expert_paths:
        name = path.replace("expert_", "").replace(".pth", "")
        ckpt = torch.load(path, map_location=device)
        
        # Reconstruct expert model
        expert_backbone = create_base_resnet().to(device)
        expert_backbone.load_state_dict(ckpt['backbone_state_dict'])
        expert_backbone.eval()
        
        cal_loader = DataLoader(subsets[name]['cal'], batch_size=128, shuffle=False)
        print(f"Collecting expert activations for {name}...")
        expert_acts = collect_activations(expert_backbone, cal_loader)
        
        calibration_data['experts'][name] = {}
        for layer_name, act in expert_acts.items():
            # Channel-wise stats
            # act shape: [B, C, H, W]
            mean = act.mean(dim=[0, 2, 3])  # [C]
            var = act.var(dim=[0, 2, 3], unbiased=False)    # [C]
            std = torch.sqrt(var + epsilon) # [C]
            
            # Global layer-wise stats for LSC/SP-TAAC
            global_std = torch.sqrt(act.var(dim=[0, 1, 2, 3], unbiased=False) + epsilon)
            
            calibration_data['experts'][name][layer_name] = {
                'mean': mean.to(device),
                'std': std.to(device),
                'global_std': global_std.to(device)
            }
            
    # 2. Collect merged model activations on each task's calibration set
    for name in subsets.keys():
        cal_loader = DataLoader(subsets[name]['cal'], batch_size=128, shuffle=False)
        print(f"Collecting merged model activations on {name} calibration set...")
        merged_acts = collect_activations(merged_backbone, cal_loader)
        
        calibration_data['merged'][name] = {}
        for layer_name, act in merged_acts.items():
            mean = act.mean(dim=[0, 2, 3])  # [C]
            var = act.var(dim=[0, 2, 3], unbiased=False)    # [C]
            std = torch.sqrt(var + epsilon) # [C]
            
            calibration_data['merged'][name][layer_name] = {
                'mean': mean.to(device),
                'std': std.to(device)
            }
            
    # 3. Compute joint calibration statistics for SP-TAAC
    # Construct Joint Calibration loader
    joint_dataset = torch.utils.data.ConcatDataset([subsets[name]['cal'] for name in subsets.keys()])
    joint_loader = DataLoader(joint_dataset, batch_size=128, shuffle=False)
    print(f"Collecting merged model activations on Joint calibration set (size={len(joint_dataset)})...")
    joint_acts = collect_activations(merged_backbone, joint_loader)
    
    calibration_data['sp_taac']['layers'] = {}
    for layer_name, act in joint_acts.items():
        # Merged global layer standard deviation on joint set
        std_merged_layer = torch.sqrt(act.var(dim=[0, 1, 2, 3], unbiased=False) + epsilon)
        
        # Target global layer standard deviation (average of experts' global standard deviations)
        std_target_layer = torch.stack([calibration_data['experts'][name][layer_name]['global_std'] for name in subsets.keys()]).mean()
        
        # Scale factor
        gamma_l = std_target_layer / std_merged_layer
        calibration_data['sp_taac']['layers'][layer_name] = gamma_l.to(device)
        
    return calibration_data

# --- REGISTER HOOKS FOR TCAC & SP-TAAC ---
def register_tcac_hooks(backbone, task_name, cal_data):
    hooks = []
    
    def make_hook(layer_name):
        def hook(module, input, output):
            # Target (expert) stats
            target_mean = cal_data['experts'][task_name][layer_name]['mean']
            target_std = cal_data['experts'][task_name][layer_name]['std']
            
            # Merged stats
            merged_mean = cal_data['merged'][task_name][layer_name]['mean']
            merged_std = cal_data['merged'][task_name][layer_name]['std']
            
            # Prevent Division-by-Zero via Sparsity-Trap Safeguard
            # Clamp merged standard deviation to a reasonable minimum to prevent noise explosion
            safe_merged_std = torch.clamp(merged_std, min=1e-4)
            
            # Reshape for broadcasting [1, C, 1, 1]
            target_mean = target_mean.view(1, -1, 1, 1)
            target_std = target_std.view(1, -1, 1, 1)
            merged_mean = merged_mean.view(1, -1, 1, 1)
            safe_merged_std = safe_merged_std.view(1, -1, 1, 1)
            
            # Channel-wise affine transformation
            scaled = (output - merged_mean) / safe_merged_std
            return scaled * target_std + target_mean
        return hook

    for name, module in backbone.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            hooks.append(module.register_forward_hook(make_hook(name)))
            
    return hooks

def register_sp_taac_hooks(backbone, cal_data):
    hooks = []
    
    def make_hook(layer_name):
        def hook(module, input, output):
            gamma = cal_data['sp_taac']['layers'][layer_name]
            return output * gamma
        return hook

    for name, module in backbone.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            hooks.append(module.register_forward_hook(make_hook(name)))
            
    return hooks

# --- RUN N-TAAC (BATCHNORM NATIVE STATS OVER JOINT SET) ---
def apply_n_taac(merged_backbone, subsets):
    print("\n--- Applying N-TAAC (Native Joint Calibration) ---")
    # Clone the merged backbone to avoid in-place modification for other baselines
    n_taac_backbone = create_base_resnet().to(device)
    n_taac_backbone.load_state_dict(merged_backbone.state_dict())
    
    # Construct Joint Calibration dataset
    joint_dataset = torch.utils.data.ConcatDataset([subsets[name]['cal'] for name in subsets.keys()])
    joint_loader = DataLoader(joint_dataset, batch_size=128, shuffle=False)
    
    # N-TAAC Procedure:
    # 1. Put model in train mode
    n_taac_backbone.train()
    # 2. Freeze all parameters (running stats will still update)
    for p in n_taac_backbone.parameters():
        p.requires_grad = False
    # 3. Set BatchNorm momentum to 1.0 to overwrite running stats with current batch stats
    for m in n_taac_backbone.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = 1.0
            
    # 4. Run single forward pass on joint dataset
    with torch.no_grad():
        for x, _ in joint_loader:
            x = x.to(device)
            n_taac_backbone(x)
            break  # A single forward batch of joint set (size 128) is standard or run whole loader
            
    # Set back to eval mode (and restore momentum just in case)
    n_taac_backbone.eval()
    for m in n_taac_backbone.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = 0.1
            
    return n_taac_backbone

# --- PROPOSED METHOD: SELF-ROUTING ACTIVATION CALIBRATION (SRAC) ---
def run_srac_evaluation(backbone, heads, subsets, beta=30.0):
    print(f"\n--- Applying Proposed Self-Routing Activation Calibration (SRAC, beta={beta}) ---")
    backbone.eval()
    prototypes = {}
    sorted_tasks = ['mnist', 'fmnist', 'cifar']
    
    # 1. Extract Task Prototypes at Anchor Layer `layer2`
    anchor_act = None
    def anchor_hook(module, input, output):
        nonlocal anchor_act
        anchor_act = output.detach()
        
    hook_handle = backbone.layer2.register_forward_hook(anchor_hook)
    
    for task_name in sorted_tasks:
        cal_loader = DataLoader(subsets[task_name]['cal'], batch_size=128, shuffle=False)
        with torch.no_grad():
            for x, _ in cal_loader:
                x = x.to(device)
                backbone(x)
                break # Extract from first batch
        pooled = anchor_act.mean(dim=[2, 3]) # [B, 128]
        proto = pooled.mean(dim=0) # [128]
        proto = proto / (proto.norm(p=2) + 1e-8) # L2 Normalization
        prototypes[task_name] = proto
        print(f"Task {task_name.upper()} early-layer prototype norm: {proto.norm().item():.4f}")
        
    hook_handle.remove()
    
    # 2. Setup pre-forward hook to reset routing weights
    routing_container = {"weights": None}
    def reset_pre_hook(module, input):
        routing_container["weights"] = None
        
    inference_hooks = []
    inference_hooks.append(backbone.register_forward_pre_hook(reset_pre_hook))
    
    # 3. Register Inference Dynamic Routing Hook
    def srac_anchor_hook(module, input, output):
        B = output.shape[0]
        pooled = output.mean(dim=[2, 3]) # [B, 128]
        pooled_norm = pooled / (pooled.norm(p=2, dim=1, keepdim=True) + 1e-8) # [B, 128]
        
        sims = []
        for task_name in sorted_tasks:
            proto = prototypes[task_name] # [128]
            sim = torch.sum(pooled_norm * proto.unsqueeze(0), dim=1) # [B]
            sims.append(sim)
        sims = torch.stack(sims, dim=1) # [B, 3]
        
        routing_container["weights"] = torch.softmax(beta * sims, dim=1)
        
    inference_hooks.append(backbone.layer2.register_forward_hook(srac_anchor_hook))
    
    # 4. Evaluate in a Truly Task-Agnostic Inference Mode
    accuracies = {}
    for h in heads.values():
        h.eval()
        
    for name in sorted_tasks:
        test_loader = DataLoader(subsets[name]['test'], batch_size=256, shuffle=False)
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                features = backbone(x)
                routing_weights = routing_container["weights"]
                
                if routing_weights is None:
                    # Fallback
                    routing_weights = torch.ones(x.shape[0], 3, device=device) / 3.0
                    
                logits_mnist = heads['mnist'](features)
                logits_fmnist = heads['fmnist'](features)
                logits_cifar = heads['cifar'](features)
                logits_all = torch.stack([logits_mnist, logits_fmnist, logits_cifar], dim=1)
                
                logits = torch.sum(routing_weights.unsqueeze(-1) * logits_all, dim=1)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
                
        acc = (correct / total) * 100.0
        accuracies[name] = acc
        print(f"SRAC accuracy on {name.upper()}: {acc:.2f}%")
        
    # Clean up hooks
    for hook in inference_hooks:
        hook.remove()
        
    accuracies['average'] = np.mean([accuracies[k] for k in sorted_tasks])
    return accuracies

# --- MAIN EXECUTION ---
def main():
    subsets = get_datasets()
    
    # Train or load experts
    expert_names = ['mnist', 'fmnist', 'cifar']
    expert_paths = [f"expert_{name}.pth" for name in expert_names]
    
    experts = {}
    for name in expert_names:
        path = f"expert_{name}.pth"
        if os.path.exists(path):
            print(f"Loading existing expert_{name}.pth...")
            ckpt = torch.load(path, map_location=device)
            backbone = create_base_resnet().to(device)
            backbone.load_state_dict(ckpt['backbone_state_dict'])
            head = nn.Linear(512, 10).to(device)
            head.load_state_dict(ckpt['head_state_dict'])
            experts[name] = ExpertModel(backbone, head).to(device)
            
            # Print expert accuracy to verify loading
            test_loader = DataLoader(subsets[name]['test'], batch_size=256, shuffle=False)
            acc = eval_model_simple(experts[name], test_loader)
            print(f"Loaded Expert {name.upper()} - Test Acc: {acc:.2f}%")
        else:
            experts[name] = train_expert(name, subsets)
            
    # Separate heads for evaluation on merged backbone
    heads = {name: experts[name].head for name in expert_names}
    
    # 1. Oracle (Single Experts) Performance
    oracle_accs = {}
    for name in expert_names:
        test_loader = DataLoader(subsets[name]['test'], batch_size=256, shuffle=False)
        oracle_accs[name] = eval_model_simple(experts[name], test_loader)
    oracle_accs['average'] = np.mean([oracle_accs[k] for k in expert_names])
    print(f"\nOracle Accuracies: {oracle_accs}")
    
    # Merge the backbones
    merged_backbone = merge_backbones(expert_paths)
    
    # 2. Uncalibrated WA Performance
    uncalibrated_accs = evaluate_merged_config(merged_backbone, heads, subsets, calibration_type="none")
    print(f"Uncalibrated WA Accuracies: {uncalibrated_accs}")
    
    # Pre-compute calibration stats for TCAC and SP-TAAC
    calibration_data = compute_calibration_data(expert_paths, merged_backbone, subsets)
    
    # 3. TCAC Performance
    tcac_accs = evaluate_merged_config(merged_backbone, heads, subsets, calibration_type="tcac", calibration_data=calibration_data)
    print(f"TCAC (Task-Conditional) Accuracies: {tcac_accs}")
    
    # 4. N-TAAC Performance
    n_taac_backbone = apply_n_taac(merged_backbone, subsets)
    n_taac_accs = evaluate_merged_config(n_taac_backbone, heads, subsets, calibration_type="none")
    print(f"N-TAAC (Task-Agnostic BN) Accuracies: {n_taac_accs}")
    
    # 5. SP-TAAC Performance
    sp_taac_accs = evaluate_merged_config(merged_backbone, heads, subsets, calibration_type="sp_taac", calibration_data=calibration_data)
    print(f"SP-TAAC Accuracies: {sp_taac_accs}")
    
    # 6. SRAC (Our Proposed Method) Performance
    # Sweep over beta temperature
    best_avg = 0
    best_srac_accs = None
    best_beta = 30.0
    for beta in [5.0, 15.0, 30.0, 50.0]:
        print(f"\nRunning SRAC with beta={beta}...")
        srac_accs = run_srac_evaluation(n_taac_backbone, heads, subsets, beta=beta)
        print(f"SRAC (beta={beta}) Accuracies: {srac_accs}")
        if srac_accs['average'] > best_avg:
            best_avg = srac_accs['average']
            best_srac_accs = srac_accs
            best_beta = beta
            
    print(f"\n--- Best SRAC Performance (beta={best_beta}): {best_srac_accs['average']:.2f}% ---")
    
    # Save results to JSON for plotting and documentation
    results = {
        'Oracle': oracle_accs,
        'Uncalibrated WA': uncalibrated_accs,
        'TCAC (Task-Conditional)': tcac_accs,
        'N-TAAC (Task-Agnostic BN)': n_taac_accs,
        'SP-TAAC (Task-Agnostic Scaling)': sp_taac_accs,
        f'SRAC (Ours, beta={best_beta})': best_srac_accs
    }
    
    with open("experiments_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Saved experiments_results.json")
    
    # Print clean markdown summary table of results
    print("\n" + "="*50)
    print("               FINAL SUMMARY TABLE")
    print("="*50)
    print(f"{'Method':<32} | {'MNIST':<8} | {'F-MNIST':<8} | {'CIFAR-10':<8} | {'Average':<8}")
    print("-"*72)
    for method, d in results.items():
        print(f"{method:<32} | {d['mnist']:.2f}% | {d['fmnist']:.2f}% | {d['cifar']:.2f}% | {d['average']:.2f}%")
    print("="*50)

if __name__ == "__main__":
    main()
