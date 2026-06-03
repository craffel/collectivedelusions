import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
import os
import copy
import numpy as np

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = False

class ToRGB(object):
    def __call__(self, pic):
        if pic.size(0) == 1:
            return pic.repeat(3, 1, 1)
        return pic

# Define identical dataloaders
def get_dataloaders(batch_size=128):
    gray_transform = transforms.Compose([
        transforms.Resize((32, 32)), # Resize to 32x32
        transforms.ToTensor(),
        ToRGB(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    color_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_mnist = datasets.MNIST(root="./data", train=False, download=True, transform=gray_transform)
    test_fmnist = datasets.FashionMNIST(root="./data", train=False, download=True, transform=gray_transform)
    test_cifar = datasets.CIFAR10(root="./data", train=False, download=True, transform=color_transform)

    # Also load small training subsets for real calibration (oracle)
    train_mnist = datasets.MNIST(root="./data", train=True, download=True, transform=gray_transform)
    train_fmnist = datasets.FashionMNIST(root="./data", train=True, download=True, transform=gray_transform)
    train_cifar = datasets.CIFAR10(root="./data", train=True, download=True, transform=color_transform)

    loaders = {
        "mnist": {
            "test": DataLoader(test_mnist, batch_size=batch_size, shuffle=False, num_workers=4),
            "calib": DataLoader(Subset(train_mnist, list(range(256))), batch_size=64, shuffle=False)
        },
        "fmnist": {
            "test": DataLoader(test_fmnist, batch_size=batch_size, shuffle=False, num_workers=4),
            "calib": DataLoader(Subset(train_fmnist, list(range(256))), batch_size=64, shuffle=False)
        },
        "cifar10": {
            "test": DataLoader(test_cifar, batch_size=batch_size, shuffle=False, num_workers=4),
            "calib": DataLoader(Subset(train_cifar, list(range(256))), batch_size=64, shuffle=False)
        }
    }
    return loaders

# Load model helper
def load_expert(name):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    path = f"models/expert_{name}.pt"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expert model path {path} does not exist. Please train experts first.")
    model.load_state_dict(torch.load(path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

# Reset BN running stats
def reset_bn_stats(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None # use exact cumulative moving average
            # Ensure stats are tracked
            m.track_running_stats = True

# BatchNorm calibration function
def calibrate_model(model, calib_loader, num_batches=10):
    model.train()
    # Freeze all parameters
    for p in model.parameters():
        p.requires_grad = False
        
    count = 0
    with torch.no_grad():
        for inputs, _ in calib_loader:
            inputs = inputs.to(device)
            _ = model(inputs)
            count += 1
            if count >= num_batches:
                break
    model.eval()
    # Restore standard momentum for subsequent evaluation steps
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = 0.1

# Generate Data-Free Calibration inputs
def get_data_free_loader(mode, size=256, batch_size=64):
    inputs_list = []
    
    if mode == "white_noise":
        # Pure gaussian noise
        for _ in range(size // batch_size):
            inputs_list.append((torch.randn(batch_size, 3, 32, 32), torch.zeros(batch_size)))
            
    elif mode == "pink_noise":
        # Spatially correlated noise via upsampling and blurring
        for _ in range(size // batch_size):
            # Generate small noise maps and upsample to 32x32 to create spatial correlations
            small_noise = torch.randn(batch_size, 3, 4, 4)
            upsampled = nn.functional.interpolate(small_noise, size=(32, 32), mode='bilinear', align_corners=False)
            # Add a bit of fine-grained white noise
            fine_noise = 0.1 * torch.randn(batch_size, 3, 32, 32)
            pink = upsampled + fine_noise
            # Normalize to match typical image statistics
            pink = (pink - pink.mean()) / (pink.std() + 1e-5)
            inputs_list.append((pink, torch.zeros(batch_size)))
            
    return inputs_list

def optimize_synthetic_data(expert_model, size=256, batch_size=64, epochs=150, lr=0.1):
    print(f"Optimizing synthetic data matching expert BN statistics ({epochs} epochs)...")
    expert_model.eval()
    
    # Initialize inputs as random normal on device, requires_grad=True
    inputs = torch.randn(size, 3, 32, 32, device=device, requires_grad=True)
    
    # Register hooks to collect layer inputs
    features = []
    def hook_fn(module, input, output):
        features.append(input[0])
        
    hooks = []
    bn_layers = []
    for m in expert_model.modules():
        if isinstance(m, nn.BatchNorm2d):
            hooks.append(m.register_forward_hook(hook_fn))
            bn_layers.append(m)
            
    optimizer = torch.optim.Adam([inputs], lr=lr)
    num_batches = size // batch_size
    
    # Helper for translation/jittering
    def jitter(x, amount=4):
        padded = nn.functional.pad(x, (amount, amount, amount, amount), mode='reflect')
        h_shift = np.random.randint(0, 2 * amount + 1)
        w_shift = np.random.randint(0, 2 * amount + 1)
        return padded[:, :, h_shift : h_shift + 32, w_shift : w_shift + 32]
        
    for epoch in range(epochs):
        epoch_loss = 0.0
        for b in range(num_batches):
            optimizer.zero_grad()
            features.clear()
            
            batch_inputs = inputs[b*batch_size : (b+1)*batch_size]
            
            # Apply jittering to force robust translation-invariant features
            jittered_inputs = jitter(batch_inputs)
            
            _ = expert_model(jittered_inputs)
            
            loss = 0.0
            for feat, bn in zip(features, bn_layers):
                batch_mean = feat.mean(dim=(0, 2, 3))
                batch_var = feat.var(dim=(0, 2, 3), unbiased=False)
                batch_std = torch.sqrt(batch_var + 1e-5)
                bn_std = torch.sqrt(bn.running_var + 1e-5)
                
                mean_loss = torch.mean((batch_mean - bn.running_mean) ** 2)
                std_loss = torch.mean((batch_std - bn_std) ** 2)
                
                loss += mean_loss + std_loss
                
            # Regularizers to ensure smooth, natural images and prevent high-frequency artifacts
            l2_loss = torch.mean(batch_inputs ** 2)
            diff_h = batch_inputs[:, :, 1:, :] - batch_inputs[:, :, :-1, :]
            diff_w = batch_inputs[:, :, :, 1:] - batch_inputs[:, :, :, :-1]
            tv_loss = torch.mean(diff_h ** 2) + torch.mean(diff_w ** 2)
            
            loss += 1e-4 * l2_loss + 1e-3 * tv_loss
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
    for h in hooks:
        h.remove()
        
    inputs_list = []
    with torch.no_grad():
        for b in range(num_batches):
            batch_inputs = inputs[b*batch_size : (b+1)*batch_size].detach()
            inputs_list.append((batch_inputs, torch.zeros(batch_size)))
            
    return inputs_list

# Evaluate model on a specific loader
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100.0 * correct / total

def merge_weights_wa(experts):
    # Perform Weight Averaging of the backbones, keep task-specific heads separate
    merged_model = models.resnet18(weights=None)
    merged_model.fc = nn.Linear(merged_model.fc.in_features, 10)
    merged_model = merged_model.to(device)
    
    merged_state = copy.deepcopy(experts[0].state_dict())
    expert_states = [e.state_dict() for e in experts]
    
    for key in merged_state.keys():
        # Only merge backbone, not the classification head fc.weight/fc.bias
        if "fc" not in key:
            # Average the weights across all experts
            weights = [state[key].float() for state in expert_states]
            merged_state[key] = torch.mean(torch.stack(weights), dim=0).to(merged_state[key].dtype)
            
    merged_model.load_state_dict(merged_state)
    return merged_model

def merge_weights_ta(experts, base_model, lam=0.5):
    # Perform Task Arithmetic model merging
    merged_model = models.resnet18(weights=None)
    merged_model.fc = nn.Linear(merged_model.fc.in_features, 10)
    merged_model = merged_model.to(device)
    
    base_state = base_model.state_dict()
    merged_state = copy.deepcopy(base_state)
    expert_states = [e.state_dict() for e in experts]
    
    for key in merged_state.keys():
        if "fc" not in key:
            # Task Vector = Expert - Base
            task_vectors = []
            for state in expert_states:
                tv = state[key].float() - base_state[key].float()
                task_vectors.append(tv)
            # Merged = Base + lam * Sum(Task Vectors)
            merged_state[key] = (base_state[key].float() + lam * torch.stack(task_vectors).sum(dim=0)).to(merged_state[key].dtype)
            
    merged_model.load_state_dict(merged_state)
    return merged_model

def run_experiment():
    print("Loading datasets...")
    loaders = get_dataloaders()
    
    print("Loading expert models...")
    expert_mnist = load_expert("mnist")
    expert_fmnist = load_expert("fmnist")
    expert_cifar = load_expert("cifar10")
    
    experts = [expert_mnist, expert_fmnist, expert_cifar]
    task_names = ["mnist", "fmnist", "cifar10"]
    
    print("\n--- Generating BN-Matching Data-Free Calibration Loaders ---")
    gen_loaders = {
        "mnist": optimize_synthetic_data(expert_mnist, size=256, batch_size=64, epochs=150, lr=0.1),
        "fmnist": optimize_synthetic_data(expert_fmnist, size=256, batch_size=64, epochs=150, lr=0.1),
        "cifar10": optimize_synthetic_data(expert_cifar, size=256, batch_size=64, epochs=150, lr=0.1)
    }
    
    # Get base progenitor
    print("Loading base progenitor...")
    base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    base_model.fc = nn.Linear(base_model.fc.in_features, 10)
    base_model = base_model.to(device)
    
    # 1. Evaluate Individual Experts (Oracle Baseline)
    print("\n=== Oracle Expert Baselines ===")
    expert_accs = {}
    for name, expert in zip(task_names, experts):
        acc = evaluate_model(expert, loaders[name]["test"])
        expert_accs[name] = acc
        print(f"Expert {name.upper()} Accuracy: {acc:.2f}%")
        
    # We will test merging with Weight Averaging and Task Arithmetic
    merge_methods = {
        "Weight Averaging (WA)": lambda: merge_weights_wa(experts),
        "Task Arithmetic (TA, lambda=0.3)": lambda: merge_weights_ta(experts, base_model, lam=0.3),
        "Task Arithmetic (TA, lambda=0.5)": lambda: merge_weights_ta(experts, base_model, lam=0.5),
        "Task Arithmetic (TA, lambda=0.7)": lambda: merge_weights_ta(experts, base_model, lam=0.7)
    }
    
    results_log = []
    
    for m_name, m_func in merge_methods.items():
        print(f"\n========================================\nMerging Method: {m_name}\n========================================")
        
        # Merge weights
        merged_model_raw = m_func()
        
        # Calibration methods to evaluate
        calib_modes = [
            ("No Calibration", "none"),
            ("Real Task-Specific Calib (Oracle)", "real_specific"),
            ("Real Joint Multi-Task Calib", "real_joint"),
            ("White-Noise Calib (Data-Free)", "white_noise"),
            ("Pink-Noise Calib (Data-Free)", "pink_noise"),
            ("OOD Calib: CIFAR for MNIST/FMNIST, MNIST for CIFAR (Data-Free)", "ood_proxy"),
            ("Generative BN-Matching Calib (Our DF-Calib)", "df_calib_gen")
        ]
        
        for c_name, c_mode in calib_modes:
            print(f"\n--- Calibration: {c_name} ---")
            
            # Evaluate on each task
            task_accs = {}
            
            for eval_task in task_names:
                # Prepare a fresh copy of the merged backbone
                model_to_eval = copy.deepcopy(merged_model_raw)
                
                # Attach the classification head of the target evaluation task
                # (since heads are task-specific and not merged)
                target_expert = expert_mnist if eval_task == "mnist" else (expert_fmnist if eval_task == "fmnist" else expert_cifar)
                model_to_eval.fc.load_state_dict(target_expert.fc.state_dict())
                
                # Perform calibration according to the mode
                if c_mode == "none":
                    pass
                elif c_mode == "real_specific":
                    reset_bn_stats(model_to_eval)
                    calibrate_model(model_to_eval, loaders[eval_task]["calib"])
                elif c_mode == "real_joint":
                    reset_bn_stats(model_to_eval)
                    # Joint calibration uses a mixture of all task calib loaders
                    joint_inputs = []
                    for name in task_names:
                        for inputs, _ in loaders[name]["calib"]:
                            joint_inputs.append((inputs, torch.zeros(inputs.size(0))))
                            if len(joint_inputs) >= 4: # limit size
                                break
                    calibrate_model(model_to_eval, joint_inputs)
                elif c_mode in ["white_noise", "pink_noise"]:
                    reset_bn_stats(model_to_eval)
                    df_loader = get_data_free_loader(c_mode, size=256, batch_size=64)
                    calibrate_model(model_to_eval, df_loader)
                elif c_mode == "ood_proxy":
                    reset_bn_stats(model_to_eval)
                    # For MNIST/FMNIST, use CIFAR-10 training data as calibration proxy
                    # For CIFAR-10, use MNIST training data as calibration proxy
                    proxy_task = "cifar10" if eval_task in ["mnist", "fmnist"] else "mnist"
                    calibrate_model(model_to_eval, loaders[proxy_task]["calib"])
                elif c_mode == "df_calib_gen":
                    reset_bn_stats(model_to_eval)
                    calibrate_model(model_to_eval, gen_loaders[eval_task])
                
                # Evaluate
                acc = evaluate_model(model_to_eval, loaders[eval_task]["test"])
                task_accs[eval_task] = acc
                print(f"Task {eval_task.upper()} Accuracy: {acc:.2f}% (Expert: {expert_accs[eval_task]:.2f}%, Gap: {expert_accs[eval_task]-acc:+.2f}%)")
            
            avg_acc = np.mean(list(task_accs.values()))
            print(f"Average Multi-Task Accuracy: {avg_acc:.2f}%")
            results_log.append({
                "merge_method": m_name,
                "calib_method": c_name,
                "mnist": task_accs["mnist"],
                "fmnist": task_accs["fmnist"],
                "cifar10": task_accs["cifar10"],
                "average": avg_acc
            })
            
    # Print comparison table at the end
    print("\n\n=======================================================")
    print("                  FINAL COMPARISON TABLE               ")
    print("=======================================================")
    print(f"{'Merge Method':<30} | {'Calibration Method':<30} | {'Avg Acc':<8}")
    print("-" * 75)
    for res in results_log:
        print(f"{res['merge_method'][:30]:<30} | {res['calib_method'][:30]:<30} | {res['average']:.2f}%")

if __name__ == "__main__":
    run_experiment()
