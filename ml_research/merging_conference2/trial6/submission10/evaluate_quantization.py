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

    # Small training subsets for real calibration (oracle)
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
        for b in range(num_batches):
            optimizer.zero_grad()
            features.clear()
            
            batch_inputs = inputs[b*batch_size : (b+1)*batch_size]
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
                
            l2_loss = torch.mean(batch_inputs ** 2)
            diff_h = batch_inputs[:, :, 1:, :] - batch_inputs[:, :, :-1, :]
            diff_w = batch_inputs[:, :, :, 1:] - batch_inputs[:, :, :, :-1]
            tv_loss = torch.mean(diff_h ** 2) + torch.mean(diff_w ** 2)
            
            loss += 1e-4 * l2_loss + 1e-3 * tv_loss
            
            loss.backward()
            optimizer.step()
            
    for h in hooks:
        h.remove()
        
    inputs_list = []
    with torch.no_grad():
        for b in range(num_batches):
            batch_inputs = inputs[b*batch_size : (b+1)*batch_size].detach()
            inputs_list.append((batch_inputs, torch.zeros(batch_size)))
            
    return inputs_list

# Evaluate model with optional FP16 or simulated weight-only INT8 quantization
def evaluate_model_quantized(model, test_loader, mode="fp32"):
    model_eval = copy.deepcopy(model)
    model_eval.eval()
    
    if mode == "fp16":
        model_eval = model_eval.half()
    elif mode == "int8":
        # Simulate symmetric weight-only INT8 quantization (per-channel)
        for name, module in model_eval.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                w = module.weight.data
                qmin, qmax = -128, 127
                for i in range(w.size(0)):
                    channel_w = w[i]
                    max_abs = channel_w.abs().max()
                    if max_abs == 0:
                        scale = 1.0
                    else:
                        scale = max_abs / 127.0
                    # Quantize and dequantize to simulate precision loss
                    q_w = torch.clamp(torch.round(channel_w / scale), qmin, qmax)
                    module.weight.data[i] = q_w * scale

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            if mode == "fp16":
                inputs = inputs.half()
            outputs = model_eval(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100.0 * correct / total

def merge_weights_wa(experts):
    merged_model = models.resnet18(weights=None)
    merged_model.fc = nn.Linear(merged_model.fc.in_features, 10)
    merged_model = merged_model.to(device)
    
    merged_state = copy.deepcopy(experts[0].state_dict())
    expert_states = [e.state_dict() for e in experts]
    
    for key in merged_state.keys():
        if "fc" not in key:
            weights = [state[key].float() for state in expert_states]
            merged_state[key] = torch.mean(torch.stack(weights), dim=0).to(merged_state[key].dtype)
            
    merged_model.load_state_dict(merged_state)
    return merged_model

def merge_weights_ta(experts, base_model, lam=0.5):
    merged_model = models.resnet18(weights=None)
    merged_model.fc = nn.Linear(merged_model.fc.in_features, 10)
    merged_model = merged_model.to(device)
    
    base_state = base_model.state_dict()
    merged_state = copy.deepcopy(base_state)
    expert_states = [e.state_dict() for e in experts]
    
    for key in merged_state.keys():
        if "fc" not in key:
            task_vectors = []
            for state in expert_states:
                tv = state[key].float() - base_state[key].float()
                task_vectors.append(tv)
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
    
    # We will test merging with Weight Averaging and Task Arithmetic (lambda=0.5)
    merge_methods = {
        "Weight Averaging (WA)": lambda: merge_weights_wa(experts),
        "Task Arithmetic (TA, lambda=0.5)": lambda: merge_weights_ta(experts, base_model, lam=0.5)
    }
    
    results = []
    
    for m_name, m_func in merge_methods.items():
        print(f"\n========================================\nMerging Method: {m_name}\n========================================")
        
        # Merge weights
        merged_model_raw = m_func()
        
        # Calibration methods to evaluate
        calib_modes = [
            ("No Calibration", "none"),
            ("Real Joint Multi-Task", "real_joint"),
            ("DF-Calib-Gen (Ours)", "df_calib_gen")
        ]
        
        for c_name, c_mode in calib_modes:
            print(f"\n--- Calibration: {c_name} ---")
            
            for p_mode in ["fp32", "fp16", "int8"]:
                task_accs = {}
                for eval_task in task_names:
                    model_to_eval = copy.deepcopy(merged_model_raw)
                    target_expert = expert_mnist if eval_task == "mnist" else (expert_fmnist if eval_task == "fmnist" else expert_cifar)
                    model_to_eval.fc.load_state_dict(target_expert.fc.state_dict())
                    
                    if c_mode == "none":
                        pass
                    elif c_mode == "real_joint":
                        reset_bn_stats(model_to_eval)
                        joint_inputs = []
                        for name in task_names:
                            for inputs, _ in loaders[name]["calib"]:
                                joint_inputs.append((inputs, torch.zeros(inputs.size(0))))
                                if len(joint_inputs) >= 4:
                                    break
                        calibrate_model(model_to_eval, joint_inputs)
                    elif c_mode == "df_calib_gen":
                        reset_bn_stats(model_to_eval)
                        calibrate_model(model_to_eval, gen_loaders[eval_task])
                    
                    # Evaluate under quantization/precision mode
                    acc = evaluate_model_quantized(model_to_eval, loaders[eval_task]["test"], mode=p_mode)
                    task_accs[eval_task] = acc
                
                avg_acc = np.mean(list(task_accs.values()))
                print(f"[{p_mode.upper()}] Average Accuracy: {avg_acc:.2f}% | MNIST: {task_accs['mnist']:.2f}%, FMNIST: {task_accs['fmnist']:.2f}%, CIFAR: {task_accs['cifar10']:.2f}%")
                
                results.append({
                    "merge": m_name,
                    "calib": c_name,
                    "precision": p_mode.upper(),
                    "mnist": task_accs["mnist"],
                    "fmnist": task_accs["fmnist"],
                    "cifar10": task_accs["cifar10"],
                    "average": avg_acc
                })
                
    # Save results to text file
    with open("quantization_results.txt", "w") as f:
        f.write("=== QUANTIZATION AND PRECISION EVALUATION RESULTS ===\n\n")
        f.write(f"{'Merge Method':<22} | {'Calibration':<22} | {'Precision':<10} | {'MNIST':<8} | {'FMNIST':<8} | {'CIFAR10':<8} | {'Average':<8}\n")
        f.write("-" * 95 + "\n")
        for r in results:
            f.write(f"{r['merge']:<22} | {r['calib']:<22} | {r['precision']:<10} | {r['mnist']:.2f}% | {r['fmnist']:.2f}% | {r['cifar10']:.2f}% | {r['average']:.2f}%\n")
            
    print("\nQuantization results saved to quantization_results.txt.")

if __name__ == "__main__":
    run_experiment()
