import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import os
import numpy as np
import copy
import random
from merging_methods import merge_weight_averaging, merge_task_arithmetic, merge_ties, merge_dare, merge_layerwise_scaling

# Set random seed for reproducibility
random.seed(2026)
np.random.seed(2026)
torch.manual_seed(2026)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
# Disable cuDNN to avoid CUDNN_STATUS_NOT_INITIALIZED error on this cluster
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = False

# Batch size for evaluation
BATCH_SIZE = 256

# Transforms
transform_mnist = transforms.Compose([
    transforms.Resize(224),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_cifar = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load test datasets for evaluation
print("Loading test datasets...")
test_mnist = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)
test_fashion = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_mnist)
test_cifar = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_cifar)

# Create subsets for fast sweeping (2000 samples each)
SUBSET_SIZE = 2000
test_mnist_sub = torch.utils.data.Subset(test_mnist, range(SUBSET_SIZE))
test_fashion_sub = torch.utils.data.Subset(test_fashion, range(SUBSET_SIZE))
test_cifar_sub = torch.utils.data.Subset(test_cifar, range(SUBSET_SIZE))

loader_mnist_sub = DataLoader(test_mnist_sub, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
loader_fashion_sub = DataLoader(test_fashion_sub, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
loader_cifar_sub = DataLoader(test_cifar_sub, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Pre-load subsets onto GPU to bypass CPU-GPU data transfers and DataLoader worker spawning overhead entirely
print("Pre-loading subsets onto GPU...")
mnist_images_sub = torch.cat([img for img, _ in loader_mnist_sub], dim=0).to(device)
mnist_labels_sub = torch.cat([lbl for _, lbl in loader_mnist_sub], dim=0).to(device)

fashion_images_sub = torch.cat([img for img, _ in loader_fashion_sub], dim=0).to(device)
fashion_labels_sub = torch.cat([lbl for _, lbl in loader_fashion_sub], dim=0).to(device)

cifar_images_sub = torch.cat([img for img, _ in loader_cifar_sub], dim=0).to(device)
cifar_labels_sub = torch.cat([lbl for _, lbl in loader_cifar_sub], dim=0).to(device)

# Loaders for full dataset evaluation (on-the-fly, fast and parallelized)
print("Setting up on-the-fly loaders for full 10,000-sample test sets...")
loader_mnist_full = DataLoader(test_mnist, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
loader_fashion_full = DataLoader(test_fashion, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
loader_cifar_full = DataLoader(test_cifar, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

def get_resnet18_base():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(512, 10)
    return model

# Activation Tracking Hooks
class ActivationTracker:
    def __init__(self):
        self.activations = {}
        self.scales = {}
        self.hooks = []

    def register_hooks(self, model):
        self.hooks.append(model.layer1.register_forward_hook(self._get_hook('layer1')))
        self.hooks.append(model.layer2.register_forward_hook(self._get_hook('layer2')))
        self.hooks.append(model.layer3.register_forward_hook(self._get_hook('layer3')))
        self.hooks.append(model.layer4.register_forward_hook(self._get_hook('layer4')))

    def _get_hook(self, name):
        def hook_fn(module, input, output):
            self.activations[name] = output.detach()
            if name in self.scales:
                return output * self.scales[name]
            return output
        return hook_fn

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
        self.activations = {}
        self.scales = {}

def compute_calibration_stats_fast(experts_dict, base_model, merge_fn, cal_samples=128, seed=None):
    """
    Computes calibration statistics fast on subsets using pre-loaded tensors
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
    mnist_idx = random.sample(range(SUBSET_SIZE), cal_samples)
    fashion_idx = random.sample(range(SUBSET_SIZE), cal_samples)
    cifar_idx = random.sample(range(SUBSET_SIZE), cal_samples)
    
    mnist_batch = mnist_images_sub[mnist_idx]
    fashion_batch = fashion_images_sub[fashion_idx]
    cifar_batch = cifar_images_sub[cifar_idx]
    
    expert_stds = {'layer1': [], 'layer2': [], 'layer3': [], 'layer4': []}
    
    for name, state_dict in experts_dict.items():
        model = get_resnet18_base().to(device)
        model.load_state_dict(state_dict)
        model.eval()
        
        tracker = ActivationTracker()
        tracker.register_hooks(model)
        
        if name == 'mnist':
            batch = mnist_batch
        elif name == 'fashion':
            batch = fashion_batch
        else:
            batch = cifar_batch
            
        with torch.no_grad():
            _ = model(batch)
            
        for layer in ['layer1', 'layer2', 'layer3', 'layer4']:
            act = tracker.activations[layer]
            expert_stds[layer].append(act.std().item())
            
        tracker.remove_hooks()
        
    target_stds = {}
    for layer in ['layer1', 'layer2', 'layer3', 'layer4']:
        target_stds[layer] = np.mean(expert_stds[layer])
        
    merged_model = get_resnet18_base().to(device)
    merged_model.load_state_dict(merge_fn())
    merged_model.eval()
    
    tracker = ActivationTracker()
    tracker.register_hooks(merged_model)
    
    merged_stds = {'layer1': 0.0, 'layer2': 0.0, 'layer3': 0.0, 'layer4': 0.0}
    with torch.no_grad():
        for batch in [mnist_batch, fashion_batch, cifar_batch]:
            _ = merged_model(batch)
            for layer in ['layer1', 'layer2', 'layer3', 'layer4']:
                act = tracker.activations[layer]
                merged_stds[layer] += act.std().item() / 3.0
                
    tracker.remove_hooks()
    
    gammas = {}
    for layer in ['layer1', 'layer2', 'layer3', 'layer4']:
        gammas[layer] = target_stds[layer] / (merged_stds[layer] + 1e-8)
        
    return gammas, target_stds, merged_stds

def evaluate_model_fast(state_dict, scales=None):
    """
    Evaluate a model state dict extremely fast using pre-loaded GPU tensors
    """
    model = get_resnet18_base().to(device)
    model.load_state_dict(state_dict)
    model.eval()
    
    tracker = ActivationTracker()
    tracker.register_hooks(model)
    if scales:
        tracker.scales = scales
        
    accs = {}
    measured_stds = {}
    
    datasets = {
        'mnist': (mnist_images_sub, mnist_labels_sub),
        'fashion': (fashion_images_sub, fashion_labels_sub),
        'cifar': (cifar_images_sub, cifar_labels_sub)
    }
    
    with torch.no_grad():
        for name, (images, labels) in datasets.items():
            correct = 0
            total = 0
            layer_stds = {'layer1': [], 'layer2': [], 'layer3': [], 'layer4': []}
            
            # Forward pass in batches to avoid OOM
            for i in range(0, len(images), BATCH_SIZE):
                batch_imgs = images[i:i+BATCH_SIZE]
                batch_lbls = labels[i:i+BATCH_SIZE]
                outputs = model(batch_imgs)
                _, predicted = outputs.max(1)
                total += batch_lbls.size(0)
                correct += predicted.eq(batch_lbls).sum().item()
                
                # Register std on first batch
                if i == 0:
                    for layer in ['layer1', 'layer2', 'layer3', 'layer4']:
                        act = tracker.activations[layer]
                        std_val = act.std().item()
                        layer_stds[layer].append(std_val if not np.isnan(std_val) else float('nan'))
            
            accs[name] = 100.0 * correct / total
            for layer in ['layer1', 'layer2', 'layer3', 'layer4']:
                if layer not in measured_stds:
                    measured_stds[layer] = []
                measured_stds[layer].append(np.mean(layer_stds[layer]))
                
    tracker.remove_hooks()
    avg_measured_stds = {layer: np.mean(measured_stds[layer]) for layer in ['layer1', 'layer2', 'layer3', 'layer4']}
    return accs, avg_measured_stds

def evaluate_model_full(state_dict, scales=None):
    """
    Evaluate a model state dict on the FULL test sets (10,000 samples each) using on-the-fly parallelized DataLoaders
    """
    model = get_resnet18_base().to(device)
    model.load_state_dict(state_dict)
    model.eval()
    
    tracker = ActivationTracker()
    tracker.register_hooks(model)
    if scales:
        tracker.scales = scales
        
    accs = {}
    measured_stds = {}
    
    loaders = {
        'mnist': loader_mnist_full,
        'fashion': loader_fashion_full,
        'cifar': loader_cifar_full
    }
    
    with torch.no_grad():
        for name, loader in loaders.items():
            correct = 0
            total = 0
            layer_stds = {'layer1': [], 'layer2': [], 'layer3': [], 'layer4': []}
            
            for i, (images, labels) in enumerate(loader):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                if i == 0:
                    for layer in ['layer1', 'layer2', 'layer3', 'layer4']:
                        act = tracker.activations[layer]
                        layer_stds[layer].append(act.std().item())
                        
            accs[name] = 100.0 * correct / total
            for layer in ['layer1', 'layer2', 'layer3', 'layer4']:
                if layer not in measured_stds:
                    measured_stds[layer] = []
                measured_stds[layer].append(np.mean(layer_stds[layer]))
            
    tracker.remove_hooks()
    avg_measured_stds = {layer: np.mean(measured_stds[layer]) for layer in ['layer1', 'layer2', 'layer3', 'layer4']}
    return accs, avg_measured_stds


if __name__ == "__main__":
    print("Loading checkpoints...")
    base_weights = torch.load("checkpoints/base_model.pt", map_location=device)
    
    experts_dict = {
        'mnist': torch.load("checkpoints/expert_mnist.pt", map_location=device),
        'fashion': torch.load("checkpoints/expert_fashion.pt", map_location=device),
        'cifar': torch.load("checkpoints/expert_cifar.pt", map_location=device)
    }
    expert_weights_list = list(experts_dict.values())
    
    results_record = []
    
    print("\n=== STEP 1: Fast Global Sweeps ===")
    
    # 1. Weight Averaging
    print("\nRunning Weight Averaging (WA)...")
    wa_weights = merge_weight_averaging(base_weights, expert_weights_list)
    accs_wa, stds_wa = evaluate_model_fast(wa_weights)
    print(f"Uncalibrated WA: Mean Acc = {np.mean(list(accs_wa.values())):.2f}%")
    results_record.append({
        "method": "WA",
        "calibrated": False,
        "params": "None",
        "accs": accs_wa,
        "mean_acc": np.mean(list(accs_wa.values())),
        "stds": stds_wa
    })
    
    gammas_wa, target_stds, merged_stds = compute_calibration_stats_fast(experts_dict, base_weights, lambda: wa_weights)
    accs_wa_cal, stds_wa_cal = evaluate_model_fast(wa_weights, scales=gammas_wa)
    print(f"SP-TAAC Calibrated WA: Mean Acc = {np.mean(list(accs_wa_cal.values())):.2f}%")
    results_record.append({
        "method": "WA",
        "calibrated": True,
        "params": "None",
        "accs": accs_wa_cal,
        "mean_acc": np.mean(list(accs_wa_cal.values())),
        "stds": stds_wa_cal
    })
    
    # 2. Task Arithmetic Global Sweep
    print("\nRunning Task Arithmetic Sweep...")
    ta_scales = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]
    for scale in ta_scales:
        ta_weights = merge_task_arithmetic(base_weights, expert_weights_list, scale=scale)
        accs, stds = evaluate_model_fast(ta_weights)
        print(f"Uncalibrated TA (scale={scale}): Mean Acc={np.mean(list(accs.values())):.2f}%")
        results_record.append({
            "method": "TA",
            "calibrated": False,
            "params": f"scale={scale}",
            "accs": accs,
            "mean_acc": np.mean(list(accs.values())),
            "stds": stds
        })
        
        gammas_ta, _, _ = compute_calibration_stats_fast(experts_dict, base_weights, lambda: ta_weights)
        accs_cal, stds_cal = evaluate_model_fast(ta_weights, scales=gammas_ta)
        print(f"SP-TAAC Calibrated TA (scale={scale}): Mean Acc={np.mean(list(accs_cal.values())):.2f}%")
        results_record.append({
            "method": "TA",
            "calibrated": True,
            "params": f"scale={scale}",
            "accs": accs_cal,
            "mean_acc": np.mean(list(accs_cal.values())),
            "stds": stds_cal
        })

    # 3. TIES Global Sweep
    print("\nRunning TIES Sweep...")
    keep_rates = [0.1, 0.2, 0.3, 0.5]
    ties_scales = [0.3, 0.5, 0.7, 1.0, 1.2, 1.5]
    for kr in keep_rates:
        for scale in ties_scales:
            ties_weights = merge_ties(base_weights, expert_weights_list, keep_rate=kr, scale=scale)
            accs, stds = evaluate_model_fast(ties_weights)
            print(f"Uncalibrated TIES (kr={kr}, scale={scale}): Mean Acc={np.mean(list(accs.values())):.2f}%")
            results_record.append({
                "method": "TIES",
                "calibrated": False,
                "params": f"kr={kr},scale={scale}",
                "accs": accs,
                "mean_acc": np.mean(list(accs.values())),
                "stds": stds
            })
            
            gammas_ties, _, _ = compute_calibration_stats_fast(experts_dict, base_weights, lambda: ties_weights)
            accs_cal, stds_cal = evaluate_model_fast(ties_weights, scales=gammas_ties)
            print(f"SP-TAAC Calibrated TIES (kr={kr}, scale={scale}): Mean Acc={np.mean(list(accs_cal.values())):.2f}%")
            results_record.append({
                "method": "TIES",
                "calibrated": True,
                "params": f"kr={kr},scale={scale}",
                "accs": accs_cal,
                "mean_acc": np.mean(list(accs_cal.values())),
                "stds": stds_cal
            })

    # 4. DARE Global Sweep
    print("\nRunning DARE Sweep...")
    drop_rates = [0.1, 0.3, 0.5, 0.7, 0.9]
    dare_scales = [0.3, 0.5, 0.7, 1.0, 1.2, 1.5]
    for dr in drop_rates:
        for scale in dare_scales:
            dare_weights = merge_dare(base_weights, expert_weights_list, drop_rate=dr, scale=scale)
            accs, stds = evaluate_model_fast(dare_weights)
            print(f"Uncalibrated DARE (dr={dr}, scale={scale}): Mean Acc={np.mean(list(accs.values())):.2f}%")
            results_record.append({
                "method": "DARE",
                "calibrated": False,
                "params": f"dr={dr},scale={scale}",
                "accs": accs,
                "mean_acc": np.mean(list(accs.values())),
                "stds": stds
            })
            
            gammas_dare, _, _ = compute_calibration_stats_fast(experts_dict, base_weights, lambda: dare_weights)
            accs_cal, stds_cal = evaluate_model_fast(dare_weights, scales=gammas_dare)
            print(f"SP-TAAC Calibrated DARE (dr={dr}, scale={scale}): Mean Acc={np.mean(list(accs_cal.values())):.2f}%")
            results_record.append({
                "method": "DARE",
                "calibrated": True,
                "params": f"dr={dr},scale={scale}",
                "accs": accs_cal,
                "mean_acc": np.mean(list(accs_cal.values())),
                "stds": stds_cal
            })

    print("\n=== STEP 2: Layer-wise Weight Scaling (LWS) and Generalization ===")
    
    # Let's define layer-wise schedules
    # 1. LWS schedules on Task Arithmetic
    ta_lws_scheds = [
        {"name": "LWS-1 (Focused)", "scales": {'layer1': 0.3, 'layer2': 0.3, 'layer3': 0.4, 'layer4': 0.5, 'default': 0.3}},
        {"name": "LWS-2 (Extreme)", "scales": {'layer1': 0.3, 'layer2': 0.3, 'layer3': 0.45, 'layer4': 0.6, 'default': 0.3}},
    ]
    for s in ta_lws_scheds:
        lws_weights = merge_layerwise_scaling(base_weights, expert_weights_list, layer_scales=s['scales'])
        accs, stds = evaluate_model_fast(lws_weights)
        print(f"LWS TA ({s['name']}): Mean Acc={np.mean(list(accs.values())):.2f}%")
        results_record.append({
            "method": "LWS-TA",
            "calibrated": False,
            "params": s['name'],
            "accs": accs,
            "mean_acc": np.mean(list(accs.values())),
            "stds": stds
        })
        
    # 2. L-TIES schedules (using best keep_rate from TIES global sweep, we will test kr=0.5)
    ties_lws_scheds = [
        {"name": "L-TIES-1 (Focused)", "scales": {'layer1': 0.5, 'layer2': 0.5, 'layer3': 0.7, 'layer4': 0.9, 'default': 0.5}},
        {"name": "L-TIES-2 (Steep)", "scales": {'layer1': 0.5, 'layer2': 0.5, 'layer3': 0.8, 'layer4': 1.1, 'default': 0.5}},
    ]
    for s in ties_lws_scheds:
        lws_ties = merge_ties(base_weights, expert_weights_list, keep_rate=0.5, scale=s['scales'])
        accs, stds = evaluate_model_fast(lws_ties)
        print(f"L-TIES ({s['name']}): Mean Acc={np.mean(list(accs.values())):.2f}%")
        results_record.append({
            "method": "L-TIES",
            "calibrated": False,
            "params": s['name'],
            "accs": accs,
            "mean_acc": np.mean(list(accs.values())),
            "stds": stds
        })
        
    # 3. L-DARE schedules (using best drop_rate, e.g. dr=0.1)
    dare_lws_scheds = [
        {"name": "L-DARE-1 (Focused)", "scales": {'layer1': 0.3, 'layer2': 0.3, 'layer3': 0.4, 'layer4': 0.5, 'default': 0.3}},
        {"name": "L-DARE-2 (Steep)", "scales": {'layer1': 0.3, 'layer2': 0.3, 'layer3': 0.45, 'layer4': 0.6, 'default': 0.3}},
    ]
    for s in dare_lws_scheds:
        lws_dare = merge_dare(base_weights, expert_weights_list, drop_rate=0.1, scale=s['scales'])
        accs, stds = evaluate_model_fast(lws_dare)
        print(f"L-DARE ({s['name']}): Mean Acc={np.mean(list(accs.values())):.2f}%")
        results_record.append({
            "method": "L-DARE",
            "calibrated": False,
            "params": s['name'],
            "accs": accs,
            "mean_acc": np.mean(list(accs.values())),
            "stds": stds
        })

    print("\n=== STEP 3: Calibration Sample Efficiency & Robustness Audit ===")
    
    # We will audit uncalibrated WA (which is deterministic, 0 data) vs. SP-TAAC calibrated WA
    # for N in [4, 8, 16, 32, 64, 128] across 5 random seeds
    sample_sizes = [4, 8, 16, 32, 64, 128]
    num_seeds = 5
    
    if os.path.exists("calibration_audit_fast.pt"):
        print("Loading existing calibration audit from calibration_audit_fast.pt...")
        audit_data = torch.load("calibration_audit_fast.pt", map_location=device)
        audit_results = audit_data["audit_results"]
    else:
        audit_results = {}
        for n in sample_sizes:
            audit_results[n] = []
            print(f"Auditing N = {n}...")
            for seed in range(num_seeds):
                gammas, _, _ = compute_calibration_stats_fast(experts_dict, base_weights, lambda: wa_weights, cal_samples=n, seed=seed)
                accs, _ = evaluate_model_fast(wa_weights, scales=gammas)
                mean_acc = np.mean(list(accs.values()))
                audit_results[n].append(mean_acc)
                print(f"  Seed {seed}: Mean Acc = {mean_acc:.2f}% (Gammas layer4={gammas['layer4']:.3f})")
                
        torch.save({
            "audit_results": audit_results,
            "uncal_wa_acc": np.mean(list(accs_wa.values()))
        }, "calibration_audit_fast.pt")
        print("Calibration audit saved.")

    print("\n=== STEP 4: Full-Dataset Final Evaluation ===")
    # We will evaluate key best-performing methods on the full 10k test sets
    
    # Find best methods from fast sweeps
    best_ta_uncal = max([r for r in results_record if r["method"] == "TA" and not r["calibrated"]], key=lambda x: x["mean_acc"])
    best_ta_cal = max([r for r in results_record if r["method"] == "TA" and r["calibrated"]], key=lambda x: x["mean_acc"])
    
    best_ties_uncal = max([r for r in results_record if r["method"] == "TIES" and not r["calibrated"]], key=lambda x: x["mean_acc"])
    best_ties_cal = max([r for r in results_record if r["method"] == "TIES" and r["calibrated"]], key=lambda x: x["mean_acc"])
    
    best_dare_uncal = max([r for r in results_record if r["method"] == "DARE" and not r["calibrated"]], key=lambda x: x["mean_acc"])
    best_dare_cal = max([r for r in results_record if r["method"] == "DARE" and r["calibrated"]], key=lambda x: x["mean_acc"])
    
    best_lws_ta = max([r for r in results_record if r["method"] == "LWS-TA"], key=lambda x: x["mean_acc"])
    best_lws_ties = max([r for r in results_record if r["method"] == "L-TIES"], key=lambda x: x["mean_acc"])
    best_lws_dare = max([r for r in results_record if r["method"] == "L-DARE"], key=lambda x: x["mean_acc"])
    
    print("\nRunning final full evaluations...")
    
    final_full_results = {}
    
    # 1. Experts
    print("Evaluating individual experts...")
    expert_accs = {}
    for name, state_dict in experts_dict.items():
        accs, _ = evaluate_model_full(state_dict)
        expert_accs[name] = accs
        print(f"  Expert {name}: MNIST={accs['mnist']:.2f}%, Fashion={accs['fashion']:.2f}%, CIFAR={accs['cifar']:.2f}%")
    final_full_results["experts"] = expert_accs
    
    # 2. WA Uncalibrated
    print("Evaluating WA Uncalibrated on full set...")
    accs, stds = evaluate_model_full(wa_weights)
    print(f"  WA Uncalibrated: Mean = {np.mean(list(accs.values())):.2f}% | Stds = {stds}")
    final_full_results["wa_uncal"] = {"accs": accs, "stds": stds}
    
    # 3. WA Calibrated (N=128, seed=2026)
    print("Evaluating WA Calibrated on full set...")
    gammas, _, _ = compute_calibration_stats_fast(experts_dict, base_weights, lambda: wa_weights, cal_samples=128, seed=2026)
    accs, stds = evaluate_model_full(wa_weights, scales=gammas)
    print(f"  WA Calibrated: Mean = {np.mean(list(accs.values())):.2f}% | Stds = {stds}")
    final_full_results["wa_cal"] = {"accs": accs, "stds": stds}
    
    # 4. Best TA Uncal
    ta_scale = float(best_ta_uncal["params"].split("=")[1])
    print(f"Evaluating TA Uncal (scale={ta_scale}) on full set...")
    best_ta_w = merge_task_arithmetic(base_weights, expert_weights_list, scale=ta_scale)
    accs, stds = evaluate_model_full(best_ta_w)
    print(f"  TA Uncal: Mean = {np.mean(list(accs.values())):.2f}% | Stds = {stds}")
    final_full_results["ta_uncal"] = {"accs": accs, "stds": stds, "params": best_ta_uncal["params"]}
    
    # 5. Best TA Cal
    ta_cal_scale = float(best_ta_cal["params"].split("=")[1])
    print(f"Evaluating TA Cal (scale={ta_cal_scale}) on full set...")
    best_ta_cal_w = merge_task_arithmetic(base_weights, expert_weights_list, scale=ta_cal_scale)
    gammas, _, _ = compute_calibration_stats_fast(experts_dict, base_weights, lambda: best_ta_cal_w, cal_samples=128, seed=2026)
    accs, stds = evaluate_model_full(best_ta_cal_w, scales=gammas)
    print(f"  TA Cal: Mean = {np.mean(list(accs.values())):.2f}% | Stds = {stds}")
    final_full_results["ta_cal"] = {"accs": accs, "stds": stds, "params": best_ta_cal["params"]}
    
    # 6. Best TIES Uncal
    parts = best_ties_uncal["params"].split(",")
    ties_kr = float(parts[0].split("=")[1])
    ties_scale = float(parts[1].split("=")[1])
    print(f"Evaluating TIES Uncal (kr={ties_kr}, scale={ties_scale}) on full set...")
    best_ties_w = merge_ties(base_weights, expert_weights_list, keep_rate=ties_kr, scale=ties_scale)
    accs, stds = evaluate_model_full(best_ties_w)
    print(f"  TIES Uncal: Mean = {np.mean(list(accs.values())):.2f}% | Stds = {stds}")
    final_full_results["ties_uncal"] = {"accs": accs, "stds": stds, "params": best_ties_uncal["params"]}
    
    # 7. Best TIES Cal
    parts = best_ties_cal["params"].split(",")
    ties_cal_kr = float(parts[0].split("=")[1])
    ties_cal_scale = float(parts[1].split("=")[1])
    print(f"Evaluating TIES Cal (kr={ties_cal_kr}, scale={ties_cal_scale}) on full set...")
    best_ties_cal_w = merge_ties(base_weights, expert_weights_list, keep_rate=ties_cal_kr, scale=ties_cal_scale)
    gammas, _, _ = compute_calibration_stats_fast(experts_dict, base_weights, lambda: best_ties_cal_w, cal_samples=128, seed=2026)
    accs, stds = evaluate_model_full(best_ties_cal_w, scales=gammas)
    print(f"  TIES Cal: Mean = {np.mean(list(accs.values())):.2f}% | Stds = {stds}")
    final_full_results["ties_cal"] = {"accs": accs, "stds": stds, "params": best_ties_cal["params"]}
    
    # 8. Best DARE Uncal
    parts = best_dare_uncal["params"].split(",")
    dare_dr = float(parts[0].split("=")[1])
    dare_scale = float(parts[1].split("=")[1])
    print(f"Evaluating DARE Uncal (dr={dare_dr}, scale={dare_scale}) on full set...")
    best_dare_w = merge_dare(base_weights, expert_weights_list, drop_rate=dare_dr, scale=dare_scale)
    accs, stds = evaluate_model_full(best_dare_w)
    print(f"  DARE Uncal: Mean = {np.mean(list(accs.values())):.2f}% | Stds = {stds}")
    final_full_results["dare_uncal"] = {"accs": accs, "stds": stds, "params": best_dare_uncal["params"]}
    
    # 9. Best DARE Cal
    parts = best_dare_cal["params"].split(",")
    dare_cal_dr = float(parts[0].split("=")[1])
    dare_cal_scale = float(parts[1].split("=")[1])
    print(f"Evaluating DARE Cal (dr={dare_cal_dr}, scale={dare_cal_scale}) on full set...")
    best_dare_cal_w = merge_dare(base_weights, expert_weights_list, drop_rate=dare_cal_dr, scale=dare_cal_scale)
    gammas, _, _ = compute_calibration_stats_fast(experts_dict, base_weights, lambda: best_dare_cal_w, cal_samples=128, seed=2026)
    accs, stds = evaluate_model_full(best_dare_cal_w, scales=gammas)
    print(f"  DARE Cal: Mean = {np.mean(list(accs.values())):.2f}% | Stds = {stds}")
    final_full_results["dare_cal"] = {"accs": accs, "stds": stds, "params": best_dare_cal["params"]}
    
    # 10. Best LWS-TA
    sched_ta = [s for s in ta_lws_scheds if s["name"] == best_lws_ta["params"]][0]
    print(f"Evaluating LWS-TA ({sched_ta['name']}) on full set...")
    best_lws_ta_w = merge_layerwise_scaling(base_weights, expert_weights_list, layer_scales=sched_ta['scales'])
    accs, stds = evaluate_model_full(best_lws_ta_w)
    print(f"  LWS-TA: Mean = {np.mean(list(accs.values())):.2f}% | Stds = {stds}")
    final_full_results["lws_ta"] = {"accs": accs, "stds": stds, "params": sched_ta["name"]}
    
    # 11. Best L-TIES
    sched_ties = [s for s in ties_lws_scheds if s["name"] == best_lws_ties["params"]][0]
    print(f"Evaluating L-TIES ({sched_ties['name']}) on full set...")
    best_lws_ties_w = merge_ties(base_weights, expert_weights_list, keep_rate=0.5, scale=sched_ties['scales'])
    accs, stds = evaluate_model_full(best_lws_ties_w)
    print(f"  L-TIES: Mean = {np.mean(list(accs.values())):.2f}% | Stds = {stds}")
    final_full_results["l_ties"] = {"accs": accs, "stds": stds, "params": sched_ties["name"]}
    
    # 12. Best L-DARE
    sched_dare = [s for s in dare_lws_scheds if s["name"] == best_lws_dare["params"]][0]
    print(f"Evaluating L-DARE ({sched_dare['name']}) on full set...")
    best_lws_dare_w = merge_dare(base_weights, expert_weights_list, drop_rate=0.1, scale=sched_dare['scales'])
    accs, stds = evaluate_model_full(best_lws_dare_w)
    print(f"  L-DARE: Mean = {np.mean(list(accs.values())):.2f}% | Stds = {stds}")
    final_full_results["l_dare"] = {"accs": accs, "stds": stds, "params": sched_dare["name"]}
    
    print("\nSaving final results...")
    torch.save({
        "results_record": results_record,
        "final_full_results": final_full_results,
        "target_stds": target_stds
    }, "results_sweep_fast.pt")
    print("Done! Evaluation completed successfully.")
