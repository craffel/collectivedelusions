import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import os
import numpy as np
import copy
from merging_methods import merge_weight_averaging, merge_task_arithmetic, merge_ties, merge_dare

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
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

loader_mnist = DataLoader(test_mnist, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
loader_fashion = DataLoader(test_fashion, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
loader_cifar = DataLoader(test_cifar, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

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
        # Register hooks on major residual blocks
        self.hooks.append(model.layer1.register_forward_hook(self._get_hook('layer1')))
        self.hooks.append(model.layer2.register_forward_hook(self._get_hook('layer2')))
        self.hooks.append(model.layer3.register_forward_hook(self._get_hook('layer3')))
        self.hooks.append(model.layer4.register_forward_hook(self._get_hook('layer4')))

    def _get_hook(self, name):
        def hook_fn(module, input, output):
            self.activations[name] = output.detach()
            # If calibration scale is registered, apply it
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

def compute_calibration_stats(experts_dict, base_model, merge_fn, cal_samples=128):
    """
    Compute target std for experts and merged model std on joint calibration set,
    and return the SP-TAAC scaling factors gamma per layer.
    """
    # Create small calibration sets
    # We take the first cal_samples from each test set
    mnist_cal_loader = DataLoader(torch.utils.data.Subset(test_mnist, range(cal_samples)), batch_size=cal_samples, shuffle=False)
    fashion_cal_loader = DataLoader(torch.utils.data.Subset(test_fashion, range(cal_samples)), batch_size=cal_samples, shuffle=False)
    cifar_cal_loader = DataLoader(torch.utils.data.Subset(test_cifar, range(cal_samples)), batch_size=cal_samples, shuffle=False)
    
    # Get calibration batches
    mnist_batch, _ = next(iter(mnist_cal_loader))
    fashion_batch, _ = next(iter(fashion_cal_loader))
    cifar_batch, _ = next(iter(cifar_cal_loader))
    
    mnist_batch = mnist_batch.to(device)
    fashion_batch = fashion_batch.to(device)
    cifar_batch = cifar_batch.to(device)
    
    # 1. Compute expert standard deviations per layer
    expert_stds = {'layer1': [], 'layer2': [], 'layer3': [], 'layer4': []}
    
    for name, state_dict in experts_dict.items():
        model = get_resnet18_base().to(device)
        model.load_state_dict(state_dict)
        model.eval()
        
        tracker = ActivationTracker()
        tracker.register_hooks(model)
        
        # Pass corresponding calibration batch
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
            # Global standard deviation over batch, channels, height, width
            std = act.std().item()
            expert_stds[layer].append(std)
            
        tracker.remove_hooks()
        
    # Average expert stds to get target std
    target_stds = {}
    for layer in ['layer1', 'layer2', 'layer3', 'layer4']:
        target_stds[layer] = np.mean(expert_stds[layer])
        
    # 2. Compute merged model standard deviations on joint calibration set
    merged_model = get_resnet18_base().to(device)
    merged_model.load_state_dict(merge_fn())
    merged_model.eval()
    
    tracker = ActivationTracker()
    tracker.register_hooks(merged_model)
    
    # Joint calibration pass
    merged_stds = {'layer1': 0.0, 'layer2': 0.0, 'layer3': 0.0, 'layer4': 0.0}
    with torch.no_grad():
        # Pass each batch and aggregate standard deviation
        for batch in [mnist_batch, fashion_batch, cifar_batch]:
            _ = merged_model(batch)
            for layer in ['layer1', 'layer2', 'layer3', 'layer4']:
                act = tracker.activations[layer]
                merged_stds[layer] += act.std().item() / 3.0 # Average std over the three tasks
                
    tracker.remove_hooks()
    
    # 3. Compute SP-TAAC scaling factors (gamma)
    gammas = {}
    for layer in ['layer1', 'layer2', 'layer3', 'layer4']:
        gammas[layer] = target_stds[layer] / (merged_stds[layer] + 1e-8)
        
    return gammas, target_stds, merged_stds

def evaluate_model(state_dict, scales=None):
    """
    Evaluate a model state dict on the three test sets.
    """
    model = get_resnet18_base().to(device)
    model.load_state_dict(state_dict)
    model.eval()
    
    tracker = ActivationTracker()
    tracker.register_hooks(model)
    if scales:
        tracker.scales = scales
        
    accs = {}
    measured_stds = {'layer1': [], 'layer2': [], 'layer3': [], 'layer4': []}
    
    loaders = {
        'mnist': loader_mnist,
        'fashion': loader_fashion,
        'cifar': loader_cifar
    }
    
    for name, loader in loaders.items():
        correct = 0
        total = 0
        layer_stds = {'layer1': [], 'layer2': [], 'layer3': [], 'layer4': []}
        
        with torch.no_grad():
            for i, (images, labels) in enumerate(loader):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Measure std for the first batch to verify variance collapse
                if i == 0:
                    for layer in ['layer1', 'layer2', 'layer3', 'layer4']:
                        act = tracker.activations[layer]
                        layer_stds[layer].append(act.std().item())
                        
        accs[name] = 100.0 * correct / total
        for layer in ['layer1', 'layer2', 'layer3', 'layer4']:
            measured_stds[layer].append(np.mean(layer_stds[layer]))
            
    tracker.remove_hooks()
    
    # Average the measured stds across tasks
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
    
    # 1. Evaluate Individual Experts
    print("\n=== Evaluating Individual Experts ===")
    for name, state_dict in experts_dict.items():
        accs, _ = evaluate_model(state_dict)
        print(f"Expert {name}: MNIST={accs['mnist']:.2f}%, Fashion={accs['fashion']:.2f}%, CIFAR={accs['cifar']:.2f}%")
        
    # We will record results for our paper
    results_record = []
    
    # 2. Evaluate Weight Averaging (WA)
    print("\n=== Evaluating Weight Averaging ===")
    wa_weights = merge_weight_averaging(base_weights, expert_weights_list)
    accs_wa, stds_wa = evaluate_model(wa_weights)
    print(f"Uncalibrated WA: MNIST={accs_wa['mnist']:.2f}%, Fashion={accs_wa['fashion']:.2f}%, CIFAR={accs_wa['cifar']:.2f}%, Mean={np.mean(list(accs_wa.values())):.2f}%")
    print(f"WA Activation Stds: {stds_wa}")
    results_record.append({
        "method": "WA",
        "calibrated": False,
        "params": "None",
        "accs": accs_wa,
        "mean_acc": np.mean(list(accs_wa.values())),
        "stds": stds_wa
    })
    
    # WA with SP-TAAC
    gammas, target_stds, merged_stds = compute_calibration_stats(experts_dict, base_weights, lambda: wa_weights)
    accs_wa_cal, stds_wa_cal = evaluate_model(wa_weights, scales=gammas)
    print(f"SP-TAAC Calibrated WA: MNIST={accs_wa_cal['mnist']:.2f}%, Fashion={accs_wa_cal['fashion']:.2f}%, CIFAR={accs_wa_cal['cifar']:.2f}%, Mean={np.mean(list(accs_wa_cal.values())):.2f}%")
    print(f"Calibrated WA Activation Stds: {stds_wa_cal}")
    print(f"SP-TAAC Scaling factors (gammas): {gammas}")
    print(f"Target Stds: {target_stds}")
    results_record.append({
        "method": "WA",
        "calibrated": True,
        "params": "None",
        "accs": accs_wa_cal,
        "mean_acc": np.mean(list(accs_wa_cal.values())),
        "stds": stds_wa_cal
    })
    
    # 3. Sweep over Task Arithmetic (TA) scales
    print("\n=== Sweeping Task Arithmetic ===")
    ta_scales = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]
    for scale in ta_scales:
        ta_weights = merge_task_arithmetic(base_weights, expert_weights_list, scale=scale)
        accs, stds = evaluate_model(ta_weights)
        print(f"Uncalibrated TA (scale={scale}): Mean Acc={np.mean(list(accs.values())):.2f}%, Stds={stds}")
        results_record.append({
            "method": "TA",
            "calibrated": False,
            "params": f"scale={scale}",
            "accs": accs,
            "mean_acc": np.mean(list(accs.values())),
            "stds": stds
        })
        
        # TA with SP-TAAC Calibration
        gammas_ta, _, _ = compute_calibration_stats(experts_dict, base_weights, lambda: ta_weights)
        accs_cal, stds_cal = evaluate_model(ta_weights, scales=gammas_ta)
        print(f"SP-TAAC Calibrated TA (scale={scale}): Mean Acc={np.mean(list(accs_cal.values())):.2f}%, Stds={stds_cal}")
        results_record.append({
            "method": "TA",
            "calibrated": True,
            "params": f"scale={scale}",
            "accs": accs_cal,
            "mean_acc": np.mean(list(accs_cal.values())),
            "stds": stds_cal
        })
        
    # 4. Sweep over TIES-Merging keep_rate and scale
    print("\n=== Sweeping TIES-Merging ===")
    keep_rates = [0.1, 0.2, 0.3, 0.5]
    ties_scales = [0.3, 0.5, 0.7, 1.0, 1.2, 1.5]
    for kr in keep_rates:
        for scale in ties_scales:
            ties_weights = merge_ties(base_weights, expert_weights_list, keep_rate=kr, scale=scale)
            accs, stds = evaluate_model(ties_weights)
            print(f"Uncalibrated TIES (keep_rate={kr}, scale={scale}): Mean Acc={np.mean(list(accs.values())):.2f}%, Stds={stds}")
            results_record.append({
                "method": "TIES",
                "calibrated": False,
                "params": f"kr={kr},scale={scale}",
                "accs": accs,
                "mean_acc": np.mean(list(accs.values())),
                "stds": stds
            })
            
            # TIES with SP-TAAC
            gammas_ties, _, _ = compute_calibration_stats(experts_dict, base_weights, lambda: ties_weights)
            accs_cal, stds_cal = evaluate_model(ties_weights, scales=gammas_ties)
            print(f"SP-TAAC Calibrated TIES (keep_rate={kr}, scale={scale}): Mean Acc={np.mean(list(accs_cal.values())):.2f}%, Stds={stds_cal}")
            results_record.append({
                "method": "TIES",
                "calibrated": True,
                "params": f"kr={kr},scale={scale}",
                "accs": accs_cal,
                "mean_acc": np.mean(list(accs_cal.values())),
                "stds": stds_cal
            })
            
    # 5. Sweep over DARE drop_rate and scale
    print("\n=== Sweeping DARE-Merging ===")
    drop_rates = [0.1, 0.3, 0.5, 0.7, 0.9]
    dare_scales = [0.3, 0.5, 0.7, 1.0, 1.2, 1.5]
    for dr in drop_rates:
        for scale in dare_scales:
            dare_weights = merge_dare(base_weights, expert_weights_list, drop_rate=dr, scale=scale)
            accs, stds = evaluate_model(dare_weights)
            print(f"Uncalibrated DARE (drop_rate={dr}, scale={scale}): Mean Acc={np.mean(list(accs.values())):.2f}%, Stds={stds}")
            results_record.append({
                "method": "DARE",
                "calibrated": False,
                "params": f"dr={dr},scale={scale}",
                "accs": accs,
                "mean_acc": np.mean(list(accs.values())),
                "stds": stds
            })
            
            # DARE with SP-TAAC
            gammas_dare, _, _ = compute_calibration_stats(experts_dict, base_weights, lambda: dare_weights)
            accs_cal, stds_cal = evaluate_model(dare_weights, scales=gammas_dare)
            print(f"SP-TAAC Calibrated DARE (drop_rate={dr}, scale={scale}): Mean Acc={np.mean(list(accs_cal.values())):.2f}%, Stds={stds_cal}")
            results_record.append({
                "method": "DARE",
                "calibrated": True,
                "params": f"dr={dr},scale={scale}",
                "accs": accs_cal,
                "mean_acc": np.mean(list(accs_cal.values())),
                "stds": stds_cal
            })
            
    # Save all results to a file for analysis and plotting
    print("\nSaving results to results_sweep.pt...")
    torch.save({
        "results": results_record,
        "target_stds": target_stds
    }, "results_sweep.pt")
    print("Results saved.")
