import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import os
import numpy as np
import copy
from merging_methods import merge_layerwise_scaling

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

class ActivationTracker:
    def __init__(self):
        self.activations = {}
        self.hooks = []

    def register_hooks(self, model):
        self.hooks.append(model.layer1.register_forward_hook(self._get_hook('layer1')))
        self.hooks.append(model.layer2.register_forward_hook(self._get_hook('layer2')))
        self.hooks.append(model.layer3.register_forward_hook(self._get_hook('layer3')))
        self.hooks.append(model.layer4.register_forward_hook(self._get_hook('layer4')))

    def _get_hook(self, name):
        def hook_fn(module, input, output):
            self.activations[name] = output.detach()
            return output
        return hook_fn

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
        self.activations = {}

def evaluate_model(state_dict):
    model = get_resnet18_base().to(device)
    model.load_state_dict(state_dict)
    model.eval()
    
    tracker = ActivationTracker()
    tracker.register_hooks(model)
    
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
                
                if i == 0:
                    for layer in ['layer1', 'layer2', 'layer3', 'layer4']:
                        act = tracker.activations[layer]
                        layer_stds[layer].append(act.std().item())
                        
        accs[name] = 100.0 * correct / total
        for layer in ['layer1', 'layer2', 'layer3', 'layer4']:
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
    
    schedules = [
        {
            "name": "Flat (scale=0.3)",
            "scales": {'layer1': 0.3, 'layer2': 0.3, 'layer3': 0.3, 'layer4': 0.3, 'default': 0.3}
        },
        {
            "name": "Linear Increase",
            "scales": {'layer1': 0.3, 'layer2': 0.33, 'layer3': 0.36, 'layer4': 0.4, 'default': 0.3}
        },
        {
            "name": "Focused Deep Scaling (LWS-1)",
            "scales": {'layer1': 0.3, 'layer2': 0.3, 'layer3': 0.4, 'layer4': 0.5, 'default': 0.3}
        },
        {
            "name": "Extreme Deep Scaling (LWS-2)",
            "scales": {'layer1': 0.3, 'layer2': 0.3, 'layer3': 0.45, 'layer4': 0.6, 'default': 0.3}
        },
        {
            "name": "Steep Deep Scaling (LWS-3)",
            "scales": {'layer1': 0.3, 'layer2': 0.3, 'layer3': 0.5, 'layer4': 0.7, 'default': 0.3}
        }
    ]
    
    print("\n=== Evaluating Layer-wise Weight Scaling (LWS) ===")
    lws_results = []
    for sched in schedules:
        print(f"\nEvaluating schedule: {sched['name']} with scales {sched['scales']}")
        lws_weights = merge_layerwise_scaling(base_weights, expert_weights_list, layer_scales=sched['scales'])
        accs, stds = evaluate_model(lws_weights)
        mean_acc = np.mean(list(accs.values()))
        print(f"Results: MNIST={accs['mnist']:.2f}%, Fashion={accs['fashion']:.2f}%, CIFAR={accs['cifar']:.2f}%, Mean={mean_acc:.2f}%")
        print(f"Activation Stds: {stds}")
        lws_results.append({
            "name": sched["name"],
            "scales": sched["scales"],
            "accs": accs,
            "mean_acc": mean_acc,
            "stds": stds
        })
        
    torch.save(lws_results, "lws_results.pt")
    print("\nLWS results saved to lws_results.pt")
