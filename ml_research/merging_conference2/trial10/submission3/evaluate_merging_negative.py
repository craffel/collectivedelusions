import torch
torch.backends.cudnn.enabled = False
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import os
import copy
import numpy as np

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Quantization helper
def quantize_tensor(tensor, bits=8):
    if bits is None or bits >= 32:
        return tensor
    qmin = -(2 ** (bits - 1))
    qmax = (2 ** (bits - 1)) - 1
    
    max_val = torch.max(torch.abs(tensor))
    if max_val == 0:
        return tensor
        
    scale = max_val / qmax
    q_tensor = torch.round(tensor / scale)
    q_tensor = torch.clamp(q_tensor, qmin, qmax)
    return q_tensor * scale

def get_quantized_model(model, bits=8):
    if bits is None or bits >= 32:
        return model
    q_model = copy.deepcopy(model)
    for name, param in q_model.named_parameters():
        param.data = quantize_tensor(param.data, bits)
    return q_model

# Activation-Driven Synaptic Resonance (ADSR) Model
class ADSRModel(nn.Module):
    def __init__(self, experts, temp=1.0, bits=None):
        super(ADSRModel, self).__init__()
        self.experts = nn.ModuleList([copy.deepcopy(exp) for exp in experts])
        self.temp = temp
        self.bits = bits
        
    def fuse_resonance(self, h_list):
        variances = []
        for h in h_list:
            var = torch.var(h, dim=(1, 2, 3), keepdim=True) # shape [Batch, 1, 1, 1]
            variances.append(var)
            
        vars_stacked = torch.stack(variances, dim=1)
        log_vars = torch.log(vars_stacked + 1e-6)
        
        alpha = torch.softmax(self.temp * log_vars, dim=1) # shape [Batch, 3, 1, 1, 1]
        
        h_stacked = torch.stack(h_list, dim=1)
        h_fused = torch.sum(alpha * h_stacked, dim=1)
        
        if self.bits is not None:
            h_fused = quantize_tensor(h_fused, self.bits)
            
        return h_fused

    def forward(self, x, task_id):
        # Initial layers
        h = [expert.conv1(x) for expert in self.experts]
        h = [self.experts[i].bn1(h[i]) for i in range(3)]
        h = [self.experts[i].relu(h[i]) for i in range(3)]
        h = [self.experts[i].maxpool(h[i]) for i in range(3)]
        
        h_fused = self.fuse_resonance(h)
        
        # Layer 1
        h = [self.experts[i].layer1(h_fused) for i in range(3)]
        h_fused = self.fuse_resonance(h)
        
        # Layer 2
        h = [self.experts[i].layer2(h_fused) for i in range(3)]
        h_fused = self.fuse_resonance(h)
        
        # Layer 3
        h = [self.experts[i].layer3(h_fused) for i in range(3)]
        h_fused = self.fuse_resonance(h)
        
        # Layer 4
        h = [self.experts[i].layer4(h_fused) for i in range(3)]
        h_fused = self.fuse_resonance(h)
        
        # Avgpool
        h_avg = self.experts[task_id].avgpool(h_fused)
        h_flat = torch.flatten(h_avg, 1)
        
        # FC
        out = self.experts[task_id].fc(h_flat)
        return out

# Load Datasets
print("Loading datasets...")
transform_cifar = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_mnist = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_mnist = torchvision.datasets.MNIST(root='./data', train=False, transform=transform_mnist)
test_fmnist = torchvision.datasets.FashionMNIST(root='./data', train=False, transform=transform_mnist)
test_cifar = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform_cifar)

batch_size = 128
loader_test_mnist = DataLoader(test_mnist, batch_size=batch_size, shuffle=False, num_workers=2)
loader_test_fmnist = DataLoader(test_fmnist, batch_size=batch_size, shuffle=False, num_workers=2)
loader_test_cifar = DataLoader(test_cifar, batch_size=batch_size, shuffle=False, num_workers=2)

loaders = [loader_test_mnist, loader_test_fmnist, loader_test_cifar]
task_names = ["MNIST", "FashionMNIST", "CIFAR-10"]

def evaluate_model(model, loader, device, task_id=None, noise_std=0.0):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if noise_std > 0:
                images = images + torch.randn_like(images) * noise_std
            
            if task_id is not None:
                outputs = model(images, task_id)
            else:
                outputs = model(images)
                
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100.0 * correct / total

def perform_evaluation():
    experts = []
    for task in ["mnist", "fmnist", "cifar10"]:
        chk_path = f"checkpoints/{task}_expert.pt"
        model = resnet18()
        model.fc = nn.Linear(model.fc.in_features, 10)
        model.load_state_dict(torch.load(chk_path, map_location=device))
        model = model.to(device)
        experts.append(model)
        
    print("\n--- Method 5 (Negative Temp): Activation-Driven Synaptic Resonance (ADSR) ---")
    temps = [-0.1, -0.5, -1.0, -2.0, -5.0, -10.0, -20.0]
    best_temp = -1.0
    best_avg = 0.0
    
    for t in temps:
        adsr_model = ADSRModel(experts, temp=t).to(device)
        adsr_accs = []
        for i, loader in enumerate(loaders):
            acc = evaluate_model(adsr_model, loader, device, task_id=i)
            adsr_accs.append(acc)
        avg_acc = np.mean(adsr_accs)
        print(f"ADSR (temp={t}) - MNIST: {adsr_accs[0]:.2f}%, FMNIST: {adsr_accs[1]:.2f}%, CIFAR10: {adsr_accs[2]:.2f}% | Avg: {avg_acc:.2f}%")
        if avg_acc > best_avg:
            best_avg = avg_acc
            best_temp = t

    print(f"\nBest temperature selected: {best_temp} with Avg Acc: {best_avg:.2f}%")

    print("\n\n=======================================================")
    print("        POST-TRAINING QUANTIZATION (PTQ) ANALYSIS      ")
    print("=======================================================")
    
    for bits in [8, 6, 4]:
        print(f"\n--- {bits}-bit PTQ Evaluation ---")
        
        # Quantized ADSR with best negative temp
        q_adsr_model = ADSRModel(experts, temp=best_temp, bits=bits).to(device)
        for exp in q_adsr_model.experts:
            for name, param in exp.named_parameters():
                param.data = quantize_tensor(param.data, bits)
                
        q_adsr_accs = []
        for i, loader in enumerate(loaders):
            acc = evaluate_model(q_adsr_model, loader, device, task_id=i)
            q_adsr_accs.append(acc)
        print(f"Quantized ADSR (temp={best_temp}) - MNIST: {q_adsr_accs[0]:.2f}%, FMNIST: {q_adsr_accs[1]:.2f}%, CIFAR10: {q_adsr_accs[2]:.2f}% | Avg: {np.mean(q_adsr_accs):.2f}%")

    print("\n\n=======================================================")
    print("        GAUSSIAN NOISE CORRUPTION ANALYSIS             ")
    print("=======================================================")
    
    for sigma in [0.1, 0.2, 0.3]:
        print(f"\n--- Gaussian Noise (std={sigma}) ---")
        
        # ADSR with best negative temp
        adsr_noise_model = ADSRModel(experts, temp=best_temp).to(device)
        adsr_noise_accs = []
        for i, loader in enumerate(loaders):
            acc = evaluate_model(adsr_noise_model, loader, device, task_id=i, noise_std=sigma)
            adsr_noise_accs.append(acc)
        print(f"ADSR (temp={best_temp}) with Noise - MNIST: {adsr_noise_accs[0]:.2f}%, FMNIST: {adsr_noise_accs[1]:.2f}%, CIFAR10: {adsr_noise_accs[2]:.2f}% | Avg: {np.mean(adsr_noise_accs):.2f}%")

if __name__ == "__main__":
    perform_evaluation()
