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
    # Symmetric uniform quantization
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
        # Do not quantize classification head if standard in PTQ (or quantize it, let's quantize everything)
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
        # h_list is a list of 3 tensors of shape [Batch, Channels, Height, Width]
        # Compute spatial and channel variance of activations for each expert
        variances = []
        for h in h_list:
            # Calculate variance across channel and spatial dimensions per batch item
            var = torch.var(h, dim=(1, 2, 3), keepdim=True) # shape [Batch, 1, 1, 1]
            variances.append(var)
            
        # Stack and apply softmax along the expert dimension
        # variances: list of 3 tensors, stack to [Batch, 3, 1, 1, 1]
        vars_stacked = torch.stack(variances, dim=1)
        # Log-variance to scale numerically
        log_vars = torch.log(vars_stacked + 1e-6)
        
        # Softmax to get resonance weights
        alpha = torch.softmax(self.temp * log_vars, dim=1) # shape [Batch, 3, 1, 1, 1]
        
        # Stack activations: [Batch, 3, Channels, Height, Width]
        h_stacked = torch.stack(h_list, dim=1)
        
        # Fused activations: weighted sum
        h_fused = torch.sum(alpha * h_stacked, dim=1)
        
        # Apply activation quantization if bits is specified
        if self.bits is not None:
            h_fused = quantize_tensor(h_fused, self.bits)
            
        return h_fused

    def forward(self, x, task_id):
        # Initial layers
        h = [expert.conv1(x) for expert in self.experts]
        h = [self.experts[i].bn1(h[i]) for i in range(3)]
        h = [self.experts[i].relu(h[i]) for i in range(3)]
        h = [self.experts[i].maxpool(h[i]) for i in range(3)]
        
        # Fuse at maxpool
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
        
        # FC (Task head)
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
                # For ADSR which requires task_id
                outputs = model(images, task_id)
            else:
                outputs = model(images)
                
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100.0 * correct / total

# BatchNorm Calibration (DE-BN) helper
def calibrate_bn(merged_model, train_datasets, device, num_samples=32):
    merged_model.train()
    # Update running stats without gradients
    for param in merged_model.parameters():
        param.requires_grad = False
        
    # We calibrate using a mixed subset of samples from all tasks
    cal_loaders = []
    for dataset in train_datasets:
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        subset = Subset(dataset, indices)
        cal_loaders.append(DataLoader(subset, batch_size=num_samples, shuffle=True))
        
    with torch.no_grad():
        # Pass calibration samples through the model
        for cal_loader in cal_loaders:
            for images, _ in cal_loader:
                images = images.to(device)
                merged_model(images)
    merged_model.eval()

def perform_evaluation():
    # Load pretrained experts
    experts = []
    for task in ["mnist", "fmnist", "cifar10"]:
        chk_path = f"checkpoints/{task}_expert.pt"
        if not os.path.exists(chk_path):
            print(f"Error: checkpoint {chk_path} not found. Train experts first.")
            return
        model = resnet18()
        model.fc = nn.Linear(model.fc.in_features, 10)
        model.load_state_dict(torch.load(chk_path, map_location=device))
        model = model.to(device)
        experts.append(model)
        
    print("\n--- Method 1: Oracle Experts (Individual Models) ---")
    oracle_accs = []
    for i, exp in enumerate(experts):
        acc = evaluate_model(exp, loaders[i], device)
        oracle_accs.append(acc)
        print(f"Expert {task_names[i]} on its own task: {acc:.2f}%")
    print(f"Average Oracle Accuracy: {np.mean(oracle_accs):.2f}%")
    
    # Define a helper to construct weight-merged models
    def get_weight_merged_model():
        merged = resnet18()
        merged.fc = nn.Linear(merged.fc.in_features, 10)
        merged = merged.to(device)
        
        # Load state dict from first expert
        merged_sd = copy.deepcopy(experts[0].state_dict())
        expert_sds = [exp.state_dict() for exp in experts]
        
        # Average all weights except fc head (since fc is task-specific, we'll route dynamically)
        for key in merged_sd.keys():
            if 'fc' not in key:
                stacked = torch.stack([sd[key].float() for sd in expert_sds])
                merged_sd[key] = torch.mean(stacked, dim=0).to(merged_sd[key].dtype)
                
        merged.load_state_dict(merged_sd)
        return merged

    # Multitask evaluation for weight-merged models
    def eval_multitask_merged(merged_model, experts_list, device, bits=None, noise_std=0.0):
        # We evaluate the merged backbone on each task using that task's expert FC head
        # We swap the FC head of the merged model with the corresponding expert head before evaluating
        accs = []
        for i, loader in enumerate(loaders):
            test_model = copy.deepcopy(merged_model)
            # Swap FC head
            test_model.fc = copy.deepcopy(experts_list[i].fc)
            test_model = test_model.to(device)
            # Apply quantization if specified
            if bits is not None:
                test_model = get_quantized_model(test_model, bits)
            acc = evaluate_model(test_model, loader, device, noise_std=noise_std)
            accs.append(acc)
        return accs

    # Standard Weight Averaging (WA)
    print("\n--- Method 2: Weight Averaging (WA) ---")
    wa_model = get_weight_merged_model()
    wa_accs = eval_multitask_merged(wa_model, experts, device)
    for i, acc in enumerate(wa_accs):
        print(f"WA on {task_names[i]}: {acc:.2f}%")
    print(f"Average WA Accuracy: {np.mean(wa_accs):.2f}%")
    
    # WA + BatchNorm Calibration (DE-BN)
    print("\n--- Method 3: WA + BatchNorm Calibration (DE-BN) ---")
    train_mnist = torchvision.datasets.MNIST(root='./data', train=True, transform=transform_mnist)
    train_fmnist = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=transform_mnist)
    train_cifar = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform_cifar)
    train_datasets = [train_mnist, train_fmnist, train_cifar]
    
    for samples in [16, 32, 64]:
        cal_model = get_weight_merged_model()
        calibrate_bn(cal_model, train_datasets, device, num_samples=samples)
        de_bn_accs = eval_multitask_merged(cal_model, experts, device)
        print(f"DE-BN ({samples} samples) - MNIST: {de_bn_accs[0]:.2f}%, FMNIST: {de_bn_accs[1]:.2f}%, CIFAR10: {de_bn_accs[2]:.2f}% | Avg: {np.mean(de_bn_accs):.2f}%")

    # WA + Parameter Scaling (HNS/IPR Style)
    print("\n--- Method 4: WA + Parameter Scaling ---")
    # Load progenitor model weights
    progenitor = resnet18(weights='IMAGENET1K_V1')
    progenitor_sd = progenitor.state_dict()
    
    for scale in [1.2, 1.4, 1.6, 1.8]:
        scaled_model = resnet18()
        scaled_model.fc = nn.Linear(scaled_model.fc.in_features, 10)
        scaled_model = scaled_model.to(device)
        
        # Load progenitor/expert states
        scaled_sd = copy.deepcopy(experts[0].state_dict())
        expert_sds = [exp.state_dict() for exp in experts]
        
        for key in scaled_sd.keys():
            if 'fc' not in key:
                # Calculate task vectors from progenitor
                task_vectors = [sd[key].float() - progenitor_sd[key].float().to(device) for sd in expert_sds]
                avg_task_vector = torch.mean(torch.stack(task_vectors), dim=0)
                # Rescale task vectors before adding back to progenitor
                scaled_sd[key] = (progenitor_sd[key].float().to(device) + scale * avg_task_vector).to(scaled_sd[key].dtype)
                
        scaled_model.load_state_dict(scaled_sd)
        scaled_accs = eval_multitask_merged(scaled_model, experts, device)
        print(f"Parameter Scaling (s={scale}) - MNIST: {scaled_accs[0]:.2f}%, FMNIST: {scaled_accs[1]:.2f}%, CIFAR10: {scaled_accs[2]:.2f}% | Avg: {np.mean(scaled_accs):.2f}%")

    # Method 5: Activation-Driven Synaptic Resonance (ADSR) - Our proposed Method
    print("\n--- Method 5: Activation-Driven Synaptic Resonance (ADSR - Our Method) ---")
    for t in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        adsr_model = ADSRModel(experts, temp=t).to(device)
        adsr_accs = []
        for i, loader in enumerate(loaders):
            acc = evaluate_model(adsr_model, loader, device, task_id=i)
            adsr_accs.append(acc)
        print(f"ADSR (temp={t}) - MNIST: {adsr_accs[0]:.2f}%, FMNIST: {adsr_accs[1]:.2f}%, CIFAR10: {adsr_accs[2]:.2f}% | Avg: {np.mean(adsr_accs):.2f}%")

    # Method 6: Robustness under PTQ (Post-Training Quantization)
    print("\n\n=======================================================")
    print("        POST-TRAINING QUANTIZATION (PTQ) ANALYSIS      ")
    print("=======================================================")
    
    for bits in [8, 6, 4]:
        print(f"\n--- {bits}-bit PTQ Evaluation ---")
        
        # Quantized WA
        q_wa_accs = eval_multitask_merged(wa_model, experts, device, bits=bits)
        print(f"Quantized WA - MNIST: {q_wa_accs[0]:.2f}%, FMNIST: {q_wa_accs[1]:.2f}%, CIFAR10: {q_wa_accs[2]:.2f}% | Avg: {np.mean(q_wa_accs):.2f}%")
        
        # Quantized DE-BN (32 samples)
        q_de_bn_model = get_weight_merged_model()
        calibrate_bn(q_de_bn_model, train_datasets, device, num_samples=32)
        q_de_bn_accs = eval_multitask_merged(q_de_bn_model, experts, device, bits=bits)
        print(f"Quantized DE-BN (32 samples) - MNIST: {q_de_bn_accs[0]:.2f}%, FMNIST: {q_de_bn_accs[1]:.2f}%, CIFAR10: {q_de_bn_accs[2]:.2f}% | Avg: {np.mean(q_de_bn_accs):.2f}%")
        
        # Quantized ADSR (temp=1.0)
        q_adsr_model = ADSRModel(experts, temp=1.0, bits=bits).to(device)
        # Apply weight quantization to each expert in ADSR module
        for exp in q_adsr_model.experts:
            for name, param in exp.named_parameters():
                param.data = quantize_tensor(param.data, bits)
                
        q_adsr_accs = []
        for i, loader in enumerate(loaders):
            acc = evaluate_model(q_adsr_model, loader, device, task_id=i)
            q_adsr_accs.append(acc)
        print(f"Quantized ADSR (temp=1.0) - MNIST: {q_adsr_accs[0]:.2f}%, FMNIST: {q_adsr_accs[1]:.2f}%, CIFAR10: {q_adsr_accs[2]:.2f}% | Avg: {np.mean(q_adsr_accs):.2f}%")

    # Method 7: Robustness to Blur/Noise (Gaussian Noise)
    print("\n\n=======================================================")
    print("        GAUSSIAN NOISE CORRUPTION ANALYSIS             ")
    print("=======================================================")
    
    for sigma in [0.1, 0.2, 0.3]:
        print(f"\n--- Gaussian Noise (std={sigma}) ---")
        
        # WA with Noise
        noise_wa_accs = eval_multitask_merged(wa_model, experts, device, noise_std=sigma)
        print(f"WA with Noise - MNIST: {noise_wa_accs[0]:.2f}%, FMNIST: {noise_wa_accs[1]:.2f}%, CIFAR10: {noise_wa_accs[2]:.2f}% | Avg: {np.mean(noise_wa_accs):.2f}%")
        
        # DE-BN (32 samples) with Noise
        noise_de_bn_model = get_weight_merged_model()
        calibrate_bn(noise_de_bn_model, train_datasets, device, num_samples=32)
        noise_de_bn_accs = eval_multitask_merged(noise_de_bn_model, experts, device, noise_std=sigma)
        print(f"DE-BN (32 samples) with Noise - MNIST: {noise_de_bn_accs[0]:.2f}%, FMNIST: {noise_de_bn_accs[1]:.2f}%, CIFAR10: {noise_de_bn_accs[2]:.2f}% | Avg: {np.mean(noise_de_bn_accs):.2f}%")
        
        # ADSR (temp=1.0) with Noise
        adsr_noise_model = ADSRModel(experts, temp=1.0).to(device)
        adsr_noise_accs = []
        for i, loader in enumerate(loaders):
            acc = evaluate_model(adsr_noise_model, loader, device, task_id=i, noise_std=sigma)
            adsr_noise_accs.append(acc)
        print(f"ADSR (temp=1.0) with Noise - MNIST: {adsr_noise_accs[0]:.2f}%, FMNIST: {adsr_noise_accs[1]:.2f}%, CIFAR10: {adsr_noise_accs[2]:.2f}% | Avg: {np.mean(adsr_noise_accs):.2f}%")

if __name__ == "__main__":
    perform_evaluation()
