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
        param.data = quantize_tensor(param.data, bits)
    return q_model

# Activation-Driven Synaptic Resonance (ADSR) Model
class ADSRModel(nn.Module):
    def __init__(self, experts, temp=-5.0, temp_cw=-0.5, bits=None, channel_wise=False, adaptive_scale=False, gamma=10.0, theta=0.8, spectral_mode=False, cutoff_ratio=0.25):
        super(ADSRModel, self).__init__()
        self.experts = nn.ModuleList([copy.deepcopy(exp) for exp in experts])
        self.temp = temp # Used for standard block-wise temperature (temp_bw)
        self.temp_cw = temp_cw # Used for channel-wise temperature
        self.bits = bits
        self.channel_wise = channel_wise
        self.adaptive_scale = adaptive_scale
        self.gamma = gamma
        self.theta = theta
        self.spectral_mode = spectral_mode
        self.cutoff_ratio = cutoff_ratio
        
    def fuse_resonance(self, h_list):
        # h_list is a list of 3 tensors of shape [Batch, Channels, Height, Width]
        if self.spectral_mode:
            # 1. Decompose each h in h_list into low and high frequency components
            h_low_list = []
            h_high_list = []
            for h in h_list:
                B, C, H_dim, W_dim = h.shape
                if H_dim > 2 and W_dim > 2:
                    # 2D RFFT
                    H_fft = torch.fft.rfft2(h, norm="ortho")
                    
                    # Create low-frequency mask
                    cutoff_h = max(1, int(H_dim * self.cutoff_ratio))
                    cutoff_w = max(1, int((W_dim // 2 + 1) * self.cutoff_ratio))
                    
                    mask = torch.zeros_like(H_fft, dtype=torch.bool)
                    y_indices = torch.arange(H_dim, device=h.device).view(-1, 1)
                    y_dist = torch.min(y_indices, H_dim - y_indices)
                    x_indices = torch.arange(H_fft.shape[-1], device=h.device).view(1, -1)
                    
                    mask_2d = (y_dist <= cutoff_h) & (x_indices <= cutoff_w)
                    mask[..., :, :] = mask_2d
                    
                    H_low = torch.where(mask, H_fft, torch.zeros_like(H_fft))
                    H_high = torch.where(~mask, H_fft, torch.zeros_like(H_fft))
                    
                    h_low = torch.fft.irfft2(H_low, s=(H_dim, W_dim), norm="ortho")
                    h_high = torch.fft.irfft2(H_high, s=(H_dim, W_dim), norm="ortho")
                else:
                    h_low = h
                    h_high = torch.zeros_like(h)
                    
                h_low_list.append(h_low)
                h_high_list.append(h_high)
                
            # 2. Route low frequencies using Block-Wise routing
            variances_low = []
            for h_low in h_low_list:
                var_low = torch.var(h_low, dim=(1, 2, 3), keepdim=True) # shape [Batch, 1, 1, 1]
                variances_low.append(var_low)
            vars_low_stacked = torch.stack(variances_low, dim=1) # shape [Batch, 3, 1, 1, 1]
            log_vars_low = torch.log(vars_low_stacked + 1e-6)
            alpha_low = torch.softmax(self.temp * log_vars_low, dim=1) # shape [Batch, 3, 1, 1, 1]
            
            # 3. Route high frequencies using Channel-Wise routing
            variances_high = []
            for h_high in h_high_list:
                if h_high.size(2) > 1 or h_high.size(3) > 1:
                    var_high = torch.var(h_high, dim=(2, 3), keepdim=True, unbiased=False) # shape [Batch, Channels, 1, 1]
                else:
                    var_high = h_high ** 2
                variances_high.append(var_high)
            vars_high_stacked = torch.stack(variances_high, dim=1) # shape [Batch, 3, Channels, 1, 1]
            log_vars_high = torch.log(vars_high_stacked + 1e-6)
            alpha_high = torch.softmax(self.temp_cw * log_vars_high, dim=1) # shape [Batch, 3, Channels, 1, 1]
            
            # 4. Perform weighted sum of spectral components
            h_low_stacked = torch.stack(h_low_list, dim=1)
            h_high_stacked = torch.stack(h_high_list, dim=1)
            
            h_fused = torch.sum(alpha_low * h_low_stacked, dim=1) + torch.sum(alpha_high * h_high_stacked, dim=1)
            
        elif self.adaptive_scale:
            # 1. Compute channel-wise variances
            variances_cw = []
            cvs = []
            for h in h_list:
                if h.size(2) > 1 or h.size(3) > 1:
                    var_c = torch.var(h, dim=(2, 3), keepdim=True, unbiased=False) # shape [Batch, Channels, 1, 1]
                else:
                    var_c = h ** 2 # shape [Batch, Channels, 1, 1]
                variances_cw.append(var_c)
                
                # Coefficient of variation of channel-wise variances
                mean_var_c = torch.mean(var_c, dim=1, keepdim=True) # shape [Batch, 1, 1, 1]
                std_var_c = torch.std(var_c, dim=1, keepdim=True, unbiased=False) # shape [Batch, 1, 1, 1]
                cv_k = std_var_c / (mean_var_c + 1e-6) # shape [Batch, 1, 1, 1]
                cvs.append(cv_k)
                
            # Compute average CV across experts
            cv_avg = torch.mean(torch.stack(cvs, dim=1), dim=1) # shape [Batch, 1, 1, 1]
            
            # Gating weight (high for clean/diverse activations, low for noisy/uniform)
            w = torch.sigmoid(self.gamma * (cv_avg - self.theta)) # shape [Batch, 1, 1, 1]
            # Ensure w can broadcast to [Batch, 3, Channels, 1, 1]
            w_expanded = w.unsqueeze(1) # shape [Batch, 1, 1, 1, 1]
            
            # 2. Compute Channel-Wise weights
            vars_cw_stacked = torch.stack(variances_cw, dim=1) # shape [Batch, 3, Channels, 1, 1]
            log_vars_cw = torch.log(vars_cw_stacked + 1e-6)
            alpha_cw = torch.softmax(self.temp_cw * log_vars_cw, dim=1) # shape [Batch, 3, Channels, 1, 1]
            
            # 3. Compute Block-Wise weights
            variances_bw = []
            for h in h_list:
                var_b = torch.var(h, dim=(1, 2, 3), keepdim=True) # shape [Batch, 1, 1, 1]
                variances_bw.append(var_b)
            vars_bw_stacked = torch.stack(variances_bw, dim=1) # shape [Batch, 3, 1, 1, 1]
            log_vars_bw = torch.log(vars_bw_stacked + 1e-6)
            alpha_bw = torch.softmax(self.temp * log_vars_bw, dim=1) # shape [Batch, 3, 1, 1, 1]
            
            # 4. Interpolate routing weights
            alpha = w_expanded * alpha_bw + (1.0 - w_expanded) * alpha_cw
            
            # Stack activations: [Batch, 3, Channels, Height, Width]
            h_stacked = torch.stack(h_list, dim=1)
            
            # Fused activations: weighted sum
            h_fused = torch.sum(alpha * h_stacked, dim=1)
            
        else:
            variances = []
            for h in h_list:
                if self.channel_wise:
                    if h.size(2) > 1 or h.size(3) > 1:
                        var = torch.var(h, dim=(2, 3), keepdim=True, unbiased=False)
                    else:
                        var = h ** 2
                else:
                    # Calculate variance across channel and spatial dimensions per batch item
                    var = torch.var(h, dim=(1, 2, 3), keepdim=True) # shape [Batch, 1, 1, 1]
                variances.append(var)
                
            # Stack and apply softmax along the expert dimension
            vars_stacked = torch.stack(variances, dim=1)
            # Log-variance to scale numerically
            log_vars = torch.log(vars_stacked + 1e-6)
            
            # Softmax to get resonance weights
            alpha = torch.softmax(self.temp * log_vars, dim=1) # shape [Batch, 3, Channels/1, 1, 1]
            
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
    for param in merged_model.parameters():
        param.requires_grad = False
        
    cal_loaders = []
    for dataset in train_datasets:
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        subset = Subset(dataset, indices)
        cal_loaders.append(DataLoader(subset, batch_size=num_samples, shuffle=True))
        
    with torch.no_grad():
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
    
    def get_weight_merged_model():
        merged = resnet18()
        merged.fc = nn.Linear(merged.fc.in_features, 10)
        merged = merged.to(device)
        
        merged_sd = copy.deepcopy(experts[0].state_dict())
        expert_sds = [exp.state_dict() for exp in experts]
        
        for key in merged_sd.keys():
            if 'fc' not in key:
                stacked = torch.stack([sd[key].float() for sd in expert_sds])
                merged_sd[key] = torch.mean(stacked, dim=0).to(merged_sd[key].dtype)
                
        merged.load_state_dict(merged_sd)
        return merged

    def eval_multitask_merged(merged_model, experts_list, device, bits=None, noise_std=0.0):
        accs = []
        for i, loader in enumerate(loaders):
            test_model = copy.deepcopy(merged_model)
            test_model.fc = copy.deepcopy(experts_list[i].fc)
            test_model = test_model.to(device)
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

    # WA + Parameter Scaling
    print("\n--- Method 4: WA + Parameter Scaling ---")
    progenitor = resnet18(weights='IMAGENET1K_V1')
    progenitor_sd = progenitor.state_dict()
    
    for scale in [1.2, 1.4, 1.6, 1.8]:
        scaled_model = resnet18()
        scaled_model.fc = nn.Linear(scaled_model.fc.in_features, 10)
        scaled_model = scaled_model.to(device)
        
        scaled_sd = copy.deepcopy(experts[0].state_dict())
        expert_sds = [exp.state_dict() for exp in experts]
        
        for key in scaled_sd.keys():
            if 'fc' not in key:
                task_vectors = [sd[key].float() - progenitor_sd[key].float().to(device) for sd in expert_sds]
                avg_task_vector = torch.mean(torch.stack(task_vectors), dim=0)
                scaled_sd[key] = (progenitor_sd[key].float().to(device) + scale * avg_task_vector).to(scaled_sd[key].dtype)
                
        scaled_model.load_state_dict(scaled_sd)
        scaled_accs = eval_multitask_merged(scaled_model, experts, device)
        print(f"Parameter Scaling (s={scale}) - MNIST: {scaled_accs[0]:.2f}%, FMNIST: {scaled_accs[1]:.2f}%, CIFAR10: {scaled_accs[2]:.2f}% | Avg: {np.mean(scaled_accs):.2f}%")

    # Method 5: Activation-Driven Synaptic Resonance (ADSR)
    print("\n--- Method 5: Block-Wise Activation-Driven Synaptic Resonance (BW-ADSR) ---")
    best_temp_bw = -5.0
    adsr_model = ADSRModel(experts, temp=best_temp_bw, channel_wise=False).to(device)
    adsr_accs = []
    for i, loader in enumerate(loaders):
        acc = evaluate_model(adsr_model, loader, device, task_id=i)
        adsr_accs.append(acc)
    best_avg_bw = np.mean(adsr_accs)
    print(f"BW-ADSR (temp={best_temp_bw}) - MNIST: {adsr_accs[0]:.2f}%, FMNIST: {adsr_accs[1]:.2f}%, CIFAR10: {adsr_accs[2]:.2f}% | Avg: {best_avg_bw:.2f}%")
    print(f"\n---> Best BW-ADSR temperature selected: {best_temp_bw} with Avg Acc: {best_avg_bw:.2f}%")

    print("\n--- Method 5b: Channel-Wise Synaptic Resonance (CW-ADSR) ---")
    best_temp_cw = -0.5
    adsr_model = ADSRModel(experts, temp=best_temp_cw, channel_wise=True).to(device)
    adsr_accs = []
    for i, loader in enumerate(loaders):
        acc = evaluate_model(adsr_model, loader, device, task_id=i)
        adsr_accs.append(acc)
    best_avg_cw = np.mean(adsr_accs)
    print(f"CW-ADSR (temp={best_temp_cw}) - MNIST: {adsr_accs[0]:.2f}%, FMNIST: {adsr_accs[1]:.2f}%, CIFAR10: {adsr_accs[2]:.2f}% | Avg: {best_avg_cw:.2f}%")
    print(f"\n---> Best CW-ADSR temperature selected: {best_temp_cw} with Avg Acc: {best_avg_cw:.2f}%")

    # Method 5c: Adaptive-Scale Synaptic Resonance (AS-ADSR)
    print("\n--- Method 5c: Adaptive-Scale Synaptic Resonance (AS-ADSR) ---")
    best_theta_as = 0.4
    best_gamma_as = 15.0
    as_model = ADSRModel(experts, temp=best_temp_bw, temp_cw=best_temp_cw, bits=None, channel_wise=False, adaptive_scale=True, gamma=best_gamma_as, theta=best_theta_as).to(device)
    as_accs = []
    for i, loader in enumerate(loaders):
        acc = evaluate_model(as_model, loader, device, task_id=i)
        as_accs.append(acc)
    best_avg_as = np.mean(as_accs)
    print(f"AS-ADSR (theta={best_theta_as}, gamma={best_gamma_as}) - MNIST: {as_accs[0]:.2f}%, FMNIST: {as_accs[1]:.2f}%, CIFAR10: {as_accs[2]:.2f}% | Avg: {best_avg_as:.2f}%")
    print(f"\n---> Best AS-ADSR (theta={best_theta_as}, gamma={best_gamma_as}) selected with Avg Acc: {best_avg_as:.2f}%")

    # Method 5d: Spatio-Spectral Synaptic Resonance (SS-ADSR)
    print("\n--- Method 5d: Spatio-Spectral Synaptic Resonance (SS-ADSR) ---")
    best_cutoff = 0.25
    ss_model = ADSRModel(experts, temp=best_temp_bw, temp_cw=best_temp_cw, bits=None, channel_wise=False, spectral_mode=True, cutoff_ratio=best_cutoff).to(device)
    ss_accs = []
    for i, loader in enumerate(loaders):
        acc = evaluate_model(ss_model, loader, device, task_id=i)
        ss_accs.append(acc)
    best_avg_ss = np.mean(ss_accs)
    print(f"SS-ADSR (cutoff={best_cutoff}) - MNIST: {ss_accs[0]:.2f}%, FMNIST: {ss_accs[1]:.2f}%, CIFAR10: {ss_accs[2]:.2f}% | Avg: {best_avg_ss:.2f}%")
    print(f"\n---> Best SS-ADSR (cutoff={best_cutoff}) selected with Avg Acc: {best_avg_ss:.2f}%")

    # Method 6: Robustness under PTQ
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
        
        # Quantized BW-ADSR
        q_bw_model = ADSRModel(experts, temp=best_temp_bw, bits=bits, channel_wise=False).to(device)
        for exp in q_bw_model.experts:
            for name, param in exp.named_parameters():
                param.data = quantize_tensor(param.data, bits)
        q_bw_accs = []
        for i, loader in enumerate(loaders):
            acc = evaluate_model(q_bw_model, loader, device, task_id=i)
            q_bw_accs.append(acc)
        print(f"Quantized BW-ADSR (temp={best_temp_bw}) - MNIST: {q_bw_accs[0]:.2f}%, FMNIST: {q_bw_accs[1]:.2f}%, CIFAR10: {q_bw_accs[2]:.2f}% | Avg: {np.mean(q_bw_accs):.2f}%")

        # Quantized CW-ADSR
        q_cw_model = ADSRModel(experts, temp=best_temp_cw, bits=bits, channel_wise=True).to(device)
        for exp in q_cw_model.experts:
            for name, param in exp.named_parameters():
                param.data = quantize_tensor(param.data, bits)
        q_cw_accs = []
        for i, loader in enumerate(loaders):
            acc = evaluate_model(q_cw_model, loader, device, task_id=i)
            q_cw_accs.append(acc)
        print(f"Quantized CW-ADSR (temp={best_temp_cw}) - MNIST: {q_cw_accs[0]:.2f}%, FMNIST: {q_cw_accs[1]:.2f}%, CIFAR10: {q_cw_accs[2]:.2f}% | Avg: {np.mean(q_cw_accs):.2f}%")

        # Quantized AS-ADSR
        q_as_model = ADSRModel(experts, temp=best_temp_bw, temp_cw=best_temp_cw, bits=bits, channel_wise=False, adaptive_scale=True, gamma=best_gamma_as, theta=best_theta_as).to(device)
        for exp in q_as_model.experts:
            for name, param in exp.named_parameters():
                param.data = quantize_tensor(param.data, bits)
        q_as_accs = []
        for i, loader in enumerate(loaders):
            acc = evaluate_model(q_as_model, loader, device, task_id=i)
            q_as_accs.append(acc)
        print(f"Quantized AS-ADSR (theta={best_theta_as}, gamma={best_gamma_as}) - MNIST: {q_as_accs[0]:.2f}%, FMNIST: {q_as_accs[1]:.2f}%, CIFAR10: {q_as_accs[2]:.2f}% | Avg: {np.mean(q_as_accs):.2f}%")

        # Quantized SS-ADSR
        q_ss_model = ADSRModel(experts, temp=best_temp_bw, temp_cw=best_temp_cw, bits=bits, channel_wise=False, spectral_mode=True, cutoff_ratio=best_cutoff).to(device)
        for exp in q_ss_model.experts:
            for name, param in exp.named_parameters():
                param.data = quantize_tensor(param.data, bits)
        q_ss_accs = []
        for i, loader in enumerate(loaders):
            acc = evaluate_model(q_ss_model, loader, device, task_id=i)
            q_ss_accs.append(acc)
        print(f"Quantized SS-ADSR (cutoff={best_cutoff}) - MNIST: {q_ss_accs[0]:.2f}%, FMNIST: {q_ss_accs[1]:.2f}%, CIFAR10: {q_ss_accs[2]:.2f}% | Avg: {np.mean(q_ss_accs):.2f}%")

    # Method 7: Robustness to Blur/Noise
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
        
        # BW-ADSR with Noise
        bw_noise_model = ADSRModel(experts, temp=best_temp_bw, channel_wise=False).to(device)
        bw_noise_accs = []
        for i, loader in enumerate(loaders):
            acc = evaluate_model(bw_noise_model, loader, device, task_id=i, noise_std=sigma)
            bw_noise_accs.append(acc)
        print(f"BW-ADSR (temp={best_temp_bw}) with Noise - MNIST: {bw_noise_accs[0]:.2f}%, FMNIST: {bw_noise_accs[1]:.2f}%, CIFAR10: {bw_noise_accs[2]:.2f}% | Avg: {np.mean(bw_noise_accs):.2f}%")

        # CW-ADSR with Noise
        cw_noise_model = ADSRModel(experts, temp=best_temp_cw, channel_wise=True).to(device)
        cw_noise_accs = []
        for i, loader in enumerate(loaders):
            acc = evaluate_model(cw_noise_model, loader, device, task_id=i, noise_std=sigma)
            cw_noise_accs.append(acc)
        print(f"CW-ADSR (temp={best_temp_cw}) with Noise - MNIST: {cw_noise_accs[0]:.2f}%, FMNIST: {cw_noise_accs[1]:.2f}%, CIFAR10: {cw_noise_accs[2]:.2f}% | Avg: {np.mean(cw_noise_accs):.2f}%")

        # AS-ADSR with Noise
        as_noise_model = ADSRModel(experts, temp=best_temp_bw, temp_cw=best_temp_cw, channel_wise=False, adaptive_scale=True, gamma=best_gamma_as, theta=best_theta_as).to(device)
        as_noise_accs = []
        for i, loader in enumerate(loaders):
            acc = evaluate_model(as_noise_model, loader, device, task_id=i, noise_std=sigma)
            as_noise_accs.append(acc)
        print(f"AS-ADSR (theta={best_theta_as}, gamma={best_gamma_as}) with Noise - MNIST: {as_noise_accs[0]:.2f}%, FMNIST: {as_noise_accs[1]:.2f}%, CIFAR10: {as_noise_accs[2]:.2f}% | Avg: {np.mean(as_noise_accs):.2f}%")

        # SS-ADSR with Noise
        ss_noise_model = ADSRModel(experts, temp=best_temp_bw, temp_cw=best_temp_cw, channel_wise=False, spectral_mode=True, cutoff_ratio=best_cutoff).to(device)
        ss_noise_accs = []
        for i, loader in enumerate(loaders):
            acc = evaluate_model(ss_noise_model, loader, device, task_id=i, noise_std=sigma)
            ss_noise_accs.append(acc)
        print(f"SS-ADSR (cutoff={best_cutoff}) with Noise - MNIST: {ss_noise_accs[0]:.2f}%, FMNIST: {ss_noise_accs[1]:.2f}%, CIFAR10: {ss_noise_accs[2]:.2f}% | Avg: {np.mean(ss_noise_accs):.2f}%")

if __name__ == "__main__":
    perform_evaluation()
