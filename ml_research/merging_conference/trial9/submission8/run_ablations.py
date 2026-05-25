import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
import os

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False

set_seed(42)

# SimpleCNN Architecture
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.pool3 = nn.AdaptiveAvgPool2d((3, 3))
        self.fc = nn.Linear(64 * 3 * 3, 10)
        self.relu = nn.ReLU()

    def forward(self, x, return_features=False):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        feat = self.pool3(x)
        feat = feat.view(feat.size(0), -1)
        out = self.fc(feat)
        if return_features:
            return out, feat
        return out

# Helper to perform differentiable model forwarding on merged parameters
def differentiable_forward(expert0, expert1, merged_model, global_coef, layer_offsets, inputs, return_features=False):
    from torch.func import functional_call
    state0 = expert0.state_dict()
    state1 = expert1.state_dict()
    merged_params_and_buffers = {}
    
    for name, param in state0.items():
        if param.dtype.is_floating_point and name in layer_offsets:
            offset = layer_offsets[name]
            lambda_val = torch.sigmoid(global_coef + offset)
            merged_params_and_buffers[name] = (1.0 - lambda_val) * state0[name] + lambda_val * state1[name]
        elif param.dtype.is_floating_point:
            lambda_val = torch.sigmoid(global_coef)
            merged_params_and_buffers[name] = ((1.0 - lambda_val) * state0[name] + lambda_val * state1[name]).detach()
        else:
            merged_params_and_buffers[name] = state0[name]
            
    if return_features:
        return functional_call(merged_model, merged_params_and_buffers, args=(inputs,), kwargs={"return_features": True})
    else:
        return functional_call(merged_model, merged_params_and_buffers, args=(inputs,))

# Helper to calculate entropy
def softmax_entropy(x):
    probs = torch.softmax(x, dim=-1)
    return -torch.sum(probs * torch.log(probs + 1e-6), dim=-1).mean()

# Test-time augmentations for SSCPA
def apply_test_time_augmentations(x):
    shift_x = random.randint(-2, 2)
    shift_y = random.randint(-2, 2)
    x_aug1 = torch.roll(x, shifts=(shift_x, shift_y), dims=(2, 3))
    x_aug1 = x_aug1 + 0.05 * torch.randn_like(x_aug1)
    
    scale = random.uniform(0.9, 1.1)
    h, w = x.shape[2], x.shape[3]
    new_h, new_w = int(h * scale), int(w * scale)
    x_resized = nn.functional.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
    if scale < 1.0:
        pad_h = (h - new_h) // 2
        pad_w = (w - new_w) // 2
        x_aug2 = nn.functional.pad(x_resized, (pad_w, h - new_h - pad_w, pad_h, w - new_w - pad_h), mode='constant', value=-1.0)
    else:
        crop_h = (new_h - h) // 2
        crop_w = (new_w - w) // 2
        x_aug2 = x_resized[:, :, crop_h:crop_h+h, crop_w:crop_w+w]
        
    x_aug2 = x_aug2 + 0.05 * torch.randn_like(x_aug2)
    return x_aug1, x_aug2

def run_sscpa(expert0, expert1, stream_batches, prototype_0, prototype_1, layer_names, device,
              beta_cons=0.5, num_steps=5, lr=1.0, T=0.1):
    set_seed(42)
    method_accuracies = []
    
    dyn_proto_0 = prototype_0.clone()
    dyn_proto_1 = prototype_1.clone()
    
    for b_idx, (inputs, targets, seg_name) in enumerate(stream_batches):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Reset merging parameters per batch
        global_coef = nn.Parameter(torch.tensor(0.0, device=device))
        layer_offsets = {name: nn.Parameter(torch.tensor(0.0, device=device)) for name in layer_names}
        all_params = [global_coef] + list(layer_offsets.values())
        optimizer = optim.AdamW(all_params, lr=lr, weight_decay=1e-4)
        
        merged_model = SimpleCNN().to(device)
        merged_model.eval()
        
        for m in merged_model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.train()
                m.momentum = 0.0
                
        for step in range(num_steps):
            optimizer.zero_grad()
            
            # 1. Dynamic routing
            with torch.no_grad():
                _, feat0 = expert0(inputs, return_features=True)
                _, feat1 = expert1(inputs, return_features=True)
                feat0_norm = feat0 / (torch.norm(feat0, dim=-1, keepdim=True) + 1e-6)
                feat1_norm = feat1 / (torch.norm(feat1, dim=-1, keepdim=True) + 1e-6)
                avg_feat0 = feat0_norm.mean(dim=0)
                avg_feat0 = avg_feat0 / (torch.norm(avg_feat0) + 1e-6)
                avg_feat1 = feat1_norm.mean(dim=0)
                avg_feat1 = avg_feat1 / (torch.norm(avg_feat1) + 1e-6)
                
                sim0 = torch.dot(avg_feat0, dyn_proto_0)
                sim1 = torch.dot(avg_feat1, dyn_proto_1)
                routing_scores = torch.stack([sim0, sim1]) / T
                pi = torch.softmax(routing_scores, dim=0)
            
            # 2. Forward raw batch
            outputs_raw = differentiable_forward(expert0, expert1, merged_model, global_coef, layer_offsets, inputs)
            
            # 3. Test-time augmentations
            inputs_aug1, inputs_aug2 = apply_test_time_augmentations(inputs)
            
            with torch.no_grad():
                _, feat_exp0_aug1 = expert0(inputs_aug1, return_features=True)
                _, feat_exp1_aug1 = expert1(inputs_aug1, return_features=True)
                _, feat_exp0_aug2 = expert0(inputs_aug2, return_features=True)
                _, feat_exp1_aug2 = expert1(inputs_aug2, return_features=True)
                
                feat_exp0_aug1 = feat_exp0_aug1 / (torch.norm(feat_exp0_aug1, dim=-1, keepdim=True) + 1e-6)
                feat_exp1_aug1 = feat_exp1_aug1 / (torch.norm(feat_exp1_aug1, dim=-1, keepdim=True) + 1e-6)
                feat_exp0_aug2 = feat_exp0_aug2 / (torch.norm(feat_exp0_aug2, dim=-1, keepdim=True) + 1e-6)
                feat_exp1_aug2 = feat_exp1_aug2 / (torch.norm(feat_exp1_aug2, dim=-1, keepdim=True) + 1e-6)
                
            _, feat_aug1 = differentiable_forward(expert0, expert1, merged_model, global_coef, layer_offsets, inputs_aug1, return_features=True)
            _, feat_aug2 = differentiable_forward(expert0, expert1, merged_model, global_coef, layer_offsets, inputs_aug2, return_features=True)
            
            feat_aug1_norm = feat_aug1 / (torch.norm(feat_aug1, dim=-1, keepdim=True) + 1e-6)
            feat_aug2_norm = feat_aug2 / (torch.norm(feat_aug2, dim=-1, keepdim=True) + 1e-6)
            
            avg_aug1 = feat_aug1_norm.mean(dim=0)
            avg_aug1 = avg_aug1 / (torch.norm(avg_aug1) + 1e-6)
            
            avg_aug2 = feat_aug2_norm.mean(dim=0)
            avg_aug2 = avg_aug2 / (torch.norm(avg_aug2) + 1e-6)
            
            loss_align = - (
                pi[0] * (feat_aug1_norm * feat_exp0_aug1).sum(dim=-1).mean() +
                pi[1] * (feat_aug1_norm * feat_exp1_aug1).sum(dim=-1).mean() +
                pi[0] * (feat_aug2_norm * feat_exp0_aug2).sum(dim=-1).mean() +
                pi[1] * (feat_aug2_norm * feat_exp1_aug2).sum(dim=-1).mean()
            )
            
            loss_consistency = - torch.dot(avg_aug1, avg_aug2)
            ent = softmax_entropy(outputs_raw)
            
            loss_total = loss_align + beta_cons * loss_consistency + 0.1 * ent
            loss_total.backward()
            optimizer.step()
            
        with torch.no_grad():
            outputs = differentiable_forward(expert0, expert1, merged_model, global_coef, layer_offsets, inputs)
            _, predicted = outputs.max(1)
            correct = predicted.eq(targets).sum().item()
            acc = 100.0 * correct / targets.size(0)
            method_accuracies.append(acc)
            
            # Prototype update on high routing confidence
            if pi[0] > 0.85:
                dyn_proto_0 = 0.9 * dyn_proto_0 + 0.1 * avg_feat0
            if pi[1] > 0.85:
                dyn_proto_1 = 0.9 * dyn_proto_1 + 0.1 * avg_feat1
                
    # Calculate Segment Accuracies
    segment_accuracies = []
    for s_idx in range(5):
        start = s_idx * 10
        end = (s_idx + 1) * 10
        seg_acc = np.mean(method_accuracies[start:end])
        segment_accuracies.append(seg_acc)
    overall = np.mean(method_accuracies)
    return segment_accuracies + [overall]

def main():
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    
    # Load test datasets
    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    fashion_test = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    kmnist_test = torchvision.datasets.KMNIST(root='./data', train=False, download=True, transform=transform)
    
    mnist_test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=64, shuffle=True)
    fashion_test_loader = torch.utils.data.DataLoader(fashion_test, batch_size=64, shuffle=True)
    
    # Load models
    base_model_path = "base_model.pt"
    expert0_path = "expert0.pt"
    expert1_path = "expert1.pt"
    
    expert0 = SimpleCNN().to(device)
    expert0.load_state_dict(torch.load(expert0_path, map_location=device))
    expert1 = SimpleCNN().to(device)
    expert1.load_state_dict(torch.load(expert1_path, map_location=device))
    
    expert0.eval()
    expert1.eval()
    
    for m in expert0.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.train()
            m.momentum = 0.0
    for m in expert1.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.train()
            m.momentum = 0.0
            
    # Set up stream
    mnist_eval_loader = torch.utils.data.DataLoader(mnist_test, batch_size=64, shuffle=True)
    fashion_eval_loader = torch.utils.data.DataLoader(fashion_test, batch_size=64, shuffle=True)
    kmnist_eval_loader = torch.utils.data.DataLoader(kmnist_test, batch_size=64, shuffle=True)
    
    mnist_iter = iter(mnist_eval_loader)
    fashion_iter = iter(fashion_eval_loader)
    kmnist_iter = iter(kmnist_eval_loader)
    
    stream_batches = []
    
    for _ in range(10):
        inputs, targets = next(mnist_iter)
        stream_batches.append((inputs, targets, "Clean MNIST"))
        
    for _ in range(10):
        inputs, targets = next(mnist_iter)
        noisy_inputs = inputs + 0.6 * torch.randn_like(inputs)
        noisy_inputs = torch.clamp(noisy_inputs, -1.0, 1.0)
        stream_batches.append((noisy_inputs, targets, "Noisy MNIST"))
        
    for _ in range(10):
        inputs, targets = next(fashion_iter)
        stream_batches.append((inputs, targets, "Clean Fashion"))
        
    for _ in range(10):
        inputs, targets = next(fashion_iter)
        noisy_inputs = inputs + 0.6 * torch.randn_like(inputs)
        noisy_inputs = torch.clamp(noisy_inputs, -1.0, 1.0)
        stream_batches.append((noisy_inputs, targets, "Noisy Fashion"))
        
    for _ in range(10):
        inputs, targets = next(kmnist_iter)
        stream_batches.append((inputs, targets, "Novel KMNIST"))
        
    layer_names = []
    for name, param in expert0.named_parameters():
        if "weight" in name or "bias" in name:
            layer_names.append(name)
            
    # Precompute prototypes
    def compute_prototype(expert_model, dataset, num_samples=512):
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
        feats = []
        with torch.no_grad():
            for inputs, _ in loader:
                inputs = inputs.to(device)
                _, feat = expert_model(inputs, return_features=True)
                feats.append(feat)
                if len(feats) * 64 >= num_samples:
                    break
        all_feats = torch.cat(feats, dim=0)[:num_samples]
        normalized_feats = all_feats / (torch.norm(all_feats, dim=-1, keepdim=True) + 1e-6)
        mean_feat = normalized_feats.mean(dim=0)
        return mean_feat / (torch.norm(mean_feat) + 1e-6)
        
    prototype_0 = compute_prototype(expert0, mnist_test)
    prototype_1 = compute_prototype(expert1, fashion_test)
    
    print("Precomputations finished. Starting Ablation sweeps...")
    
    # Sweep 1: Consistency Loss Weight
    print("\n--- Sweeping Consistency Loss Weight ---")
    results_beta = {}
    for beta in [0.0, 0.1, 0.5, 1.0]:
        res = run_sscpa(expert0, expert1, stream_batches, prototype_0, prototype_1, layer_names, device,
                        beta_cons=beta)
        results_beta[beta] = res
        print(f"Beta_cons = {beta:.1f} | Clean MNIST: {res[0]:.2f} | Noisy MNIST: {res[1]:.2f} | Clean Fashion: {res[2]:.2f} | Noisy Fashion: {res[3]:.2f} | Novel KMNIST: {res[4]:.2f} | Overall: {res[5]:.2f}")
        
    # Sweep 2: Optimization Steps
    print("\n--- Sweeping Optimization Steps ---")
    results_steps = {}
    for steps in [1, 3, 5, 10]:
        res = run_sscpa(expert0, expert1, stream_batches, prototype_0, prototype_1, layer_names, device,
                        num_steps=steps)
        results_steps[steps] = res
        print(f"Steps = {steps} | Clean MNIST: {res[0]:.2f} | Noisy MNIST: {res[1]:.2f} | Clean Fashion: {res[2]:.2f} | Noisy Fashion: {res[3]:.2f} | Novel KMNIST: {res[4]:.2f} | Overall: {res[5]:.2f}")
        
    # Sweep 3: Learning Rate
    print("\n--- Sweeping Learning Rate ---")
    results_lr = {}
    for lr in [0.1, 0.5, 1.0, 2.0]:
        res = run_sscpa(expert0, expert1, stream_batches, prototype_0, prototype_1, layer_names, device,
                        lr=lr)
        results_lr[lr] = res
        print(f"LR = {lr:.1f} | Clean MNIST: {res[0]:.2f} | Noisy MNIST: {res[1]:.2f} | Clean Fashion: {res[2]:.2f} | Noisy Fashion: {res[3]:.2f} | Novel KMNIST: {res[4]:.2f} | Overall: {res[5]:.2f}")
        
    # Sweep 4: Routing Temperature
    print("\n--- Sweeping Routing Temperature ---")
    results_temp = {}
    for temp in [0.01, 0.05, 0.1, 0.2]:
        res = run_sscpa(expert0, expert1, stream_batches, prototype_0, prototype_1, layer_names, device,
                        T=temp)
        results_temp[temp] = res
        print(f"T_routing = {temp:.2f} | Clean MNIST: {res[0]:.2f} | Noisy MNIST: {res[1]:.2f} | Clean Fashion: {res[2]:.2f} | Noisy Fashion: {res[3]:.2f} | Novel KMNIST: {res[4]:.2f} | Overall: {res[5]:.2f}")

if __name__ == "__main__":
    main()
