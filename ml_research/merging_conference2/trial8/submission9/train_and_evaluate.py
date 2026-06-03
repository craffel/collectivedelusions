import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import copy
import time

# Create directories
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.enabled = False  # Disable cuDNN to avoid CUDNN_STATUS_NOT_INITIALIZED on server nodes

# Global variables
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Define Dataset transforms
mnist_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

fashion_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

cifar_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Multi-head ResNet-18 Architecture
class MultiHeadResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Load pre-trained ResNet18
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # replace fc with Identity to get embeddings
        
        # Define independent task classification heads
        self.heads = nn.ModuleDict({
            'mnist': nn.Linear(in_features, num_classes),
            'fashion_mnist': nn.Linear(in_features, num_classes),
            'cifar10': nn.Linear(in_features, num_classes)
        })
        
    def forward(self, x, task_name=None):
        features = self.backbone(x)
        if task_name is not None:
            return self.heads[task_name](features)
        return features

def get_dataloader(task_name, batch_size=128, train=True):
    if task_name == 'mnist':
        dataset = torchvision.datasets.MNIST(root='./data', train=train, download=True, transform=mnist_transform)
    elif task_name == 'fashion_mnist':
        dataset = torchvision.datasets.FashionMNIST(root='./data', train=train, download=True, transform=fashion_transform)
    elif task_name == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(root='./data', train=train, download=True, transform=cifar_transform)
    else:
        raise ValueError(f"Unknown task: {task_name}")
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=4, pin_memory=True)

# Training function for a single expert
def train_expert(task_name, epochs=5, lr=5e-4):
    print(f"\n--- Training Expert for {task_name} ---")
    model = MultiHeadResNet().to(DEVICE)
    
    progenitor_path = "checkpoints/progenitor.pth"
    if not os.path.exists(progenitor_path):
        torch.save(model.state_dict(), progenitor_path)
        print(f"Saved initial progenitor checkpoint to {progenitor_path}")
    else:
        model.load_state_dict(torch.load(progenitor_path, map_location=DEVICE))
    
    train_loader = get_dataloader(task_name, batch_size=128, train=True)
    
    for name, param in model.heads.items():
        if name != task_name:
            param.requires_grad = False
            
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs, task_name)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.2f}% - Time: {elapsed:.1f}s")
        
    ckpt_path = f"checkpoints/expert_{task_name}.pth"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved expert checkpoint to {ckpt_path}")
    return model

# Evaluation function
def evaluate_model(model, task_name, split='test'):
    loader = get_dataloader(task_name, batch_size=256, train=(split=='train'))
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs, task_name)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    acc = 100.0 * correct / total
    return acc

# --- Helper Functions for Merging ---

def load_models():
    progenitor = MultiHeadResNet().to(DEVICE)
    progenitor.load_state_dict(torch.load("checkpoints/progenitor.pth", map_location=DEVICE))
    
    experts = {}
    for task in ['mnist', 'fashion_mnist', 'cifar10']:
        path = f"checkpoints/expert_{task}.pth"
        model = MultiHeadResNet().to(DEVICE)
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        experts[task] = model
        
    return progenitor, experts

def is_bn(key):
    return 'bn' in key or 'downsample.1' in key or 'running' in key or 'tracked' in key

def get_task_vectors(progenitor, experts):
    task_vectors = {}
    progenitor_state = progenitor.state_dict()
    for task_name, expert in experts.items():
        expert_state = expert.state_dict()
        task_vector = {}
        for k in progenitor_state.keys():
            if 'backbone' in k and not is_bn(k):
                if progenitor_state[k].dtype.is_floating_point:
                    task_vector[k] = expert_state[k] - progenitor_state[k]
        task_vectors[task_name] = task_vector
    return task_vectors

# Modular assembly function
def assemble_merged_model(progenitor, experts, backbone_weights, bn_mode='average', bn_target_task=None):
    merged_model = copy.deepcopy(progenitor)
    merged_state = merged_model.state_dict()
    
    for k in backbone_weights.keys():
        merged_state[k] = backbone_weights[k]
        
    for task in ['mnist', 'fashion_mnist', 'cifar10']:
        expert_state = experts[task].state_dict()
        for k in expert_state.keys():
            if f'heads.{task}' in k:
                merged_state[k] = expert_state[k]
                
    for k in progenitor.state_dict().keys():
        if is_bn(k):
            if bn_mode == 'average':
                stats = []
                for task in ['mnist', 'fashion_mnist', 'cifar10']:
                    stats.append(experts[task].state_dict()[k].to(torch.float32))
                avg_stat = torch.stack(stats, dim=0).mean(dim=0)
                if progenitor.state_dict()[k].dtype == torch.int64:
                    merged_state[k] = avg_stat.to(torch.int64)
                else:
                    merged_state[k] = avg_stat.to(progenitor.state_dict()[k].dtype)
            elif bn_mode == 'specialized':
                assert bn_target_task is not None
                merged_state[k] = experts[bn_target_task].state_dict()[k]
                
    merged_model.load_state_dict(merged_state)
    return merged_model

# --- Merging Backbone Algorithms ---

def merge_backbone_task_arithmetic(progenitor, task_vectors, scaling_factor=1.0):
    progenitor_state = progenitor.state_dict()
    merged_backbone = {}
    for k in task_vectors['mnist'].keys():
        updates = []
        for task in ['mnist', 'fashion_mnist', 'cifar10']:
            updates.append(task_vectors[task][k])
        avg_update = torch.stack(updates, dim=0).mean(dim=0)
        merged_backbone[k] = progenitor_state[k] + scaling_factor * avg_update
    return merged_backbone

def merge_backbone_u_ipr(progenitor, task_vectors):
    progenitor_state = progenitor.state_dict()
    merged_backbone = {}
    for k in task_vectors['mnist'].keys():
        updates = []
        for task in ['mnist', 'fashion_mnist', 'cifar10']:
            updates.append(task_vectors[task][k])
        merged_update = torch.stack(updates, dim=0).mean(dim=0)
        
        expert_norms = [torch.norm(u, p='fro').item() for u in updates]
        avg_expert_norm = np.mean(expert_norms)
        merged_norm = torch.norm(merged_update, p='fro').item()
        
        u_ipr_scale = avg_expert_norm / (merged_norm + 1e-8)
        u_ipr_scale = np.clip(u_ipr_scale, 0.1, 10.0)
        merged_backbone[k] = progenitor_state[k] + u_ipr_scale * merged_update
    return merged_backbone

def merge_backbone_s_ipr(progenitor, task_vectors):
    progenitor_state = progenitor.state_dict()
    merged_backbone = {}
    for k in task_vectors['mnist'].keys():
        updates = []
        for task in ['mnist', 'fashion_mnist', 'cifar10']:
            updates.append(task_vectors[task][k])
        merged_update = torch.stack(updates, dim=0).mean(dim=0)
        
        if merged_update.dim() >= 2:
            orig_shape = merged_update.shape
            h_dim = orig_shape[0]
            w_dim = int(merged_update.numel() / h_dim)
            
            try:
                merged_2d = merged_update.view(h_dim, w_dim)
                U, S_merged, V = torch.linalg.svd(merged_2d, full_matrices=False)
                expert_spectrums = []
                for u in updates:
                    u_2d = u.view(h_dim, w_dim)
                    _, S_expert, _ = torch.linalg.svd(u_2d, full_matrices=False)
                    expert_spectrums.append(S_expert)
                avg_spectrum = torch.stack(expert_spectrums, dim=0).mean(dim=0)
                corrected_2d = U @ torch.diag(avg_spectrum) @ V
                corrected_update = corrected_2d.view(orig_shape)
            except Exception as e:
                expert_norms = [torch.norm(u, p='fro').item() for u in updates]
                avg_expert_norm = np.mean(expert_norms)
                merged_norm = torch.norm(merged_update, p='fro').item()
                u_ipr_scale = avg_expert_norm / (merged_norm + 1e-8)
                u_ipr_scale = np.clip(u_ipr_scale, 0.1, 10.0)
                corrected_update = u_ipr_scale * merged_update
        else:
            expert_norms = [torch.norm(u, p='fro').item() for u in updates]
            avg_expert_norm = np.mean(expert_norms)
            merged_norm = torch.norm(merged_update, p='fro').item()
            u_ipr_scale = avg_expert_norm / (merged_norm + 1e-8)
            u_ipr_scale = np.clip(u_ipr_scale, 0.1, 10.0)
            corrected_update = u_ipr_scale * merged_update
            
        merged_backbone[k] = progenitor_state[k] + corrected_update
    return merged_backbone

def merge_backbone_hns(progenitor, task_vectors, target_task):
    progenitor_state = progenitor.state_dict()
    merged_backbone = {}
    for k in task_vectors['mnist'].keys():
        updates = []
        for task in ['mnist', 'fashion_mnist', 'cifar10']:
            updates.append(task_vectors[task][k])
        merged_update = torch.stack(updates, dim=0).mean(dim=0)
        expert_update = task_vectors[target_task][k]
        
        if merged_update.dim() >= 2:
            c_out = merged_update.shape[0]
            corrected_update = torch.zeros_like(merged_update)
            for c in range(c_out):
                exp_chan_norm = torch.norm(expert_update[c], p=2).item()
                mer_chan_norm = torch.norm(merged_update[c], p=2).item()
                gamma = exp_chan_norm / (mer_chan_norm + 1e-8)
                gamma = np.clip(gamma, 0.1, 10.0)
                corrected_update[c] = gamma * merged_update[c]
        else:
            exp_norm = torch.norm(expert_update, p=2).item()
            mer_norm = torch.norm(merged_update, p=2).item()
            gamma = exp_norm / (mer_norm + 1e-8)
            gamma = np.clip(gamma, 0.1, 10.0)
            corrected_update = gamma * merged_update
            
        merged_backbone[k] = progenitor_state[k] + corrected_update
    return merged_backbone

def merge_backbone_ties(progenitor, task_vectors, density=0.2):
    progenitor_state = progenitor.state_dict()
    merged_backbone = {}
    
    for k in task_vectors['mnist'].keys():
        updates = []
        for task in ['mnist', 'fashion_mnist', 'cifar10']:
            updates.append(task_vectors[task][k])
            
        # 1. Trim
        trimmed_updates = []
        for u in updates:
            flat_u = u.flatten()
            num_el = flat_u.numel()
            k_val = int(density * num_el)
            if k_val == 0:
                k_val = 1
            # Get top k by absolute value
            abs_u = torch.abs(flat_u)
            threshold = torch.topk(abs_u, k_val).values[-1]
            mask = (abs_u >= threshold).view_as(u)
            trimmed_updates.append(u * mask)
            
        # Stack trimmed updates
        stacked_trimmed = torch.stack(trimmed_updates, dim=0) # [K, ...]
        
        # 2. Elect Sign
        signs = torch.sign(stacked_trimmed)
        sign_sum = signs.sum(dim=0)
        consensus_sign = torch.sign(sign_sum) # [...]
        
        # 3. Disjoint Merge
        matching_mask = (signs == consensus_sign.unsqueeze(0)) & (stacked_trimmed != 0)
        sum_matching = (stacked_trimmed * matching_mask).sum(dim=0)
        count_matching = matching_mask.sum(dim=0)
        
        merged_update = torch.where(count_matching > 0, sum_matching / (count_matching + 1e-8), torch.zeros_like(sum_matching))
        merged_update = merged_update * (consensus_sign != 0).float()
        
        merged_backbone[k] = progenitor_state[k] + merged_update
    return merged_backbone

def merge_backbone_dare(progenitor, task_vectors, drop_rate=0.2):
    progenitor_state = progenitor.state_dict()
    merged_backbone = {}
    
    for k in task_vectors['mnist'].keys():
        updates = []
        for task in ['mnist', 'fashion_mnist', 'cifar10']:
            updates.append(task_vectors[task][k])
            
        dare_updates = []
        for u in updates:
            torch.manual_seed(42)
            mask = (torch.rand_like(u) > drop_rate).float()
            rescaled_u = (u * mask) / (1.0 - drop_rate)
            dare_updates.append(rescaled_u)
            
        merged_update = torch.stack(dare_updates, dim=0).mean(dim=0)
        merged_backbone[k] = progenitor_state[k] + merged_update
    return merged_backbone

def compute_fisher_diagonal(experts):
    print("\n--- Estimating Diagonal Fisher Information for Experts ---")
    fisher_dicts = {}
    criterion = nn.CrossEntropyLoss()
    
    for task_name, expert in experts.items():
        print(f"Estimating Fisher on task: {task_name}...")
        model = copy.deepcopy(expert).to(DEVICE)
        model.eval()
        
        for p in model.parameters():
            p.requires_grad = True
            
        fisher = {k: torch.zeros_like(p) for k, p in model.named_parameters() if 'backbone' in k and not is_bn(k)}
        
        train_loader = get_dataloader(task_name, batch_size=64, train=True)
        num_batches = 20
        batch_count = 0
        
        for inputs, targets in train_loader:
            if batch_count >= num_batches:
                break
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            model.zero_grad()
            outputs = model(inputs, task_name)
            loss = criterion(outputs, targets)
            loss.backward()
            
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if name in fisher and p.grad is not None:
                        fisher[name] += p.grad.data ** 2
                        
            batch_count += 1
            
        with torch.no_grad():
            for k in fisher.keys():
                fisher[k] /= float(batch_count)
                
        fisher_dicts[task_name] = fisher
        print(f"Fisher estimation complete for {task_name}.")
        
    return fisher_dicts

def merge_backbone_fisher(progenitor, task_vectors, fisher_dicts):
    progenitor_state = progenitor.state_dict()
    merged_backbone = {}
    
    for k in task_vectors['mnist'].keys():
        updates = []
        fishers = []
        for task in ['mnist', 'fashion_mnist', 'cifar10']:
            updates.append(task_vectors[task][k])
            fishers.append(fisher_dicts[task][k])
            
        weighted_sum_update = torch.zeros_like(updates[0])
        sum_fisher = torch.zeros_like(updates[0])
        
        for u, f in zip(updates, fishers):
            weighted_sum_update += f * u
            sum_fisher += f
            
        merged_update = weighted_sum_update / (sum_fisher + 1e-8)
        merged_backbone[k] = progenitor_state[k] + merged_update
    return merged_backbone

def merge_backbone_gmpr(progenitor, task_vectors):
    progenitor_state = progenitor.state_dict()
    merged_backbone = {}
    for k in task_vectors['mnist'].keys():
        updates = []
        for task in ['mnist', 'fashion_mnist', 'cifar10']:
            updates.append(task_vectors[task][k])
        merged_update = torch.stack(updates, dim=0).mean(dim=0)
        
        shape = merged_update.shape
        corrected_update = torch.zeros_like(merged_update)
        
        if merged_update.dim() >= 2:
            c_out = shape[0]
            for c in range(c_out):
                m_slice = merged_update[c]
                
                exp_means = []
                exp_stds = []
                for exp_update in updates:
                    exp_means.append(exp_update[c].mean().item())
                    exp_stds.append(exp_update[c].std().item())
                
                mean_target = np.mean(exp_means)
                std_target = np.mean(exp_stds)
                
                mean_merged = m_slice.mean().item()
                std_merged = m_slice.std().item()
                
                standardized = (m_slice - mean_merged) / (std_merged + 1e-8)
                corrected_update[c] = mean_target + std_target * standardized
        else:
            expert_norms = [torch.norm(u, p='fro').item() for u in updates]
            avg_expert_norm = np.mean(expert_norms)
            merged_norm = torch.norm(merged_update, p='fro').item()
            u_ipr_scale = avg_expert_norm / (merged_norm + 1e-8)
            u_ipr_scale = np.clip(u_ipr_scale, 0.1, 10.0)
            corrected_update = u_ipr_scale * merged_update
            
        merged_backbone[k] = progenitor_state[k] + corrected_update
    return merged_backbone

# WCPR - Flexible Granularity Calibration (for Ablations)
def merge_backbone_wcpr(progenitor, task_vectors, mode='unified', target_task=None, granularity='channel', merge_method='average', fisher_dicts=None):
    progenitor_state = progenitor.state_dict()
    merged_backbone = {}
    
    for k in task_vectors['mnist'].keys():
        updates = []
        for task in ['mnist', 'fashion_mnist', 'cifar10']:
            updates.append(task_vectors[task][k])
            
        if merge_method == 'average':
            merged_update = torch.stack(updates, dim=0).mean(dim=0)
        elif merge_method == 'ties':
            density = 0.2
            trimmed_updates = []
            for u in updates:
                flat_u = u.flatten()
                num_el = flat_u.numel()
                k_val = int(density * num_el)
                if k_val == 0:
                    k_val = 1
                abs_u = torch.abs(flat_u)
                threshold = torch.topk(abs_u, k_val).values[-1]
                mask = (abs_u >= threshold).view_as(u)
                trimmed_updates.append(u * mask)
            stacked_trimmed = torch.stack(trimmed_updates, dim=0)
            signs = torch.sign(stacked_trimmed)
            sign_sum = signs.sum(dim=0)
            consensus_sign = torch.sign(sign_sum)
            matching_mask = (signs == consensus_sign.unsqueeze(0)) & (stacked_trimmed != 0)
            sum_matching = (stacked_trimmed * matching_mask).sum(dim=0)
            count_matching = matching_mask.sum(dim=0)
            merged_update = torch.where(count_matching > 0, sum_matching / (count_matching + 1e-8), torch.zeros_like(sum_matching))
            merged_update = merged_update * (consensus_sign != 0).float()
        elif merge_method == 'dare':
            drop_rate = 0.2
            dare_updates = []
            for u in updates:
                torch.manual_seed(42)
                mask = (torch.rand_like(u) > drop_rate).float()
                rescaled_u = (u * mask) / (1.0 - drop_rate)
                dare_updates.append(rescaled_u)
            merged_update = torch.stack(dare_updates, dim=0).mean(dim=0)
        elif merge_method == 'fisher':
            assert fisher_dicts is not None
            weighted_sum_update = torch.zeros_like(updates[0])
            sum_fisher = torch.zeros_like(updates[0])
            for idx, task in enumerate(['mnist', 'fashion_mnist', 'cifar10']):
                u = updates[idx]
                f = fisher_dicts[task][k]
                weighted_sum_update += f * u
                sum_fisher += f
            merged_update = weighted_sum_update / (sum_fisher + 1e-8)
        
        shape = merged_update.shape
        corrected_update = torch.zeros_like(merged_update)
        
        if merged_update.dim() >= 2 and granularity == 'channel':
            # Proposed: Channel-wise calibration
            c_out = shape[0]
            for c in range(c_out):
                m_slice = merged_update[c]
                flat_m = m_slice.flatten()
                sort_idx = torch.argsort(flat_m)
                
                sorted_experts = []
                for exp_update in updates:
                    sorted_experts.append(torch.sort(exp_update[c].flatten())[0])
                    
                if mode == 'unified':
                    flat_target = torch.stack(sorted_experts, dim=0).mean(dim=0)
                else:
                    assert target_task is not None
                    tgt_idx = ['mnist', 'fashion_mnist', 'cifar10'].index(target_task)
                    flat_target = sorted_experts[tgt_idx]
                    
                calibrated_flat = torch.zeros_like(flat_m)
                calibrated_flat[sort_idx] = flat_target
                corrected_update[c] = calibrated_flat.view(m_slice.shape)
                
        elif merged_update.dim() >= 2 and granularity == 'global':
            # Ablation: Global layer-wise sorting calibration (catastrophic mix across channels)
            flat_m = merged_update.flatten()
            sort_idx = torch.argsort(flat_m)
            
            sorted_experts = []
            for exp_update in updates:
                sorted_experts.append(torch.sort(exp_update.flatten())[0])
                
            if mode == 'unified':
                flat_target = torch.stack(sorted_experts, dim=0).mean(dim=0)
            else:
                assert target_task is not None
                tgt_idx = ['mnist', 'fashion_mnist', 'cifar10'].index(target_task)
                flat_target = sorted_experts[tgt_idx]
                
            calibrated_flat = torch.zeros_like(flat_m)
            calibrated_flat[sort_idx] = flat_target
            corrected_update = calibrated_flat.view(shape)
            
        else:
            # 1D parameters or non-channel fallback (isotropic)
            expert_norms = [torch.norm(u, p='fro').item() for u in updates]
            avg_expert_norm = np.mean(expert_norms)
            merged_norm = torch.norm(merged_update, p='fro').item()
            u_ipr_scale = avg_expert_norm / (merged_norm + 1e-8)
            u_ipr_scale = np.clip(u_ipr_scale, 0.1, 10.0)
            corrected_update = u_ipr_scale * merged_update
            
        merged_backbone[k] = progenitor_state[k] + corrected_update
        
    return merged_backbone


def analyze_moments(progenitor, experts, task_vectors):
    print("\n" + "="*20 + " STATISTICAL MOMENT ANALYSIS (LAYER 4) " + "="*20)
    k = 'backbone.layer4.1.conv2.weight'
    if k not in task_vectors['mnist']:
        for key in task_vectors['mnist'].keys():
            if 'conv' in key:
                k = key
                break
                
    print(f"Analyzing layer: {k}")
    
    exp_updates = [task_vectors[t][k] for t in ['mnist', 'fashion_mnist', 'cifar10']]
    merged_update = torch.stack(exp_updates, dim=0).mean(dim=0)
    
    wcpr_update = torch.zeros_like(merged_update)
    c_out = merged_update.shape[0]
    for c in range(c_out):
        m_slice = merged_update[c]
        flat_m = m_slice.flatten()
        sort_idx = torch.argsort(flat_m)
        
        sorted_experts = []
        for exp in exp_updates:
            sorted_experts.append(torch.sort(exp[c].flatten())[0])
            
        flat_target = torch.stack(sorted_experts, dim=0).mean(dim=0)
        calibrated_flat = torch.zeros_like(flat_m)
        calibrated_flat[sort_idx] = flat_target
        wcpr_update[c] = calibrated_flat.view(m_slice.shape)
        
    gmpr_update = torch.zeros_like(merged_update)
    for c in range(c_out):
        m_slice = merged_update[c]
        
        exp_means = []
        exp_stds = []
        for exp in exp_updates:
            exp_means.append(exp[c].mean().item())
            exp_stds.append(exp[c].std().item())
            
        mean_target = np.mean(exp_means)
        std_target = np.mean(exp_stds)
        
        mean_merged = m_slice.mean().item()
        std_merged = m_slice.std().item()
        
        standardized = (m_slice - mean_merged) / (std_merged + 1e-8)
        gmpr_update[c] = mean_target + std_target * standardized
        
    def get_stats(tensor):
        flat = tensor.flatten()
        mean = flat.mean().item()
        var = flat.var().item()
        std = flat.std().item()
        
        skew = torch.mean(((flat - mean) / (std + 1e-8))**3).item()
        kurt = torch.mean(((flat - mean) / (std + 1e-8))**4).item()
        
        return mean, var, skew, kurt

    print(f"{'Method':<20} | {'Mean':<10} | {'Variance':<10} | {'Skewness':<10} | {'Kurtosis':<10}")
    print("-" * 70)
    
    for t in ['mnist', 'fashion_mnist', 'cifar10']:
        m, v, s, ku = get_stats(task_vectors[t][k])
        print(f"Expert {t:<13} | {m:<10.6f} | {v:<10.6f} | {s:<10.4f} | {ku:<10.4f}")
        
    m_exp, v_exp, s_exp, ku_exp = zip(*[get_stats(task_vectors[t][k]) for t in ['mnist', 'fashion_mnist', 'cifar10']])
    print("-" * 70)
    print(f"{'Experts Target Avg':<20} | {np.mean(m_exp):<10.6f} | {np.mean(v_exp):<10.6f} | {np.mean(s_exp):<10.4f} | {np.mean(ku_exp):<10.4f}")
    
    m_un, v_un, s_un, ku_un = get_stats(merged_update)
    print(f"{'Merged (WA) Uncal.':<20} | {m_un:<10.6f} | {v_un:<10.6f} | {s_un:<10.4f} | {ku_un:<10.4f}")
    
    m_gm, v_gm, s_gm, ku_gm = get_stats(gmpr_update)
    print(f"{'GMPR (Parametric)':<20} | {m_gm:<10.6f} | {v_gm:<10.6f} | {s_gm:<10.4f} | {ku_gm:<10.4f}")
    
    m_wc, v_wc, s_wc, ku_wc = get_stats(wcpr_update)
    print(f"{'WCPR (Ours)':<20} | {m_wc:<10.6f} | {v_wc:<10.6f} | {s_wc:<10.4f} | {ku_wc:<10.4f}")
    print("="*79 + "\n")
    
    with open("results/comparison_results.txt", "a") as f:
        f.write("\n" + "="*20 + " STATISTICAL MOMENT ANALYSIS (LAYER 4) " + "="*20 + "\n")
        f.write(f"Analyzing layer: {k}\n")
        f.write(f"{'Method':<20} | {'Mean':<10} | {'Variance':<10} | {'Skewness':<10} | {'Kurtosis':<10}\n")
        f.write("-" * 70 + "\n")
        for t in ['mnist', 'fashion_mnist', 'cifar10']:
            m, v, s, ku = get_stats(task_vectors[t][k])
            f.write(f"Expert {t:<13} | {m:<10.6f} | {v:<10.6f} | {s:<10.4f} | {ku:<10.4f}\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Experts Target Avg':<20} | {np.mean(m_exp):<10.6f} | {np.mean(v_exp):<10.6f} | {np.mean(s_exp):<10.4f} | {np.mean(ku_exp):<10.4f}\n")
        f.write(f"{'Merged (WA) Uncal.':<20} | {m_un:<10.6f} | {v_un:<10.6f} | {s_un:<10.4f} | {ku_un:<10.4f}\n")
        f.write(f"{'GMPR (Parametric)':<20} | {m_gm:<10.6f} | {v_gm:<10.6f} | {s_gm:<10.4f} | {ku_gm:<10.4f}\n")
        f.write(f"{'WCPR (Ours)':<20} | {m_wc:<10.6f} | {v_wc:<10.6f} | {s_wc:<10.4f} | {ku_wc:<10.4f}\n")
        f.write("="*79 + "\n")


# --- MAIN PIPELINE ---

def main():
    tasks = ['mnist', 'fashion_mnist', 'cifar10']
    
    progenitor, experts = load_models()
    
    print("\nEvaluating Individual Oracle Experts (Upper Bound):")
    oracle_scores = {}
    for task in tasks:
        score = evaluate_model(experts[task], task)
        oracle_scores[task] = score
        print(f"Oracle Expert on {task}: {score:.2f}%")
        
    task_vectors = get_task_vectors(progenitor, experts)
    
    # Estimate diagonal Fisher information for Fisher and Fisher+WCPR configs
    fisher_dicts = compute_fisher_diagonal(experts)
    
    # 1. Main Configurations Evaluation
    configs = [
        ("Weight Averaging (WA) Uncalibrated", lambda p, exps, tv: assemble_merged_model(p, exps, merge_backbone_task_arithmetic(p, tv, scaling_factor=1.0), bn_mode='average')),
        ("Task Arithmetic (TA, lambda=0.2)", lambda p, exps, tv: assemble_merged_model(p, exps, merge_backbone_task_arithmetic(p, tv, scaling_factor=0.2), bn_mode='average')),
        ("Task Arithmetic (TA, lambda=0.5)", lambda p, exps, tv: assemble_merged_model(p, exps, merge_backbone_task_arithmetic(p, tv, scaling_factor=0.5), bn_mode='average')),
        ("Update-level IPR (U-IPR)", lambda p, exps, tv: assemble_merged_model(p, exps, merge_backbone_u_ipr(p, tv), bn_mode='average')),
        ("Spectral Parameter Resonance (S-IPR)", lambda p, exps, tv: assemble_merged_model(p, exps, merge_backbone_s_ipr(p, tv), bn_mode='average')),
        ("HNS (Task-Specific Calibration)", None),
        ("Gaussian-Matching Parameter Resonance (GMPR)", lambda p, exps, tv: assemble_merged_model(p, exps, merge_backbone_gmpr(p, tv), bn_mode='average')),
        ("TIES-Merging", lambda p, exps, tv: assemble_merged_model(p, exps, merge_backbone_ties(p, tv), bn_mode='average')),
        ("DARE-Merging", lambda p, exps, tv: assemble_merged_model(p, exps, merge_backbone_dare(p, tv), bn_mode='average')),
        ("Fisher-Weighted Averaging", lambda p, exps, tv: assemble_merged_model(p, exps, merge_backbone_fisher(p, tv, fisher_dicts), bn_mode='average')),
        ("WCPR (Ours, Unified Static Model)", lambda p, exps, tv: assemble_merged_model(p, exps, merge_backbone_wcpr(p, tv, mode='unified', granularity='channel'), bn_mode='average')),
        ("TIES + WCPR (Ours)", lambda p, exps, tv: assemble_merged_model(p, exps, merge_backbone_wcpr(p, tv, mode='unified', granularity='channel', merge_method='ties'), bn_mode='average')),
        ("DARE + WCPR (Ours)", lambda p, exps, tv: assemble_merged_model(p, exps, merge_backbone_wcpr(p, tv, mode='unified', granularity='channel', merge_method='dare'), bn_mode='average')),
        ("Fisher + WCPR (Ours)", lambda p, exps, tv: assemble_merged_model(p, exps, merge_backbone_wcpr(p, tv, mode='unified', granularity='channel', merge_method='fisher', fisher_dicts=fisher_dicts), bn_mode='average')),
        ("WCPR (Ours, Task-Specific Calibration)", None),
    ]
    
    results = []
    
    for name, merge_fn in configs:
        print(f"\n--- Evaluating Method: {name} ---")
        scores = {}
        
        if name == "HNS (Task-Specific Calibration)":
            for task in tasks:
                hns_backbone = merge_backbone_hns(progenitor, task_vectors, task)
                hns_model = assemble_merged_model(progenitor, experts, hns_backbone, bn_mode='specialized', bn_target_task=task)
                scores[task] = evaluate_model(hns_model, task)
        elif name == "WCPR (Ours, Task-Specific Calibration)":
            for task in tasks:
                wcpr_spec_backbone = merge_backbone_wcpr(progenitor, task_vectors, mode='specialized', target_task=task, granularity='channel')
                wcpr_spec_model = assemble_merged_model(progenitor, experts, wcpr_spec_backbone, bn_mode='specialized', bn_target_task=task)
                scores[task] = evaluate_model(wcpr_spec_model, task)
        else:
            merged_model = merge_fn(progenitor, experts, task_vectors)
            if "WCPR (Ours, Unified Static Model)" in name:
                torch.save(merged_model.state_dict(), "checkpoints/submission_wcpr.pth")
                torch.save(merged_model.state_dict(), "checkpoints/submission.pth")
            for task in tasks:
                scores[task] = evaluate_model(merged_model, task)
                
        avg_score = np.mean([scores[t] for t in tasks])
        print(f"Scores - MNIST: {scores['mnist']:.2f}%, F-MNIST: {scores['fashion_mnist']:.2f}%, CIFAR-10: {scores['cifar10']:.2f}%, Avg: {avg_score:.2f}%")
        results.append((name, scores['mnist'], scores['fashion_mnist'], scores['cifar10'], avg_score))
        
    # 2. Ablation Studies Evaluation: Sorting Granularity
    print("\n" + "="*20 + " ABLATION STUDY: SORTING GRANULARITY " + "="*20)
    
    ablation_configs = [
        ("WCPR (Global Layer-wise Sorting)", lambda p, exps, tv: assemble_merged_model(p, exps, merge_backbone_wcpr(p, tv, mode='unified', granularity='global'), bn_mode='average')),
        ("WCPR (Channel-wise Sorting - Proposed)", lambda p, exps, tv: assemble_merged_model(p, exps, merge_backbone_wcpr(p, tv, mode='unified', granularity='channel'), bn_mode='average')),
    ]
    
    ablation_results = []
    for name, merge_fn in ablation_configs:
        print(f"\n--- Evaluating Ablation: {name} ---")
        scores = {}
        merged_model = merge_fn(progenitor, experts, task_vectors)
        for task in tasks:
            scores[task] = evaluate_model(merged_model, task)
        avg_score = np.mean([scores[t] for t in tasks])
        print(f"Scores - MNIST: {scores['mnist']:.2f}%, F-MNIST: {scores['fashion_mnist']:.2f}%, CIFAR-10: {scores['cifar10']:.2f}%, Avg: {avg_score:.2f}%")
        ablation_results.append((name, scores['mnist'], scores['fashion_mnist'], scores['cifar10'], avg_score))

    # Print main comparison table
    print("\n" + "="*80)
    print(f"{'Method':<45} | {'MNIST':<7} | {'F-MNIST':<7} | {'CIFAR-10':<8} | {'Average':<7}")
    print("-"*80)
    print(f"{'Oracle Experts (Individual)':<45} | {oracle_scores['mnist']:<7.2f} | {oracle_scores['fashion_mnist']:<7.2f} | {oracle_scores['cifar10']:<8.2f} | {np.mean([oracle_scores[t] for t in tasks]):<7.2f}")
    print("-"*80)
    for res in results:
        print(f"{res[0]:<45} | {res[1]:<7.2f} | {res[2]:<7.2f} | {res[3]:<8.2f} | {res[4]:<7.2f}")
    print("="*80)
    
    # Print ablation study table
    print("\n" + "="*80)
    print(f"{'Ablation: Sorting Granularity':<45} | {'MNIST':<7} | {'F-MNIST':<7} | {'CIFAR-10':<8} | {'Average':<7}")
    print("-"*80)
    for res in ablation_results:
        print(f"{res[0]:<45} | {res[1]:<7.2f} | {res[2]:<7.2f} | {res[3]:<8.2f} | {res[4]:<7.2f}")
    print("="*80)
    
    # Save all results to txt file
    with open("results/comparison_results.txt", "w") as f:
        f.write("Method | MNIST | F-MNIST | CIFAR-10 | Average\n")
        f.write("-" * 60 + "\n")
        f.write(f"Oracle Experts (Individual) | {oracle_scores['mnist']:.2f} | {oracle_scores['fashion_mnist']:.2f} | {oracle_scores['cifar10']:.2f} | {np.mean([oracle_scores[t] for t in tasks]):.2f}\n")
        for res in results:
            f.write(f"{res[0]} | {res[1]:.2f} | {res[2]:.2f} | {res[3]:.2f} | {res[4]:.2f}\n")
        f.write("\nABLATION STUDY: SORTING GRANULARITY\n")
        f.write("Ablation Config | MNIST | F-MNIST | CIFAR-10 | Average\n")
        f.write("-" * 60 + "\n")
        for res in ablation_results:
            f.write(f"{res[0]} | {res[1]:.2f} | {res[2]:.2f} | {res[3]:.2f} | {res[4]:.2f}\n")
            
    print("\nResults and Ablations successfully written to results/comparison_results.txt")
    
    # Run statistical moment analysis
    analyze_moments(progenitor, experts, task_vectors)

if __name__ == "__main__":
    main()
