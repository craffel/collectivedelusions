import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import Subset, DataLoader, TensorDataset
import numpy as np
import random
import os
import json
import time

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False  # Disable cuDNN to bypass custom cluster initialization issues

# Replicate grayscale channel to 3 channels
class ReplicateChannels(object):
    def __call__(self, img):
        if img.shape[0] == 1:
            return img.repeat(3, 1, 1)
        return img

def get_dataloaders(seed=42):
    set_seed(seed)
    
    transform_mnist = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        ReplicateChannels(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    transform_fmnist = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        ReplicateChannels(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    transform_cifar = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load raw datasets (downloading to './data')
    mnist_train_full = torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=transform_mnist)
    mnist_test_full = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform_mnist)
    
    fmnist_train_full = torchvision.datasets.FashionMNIST(root='./data', train=True, download=False, transform=transform_fmnist)
    fmnist_test_full = torchvision.datasets.FashionMNIST(root='./data', train=False, download=False, transform=transform_fmnist)
    
    cifar_train_full = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_cifar)
    cifar_test_full = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_cifar)
    
    # Deterministically select 5,000 samples for training experts
    def get_subset(dataset, num_samples=5000):
        indices = list(range(len(dataset)))
        rng = random.Random(42)
        rng.shuffle(indices)
        return Subset(dataset, indices[:num_samples])
        
    mnist_train_subset = get_subset(mnist_train_full, 5000)
    fmnist_train_subset = get_subset(fmnist_train_full, 5000)
    cifar_train_subset = get_subset(cifar_train_full, 5000)
    
    loaders = {
        'mnist': {
            'train_expert': DataLoader(mnist_train_subset, batch_size=64, shuffle=True, num_workers=2),
            'test': DataLoader(mnist_test_full, batch_size=256, shuffle=False, num_workers=2),
        },
        'fmnist': {
            'train_expert': DataLoader(fmnist_train_subset, batch_size=64, shuffle=True, num_workers=2),
            'test': DataLoader(fmnist_test_full, batch_size=256, shuffle=False, num_workers=2),
        },
        'cifar': {
            'train_expert': DataLoader(cifar_train_subset, batch_size=64, shuffle=True, num_workers=2),
            'test': DataLoader(cifar_test_full, batch_size=256, shuffle=False, num_workers=2),
        }
    }
    return loaders

def get_resnet18_expert():
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.1),
        nn.Linear(512, 10)
    )
    return model

def train_expert(name, train_loader, device, epochs=5):
    print(f"Training expert for {name}...")
    model = get_resnet18_expert().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.2f}%")
        
    model.eval()
    return model

# Merge Backbones (WA)
def merge_wa(expert_models):
    merged_state_dict = {}
    keys = expert_models[0].state_dict().keys()
    for key in keys:
        if 'fc' in key:
            continue
        weights = [model.state_dict()[key].float() for model in expert_models]
        merged_state_dict[key] = torch.stack(weights).mean(dim=0)
    return merged_state_dict

# Merge Backbones (TA)
def merge_ta(expert_models, base_model, lam=0.3):
    merged_state_dict = {}
    base_state = base_model.state_dict()
    keys = base_state.keys()
    for key in keys:
        if 'fc' in key:
            continue
        task_vectors = []
        for model in expert_models:
            task_vectors.append(model.state_dict()[key].float() - base_state[key].float())
        merged_state_dict[key] = base_state[key].float() + lam * torch.stack(task_vectors).sum(dim=0)
    return merged_state_dict

# BN Calibration (N-TAAC)
def calibrate_bn(model, calibration_loader, device):
    model.to(device)
    model.train()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None  # Cumulative average
            
    with torch.no_grad():
        for inputs, _ in calibration_loader:
            inputs = inputs.to(device)
            _ = model(inputs)
            
    # Reset momentum back to default 0.1
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = 0.1
    model.eval()

# Feature Extractor for MSPR (Layer 2 Hook)
class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.features = None
        self.hook = self.model.layer2.register_forward_hook(self.save_features)
        
    def save_features(self, module, input, output):
        self.features = torch.mean(output, dim=(2, 3)) # GAP
        
    def forward(self, x):
        self.features = None
        _ = self.model(x)
        return self.features
        
    def close(self):
        self.hook.remove()

# Extract 1D Intensity Histogram
def extract_histograms(x, bins=16):
    intensity = x.mean(dim=1)  # (B, H, W)
    histograms = []
    for i in range(x.shape[0]):
        h = torch.histc(intensity[i], bins=bins, min=-1.0, max=1.0)
        h = h / (h.sum() + 1e-8)
        histograms.append(h)
    return torch.stack(histograms)

# Extract Downsampled Intensity
def extract_downsampled(x, size=6):
    intensity = x.mean(dim=1, keepdim=True)  # (B, 1, H, W)
    down = torch.nn.functional.adaptive_avg_pool2d(intensity, (size, size))
    down = down.view(x.shape[0], -1)
    down_norm = down / (down.norm(p=2, dim=1, keepdim=True) + 1e-8)
    return down_norm

# Extract Hybrid Representation
def extract_hybrid(x, bins=16, size=6, hist_weight=1.0, spatial_weight=1.0):
    hists = extract_histograms(x, bins=bins)
    hists_norm = hists / (hists.norm(p=2, dim=1, keepdim=True) + 1e-8)
    spat = extract_downsampled(x, size=size)
    hybrid = torch.cat([hist_weight * hists_norm, spatial_weight * spat], dim=1)
    hybrid_norm = hybrid / (hybrid.norm(p=2, dim=1, keepdim=True) + 1e-8)
    return hybrid_norm

# Train a microscopic linear boundary classifier
def train_micro_linear_router(calib_images, feature_fn, feat_dim, device, lr=0.1, epochs=150):
    tasks = ['mnist', 'fmnist', 'cifar']
    task_to_idx = {name: i for i, name in enumerate(tasks)}
    
    train_feats_list = []
    train_labels_list = []
    for name, inputs in calib_images.items():
        feats = feature_fn(inputs)
        train_feats_list.append(feats)
        train_labels_list.append(torch.full((inputs.size(0),), task_to_idx[name], dtype=torch.long))
        
    train_feats = torch.cat(train_feats_list).to(device)
    train_labels = torch.cat(train_labels_list).to(device)
    
    model = nn.Linear(feat_dim, 3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    dataset = TensorDataset(train_feats, train_labels)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model.train()
    for _ in range(epochs):
        for bx, by in loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            
    model.eval()
    return model

# Evaluate a model on a single task
def evaluate_single_task(model, loader, head, device):
    model.eval()
    head.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            # Forward pass except fc layer
            x = model.conv1(inputs)
            x = model.bn1(x)
            x = model.relu(x)
            x = model.maxpool(x)
            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3(x)
            x = model.layer4(x)
            x = model.avgpool(x)
            x = torch.flatten(x, 1)
            outputs = head(x)
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100.0 * correct / total

# Run joint evaluation with a routing mechanism
def evaluate_routed_tasks(backbone, loaders, expert_heads, router_fn, device, router_type='IHTR'):
    backbone.eval()
    for head in expert_heads.values():
        head.eval()
        
    results = {}
    routing_correct = {task: 0 for task in loaders.keys()}
    routing_total = {task: 0 for task in loaders.keys()}
    
    # Evaluate
    for current_task, task_loader in loaders.items():
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in task_loader['test']:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Determine routed task decisions
                if router_type in ['IHTR', 'HLBR']:
                    routed_tasks = router_fn(inputs)
                elif router_type == 'MSPR':
                    routed_tasks = router_fn(inputs, backbone)
                else:
                    # Oracle
                    routed_tasks = [current_task] * inputs.size(0)
                
                # Forward pass up to classifier
                x = backbone.conv1(inputs)
                x = backbone.bn1(x)
                x = backbone.relu(x)
                x = backbone.maxpool(x)
                x = backbone.layer1(x)
                x = backbone.layer2(x)
                x = backbone.layer3(x)
                x = backbone.layer4(x)
                x = backbone.avgpool(x)
                x = torch.flatten(x, 1)
                
                for i in range(inputs.size(0)):
                    routed_task = routed_tasks[i]
                    routing_total[current_task] += 1
                    if routed_task == current_task:
                        routing_correct[current_task] += 1
                        
                    head = expert_heads[routed_task]
                    output = head(x[i:i+1])
                    _, predicted = output.max(1)
                    total += 1
                    if predicted.item() == targets[i].item():
                        correct += 1
                        
        results[current_task] = 100.0 * correct / total
        
    avg_acc = np.mean(list(results.values()))
    routing_accs = {task: 100.0 * routing_correct[task] / routing_total[task] for task in loaders.keys()}
    avg_routing_acc = np.mean(list(routing_accs.values()))
    
    return results, avg_acc, routing_accs, avg_routing_acc

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load data loaders
    loaders = get_dataloaders(seed=42)
    expert_names = ['mnist', 'fmnist', 'cifar']
    expert_models = {}
    
    # 2. Expert checkpoints
    for name in expert_names:
        chk_path = f"expert_{name}.pt"
        if os.path.exists(chk_path):
            print(f"Loading checkpoint for {name} expert from {chk_path}...")
            model = get_resnet18_expert()
            model.load_state_dict(torch.load(chk_path, map_location=device))
            model = model.to(device)
            expert_models[name] = model
        else:
            model = train_expert(name, loaders[name]['train_expert'], device, epochs=5)
            torch.save(model.state_dict(), chk_path)
            expert_models[name] = model
            
    # Verify individual expert test accuracies
    print("\n--- Verifying Expert Accuracies ---")
    expert_accs = {}
    for name in expert_names:
        acc = evaluate_single_task(expert_models[name], loaders[name]['test'], expert_models[name].fc, device)
        expert_accs[name] = acc
        print(f"Expert {name} Test Accuracy: {acc:.2f}%")
        
    base_model = get_resnet18_expert().to(device)
    expert_heads = {name: model.fc for name, model in expert_models.items()}
    
    # 3. Model Merging (SWA & TA)
    print("\n--- Model Merging ---")
    wa_backbone_state = merge_wa(list(expert_models.values()))
    ta_backbone_state = merge_ta(list(expert_models.values()), base_model, lam=0.3)
    
    backbone_wa = get_resnet18_expert().to(device)
    backbone_wa.load_state_dict(wa_backbone_state, strict=False)
    
    backbone_ta = get_resnet18_expert().to(device)
    backbone_ta.load_state_dict(ta_backbone_state, strict=False)
    
    # Evaluate Uncalibrated Baselines
    print("\n--- Uncalibrated Baselines (Oracle Gating) ---")
    _, wa_oracle_acc, _, _ = evaluate_routed_tasks(backbone_wa, loaders, expert_heads, None, device, router_type='Oracle')
    _, ta_oracle_acc, _, _ = evaluate_routed_tasks(backbone_ta, loaders, expert_heads, None, device, router_type='Oracle')
    print(f"WA Oracle Avg Acc: {wa_oracle_acc:.2f}%")
    print(f"TA Oracle Avg Acc: {ta_oracle_acc:.2f}%")
    
    results_dict = {
        'expert_accs': expert_accs,
        'uncalibrated': {
            'WA_oracle': wa_oracle_acc,
            'TA_oracle': ta_oracle_acc
        },
        'calibrated': {}
    }
    
    # 4. Calibration Sets Generation (N=128)
    N_eval = 128
    print(f"\n--- Extracting Calibration Sets (N={N_eval}) ---")
    calib_images = {}
    joint_inputs = []
    joint_targets = []
    
    for task_idx, (name, task_loader) in enumerate(loaders.items()):
        inputs_list = []
        count = 0
        for inputs, targets in task_loader['train_expert']:
            inputs_list.append(inputs)
            count += inputs.size(0)
            if count >= N_eval:
                break
        task_inputs = torch.cat(inputs_list)[:N_eval]
        calib_images[name] = task_inputs
        joint_inputs.append(task_inputs)
        joint_targets.append(torch.full((N_eval,), task_idx, dtype=torch.long))
        
    joint_inputs_tensor = torch.cat(joint_inputs)
    joint_targets_tensor = torch.cat(joint_targets)
    joint_calib_dataset = TensorDataset(joint_inputs_tensor, joint_targets_tensor)
    joint_calib_loader = DataLoader(joint_calib_dataset, batch_size=32, shuffle=True)
    
    # 5. BN Calibration (N-TAAC)
    print("\n--- Running N-TAAC Calibration (N=128) ---")
    backbone_wa_calib = get_resnet18_expert().to(device)
    backbone_wa_calib.load_state_dict(wa_backbone_state, strict=False)
    calibrate_bn(backbone_wa_calib, joint_calib_loader, device)
    
    backbone_ta_calib = get_resnet18_expert().to(device)
    backbone_ta_calib.load_state_dict(ta_backbone_state, strict=False)
    calibrate_bn(backbone_ta_calib, joint_calib_loader, device)
    
    # Oracle Calibrated
    _, wa_calib_oracle_acc, _, _ = evaluate_routed_tasks(backbone_wa_calib, loaders, expert_heads, None, device, router_type='Oracle')
    _, ta_calib_oracle_acc, _, _ = evaluate_routed_tasks(backbone_ta_calib, loaders, expert_heads, None, device, router_type='Oracle')
    print(f"N-TAAC WA Oracle Avg Acc: {wa_calib_oracle_acc:.2f}%")
    print(f"N-TAAC TA Oracle Avg Acc: {ta_calib_oracle_acc:.2f}%")
    
    # 6. Extract Routing Prototypes and Train HLBR
    print("\n--- Setting Up Routers (N=128) ---")
    
    # A. MSPR (Layer 2)
    mspr_extractor_wa = FeatureExtractor(backbone_wa_calib).to(device)
    mspr_prototypes = {}
    for name, inputs in calib_images.items():
        inputs = inputs.to(device)
        feats = mspr_extractor_wa(inputs)
        prototype = feats.mean(dim=0)
        mspr_prototypes[name] = prototype / (prototype.norm(p=2) + 1e-8)
    mspr_extractor_wa.close()
    
    def mspr_router(inputs, backbone_model):
        mspr_extractor = FeatureExtractor(backbone_model).to(device)
        feats = mspr_extractor(inputs)
        feats_norm = feats / (feats.norm(p=2, dim=1, keepdim=True) + 1e-8)
        decisions = []
        for i in range(inputs.size(0)):
            scores = {}
            for name, proto in mspr_prototypes.items():
                scores[name] = torch.dot(feats_norm[i], proto).item()
            routed_task = max(scores, key=scores.get)
            decisions.append(routed_task)
        mspr_extractor.close()
        return decisions
        
    # B. IHTR (16-bins, Cosine)
    ihtr_prototypes = {}
    for name, inputs in calib_images.items():
        hists = extract_histograms(inputs, bins=16)
        prototype = hists.mean(dim=0)
        ihtr_prototypes[name] = prototype / (prototype.norm(p=2) + 1e-8)
        
    def ihtr_router(inputs):
        hists = extract_histograms(inputs, bins=16)
        decisions = []
        for i in range(inputs.size(0)):
            scores = {}
            for name, proto in ihtr_prototypes.items():
                proto_dev = proto.to(hists.device)
                h_norm = hists[i] / (hists[i].norm(p=2) + 1e-8)
                scores[name] = torch.dot(h_norm, proto_dev).item()
            routed_task = max(scores, key=scores.get)
            decisions.append(routed_task)
        return decisions
        
    # C. HLBR (Hybrid Linear Boundary Router: Bins=16, Size=6x6, Dim=52)
    hlbr_feat_fn = lambda x: extract_hybrid(x, bins=16, size=6)
    hlbr_dim = 52
    hlbr_classifier = train_micro_linear_router(calib_images, hlbr_feat_fn, hlbr_dim, device)
    
    def hlbr_router(inputs):
        feats = hlbr_feat_fn(inputs).to(device)
        with torch.no_grad():
            outputs = hlbr_classifier(feats)
            _, predicted = outputs.max(dim=1)
        tasks = ['mnist', 'fmnist', 'cifar']
        return [tasks[idx.item()] for idx in predicted]

    # Evaluate Routers on Calibrated Backbone
    print("\n--- Evaluating MSPR Router ---")
    wa_mspr_res, wa_mspr_acc, wa_mspr_rout, wa_mspr_avg_rout = evaluate_routed_tasks(
        backbone_wa_calib, loaders, expert_heads, mspr_router, device, router_type='MSPR'
    )
    ta_mspr_res, ta_mspr_acc, ta_mspr_rout, ta_mspr_avg_rout = evaluate_routed_tasks(
        backbone_ta_calib, loaders, expert_heads, mspr_router, device, router_type='MSPR'
    )
    print(f"WA + MSPR Avg Acc: {wa_mspr_acc:.2f}% (Routing Acc: {wa_mspr_avg_rout:.2f}%)")
    print(f"TA + MSPR Avg Acc: {ta_mspr_acc:.2f}% (Routing Acc: {ta_mspr_avg_rout:.2f}%)")
    
    print("\n--- Evaluating Nearest-Centroid IHTR ---")
    wa_ihtr_res, wa_ihtr_acc, wa_ihtr_rout, wa_ihtr_avg_rout = evaluate_routed_tasks(
        backbone_wa_calib, loaders, expert_heads, ihtr_router, device, router_type='IHTR'
    )
    ta_ihtr_res, ta_ihtr_acc, ta_ihtr_rout, ta_ihtr_avg_rout = evaluate_routed_tasks(
        backbone_ta_calib, loaders, expert_heads, ihtr_router, device, router_type='IHTR'
    )
    print(f"WA + IHTR Avg Acc: {wa_ihtr_acc:.2f}% (Routing Acc: {wa_ihtr_avg_rout:.2f}%)")
    print(f"TA + IHTR Avg Acc: {ta_ihtr_acc:.2f}% (Routing Acc: {ta_ihtr_avg_rout:.2f}%)")
    
    print("\n--- Evaluating Hybrid Linear Boundary Router (HLBR, Ours) ---")
    wa_hlbr_res, wa_hlbr_acc, wa_hlbr_rout, wa_hlbr_avg_rout = evaluate_routed_tasks(
        backbone_wa_calib, loaders, expert_heads, hlbr_router, device, router_type='HLBR'
    )
    ta_hlbr_res, ta_hlbr_acc, ta_hlbr_rout, ta_hlbr_avg_rout = evaluate_routed_tasks(
        backbone_ta_calib, loaders, expert_heads, hlbr_router, device, router_type='HLBR'
    )
    print(f"WA + HLBR Avg Acc: {wa_hlbr_acc:.2f}% (Routing Acc: {wa_hlbr_avg_rout:.2f}%)")
    print(f"TA + HLBR Avg Acc: {ta_hlbr_acc:.2f}% (Routing Acc: {ta_hlbr_avg_rout:.2f}%)")
    
    # Save primary calibrated results
    results_dict['calibrated']['N128'] = {
        'WA_NTAAC_Oracle': wa_calib_oracle_acc,
        'TA_NTAAC_Oracle': ta_calib_oracle_acc,
        'WA_MSPR_Acc': wa_mspr_acc,
        'WA_MSPR_Routing_Acc': wa_mspr_avg_rout,
        'WA_MSPR_Per_Task_Acc': wa_mspr_res,
        'WA_MSPR_Per_Task_Routing': wa_mspr_rout,
        'TA_MSPR_Acc': ta_mspr_acc,
        'TA_MSPR_Routing_Acc': ta_mspr_avg_rout,
        'TA_MSPR_Per_Task_Acc': ta_mspr_res,
        'TA_MSPR_Per_Task_Routing': ta_mspr_rout,
        
        'WA_IHTR_Acc': wa_ihtr_acc,
        'WA_IHTR_Routing_Acc': wa_ihtr_avg_rout,
        'WA_IHTR_Per_Task_Acc': wa_ihtr_res,
        'WA_IHTR_Per_Task_Routing': wa_ihtr_rout,
        'TA_IHTR_Acc': ta_ihtr_acc,
        'TA_IHTR_Routing_Acc': ta_ihtr_avg_rout,
        'TA_IHTR_Per_Task_Acc': ta_ihtr_res,
        'TA_IHTR_Per_Task_Routing': ta_ihtr_rout,
        
        'WA_HLBR_Acc': wa_hlbr_acc,
        'WA_HLBR_Routing_Acc': wa_hlbr_avg_rout,
        'WA_HLBR_Per_Task_Acc': wa_hlbr_res,
        'WA_HLBR_Per_Task_Routing': wa_hlbr_rout,
        'TA_HLBR_Acc': ta_hlbr_acc,
        'TA_HLBR_Routing_Acc': ta_hlbr_avg_rout,
        'TA_HLBR_Per_Task_Acc': ta_hlbr_res,
        'TA_HLBR_Per_Task_Routing': ta_hlbr_rout
    }
    
    # 7. Sweep: HLBR Feature Configurations (Bins, Size)
    print("\n--- Sweep: HLBR Configurations ---")
    hlbr_configs = {}
    for bins, size in [(16, 4), (16, 6), (32, 6), (32, 8)]:
        f_fn = lambda x: extract_hybrid(x, bins=bins, size=size)
        f_dim = bins + size * size
        
        # Train
        sweep_classifier = train_micro_linear_router(calib_images, f_fn, f_dim, device)
        
        def sweep_router(inputs):
            feats = f_fn(inputs).to(device)
            with torch.no_grad():
                outputs = sweep_classifier(feats)
                _, predicted = outputs.max(dim=1)
            tasks = ['mnist', 'fmnist', 'cifar']
            return [tasks[idx.item()] for idx in predicted]
            
        _, sweep_acc, _, sweep_rout_acc = evaluate_routed_tasks(
            backbone_wa_calib, loaders, expert_heads, sweep_router, device, router_type='HLBR'
        )
        hlbr_configs[f"Hybrid_Bins{bins}_Size{size}"] = {'acc': sweep_acc, 'routing_acc': sweep_rout_acc}
        print(f"Hybrid Bins={bins:2d}, Size={size}x{size} | Avg Acc: {sweep_acc:.2f}% | Routing Acc: {sweep_rout_acc:.2f}%")
    results_dict['ablation_configs'] = hlbr_configs
    
    # 8. Sweep: Calibration Sizes N in [16, 64, 128, 256] for HLBR (Bins=16, Size=6x6)
    print("\n--- Sweep: Calibration Dataset Size N ---")
    n_results = {}
    for N in [16, 64, 128, 256]:
        # Extract N calibration images
        sweep_calib_images = {}
        sweep_joint_inputs = []
        sweep_joint_targets = []
        for task_idx, (name, task_loader) in enumerate(loaders.items()):
            inputs_list = []
            count = 0
            for inputs, targets in task_loader['train_expert']:
                inputs_list.append(inputs)
                count += inputs.size(0)
                if count >= N:
                    break
            task_inputs = torch.cat(inputs_list)[:N]
            sweep_calib_images[name] = task_inputs
            sweep_joint_inputs.append(task_inputs)
            sweep_joint_targets.append(torch.full((N,), task_idx, dtype=torch.long))
            
        sweep_joint_inputs_tensor = torch.cat(sweep_joint_inputs)
        sweep_joint_targets_tensor = torch.cat(sweep_joint_targets)
        sweep_joint_calib_dataset = TensorDataset(sweep_joint_inputs_tensor, sweep_joint_targets_tensor)
        sweep_joint_calib_loader = DataLoader(sweep_joint_calib_dataset, batch_size=32, shuffle=True)
        
        # BN calibration
        sweep_backbone_wa = get_resnet18_expert().to(device)
        sweep_backbone_wa.load_state_dict(wa_backbone_state, strict=False)
        calibrate_bn(sweep_backbone_wa, sweep_joint_calib_loader, device)
        
        # Train HLBR
        sweep_hlbr_classifier = train_micro_linear_router(sweep_calib_images, hlbr_feat_fn, hlbr_dim, device)
        
        def sweep_hlbr_router(inputs):
            feats = hlbr_feat_fn(inputs).to(device)
            with torch.no_grad():
                outputs = sweep_hlbr_classifier(feats)
                _, predicted = outputs.max(dim=1)
            tasks = ['mnist', 'fmnist', 'cifar']
            return [tasks[idx.item()] for idx in predicted]
            
        _, sweep_acc, _, sweep_rout_acc = evaluate_routed_tasks(
            sweep_backbone_wa, loaders, expert_heads, sweep_hlbr_router, device, router_type='HLBR'
        )
        n_results[N] = {'acc': sweep_acc, 'routing_acc': sweep_rout_acc}
        print(f"Size N: {N:3d} | Avg Acc: {sweep_acc:.2f}% | Routing Acc: {sweep_rout_acc:.2f}%")
    results_dict['ablation_sizes'] = n_results
    
    # 9. Statistical Stability over 5 Seeds (N=128)
    print("\n--- Statistical Stability (5 seeds) ---")
    seeds = [42, 43, 44, 45, 46]
    seed_accs_hlbr = []
    seed_rout_accs_hlbr = []
    seed_accs_ihtr = []
    seed_rout_accs_ihtr = []
    
    for s in seeds:
        seed_loaders = get_dataloaders(seed=s)
        
        seed_calib_images = {}
        seed_joint_inputs = []
        seed_joint_targets = []
        for task_idx, (name, task_loader) in enumerate(seed_loaders.items()):
            inputs_list = []
            count = 0
            for inputs, targets in task_loader['train_expert']:
                inputs_list.append(inputs)
                count += inputs.size(0)
                if count >= N_eval:
                    break
            task_inputs = torch.cat(inputs_list)[:N_eval]
            seed_calib_images[name] = task_inputs
            seed_joint_inputs.append(task_inputs)
            seed_joint_targets.append(torch.full((N_eval,), task_idx, dtype=torch.long))
            
        seed_joint_inputs_tensor = torch.cat(seed_joint_inputs)
        seed_joint_targets_tensor = torch.cat(seed_joint_targets)
        seed_joint_calib_dataset = TensorDataset(seed_joint_inputs_tensor, seed_joint_targets_tensor)
        seed_joint_calib_loader = DataLoader(seed_joint_calib_dataset, batch_size=32, shuffle=True)
        
        # BN calibration
        seed_backbone_wa = get_resnet18_expert().to(device)
        seed_backbone_wa.load_state_dict(wa_backbone_state, strict=False)
        calibrate_bn(seed_backbone_wa, seed_joint_calib_loader, device)
        
        # Set up Routers
        # A. Nearest Centroid IHTR (16 bins)
        seed_ihtr_prototypes = {}
        for name, inputs in seed_calib_images.items():
            hists = extract_histograms(inputs, bins=16)
            proto = hists.mean(dim=0)
            seed_ihtr_prototypes[name] = proto / (proto.norm(p=2) + 1e-8)
            
        def seed_ihtr_router(inputs):
            hists = extract_histograms(inputs, bins=16)
            decisions = []
            for i in range(inputs.size(0)):
                scores = {}
                for name, proto in seed_ihtr_prototypes.items():
                    proto_dev = proto.to(hists.device)
                    h_norm = hists[i] / (hists[i].norm(p=2) + 1e-8)
                    scores[name] = torch.dot(h_norm, proto_dev).item()
                routed_task = max(scores, key=scores.get)
                decisions.append(routed_task)
            return decisions
            
        # B. HLBR (Hybrid Linear Boundary Router)
        seed_hlbr_classifier = train_micro_linear_router(seed_calib_images, hlbr_feat_fn, hlbr_dim, device)
        
        def seed_hlbr_router(inputs):
            feats = hlbr_feat_fn(inputs).to(device)
            with torch.no_grad():
                outputs = seed_hlbr_classifier(feats)
                _, predicted = outputs.max(dim=1)
            tasks = ['mnist', 'fmnist', 'cifar']
            return [tasks[idx.item()] for idx in predicted]
            
        # Evaluate
        _, i_acc, _, i_rout = evaluate_routed_tasks(seed_backbone_wa, seed_loaders, expert_heads, seed_ihtr_router, device, router_type='IHTR')
        _, h_acc, _, h_rout = evaluate_routed_tasks(seed_backbone_wa, seed_loaders, expert_heads, seed_hlbr_router, device, router_type='HLBR')
        
        seed_accs_ihtr.append(i_acc)
        seed_rout_accs_ihtr.append(i_rout)
        seed_accs_hlbr.append(h_acc)
        seed_rout_accs_hlbr.append(h_rout)
        print(f"Seed {s} | IHTR Acc: {i_acc:.2f}% (Rout: {i_rout:.2f}%) | HLBR Acc: {h_acc:.2f}% (Rout: {h_rout:.2f}%)")
        
    results_dict['stability'] = {
        'seeds': seeds,
        'ihtr_accs': seed_accs_ihtr,
        'ihtr_rout_accs': seed_rout_accs_ihtr,
        'ihtr_mean_acc': np.mean(seed_accs_ihtr),
        'ihtr_std_acc': np.std(seed_accs_ihtr),
        'ihtr_mean_rout': np.mean(seed_rout_accs_ihtr),
        'ihtr_std_rout': np.std(seed_rout_accs_ihtr),
        
        'hlbr_accs': seed_accs_hlbr,
        'hlbr_rout_accs': seed_rout_accs_hlbr,
        'hlbr_mean_acc': np.mean(seed_accs_hlbr),
        'hlbr_std_acc': np.std(seed_accs_hlbr),
        'hlbr_mean_rout': np.mean(seed_rout_accs_hlbr),
        'hlbr_std_rout': np.std(seed_rout_accs_hlbr),
    }
    print(f"\nIHTR Stability | Acc: {results_dict['stability']['ihtr_mean_acc']:.2f}% ± {results_dict['stability']['ihtr_std_acc']:.2f}% | Rout: {results_dict['stability']['ihtr_mean_rout']:.2f}% ± {results_dict['stability']['ihtr_std_rout']:.2f}%")
    print(f"HLBR Stability | Acc: {results_dict['stability']['hlbr_mean_acc']:.2f}% ± {results_dict['stability']['hlbr_std_acc']:.2f}% | Rout: {results_dict['stability']['hlbr_mean_rout']:.2f}% ± {results_dict['stability']['hlbr_std_rout']:.2f}%")
    
    # 10. Latency & Compiler compatibility profiling
    print("\n--- Latency and Compiler Compatibility Profiling ---")
    test_inputs = torch.randn(128, 3, 32, 32).to(device)
    dummy_backbone = get_resnet18_expert().to(device)
    
    # MSPR Latency
    for _ in range(10):
        _ = dummy_backbone(test_inputs)
    start_time = time.time()
    for _ in range(50):
        _ = mspr_router(test_inputs, dummy_backbone)
    mspr_latency_ms = 1000.0 * (time.time() - start_time) / 50.0
    print(f"MSPR Routing Latency per batch: {mspr_latency_ms:.4f} ms")
    
    # IHTR Latency
    start_time = time.time()
    for _ in range(50):
        _ = ihtr_router(test_inputs)
    ihtr_latency_ms = 1000.0 * (time.time() - start_time) / 50.0
    print(f"IHTR Routing Latency per batch: {ihtr_latency_ms:.4f} ms")
    
    # HLBR Latency
    start_time = time.time()
    for _ in range(50):
        _ = hlbr_router(test_inputs)
    hlbr_latency_ms = 1000.0 * (time.time() - start_time) / 50.0
    print(f"HLBR Routing Latency per batch: {hlbr_latency_ms:.4f} ms")
    
    # Compiler compatibility
    compiled_backbone = torch.compile(backbone_wa_calib)
    try:
        _ = compiled_backbone(test_inputs)
        print("Success! Merged backbone compiled successfully with torch.compile.")
        compiled_ok = True
    except Exception as e:
        print(f"Compilation failed: {e}")
        compiled_ok = False
        
    results_dict['profiling'] = {
        'mspr_routing_latency_ms': mspr_latency_ms,
        'ihtr_routing_latency_ms': ihtr_latency_ms,
        'hlbr_routing_latency_ms': hlbr_latency_ms,
        'torch_compile_compatible': compiled_ok
    }
    
    # 11. Write results
    with open('results_ihtr.json', 'w') as f:
        json.dump(results_dict, f, indent=4)
    print("\nSuccessfully wrote all experimental results to results_ihtr.json!")

if __name__ == '__main__':
    main()
