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
    torch.backends.cudnn.enabled = False

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
    
    mnist_train_full = torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=transform_mnist)
    mnist_test_full = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform_mnist)
    
    fmnist_train_full = torchvision.datasets.FashionMNIST(root='./data', train=True, download=False, transform=transform_fmnist)
    fmnist_test_full = torchvision.datasets.FashionMNIST(root='./data', train=False, download=False, transform=transform_fmnist)
    
    cifar_train_full = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_cifar)
    cifar_test_full = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_cifar)
    
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

def merge_wa(expert_models):
    merged_state_dict = {}
    keys = expert_models[0].state_dict().keys()
    for key in keys:
        if 'fc' in key:
            continue
        weights = [model.state_dict()[key].float() for model in expert_models]
        merged_state_dict[key] = torch.stack(weights).mean(dim=0)
    return merged_state_dict

def calibrate_bn(model, calibration_loader, device):
    model.to(device)
    model.train()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None
            
    with torch.no_grad():
        for inputs, _ in calibration_loader:
            inputs = inputs.to(device)
            _ = model(inputs)
            
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = 0.1
    model.eval()

class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.features = None
        self.hook = self.model.layer2.register_forward_hook(self.save_features)
        
    def save_features(self, module, input, output):
        self.features = torch.mean(output, dim=(2, 3))
        
    def forward(self, x):
        self.features = None
        _ = self.model(x)
        return self.features
        
    def close(self):
        self.hook.remove()

def extract_histograms(x, bins=16):
    intensity = x.mean(dim=1)
    histograms = []
    for i in range(x.shape[0]):
        h = torch.histc(intensity[i], bins=bins, min=-1.0, max=1.0)
        h = h / (h.sum() + 1e-8)
        histograms.append(h)
    return torch.stack(histograms)

def extract_downsampled(x, size=6):
    intensity = x.mean(dim=1, keepdim=True)
    d = nn.functional.adaptive_avg_pool2d(intensity, (size, size))
    d_flat = d.view(x.shape[0], -1)
    d_flat_norm = d_flat / (d_flat.norm(p=2, dim=1, keepdim=True) + 1e-8)
    return d_flat_norm

def extract_hybrid(x, bins=16, size=6):
    hists = extract_histograms(x, bins=bins)
    ds = extract_downsampled(x, size=size)
    hybrid = torch.cat([hists, ds], dim=1)
    hybrid_norm = hybrid / (hybrid.norm(p=2, dim=1, keepdim=True) + 1e-8)
    return hybrid_norm

def train_micro_linear_router(calib_images, feature_fn, feat_dim, device, lr=0.1, epochs=150):
    train_feats_list = []
    train_labels_list = []
    for task_idx, name in enumerate(['mnist', 'fmnist', 'cifar']):
        inputs = calib_images[name]
        
        # Add clean calibration inputs
        feats = feature_fn(inputs)
        train_feats_list.append(feats)
        train_labels_list.append(torch.full((inputs.size(0),), task_idx, dtype=torch.long))
        
        # Add offline brightness-augmented calibration inputs for brightness robustness!
        for shift in [-0.3, -0.2, -0.1, 0.1, 0.2, 0.3]:
            inputs_aug = torch.clamp(inputs + shift, -1.0, 1.0)
            feats_aug = feature_fn(inputs_aug)
            train_feats_list.append(feats_aug)
            train_labels_list.append(torch.full((inputs.size(0),), task_idx, dtype=torch.long))
        
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

def evaluate_routed_tasks_robust(backbone, loaders, expert_heads, router_fn, device, router_type, perturb_fn):
    backbone.eval()
    for head in expert_heads.values():
        head.eval()
        
    results = {}
    routing_correct = {task: 0 for task in loaders.keys()}
    routing_total = {task: 0 for task in loaders.keys()}
    
    for current_task, task_loader in loaders.items():
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in task_loader['test']:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Apply environmental perturbation at boundary!
                inputs_perturbed = perturb_fn(inputs)
                
                # Determine routed task decisions
                if router_type in ['IHTR', 'HLBR']:
                    routed_tasks = router_fn(inputs_perturbed)
                elif router_type == 'MSPR':
                    routed_tasks = router_fn(inputs_perturbed, backbone)
                else:
                    routed_tasks = [current_task] * inputs.size(0)
                    
                # Forward pass up to classifier
                x = backbone.conv1(inputs_perturbed)
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
    
    loaders = get_dataloaders(seed=42)
    expert_names = ['mnist', 'fmnist', 'cifar']
    expert_models = {}
    
    for name in expert_names:
        chk_path = f"expert_{name}.pt"
        model = get_resnet18_expert()
        model.load_state_dict(torch.load(chk_path, map_location=device))
        model = model.to(device)
        expert_models[name] = model
        
    expert_heads = {name: model.fc for name, model in expert_models.items()}
    
    # WA Merger
    wa_backbone_state = merge_wa(list(expert_models.values()))
    backbone_wa = get_resnet18_expert().to(device)
    backbone_wa.load_state_dict(wa_backbone_state, strict=False)
    
    # Calibration Sets (N=128)
    N_eval = 128
    calib_images = {}
    joint_inputs = []
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
        
    joint_inputs_tensor = torch.cat(joint_inputs)
    joint_targets_tensor = torch.zeros(len(joint_inputs_tensor))
    joint_calibration_loader = DataLoader(
        TensorDataset(joint_inputs_tensor, joint_targets_tensor),
        batch_size=32, shuffle=True
    )
    
    # Calibrate BN
    calibrate_bn(backbone_wa, joint_calibration_loader, device)
    
    # Setup Routers
    # A. MSPR (Layer 2)
    mspr_extractor_wa = FeatureExtractor(backbone_wa).to(device)
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
        
    # B. IHTR (16 bins)
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
        
    # C. HLBR
    hlbr_feat_fn = lambda x: extract_hybrid(x, bins=16, size=6)
    hlbr_classifier = train_micro_linear_router(calib_images, hlbr_feat_fn, 52, device)
    
    def hlbr_router(inputs):
        feats = hlbr_feat_fn(inputs).to(device)
        with torch.no_grad():
            outputs = hlbr_classifier(feats)
            _, predicted = outputs.max(dim=1)
        tasks = ['mnist', 'fmnist', 'cifar']
        return [tasks[idx.item()] for idx in predicted]
        
    # Perturbations
    perturbations = {
        'Clean': lambda x: x,
        'Gaussian_Noise_Soft': lambda x: torch.clamp(x + 0.05 * torch.randn_like(x), -1.0, 1.0),
        'Gaussian_Noise_Med': lambda x: torch.clamp(x + 0.15 * torch.randn_like(x), -1.0, 1.0),
        'Brightness_Shift': lambda x: torch.clamp(x + 0.15, -1.0, 1.0),
        'Translation_Shift': lambda x: torch.roll(x, shifts=(2, 2), dims=(2, 3))
    }
    
    results = {}
    for p_name, p_fn in perturbations.items():
        print(f"\n--- Evaluating under perturbation: {p_name} ---")
        results[p_name] = {}
        
        # 1. MSPR
        t0 = time.time()
        _, m_acc, _, r_acc = evaluate_routed_tasks_robust(backbone_wa, loaders, expert_heads, mspr_router, device, 'MSPR', p_fn)
        results[p_name]['MSPR'] = {'acc': m_acc, 'routing_acc': r_acc}
        print(f"MSPR: Routing Acc: {r_acc:.2f}%, Model Acc: {m_acc:.2f}% (Time: {time.time()-t0:.2f}s)")
        
        # 2. IHTR
        t0 = time.time()
        _, m_acc, _, r_acc = evaluate_routed_tasks_robust(backbone_wa, loaders, expert_heads, ihtr_router, device, 'IHTR', p_fn)
        results[p_name]['IHTR'] = {'acc': m_acc, 'routing_acc': r_acc}
        print(f"IHTR: Routing Acc: {r_acc:.2f}%, Model Acc: {m_acc:.2f}% (Time: {time.time()-t0:.2f}s)")
        
        # 3. HLBR
        t0 = time.time()
        _, m_acc, _, r_acc = evaluate_routed_tasks_robust(backbone_wa, loaders, expert_heads, hlbr_router, device, 'HLBR', p_fn)
        results[p_name]['HLBR'] = {'acc': m_acc, 'routing_acc': r_acc}
        print(f"HLBR: Routing Acc: {r_acc:.2f}%, Model Acc: {m_acc:.2f}% (Time: {time.time()-t0:.2f}s)")
        
    with open('results_robustness.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("\nRobustness results successfully written to results_robustness.json!")

if __name__ == '__main__':
    main()
