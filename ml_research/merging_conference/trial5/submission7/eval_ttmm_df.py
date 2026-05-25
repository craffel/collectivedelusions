import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.func import functional_call
from tqdm import tqdm
import numpy as np

# Disable cuDNN to avoid CUDNN_STATUS_NOT_INITIALIZED errors on cluster nodes
torch.backends.cudnn.enabled = False

def get_dataset(name, train=True, transform=None):
    if name == 'mnist':
        return datasets.MNIST(root='./data', train=train, download=True, transform=transform)
    elif name == 'fashion':
        return datasets.FashionMNIST(root='./data', train=train, download=True, transform=transform)
    elif name == 'kmnist':
        return datasets.KMNIST(root='./data', train=train, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {name}")

class CorrectContrastTransform(object):
    def __init__(self, factor=0.3):
        self.factor = factor
        
    def __call__(self, img_tensor):
        img_raw = img_tensor * 0.5 + 0.5
        mean = img_raw.mean()
        img_corrupted = mean + self.factor * (img_raw - mean)
        img_corrupted = torch.clamp(img_corrupted, 0.0, 1.0)
        return (img_corrupted - 0.5) / 0.5

def get_transforms(corruption, noise_sigma=0.15, blur_sigma=1.6, contrast_factor=0.3):
    base_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    if corruption == 'clean':
        return base_transform
    elif corruption == 'noise':
        return transforms.Compose([
            base_transform,
            transforms.Lambda(lambda x: torch.clamp(x + torch.randn_like(x) * noise_sigma, -1.0, 1.0))
        ])
    elif corruption == 'blur':
        return transforms.Compose([
            base_transform,
            transforms.GaussianBlur(kernel_size=5, sigma=blur_sigma)
        ])
    elif corruption == 'contrast':
        return transforms.Compose([
            base_transform,
            CorrectContrastTransform(factor=contrast_factor)
        ])
    else:
        raise ValueError(f"Unknown corruption: {corruption}")

def load_base_model():
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    pretrained_conv1 = model.conv1.weight
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        model.conv1.weight.copy_(pretrained_conv1.mean(dim=1, keepdim=True))
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model

def load_experts():
    experts = {}
    tasks = ['mnist', 'fashion', 'kmnist']
    for task in tasks:
        model = load_base_model()
        model.load_state_dict(torch.load(f'experts/{task}_expert.pt', map_location='cpu'))
        experts[task] = model.state_dict()
    return experts

def extract_prototypes(experts, num_samples=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    prototypes = {}
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    tasks = ['mnist', 'fashion', 'kmnist']
    for k, task in enumerate(tasks):
        model = load_base_model()
        model.load_state_dict(experts[task])
        model = model.to(device)
        model.eval()
        train_dataset = get_dataset(task, train=True, transform=transform)
        class_samples = {c: [] for c in range(10)}
        for img, label in train_dataset:
            if len(class_samples[label]) < num_samples:
                class_samples[label].append(img)
            if all(len(class_samples[c]) == num_samples for c in range(10)):
                break
        original_fc = model.fc
        model.fc = nn.Identity()
        task_prototypes = []
        with torch.no_grad():
            for c in range(10):
                imgs = torch.stack(class_samples[c]).to(device)
                features = model(imgs)
                mean_feature = features.mean(dim=0)
                mean_feature = mean_feature / mean_feature.norm(p=2)
                task_prototypes.append(mean_feature.cpu())
        model.fc = original_fc
        prototypes[task] = torch.stack(task_prototypes)
    return prototypes

def get_blended_params(base_weights_dev, experts_dev, lambdas, backbone_layers, task_name):
    blended_params = {}
    for name, param in base_weights_dev.items():
        if 'fc' not in name:
            if name in lambdas and param.is_floating_point():
                l = lambdas[name] if name in backbone_layers else lambdas[name].detach()
                v1 = experts_dev['mnist'][name] - base_weights_dev[name]
                v2 = experts_dev['fashion'][name] - base_weights_dev[name]
                v3 = experts_dev['kmnist'][name] - base_weights_dev[name]
                blended_params[name] = base_weights_dev[name] + l[0] * v1 + l[1] * v2 + l[2] * v3
            else:
                blended_params[name] = param
        else:
            blended_params[name] = experts_dev[task_name][name]
    return blended_params

def project_simplex(v):
    v_np = v.cpu().numpy()
    u = np.sort(v_np)[::-1]
    cssv = np.cumsum(u)
    ind = np.arange(len(v_np)) + 1
    cond = u - (cssv - 1) / ind > 0
    rho = ind[cond][-1]
    theta = (cssv[cond][-1] - 1) / rho
    w = np.maximum(v_np - theta, 0)
    res = torch.tensor(w, device=v.device, dtype=v.dtype)
    if v.requires_grad:
        res.requires_grad_(True)
    return res

def run_evaluation(stream_type, corruption, experts, prototypes, method, lr=0.1, alpha=0.5, beta=0.1, num_steps=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = get_transforms(corruption)
    test_datasets = {
        'mnist': get_dataset('mnist', train=False, transform=transform),
        'fashion': get_dataset('fashion', train=False, transform=transform),
        'kmnist': get_dataset('kmnist', train=False, transform=transform)
    }
    
    torch.manual_seed(42)
    subset_indices = {task: torch.randperm(len(test_datasets[task]))[:1600] for task in test_datasets}
    loaders = {
        task: DataLoader(Subset(test_datasets[task], subset_indices[task]), batch_size=32, shuffle=False)
        for task in test_datasets
    }
    
    mnist_batches = list(loaders['mnist'])
    fashion_batches = list(loaders['fashion'])
    kmnist_batches = list(loaders['kmnist'])
    
    stream_batches = []
    if stream_type == 'alternating':
        for i in range(50):
            stream_batches.append(('mnist', mnist_batches[i]))
            stream_batches.append(('fashion', fashion_batches[i]))
            stream_batches.append(('kmnist', kmnist_batches[i]))
    elif stream_type == 'sequential':
        for b in mnist_batches:
            stream_batches.append(('mnist', b))
        for b in fashion_batches:
            stream_batches.append(('fashion', b))
        for b in kmnist_batches:
            stream_batches.append(('kmnist', b))
            
    base_weights = load_base_model().state_dict()
    base_weights_dev = {k: v.to(device) for k, v in base_weights.items()}
    experts_dev = {
        'mnist': {k: v.to(device) for k, v in experts['mnist'].items()},
        'fashion': {k: v.to(device) for k, v in experts['fashion'].items()},
        'kmnist': {k: v.to(device) for k, v in experts['kmnist'].items()}
    }
    merged_model = load_base_model().to(device)
    
    backbone_layers = [name for name, param in merged_model.named_parameters() if 'fc' not in name]
    all_backbone_keys = [name for name, param in merged_model.state_dict().items() if 'fc' not in name and param.is_floating_point()]
    
    lambdas = {name: torch.tensor([1/3, 1/3, 1/3], device=device, requires_grad=True) for name in all_backbone_keys}
    
    running_grad_sq = {name: torch.zeros(3, device=device) for name in backbone_layers}
    
    ema_loss = 0.0
    beta_ema = 0.90
    opr_alpha = 2.5 if corruption != 'clean' else 4.0
    
    correct = 0
    total = 0
    
    pbar = tqdm(stream_batches, desc=f"{method.upper()} ({stream_type}, {corruption})")
    for step, (task_name, (images, labels)) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        # Prototype-driven Dynamic Routing (PD-Routing)
        temp_lambdas = {name: torch.tensor([1/3, 1/3, 1/3], device=device) for name in all_backbone_keys}
        blended_params = get_blended_params(base_weights_dev, experts_dev, temp_lambdas, backbone_layers, task_name)
        
        original_fc = merged_model.fc
        merged_model.fc = nn.Identity()
        with torch.no_grad():
            anchor_features = functional_call(merged_model, blended_params, images)
            anchor_features = anchor_features / anchor_features.norm(p=2, dim=1, keepdim=True)
        merged_model.fc = original_fc
        
        S = []
        for t_idx, t in enumerate(['mnist', 'fashion', 'kmnist']):
            t_proto = prototypes[t].to(device)
            sims = torch.mm(anchor_features, t_proto.t())
            max_sims, _ = sims.max(dim=1)
            S_k = max_sims.mean().item()
            S.append(S_k)
            
        S_tensor = torch.tensor(S, device=device)
        lambda_prior = F.softmax(S_tensor / 0.02, dim=0)
        
        with torch.no_grad():
            for name in all_backbone_keys:
                lambdas[name].copy_(lambda_prior)
                
        for sub_step in range(num_steps):
            blended_params = get_blended_params(base_weights_dev, experts_dev, lambdas, backbone_layers, task_name)
            outputs = functional_call(merged_model, blended_params, images)
            
            probs = F.softmax(outputs, dim=1)
            entropy_loss_samples = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            
            original_fc = merged_model.fc
            merged_model.fc = nn.Identity()
            features = functional_call(merged_model, blended_params, images)
            features_norm = features / features.norm(p=2, dim=1, keepdim=True)
            merged_model.fc = original_fc
            
            active_proto = prototypes[task_name].to(device)
            M = torch.mm(features_norm, active_proto.t())
            
            max_probs, pred_labels = probs.max(dim=1)
            mask = max_probs > 0.85
            
            if mask.sum() > 0:
                M_masked = M[mask]
                pred_labels_masked = pred_labels[mask]
                exp_sims = torch.exp(M_masked / 0.1)
                pos_sims = torch.exp(M_masked[torch.arange(M_masked.size(0)), pred_labels_masked] / 0.1)
                contrast_loss_samples = -torch.log(pos_sims / exp_sims.sum(dim=1))
                contrast_loss = torch.zeros(images.size(0), device=device)
                contrast_loss[mask] = contrast_loss_samples
            else:
                contrast_loss = torch.zeros(images.size(0), device=device)
                
            loss_samples = entropy_loss_samples + beta * contrast_loss
            
            mean_loss = loss_samples.mean().item()
            if step == 0:
                ema_loss = mean_loss
            else:
                if mean_loss > opr_alpha * ema_loss:
                    with torch.no_grad():
                        for name in all_backbone_keys:
                            lambdas[name].copy_(torch.tensor([1/3, 1/3, 1/3], device=device))
                    ema_loss = mean_loss
                    with torch.no_grad():
                        for name in backbone_layers:
                            running_grad_sq[name].zero_()
                else:
                    ema_loss = beta_ema * ema_loss + (1 - beta_ema) * mean_loss
                    
            active_classes = torch.unique(pred_labels)
            class_grads = {}
            for c in active_classes:
                c_mask = pred_labels == c
                if c_mask.sum() > 0:
                    c_loss = loss_samples[c_mask].mean()
                    c_grads = torch.autograd.grad(c_loss, [lambdas[name] for name in backbone_layers], retain_graph=True, allow_unused=True)
                    class_grads[c.item()] = {name: grad.clone() for name, grad in zip(backbone_layers, c_grads) if grad is not None}
            
            # Update running squared gradients EMA using class-averaged gradients
            with torch.no_grad():
                for name in backbone_layers:
                    g_total = torch.zeros(3, device=device)
                    count = 0
                    for c in class_grads.keys():
                        if name in class_grads[c]:
                            g_total += class_grads[c][name]
                            count += 1
                    if count > 0:
                        g_total /= count
                        running_grad_sq[name] = 0.95 * running_grad_sq[name] + 0.05 * (g_total ** 2)
                        
            # Construct online Fisher preconditioning weights
            online_fisher_weights = {}
            for name in backbone_layers:
                f_val = running_grad_sq[name].mean().item()
                # 1e-4 stabilizer baseline prevents preconditioning exploding
                g_val = (f_val + 1e-4) ** alpha
                online_fisher_weights[name] = g_val
                
            # Pairwise gradient surgery in estimated Fisher space
            for c_a in class_grads.keys():
                for c_b in class_grads.keys():
                    if c_a != c_b:
                        dot_prod = 0.0
                        norm_b_sq = 0.0
                        for name in backbone_layers:
                            if name in class_grads[c_a] and name in class_grads[c_b]:
                                metric_w = online_fisher_weights[name]
                                dot_prod += metric_w * torch.sum(class_grads[c_a][name] * class_grads[c_b][name]).item()
                                norm_b_sq += metric_w * torch.sum(class_grads[c_b][name] * class_grads[c_b][name]).item()
                                
                        if dot_prod < 0:
                            for name in backbone_layers:
                                if name in class_grads[c_a] and name in class_grads[c_b]:
                                    class_grads[c_a][name] -= (dot_prod / (norm_b_sq + 1e-8)) * class_grads[c_b][name]
                                    
            # Update step
            with torch.no_grad():
                for name in backbone_layers:
                    final_grad = torch.zeros_like(lambdas[name])
                    for c in class_grads.keys():
                        if name in class_grads[c]:
                            final_grad += class_grads[c][name]
                    scaled_lr = lr / online_fisher_weights[name]
                    lambdas[name] -= scaled_lr * final_grad
                    lambdas[name] = project_simplex(lambdas[name])
                    
        with torch.no_grad():
            final_blended_params = get_blended_params(base_weights_dev, experts_dev, lambdas, backbone_layers, task_name)
            final_outputs = functional_call(merged_model, final_blended_params, images)
            _, predicted = torch.max(final_outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            pbar.set_postfix(Acc=f"{100 * correct / total:.2f}%")
            
    final_acc = 100 * correct / total
    print(f"[{method.upper()}] Final Accuracy on {stream_type} ({corruption}): {final_acc:.2f}%\n")
    return final_acc

if __name__ == '__main__':
    experts = load_experts()
    prototypes = extract_prototypes(experts, num_samples=100)
    
    streams = ['alternating', 'sequential']
    corruptions = ['clean', 'noise', 'blur', 'contrast']
    
    results = {}
    for stream in streams:
        results[stream] = {}
        for corr in corruptions:
            acc = run_evaluation(stream, corr, experts, prototypes, 'iggs-merge-df')
            results[stream][corr] = acc
            
    np.savez('evaluation_results_df.npz', results=results)
    print("Data-Free evaluations complete! Results saved to evaluation_results_df.npz.")
