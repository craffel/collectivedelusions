import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.nn.utils.stateless import functional_call
import numpy as np
import copy
import random

# Set seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Set device to CPU
device = torch.device("cpu")
torch.backends.cudnn.enabled = False

# Transforms
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define the model wrapper
class ResNetExpert(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNetExpert, self).__init__()
        self.resnet = resnet18()
        self.feature_extractor = nn.Sequential(*list(self.resnet.children())[:-1])
        self.fc = nn.Linear(512, num_classes)
        
    def forward(self, x, return_features=False):
        features = self.feature_extractor(x)
        features = torch.flatten(features, 1)
        out = self.fc(features)
        if return_features:
            return out, features
        return out

def get_layer_group(param_name):
    if "conv1" in param_name or "bn1" in param_name:
        return 0
    elif "layer1" in param_name:
        return 1
    elif "layer2" in param_name:
        return 2
    elif "layer3" in param_name:
        return 3
    elif "layer4" in param_name:
        return 4
    else:
        return 0

# Apply OOD Corruption: Gaussian noise + Average Blur
def apply_corruption(inputs):
    noise = torch.randn_like(inputs) * 0.25
    corrupted = inputs + noise
    import torch.nn.functional as F
    corrupted = F.avg_pool2d(corrupted, kernel_size=3, stride=1, padding=1)
    return corrupted

def make_test_stream(batch_size=64, seed=2026):
    set_seed(seed)
    mnist_test = torchvision.datasets.MNIST(root="./data", train=False, download=False, transform=transform)
    fashion_test = torchvision.datasets.FashionMNIST(root="./data", train=False, download=False, transform=transform)
    kmnist_test = torchvision.datasets.KMNIST(root="./data", train=False, download=False, transform=transform)
    
    mnist_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=True)
    fashion_loader = torch.utils.data.DataLoader(fashion_test, batch_size=batch_size, shuffle=True)
    kmnist_loader = torch.utils.data.DataLoader(kmnist_test, batch_size=batch_size, shuffle=True)
    
    mnist_iter = iter(mnist_loader)
    fashion_iter = iter(fashion_loader)
    kmnist_iter = iter(kmnist_loader)
    
    blocks = [
        ("mnist", mnist_iter, 0),
        ("fashion", fashion_iter, 1),
        ("kmnist", kmnist_iter, 2),
        ("mnist", mnist_iter, 0),
        ("fashion", fashion_iter, 1),
        ("kmnist", kmnist_iter, 2)
    ]
    
    stream = []
    for name, iterator, task_id in blocks:
        for _ in range(10):
            try:
                inputs, targets = next(iterator)
            except StopIteration:
                if name == "mnist":
                    mnist_iter = iter(mnist_loader)
                    inputs, targets = next(mnist_iter)
                elif name == "fashion":
                    fashion_iter = iter(fashion_loader)
                    inputs, targets = next(fashion_iter)
                else:
                    kmnist_iter = iter(kmnist_loader)
                    inputs, targets = next(kmnist_iter)
            stream.append((inputs, targets, task_id))
    return stream

def run_experiment(method, stream, experts, original_heads, anchors=None, lr=0.01, seed=2026):
    set_seed(seed)
    alphas = torch.zeros(5, 3, device=device, requires_grad=(method != "static"))
    heads = [copy.deepcopy(h).to(device) for h in original_heads]
    
    for idx, h in enumerate(heads):
        for p in h.parameters():
            if method in ["unconstrained", "caba", "caba_no_distill", "caba_no_align"]:
                p.requires_grad = True
            else:
                p.requires_grad = False
                
    params_to_opt = []
    if method != "static":
        lr_alpha = 0.005 if method.startswith("caba") else lr
        params_to_opt.append({"params": [alphas], "lr": lr_alpha})
    if method in ["unconstrained", "caba", "caba_no_distill", "caba_no_align"]:
        lr_head = 0.001 if method.startswith("caba") else lr
        for h in heads:
            params_to_opt.append({"params": h.parameters(), "lr": lr_head})
            
    optimizer = optim.Adam(params_to_opt) if params_to_opt else None
    
    base_model = ResNetExpert().to(device)
    base_fe = base_model.resnet
    base_fe.fc = nn.Identity()
    
    expert_params = []
    for exp in experts:
        sd = exp.state_dict()
        fe_sd = {k[7:]: v for k, v in sd.items() if k.startswith("resnet.") and not k.startswith("resnet.fc")}
        expert_params.append(fe_sd)
        
    fe_param_names = list(expert_params[0].keys())
    
    def merge_weights(lambdas):
        merged = {}
        for name in fe_param_names:
            g = get_layer_group(name)
            c = lambdas[g]
            if "running_mean" in name or "running_var" in name or "num_batches_tracked" in name:
                c = c.detach()
            merged[name] = (
                c[0] * expert_params[0][name] +
                c[1] * expert_params[1][name] +
                c[2] * expert_params[2][name]
            )
        return merged
    
    step_correct = []
    step_totals = []
    
    for step, (inputs, targets, true_task_id) in enumerate(stream):
        inputs, targets = inputs.to(device), targets.to(device)
        corrupted_inputs = apply_corruption(inputs)
        inferred_task_id = true_task_id
            
        if method != "static" and optimizer is not None:
            optimizer.zero_grad()
            lambdas = torch.softmax(alphas, dim=1)
            merged_params = merge_weights(lambdas)
                
            features_raw = functional_call(base_fe, merged_params, corrupted_inputs)
            features = torch.flatten(features_raw, 1)
            
            logits = heads[inferred_task_id](features)
            probs = torch.softmax(logits, dim=-1)
            log_probs = torch.log_softmax(logits, dim=-1)
            
            loss_ent = -torch.sum(probs * log_probs, dim=-1).mean()
            
            augmented_inputs = corrupted_inputs + torch.randn_like(corrupted_inputs) * 0.1
            features_aug_raw = functional_call(base_fe, merged_params, augmented_inputs)
            features_aug = torch.flatten(features_aug_raw, 1)
            logits_aug = heads[inferred_task_id](features_aug)
            log_probs_aug = torch.log_softmax(logits_aug, dim=-1)
            loss_const = nn.functional.kl_div(log_probs_aug, probs.detach(), reduction="batchmean")
            
            loss_align = 0.0
            loss_distill = 0.0
            if method.startswith("caba") and anchors is not None:
                task_anchors = torch.tensor(anchors[inferred_task_id], device=device)
                features_norm = nn.functional.normalize(features, p=2, dim=1)
                sims = torch.mm(features_norm, task_anchors.t())
                temp = 0.1
                p_anc = torch.softmax(sims / temp, dim=-1)
                loss_align = nn.functional.kl_div(log_probs, p_anc.detach(), reduction="batchmean")
                
                with torch.no_grad():
                    orig_logits = original_heads[inferred_task_id].to(device)(features)
                    orig_probs = torch.softmax(orig_logits, dim=-1)
                loss_distill = nn.functional.kl_div(log_probs, orig_probs.detach(), reduction="batchmean")
                
            gamma = 5.0 if method in ["caba", "caba_no_distill"] else 0.0
            distill_weight = 5.0 if method in ["caba", "caba_no_align"] else 0.0
            beta = 5.0
            loss = loss_ent + beta * loss_const + gamma * loss_align + distill_weight * loss_distill
            
            loss.backward()
            optimizer.step()
            
        with torch.no_grad():
            lambdas = torch.softmax(alphas, dim=1) if method != "static" else torch.ones(5, 3, device=device) / 3.0
            merged_params = merge_weights(lambdas)
            features_raw = functional_call(base_fe, merged_params, corrupted_inputs)
            features = torch.flatten(features_raw, 1)
            logits = heads[inferred_task_id](features)
            _, predicted = logits.max(1)
            
            correct = predicted.eq(targets).sum().item()
            step_correct.append(correct)
            step_totals.append(targets.size(0))
            
    return step_correct, step_totals

def main():
    experts = []
    original_heads = []
    anchors = []
    
    names = ["mnist", "fashion", "kmnist"]
    for name in names:
        model = ResNetExpert(num_classes=10).to(device)
        model.load_state_dict(torch.load(f"models/expert_{name}.pth", map_location=device, weights_only=True))
        model.eval()
        experts.append(model)
        original_heads.append(copy.deepcopy(model.fc))
        
        anc = np.load(f"anchors/anchors_{name}.npy")
        anchors.append(anc)
        
    seeds = [2026, 2027, 2028]
    methods = ["static", "s2c", "unconstrained", "caba", "caba_no_distill", "caba_no_align"]
    
    results = {m: {s: {} for s in seeds} for m in methods}
    
    for seed in seeds:
        print(f"--- Running seed {seed} ---")
        stream = make_test_stream(batch_size=64, seed=seed)
        for method in methods:
            correct, totals = run_experiment(method, stream, experts, original_heads, anchors=anchors, seed=seed)
            
            mnist_correct = sum(correct[0:10]) + sum(correct[30:40])
            mnist_total = sum(totals[0:10]) + sum(totals[30:40])
            
            fashion_correct = sum(correct[10:20]) + sum(correct[40:50])
            fashion_total = sum(totals[10:20]) + sum(totals[40:50])
            
            kmnist_correct = sum(correct[20:30]) + sum(correct[50:60])
            kmnist_total = sum(totals[20:30]) + sum(totals[50:60])
            
            overall_correct = sum(correct)
            overall_total = sum(totals)
            
            results[method][seed]["mnist"] = mnist_correct / mnist_total * 100
            results[method][seed]["fashion"] = fashion_correct / fashion_total * 100
            results[method][seed]["kmnist"] = kmnist_correct / kmnist_total * 100
            results[method][seed]["overall"] = overall_correct / overall_total * 100
            
    print("\n" + "="*80)
    print(f"{'Method':<20} | {'MNIST':<15} | {'Fashion':<15} | {'KMNIST':<15} | {'Overall':<15}")
    print("-"*80)
    
    for method in methods:
        row = f"{method:<20}"
        for metric in ["mnist", "fashion", "kmnist", "overall"]:
            vals = [results[method][s][metric] for s in seeds]
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            row += f" | {mean_val:.2f} ± {std_val:.2f}%"
        print(row)
    print("="*80)

if __name__ == "__main__":
    main()
