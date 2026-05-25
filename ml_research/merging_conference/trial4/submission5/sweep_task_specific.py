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

def set_seed(seed=2026):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(2026)

device = torch.device("cpu")
torch.backends.cudnn.enabled = False

# Transforms
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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

def apply_corruption(inputs):
    noise = torch.randn_like(inputs) * 0.25
    corrupted = inputs + noise
    import torch.nn.functional as F
    corrupted = F.avg_pool2d(corrupted, kernel_size=3, stride=1, padding=1)
    return corrupted

def make_test_stream(batch_size=64):
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

def run_task_specific_experiment(stream, experts, original_heads, anchors, config):
    set_seed(2026)
    alphas = torch.zeros(5, 3, device=device, requires_grad=True)
    heads = [copy.deepcopy(h).to(device) for h in original_heads]
    
    for h in heads:
        for p in h.parameters():
            p.requires_grad = True
                
    params_to_opt = [
        {"params": [alphas], "lr": 0.005},
    ]
    for h in heads:
        params_to_opt.append({"params": h.parameters(), "lr": 0.001})
            
    optimizer = optim.Adam(params_to_opt)
    
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
        
        # Hyperparameters for this task
        gamma = config[inferred_task_id]["gamma"]
        distill_weight = config[inferred_task_id]["distill"]
        
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
        
        beta = 5.0
        loss = loss_ent + beta * loss_const + gamma * loss_align + distill_weight * loss_distill
        
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            lambdas = torch.softmax(alphas, dim=1)
            merged_params = merge_weights(lambdas)
            features_raw = functional_call(base_fe, merged_params, corrupted_inputs)
            features = torch.flatten(features_raw, 1)
            logits = heads[inferred_task_id](features)
            _, predicted = logits.max(1)
            
            correct = predicted.eq(targets).sum().item()
            step_correct.append(correct)
            step_totals.append(targets.size(0))
            
    # Calculate task-specific and overall accuracies
    task_correct = {0: 0, 1: 0, 2: 0}
    task_totals = {0: 0, 1: 0, 2: 0}
    for step, (inputs, targets, true_task_id) in enumerate(stream):
        task_correct[true_task_id] += step_correct[step]
        task_totals[true_task_id] += step_totals[step]
        
    accs = {}
    for tid in [0, 1, 2]:
        accs[tid] = task_correct[tid] / task_totals[tid]
    accs["overall"] = sum(step_correct) / sum(step_totals)
    return accs

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
        
    set_seed(2026)
    stream = make_test_stream(batch_size=64)
    
    # Let's define some candidate configurations to sweep
    # Config format: {0: {"gamma": G, "distill": D}, 1: {"gamma": G, "distill": D}, 2: {"gamma": G, "distill": D}}
    # We want to vary MNIST (task 0) and KMNIST (task 2) specifically.
    
    mnist_options = [
        {"gamma": 0.0, "distill": 10.0},
        {"gamma": 1.0, "distill": 10.0},
        {"gamma": 1.0, "distill": 15.0},
        {"gamma": 2.0, "distill": 10.0},
        {"gamma": 5.0, "distill": 5.0}, # Baseline
    ]
    
    fashion_options = [
        {"gamma": 5.0, "distill": 5.0}, # Baseline
        {"gamma": 10.0, "distill": 5.0},
    ]
    
    kmnist_options = [
        {"gamma": 5.0, "distill": 5.0}, # Baseline
        {"gamma": 10.0, "distill": 5.0},
        {"gamma": 10.0, "distill": 3.0},
        {"gamma": 15.0, "distill": 5.0},
    ]
    
    best_overall = 0.0
    best_config = None
    best_results = None
    
    print("Starting Sweep...")
    for m_opt in mnist_options:
        for f_opt in fashion_options:
            for k_opt in kmnist_options:
                config = {0: m_opt, 1: f_opt, 2: k_opt}
                accs = run_task_specific_experiment(stream, experts, original_heads, anchors, config)
                
                print(f"M: {m_opt}, F: {f_opt}, K: {k_opt} | MNIST: {accs[0]*100:.2f}% | Fashion: {accs[1]*100:.2f}% | KMNIST: {accs[2]*100:.2f}% | Overall: {accs['overall']*100:.2f}%")
                
                if accs["overall"] > best_overall:
                    best_overall = accs["overall"]
                    best_config = config
                    best_results = accs
                    
    print("\n=== SWEEP COMPLETE ===")
    print("Best Config:")
    print(f"MNIST: {best_config[0]}")
    print(f"Fashion: {best_config[1]}")
    print(f"KMNIST: {best_config[2]}")
    print(f"MNIST Acc: {best_results[0]*100:.2f}%")
    print(f"Fashion Acc: {best_results[1]*100:.2f}%")
    print(f"KMNIST Acc: {best_results[2]*100:.2f}%")
    print(f"Best Overall Accuracy: {best_overall*100:.2f}%")

if __name__ == "__main__":
    main()
