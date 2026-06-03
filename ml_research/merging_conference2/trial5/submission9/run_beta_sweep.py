import os
import copy
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def get_datasets():
    transform_mnist = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    transform_cifar = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    mnist_train = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform_mnist)
    mnist_test = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform_mnist)
    
    fmnist_train = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform_mnist)
    fmnist_test = torchvision.datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform_mnist)
    
    cifar_train = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_cifar)
    cifar_test = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_cifar)
    
    datasets = {}
    for name, train_data, test_data in [("MNIST", mnist_train, mnist_test), 
                                        ("FMNIST", fmnist_train, fmnist_test), 
                                        ("CIFAR10", cifar_train, cifar_test)]:
        datasets[name] = {
            "finetune": Subset(train_data, list(range(5000))),
            "calibration": Subset(train_data, list(range(5000, 5256))), # 256 samples
            "test": test_data
        }
    return datasets

print("Preparing datasets...")
datasets = get_datasets()

def get_resnet18():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(512, 10)
    return model

def load_experts():
    experts = {}
    for name in ["MNIST", "FMNIST", "CIFAR10"]:
        save_path = f"expert_{name.lower()}.pth"
        print(f"Loading saved expert for {name} from {save_path}...")
        model = get_resnet18()
        model.load_state_dict(torch.load(save_path, map_location=device))
        model = model.to(device)
        experts[name] = model
    return experts

experts = load_experts()

def evaluate_model(model, test_dataset):
    model.eval()
    dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
    return correct / total * 100.0

def merge_models_wa(experts):
    merged = get_resnet18().to(device)
    merged_state = merged.state_dict()
    expert_states = {name: experts[name].state_dict() for name in experts}
    keys = [k for k in merged_state.keys() if "fc" not in k] # exclude heads
    for key in keys:
        tensors = [expert_states[name][key].float() for name in experts]
        merged_state[key] = torch.stack(tensors, dim=0).mean(dim=0).to(merged_state[key].dtype)
    merged.load_state_dict(merged_state)
    return merged

# BN Layer stats
bn_layer_names = [name for name, module in merge_models_wa(experts).named_modules() if isinstance(module, nn.BatchNorm2d)]

def collect_layer_stats(model, dataset, layer_names):
    hooks = {}
    stats_hooks = {}
    
    class StatsHook:
        def __init__(self):
            self.outputs = []
        def __call__(self, module, input, output):
            self.outputs.append(output.detach().cpu().clone()) # clone is CPU-safe
            
    for name, module in model.named_modules():
        if name in layer_names:
            h = StatsHook()
            stats_hooks[name] = h
            hooks[name] = module.register_forward_hook(h)
            
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
    model.eval()
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            _ = model(x)
            
    for h in hooks.values():
        h.remove()
        
    stats = {}
    for name, h in stats_hooks.items():
        acts = torch.cat(h.outputs, dim=0) # [N, C, H, W]
        mean_c = acts.mean(dim=(0, 2, 3)).to(device)
        std_c = acts.std(dim=(0, 2, 3), unbiased=False).to(device)
        std_g = acts.std(unbiased=False).item()
        stats[name] = {"mean_c": mean_c, "std_c": std_c, "std_g": std_g}
    return stats

# N-TAAC implementation
def apply_ntaac(model, datasets):
    joint_cal_sets = [datasets[name]["calibration"] for name in ["MNIST", "FMNIST", "CIFAR10"]]
    joint_cal_dataset = torch.utils.data.ConcatDataset(joint_cal_sets)
    loader = DataLoader(joint_cal_dataset, batch_size=64, shuffle=True)

    cal_model = copy.deepcopy(model)
    cal_model.eval()
    for module in cal_model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            module.train()

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            _ = cal_model(x)

    cal_model.eval()
    return cal_model

def get_srac_scales(merged_backbone, experts, datasets):
    expert_stats = {name: collect_layer_stats(experts[name], datasets[name]["calibration"], bn_layer_names) 
                    for name in ["MNIST", "FMNIST", "CIFAR10"]}
                    
    merged_stats_per_task = {name: collect_layer_stats(merged_backbone, datasets[name]["calibration"], bn_layer_names) 
                             for name in ["MNIST", "FMNIST", "CIFAR10"]}
                             
    scales_dict = {"layer3": [], "layer4": []}
    for layer in ["layer3", "layer4"]:
        bn_sublayers = [bn for bn in bn_layer_names if layer in bn]
        for task_idx, name in enumerate(["MNIST", "FMNIST", "CIFAR10"]):
            scale_val_list = []
            for bn in bn_sublayers:
                std_exp = expert_stats[name][bn]["std_g"]
                std_m = merged_stats_per_task[name][bn]["std_g"]
                scale_val_list.append(std_exp / (std_m + 1e-8))
            avg_scale = np.mean(scale_val_list)
            scales_dict[layer].append(avg_scale)
        scales_dict[layer] = torch.tensor(scales_dict[layer], dtype=torch.float32, device=device)
    return scales_dict

def extract_prototypes(base_backbone, datasets):
    prototypes = {}
    class Layer2Hook:
        def __init__(self):
            self.outputs = []
        def __call__(self, module, input, output):
            self.outputs.append(output.detach().cpu().clone())
            
    h = Layer2Hook()
    hook = base_backbone.layer2.register_forward_hook(h)
    
    for name in ["MNIST", "FMNIST", "CIFAR10"]:
        loader = DataLoader(datasets[name]["calibration"], batch_size=64, shuffle=False)
        base_backbone.eval()
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device)
                _ = base_backbone(x)
                
        acts = torch.cat(h.outputs, dim=0) # [N, C, H, W]
        pooled = torch.mean(acts, dim=(2, 3)) # [N, C]
        avg_proto = torch.mean(pooled, dim=0) # [C]
        proto_norm = avg_proto / (torch.norm(avg_proto, p=2) + 1e-8)
        prototypes[name] = proto_norm.to(device)
        h.outputs = []
        
    hook.remove()
    return prototypes

# SRAC implementation
class SRACModel(nn.Module):
    def __init__(self, base_backbone, experts, prototypes, scales_dict, beta=30.0):
        super().__init__()
        self.base_backbone = copy.deepcopy(base_backbone)
        self.experts = nn.ModuleList([experts[name] for name in ["MNIST", "FMNIST", "CIFAR10"]])
        self.prototypes = nn.Parameter(torch.stack([prototypes[name] for name in ["MNIST", "FMNIST", "CIFAR10"]]), requires_grad=False)
        self.scales_dict = scales_dict
        self.beta = beta
        
    def forward(self, x):
        feat = self.base_backbone.conv1(x)
        feat = self.base_backbone.bn1(feat)
        feat = self.base_backbone.relu(feat)
        feat = self.base_backbone.maxpool(feat)
        feat = self.base_backbone.layer1(feat)
        feat_anchor = self.base_backbone.layer2(feat)
        
        v = torch.mean(feat_anchor, dim=(2, 3))
        v_norm = v / (torch.norm(v, p=2, dim=1, keepdim=True) + 1e-8)
        sims = torch.matmul(v_norm, self.prototypes.t())
        w = torch.softmax(self.beta * sims, dim=1)
        
        feat = self.base_backbone.layer3(feat_anchor)
        gamma_3 = (w * self.scales_dict["layer3"].unsqueeze(0)).sum(dim=1, keepdim=True)
        feat = feat * gamma_3.unsqueeze(-1).unsqueeze(-1)
        
        feat = self.base_backbone.layer4(feat)
        gamma_4 = (w * self.scales_dict["layer4"].unsqueeze(0)).sum(dim=1, keepdim=True)
        feat = feat * gamma_4.unsqueeze(-1).unsqueeze(-1)
        
        feat = self.base_backbone.avgpool(feat)
        feat = torch.flatten(feat, 1)
        
        logits_0 = self.experts[0].fc(feat)
        logits_1 = self.experts[1].fc(feat)
        logits_2 = self.experts[2].fc(feat)
        
        logits = (w[:, 0:1] * logits_0 + 
                  w[:, 1:2] * logits_1 + 
                  w[:, 2:3] * logits_2)
        return logits

# MSPR implementation
class MSPRModel(nn.Module):
    def __init__(self, backbone, experts, prototypes):
        super().__init__()
        self.backbone = copy.deepcopy(backbone)
        self.experts = nn.ModuleList([experts[name] for name in ["MNIST", "FMNIST", "CIFAR10"]])
        self.prototypes = nn.Parameter(torch.stack([prototypes[name] for name in ["MNIST", "FMNIST", "CIFAR10"]]), requires_grad=False)
        
    def forward(self, x):
        feat = self.backbone.conv1(x)
        feat = self.backbone.bn1(feat)
        feat = self.backbone.relu(feat)
        feat = self.backbone.maxpool(feat)
        feat = self.backbone.layer1(feat)
        feat_anchor = self.backbone.layer2(feat)
        
        v = torch.mean(feat_anchor, dim=(2, 3))
        v_norm = v / (torch.norm(v, p=2, dim=1, keepdim=True) + 1e-8)
        sims = torch.matmul(v_norm, self.prototypes.t())
        task_ids = torch.argmax(sims, dim=1)
        
        feat = self.backbone.layer3(feat_anchor)
        feat = self.backbone.layer4(feat)
        feat = self.backbone.avgpool(feat)
        feat = torch.flatten(feat, 1)
        
        logits = torch.zeros(x.size(0), 10, device=x.device)
        for k in range(3):
            mask = (task_ids == k)
            if mask.sum() > 0:
                logits[mask] = self.experts[k].fc(feat[mask])
        return logits

def run_beta_sweep():
    # 1. Merge models
    wa_backbone = merge_models_wa(experts)
    
    # 2. Calibrate BN
    ntaac_backbone = apply_ntaac(wa_backbone, datasets)
    
    # 3. Extract Prototypes and scale factors
    prototypes = extract_prototypes(ntaac_backbone, datasets)
    scales_dict = get_srac_scales(ntaac_backbone, experts, datasets)
    
    # Evaluate MSPR (No beta, completely parameter-free)
    mspr_model = MSPRModel(ntaac_backbone, experts, prototypes)
    mspr_accs = {}
    for name in ["MNIST", "FMNIST", "CIFAR10"]:
        mspr_accs[name] = evaluate_model(mspr_model, datasets[name]["test"])
    mspr_avg = np.mean(list(mspr_accs.values()))
    print(f"[MSPR (Ours, No beta)] MNIST: {mspr_accs['MNIST']:.2f}%, FMNIST: {mspr_accs['FMNIST']:.2f}%, CIFAR10: {mspr_accs['CIFAR10']:.2f}%, Avg: {mspr_avg:.2f}%")
    
    betas = [0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 100.0, 500.0, 1000.0]
    srac_results = []
    
    for beta in betas:
        srac_model = SRACModel(ntaac_backbone, experts, prototypes, scales_dict, beta=beta)
        srac_accs = {}
        for name in ["MNIST", "FMNIST", "CIFAR10"]:
            srac_accs[name] = evaluate_model(srac_model, datasets[name]["test"])
        srac_avg = np.mean(list(srac_accs.values()))
        print(f"[SRAC beta={beta}] MNIST: {srac_accs['MNIST']:.2f}%, FMNIST: {srac_accs['FMNIST']:.2f}%, CIFAR10: {srac_accs['CIFAR10']:.2f}%, Avg: {srac_avg:.2f}%")
        srac_results.append({
            "beta": beta,
            "MNIST": srac_accs["MNIST"],
            "FMNIST": srac_accs["FMNIST"],
            "CIFAR10": srac_accs["CIFAR10"],
            "Avg": srac_avg
        })
        
    print("\n--- Summary of SRAC Temperature Sensitivity ---")
    print("| Method (Beta) | MNIST (%) | FMNIST (%) | CIFAR-10 (%) | Average (%) |")
    print("|---|---|---|---|---|")
    for r in srac_results:
        print(f"| SRAC (\\beta={r['beta']}) | {r['MNIST']:.2f}% | {r['FMNIST']:.2f}% | {r['CIFAR10']:.2f}% | {r['Avg']:.2f}% |")
    print(f"| **MSPR (Ours, No \\beta)** | {mspr_accs['MNIST']:.2f}% | {mspr_accs['FMNIST']:.2f}% | {mspr_accs['CIFAR10']:.2f}% | {mspr_avg:.2f}% |")

if __name__ == "__main__":
    run_beta_sweep()
