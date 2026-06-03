import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import copy

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    torch.backends.cudnn.enabled = False
    print("cuDNN disabled to prevent initialization issues.")

# Helper to load and split dataset
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
            "calibration": Subset(train_data, list(range(5000, 5256))),
            "test": test_data
        }
    return datasets

print("Preparing datasets...")
datasets = get_datasets()
print("Datasets prepared.")

# Helper to get pretrained ResNet-18
def get_resnet18():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(512, 10)
    return model

# Train expert models if not saved
def train_experts():
    experts = {}
    for name in ["MNIST", "FMNIST", "CIFAR10"]:
        save_path = f"expert_{name.lower()}.pth"
        if os.path.exists(save_path):
            print(f"Loading saved expert for {name} from {save_path}...")
            model = get_resnet18()
            model.load_state_dict(torch.load(save_path, map_location=device))
            model = model.to(device)
            experts[name] = model
        else:
            print(f"Training expert for {name}...")
            model = get_resnet18().to(device)
            dataloader = DataLoader(datasets[name]["finetune"], batch_size=128, shuffle=True)
            optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
            criterion = nn.CrossEntropyLoss()
            
            model.train()
            for epoch in range(5):
                running_loss = 0.0
                correct = 0
                total = 0
                for x, y in dataloader:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    outputs = model(x)
                    loss = criterion(outputs, y)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item() * x.size(0)
                    _, predicted = outputs.max(1)
                    total += y.size(0)
                    correct += predicted.eq(y).sum().item()
                epoch_loss = running_loss / total
                epoch_acc = correct / total
                print(f"Epoch {epoch+1}/5 - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}")
            
            torch.save(model.state_dict(), save_path)
            experts[name] = model
    return experts

experts = train_experts()

# Get pretrained model state_dict for base_weights (for Task Arithmetic)
pretrained_model = get_resnet18()
base_weights = {k: v.clone() for k, v in pretrained_model.state_dict().items()}

# Evaluation helper
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

# Print single-expert oracle accuracies
print("\n--- Oracle (Single Expert) Accuracies ---")
oracle_accs = {}
for name in ["MNIST", "FMNIST", "CIFAR10"]:
    acc = evaluate_model(experts[name], datasets[name]["test"])
    oracle_accs[name] = acc
    print(f"{name} Oracle: {acc:.2f}%")
print(f"Average Oracle: {np.mean(list(oracle_accs.values())):.2f}%")

# Helper to merge model backbones via Weight Averaging
def merge_models_wa(experts):
    merged = get_resnet18().to(device)
    merged_state = merged.state_dict()
    
    expert_states = {k: v.state_dict() for k, v in experts.items()}
    keys = [k for k in merged_state.keys() if "fc" not in k] # exclude heads
    
    for key in keys:
        tensors = [expert_states[name][key].float() for name in experts]
        merged_state[key] = torch.stack(tensors, dim=0).mean(dim=0).to(merged_state[key].dtype)
        
    merged.load_state_dict(merged_state)
    return merged

# Helper to merge model backbones via Task Arithmetic
def merge_models_ta(experts, base_weights, lam=0.3):
    merged = get_resnet18().to(device)
    merged_state = merged.state_dict()
    
    expert_states = {k: v.state_dict() for k, v in experts.items()}
    keys = [k for k in merged_state.keys() if "fc" not in k]
    
    for key in keys:
        task_vectors = [expert_states[name][key].float() - base_weights[key].to(device).float() for name in experts]
        merged_state[key] = (base_weights[key].to(device).float() + lam * torch.stack(task_vectors, dim=0).sum(dim=0)).to(merged_state[key].dtype)
        
    merged.load_state_dict(merged_state)
    return merged

# Build base merged models
print("\n--- Model Merging (Uncalibrated Baselines) ---")
wa_backbone = merge_models_wa(experts)
ta_backbone = merge_models_ta(experts, base_weights, lam=0.3)

# Collect statistics from BatchNorm layers
bn_layer_names = [name for name, module in wa_backbone.named_modules() if isinstance(module, nn.BatchNorm2d)]

def collect_layer_stats(model, dataset, layer_names):
    hooks = {}
    stats_hooks = {}
    
    class StatsHook:
        def __init__(self):
            self.outputs = []
        def __call__(self, module, input, output):
            self.outputs.append(output.detach().cpu())
            
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

# ZIO-CF parameter fuser
def fuse_calibration_to_bn(model, scales, shifts=None):
    fused_model = copy.deepcopy(model)
    for name, module in fused_model.named_modules():
        if name in scales:
            s = scales[name]
            with torch.no_grad():
                if isinstance(s, float) or s.ndim == 0:
                    module.weight.copy_(module.weight * s)
                    module.bias.copy_(module.bias * s)
                else:
                    module.weight.copy_(module.weight * s)
                    if shifts is not None and name in shifts:
                        module.bias.copy_(module.bias * s + shifts[name])
                    else:
                        module.bias.copy_(module.bias * s)
    return fused_model

# SP-TAAC calibration using ZIO-CF
def apply_sptaac(merged_backbone, experts, datasets):
    expert_stats = {name: collect_layer_stats(experts[name], datasets[name]["calibration"], bn_layer_names) 
                    for name in ["MNIST", "FMNIST", "CIFAR10"]}
                    
    joint_cal_sets = [datasets[name]["calibration"] for name in ["MNIST", "FMNIST", "CIFAR10"]]
    joint_cal_dataset = torch.utils.data.ConcatDataset(joint_cal_sets)
    merged_stats = collect_layer_stats(merged_backbone, joint_cal_dataset, bn_layer_names)
    
    scales = {}
    for layer in bn_layer_names:
        avg_std_exp = np.mean([expert_stats[name][layer]["std_g"] for name in ["MNIST", "FMNIST", "CIFAR10"]])
        std_m = merged_stats[layer]["std_g"]
        scales[layer] = avg_std_exp / (std_m + 1e-8)
        
    return fuse_calibration_to_bn(merged_backbone, scales)

def apply_tcac_expert_fused(merged_backbone, experts, datasets):
    fused_experts = {}
    for name in ["MNIST", "FMNIST", "CIFAR10"]:
        exp_stats = collect_layer_stats(experts[name], datasets[name]["calibration"], bn_layer_names)
        m_stats = collect_layer_stats(merged_backbone, datasets[name]["calibration"], bn_layer_names)
        
        scales = {}
        shifts = {}
        for layer in bn_layer_names:
            std_exp = exp_stats[layer]["std_c"]
            mean_exp = exp_stats[layer]["mean_c"]
            std_m = m_stats[layer]["std_c"]
            mean_m = m_stats[layer]["mean_c"]
            
            s = std_exp / (std_m + 1e-5)
            b = mean_exp - s * mean_m
            scales[layer] = s
            shifts[layer] = b
            
        fused_experts[name] = fuse_calibration_to_bn(merged_backbone, scales, shifts)
    return fused_experts


# Prototype extraction helper for Layer 2
def extract_prototypes(base_backbone, datasets):
    prototypes = {}
    class Layer2Hook:
        def __init__(self):
            self.outputs = []
        def __call__(self, module, input, output):
            self.outputs.append(output.detach().cpu())
            
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
        pooled = torch.mean(acts, dim=(2, 3)) # [N, C] where C=128
        avg_proto = torch.mean(pooled, dim=0) # [C]
        proto_norm = avg_proto / (torch.norm(avg_proto, p=2) + 1e-8)
        prototypes[name] = proto_norm.to(device)
        h.outputs = []
        
    hook.remove()
    return prototypes


# precompute prototypes
print("\nExtracting Layer 2 prototypes...")
wa_prototypes = extract_prototypes(wa_backbone, datasets)
ta_prototypes = extract_prototypes(ta_backbone, datasets)


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
        # 1. Run up to layer2
        feat = self.base_backbone.conv1(x)
        feat = self.base_backbone.bn1(feat)
        feat = self.base_backbone.relu(feat)
        feat = self.base_backbone.maxpool(feat)
        feat = self.base_backbone.layer1(feat)
        feat_anchor = self.base_backbone.layer2(feat)
        
        # 2. Extract v(x) and compute routing weights w_k(x)
        v = torch.mean(feat_anchor, dim=(2, 3)) # [B, 128]
        v_norm = v / (torch.norm(v, p=2, dim=1, keepdim=True) + 1e-8) # [B, 128]
        sims = torch.matmul(v_norm, self.prototypes.t()) # [B, 3]
        w = torch.softmax(self.beta * sims, dim=1) # [B, 3]
        
        # 3. Dynamic activation scaling for Layer 3
        feat = self.base_backbone.layer3(feat_anchor)
        gamma_3 = (w * self.scales_dict["layer3"].unsqueeze(0)).sum(dim=1, keepdim=True) # [B, 1]
        feat = feat * gamma_3.unsqueeze(-1).unsqueeze(-1)
        
        # Dynamic activation scaling for Layer 4
        feat = self.base_backbone.layer4(feat)
        gamma_4 = (w * self.scales_dict["layer4"].unsqueeze(0)).sum(dim=1, keepdim=True) # [B, 1]
        feat = feat * gamma_4.unsqueeze(-1).unsqueeze(-1)
        
        feat = self.base_backbone.avgpool(feat)
        feat = torch.flatten(feat, 1)
        
        # 4. Route final heads
        logits_0 = self.experts[0].fc(feat)
        logits_1 = self.experts[1].fc(feat)
        logits_2 = self.experts[2].fc(feat)
        
        logits = (w[:, 0:1] * logits_0 + 
                  w[:, 1:2] * logits_1 + 
                  w[:, 2:3] * logits_2)
        return logits


# MSPR (Minimalist Static Prototype Routing) implementation (Hard Head-Routing)
class MSPRModel(nn.Module):
    def __init__(self, backbone, experts, prototypes):
        super().__init__()
        self.backbone = copy.deepcopy(backbone)
        self.experts = nn.ModuleList([experts[name] for name in ["MNIST", "FMNIST", "CIFAR10"]])
        self.prototypes = nn.Parameter(torch.stack([prototypes[name] for name in ["MNIST", "FMNIST", "CIFAR10"]]), requires_grad=False)
        
    def forward(self, x):
        # 1. Run early layers up to layer2 output
        feat = self.backbone.conv1(x)
        feat = self.backbone.bn1(feat)
        feat = self.backbone.relu(feat)
        feat = self.backbone.maxpool(feat)
        feat = self.backbone.layer1(feat)
        feat_anchor = self.backbone.layer2(feat)
        
        # 2. Extract v(x) and classify task ID (arg max cosine similarity)
        v = torch.mean(feat_anchor, dim=(2, 3)) # [B, 128]
        v_norm = v / (torch.norm(v, p=2, dim=1, keepdim=True) + 1e-8) # [B, 128]
        sims = torch.matmul(v_norm, self.prototypes.t()) # [B, 3]
        task_ids = torch.argmax(sims, dim=1) # [B]
        
        # 3. Run rest of backbone (layer3, layer4)
        feat = self.backbone.layer3(feat_anchor)
        feat = self.backbone.layer4(feat)
        feat = self.backbone.avgpool(feat)
        feat = torch.flatten(feat, 1)
        
        # 4. Route final linear heads without any soft summing
        logits = torch.zeros(x.size(0), 10, device=x.device)
        for k in range(3):
            mask = (task_ids == k)
            if mask.sum() > 0:
                logits[mask] = self.experts[k].fc(feat[mask])
                
        return logits


# Build scales_dict for SRAC
def get_srac_scales(merged_backbone, experts, datasets):
    expert_stats = {name: collect_layer_stats(experts[name], datasets[name]["calibration"], bn_layer_names) 
                    for name in ["MNIST", "FMNIST", "CIFAR10"]}
                    
    # For merged, compute stats on each task's calibration set
    merged_stats_per_task = {name: collect_layer_stats(merged_backbone, datasets[name]["calibration"], bn_layer_names) 
                             for name in ["MNIST", "FMNIST", "CIFAR10"]}
                             
    scales_dict = {"layer3": [], "layer4": []}
    for layer in ["layer3", "layer4"]:
        # Find all BN modules inside this layer
        bn_sublayers = [bn for bn in bn_layer_names if layer in bn]
        
        # Compute average scale for this block across its BN sublayers
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


# Print routing statistics helper
def print_routing_stats(model, datasets):
    model.eval()
    for name in ["MNIST", "FMNIST", "CIFAR10"]:
        dataloader = DataLoader(datasets[name]["test"], batch_size=256, shuffle=False)
        routes = [0, 0, 0] # MNIST, FMNIST, CIFAR10
        with torch.no_grad():
            for x, _ in dataloader:
                x = x.to(device)
                feat = model.backbone.conv1(x)
                feat = model.backbone.bn1(feat)
                feat = model.backbone.relu(feat)
                feat = model.backbone.maxpool(feat)
                feat = model.backbone.layer1(feat)
                feat_anchor = model.backbone.layer2(feat)
                
                v = torch.mean(feat_anchor, dim=(2, 3))
                v_norm = v / (torch.norm(v, p=2, dim=1, keepdim=True) + 1e-8)
                sims = torch.matmul(v_norm, model.prototypes.t())
                task_ids = torch.argmax(sims, dim=1)
                
                for k in range(3):
                    routes[k] += (task_ids == k).sum().item()
        total = sum(routes)
        print(f"[{name} Dataset Routing] MNIST head: {routes[0]/total*100.0:.2f}%, FMNIST head: {routes[1]/total*100.0:.2f}%, CIFAR10 head: {routes[2]/total*100.0:.2f}%")

# Latency profiling helper
def profile_latency(model, x, name, num_runs=50):
    model.eval()
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    start_time = time.perf_counter()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    avg_latency = (end_time - start_time) / num_runs * 1000.0 # ms
    print(f"[{name}] Avg Latency: {avg_latency:.2f} ms")
    return avg_latency

def test_compilation_and_profile(model, x, name):
    model.eval()
    try:
        print(f"\nCompiling {name} using torch.compile...")
        compiled_model = torch.compile(model)
        # Trigger compilation
        with torch.no_grad():
            _ = compiled_model(x)
        print(f"Compilation successful for {name}!")
        latency = profile_latency(compiled_model, x, f"{name} (Compiled)")
        return latency
    except Exception as e:
        print(f"Compilation FAILED for {name}! Error: {str(e)}")
        return None

# Build and evaluate model versions
def run_evaluation_suite(backbone, prototypes, merge_name):
    print(f"\n--- Evaluation Suite for {merge_name} ---")
    
    # 1. N-TAAC + SRAC (The previous state-of-the-art)
    ntaac_backbone = apply_ntaac(backbone, datasets)
    srac_scales = get_srac_scales(ntaac_backbone, experts, datasets)
    srac_model = SRACModel(ntaac_backbone, experts, prototypes, srac_scales, beta=30.0)
    
    # 2. N-TAAC + MSPR (Our minimalist proposed alternative)
    mspr_model = MSPRModel(ntaac_backbone, experts, prototypes)
    
    # Let's evaluate on each test set task-agnostically!
    results = {}
    for model_name, model in [("N-TAAC + SRAC", srac_model), ("N-TAAC + MSPR (Ours)", mspr_model)]:
        accs = {}
        for name in ["MNIST", "FMNIST", "CIFAR10"]:
            acc = evaluate_model(model, datasets[name]["test"])
            accs[name] = acc
        avg_acc = np.mean(list(accs.values()))
        print(f"[{model_name}] MNIST: {accs['MNIST']:.2f}%, FMNIST: {accs['FMNIST']:.2f}%, CIFAR10: {accs['CIFAR10']:.2f}%, Avg: {avg_acc:.2f}%")
        results[model_name] = accs
        
    print("\n--- Routing Statistics ---")
    print_routing_stats(mspr_model, datasets)
    
    print("\n--- Latency and Compiler Profiling ---")
    # Sample batch
    sample_x = torch.randn(128, 3, 32, 32, device=device)
    
    # Profile base uncalibrated backbone
    profile_latency(ntaac_backbone, sample_x, "Uncalibrated Backbone")
    
    # Profile SRAC
    profile_latency(srac_model, sample_x, "N-TAAC + SRAC (Uncompiled)")
    
    # Profile MSPR
    profile_latency(mspr_model, sample_x, "N-TAAC + MSPR (Ours, Uncompiled)")
    
    # Test compilation
    test_compilation_and_profile(srac_model, sample_x, "N-TAAC + SRAC")
    test_compilation_and_profile(mspr_model, sample_x, "N-TAAC + MSPR")
    
    return results

print("\n--- Running Evaluation under Weight Averaging (WA) ---")
wa_results = run_evaluation_suite(wa_backbone, wa_prototypes, "WA")

print("\n--- Running Evaluation under Task Arithmetic (TA) ---")
ta_results = run_evaluation_suite(ta_backbone, ta_prototypes, "TA")


def run_lambda_sweep():
    print("\n--- Running Task Arithmetic Merging Coefficient (Lambda) Sweep ---")
    lambdas = [0.1, 0.3, 0.5, 0.7, 0.9]
    sweep_results = []
    
    for lam in lambdas:
        print(f"\nEvaluating Lambda = {lam}...")
        ta_back_lam = merge_models_ta(experts, base_weights, lam=lam)
        
        # Uncalibrated baseline
        uncal_accs = {}
        for name in ["MNIST", "FMNIST", "CIFAR10"]:
            model_to_eval = copy.deepcopy(ta_back_lam)
            model_to_eval.fc.load_state_dict(experts[name].fc.state_dict())
            acc = evaluate_model(model_to_eval, datasets[name]["test"])
            uncal_accs[name] = acc
        uncal_avg = np.mean(list(uncal_accs.values()))
        print(f"[TA Lambda={lam} Uncalibrated] MNIST: {uncal_accs['MNIST']:.2f}%, FMNIST: {uncal_accs['FMNIST']:.2f}%, CIFAR10: {uncal_accs['CIFAR10']:.2f}%, Avg: {uncal_avg:.2f}%")
        
        # N-TAAC + SRAC
        ntaac_backbone = apply_ntaac(ta_back_lam, datasets)
        srac_scales = get_srac_scales(ntaac_backbone, experts, datasets)
        srac_model = SRACModel(ntaac_backbone, experts, ta_prototypes, srac_scales, beta=30.0)
        srac_accs = {}
        for name in ["MNIST", "FMNIST", "CIFAR10"]:
            acc = evaluate_model(srac_model, datasets[name]["test"])
            srac_accs[name] = acc
        srac_avg = np.mean(list(srac_accs.values()))
        print(f"[TA Lambda={lam} N-TAAC+SRAC] MNIST: {srac_accs['MNIST']:.2f}%, FMNIST: {srac_accs['FMNIST']:.2f}%, CIFAR10: {srac_accs['CIFAR10']:.2f}%, Avg: {srac_avg:.2f}%")
        
        # N-TAAC + MSPR
        mspr_model = MSPRModel(ntaac_backbone, experts, ta_prototypes)
        mspr_accs = {}
        for name in ["MNIST", "FMNIST", "CIFAR10"]:
            acc = evaluate_model(mspr_model, datasets[name]["test"])
            mspr_accs[name] = acc
        mspr_avg = np.mean(list(mspr_accs.values()))
        print(f"[TA Lambda={lam} N-TAAC+MSPR (Ours)] MNIST: {mspr_accs['MNIST']:.2f}%, FMNIST: {mspr_accs['FMNIST']:.2f}%, CIFAR10: {mspr_accs['CIFAR10']:.2f}%, Avg: {mspr_avg:.2f}%")
        
        sweep_results.append({
            "lambda": lam,
            "uncal_avg": uncal_avg,
            "srac_avg": srac_avg,
            "mspr_avg": mspr_avg
        })
        
    print("\n--- Summary of Lambda Sweep ---")
    for r in sweep_results:
        print(f"Lambda {r['lambda']}: Uncalibrated Avg = {r['uncal_avg']:.2f}%, SRAC Avg = {r['srac_avg']:.2f}%, MSPR Avg = {r['mspr_avg']:.2f}%")

run_lambda_sweep()


# --- MSPR Routing Layer Ablation Sweep ---

class AblationMSPRModel(nn.Module):
    def __init__(self, backbone, experts, prototypes, layer_name):
        super().__init__()
        self.backbone = copy.deepcopy(backbone)
        self.experts = nn.ModuleList([experts[name] for name in ["MNIST", "FMNIST", "CIFAR10"]])
        self.prototypes = nn.Parameter(torch.stack([prototypes[name] for name in ["MNIST", "FMNIST", "CIFAR10"]]), requires_grad=False)
        self.layer_name = layer_name
        
    def forward(self, x):
        feat = self.backbone.conv1(x)
        feat = self.backbone.bn1(feat)
        feat = self.backbone.relu(feat)
        feat = self.backbone.maxpool(feat)
        
        if self.layer_name == 'layer1':
            feat_anchor = self.backbone.layer1(feat)
            feat = self.backbone.layer2(feat_anchor)
            feat = self.backbone.layer3(feat)
            feat = self.backbone.layer4(feat)
        elif self.layer_name == 'layer2':
            feat = self.backbone.layer1(feat)
            feat_anchor = self.backbone.layer2(feat)
            feat = self.backbone.layer3(feat_anchor)
            feat = self.backbone.layer4(feat)
        elif self.layer_name == 'layer3':
            feat = self.backbone.layer1(feat)
            feat = self.backbone.layer2(feat)
            feat_anchor = self.backbone.layer3(feat)
            feat = self.backbone.layer4(feat_anchor)
        elif self.layer_name == 'layer4':
            feat = self.backbone.layer1(feat)
            feat = self.backbone.layer2(feat)
            feat = self.backbone.layer3(feat)
            feat_anchor = self.backbone.layer4(feat)
            feat = feat_anchor
            
        v = torch.mean(feat_anchor, dim=(2, 3))
        v_norm = v / (torch.norm(v, p=2, dim=1, keepdim=True) + 1e-8)
        sims = torch.matmul(v_norm, self.prototypes.t())
        task_ids = torch.argmax(sims, dim=1)
        
        feat = self.backbone.avgpool(feat)
        feat = torch.flatten(feat, 1)
        
        logits = torch.zeros(x.size(0), 10, device=x.device)
        for k in range(3):
            mask = (task_ids == k)
            if mask.sum() > 0:
                logits[mask] = self.experts[k].fc(feat[mask])
        return logits

def extract_prototypes_for_layer(base_backbone, datasets, layer_name):
    prototypes = {}
    class LayerHook:
        def __init__(self):
            self.outputs = []
        def __call__(self, module, input, output):
            self.outputs.append(output.detach().cpu())
            
    h = LayerHook()
    target_module = getattr(base_backbone, layer_name)
    hook = target_module.register_forward_hook(h)
    
    for name in ["MNIST", "FMNIST", "CIFAR10"]:
        loader = DataLoader(datasets[name]["calibration"], batch_size=64, shuffle=False)
        base_backbone.eval()
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device)
                _ = base_backbone(x)
                
        acts = torch.cat(h.outputs, dim=0)
        pooled = torch.mean(acts, dim=(2, 3))
        avg_proto = torch.mean(pooled, dim=0)
        proto_norm = avg_proto / (torch.norm(avg_proto, p=2) + 1e-8)
        prototypes[name] = proto_norm.to(device)
        h.outputs = []
        
    hook.remove()
    return prototypes

def run_layer_ablation_sweep():
    print("\n--- Running MSPR Routing Layer Ablation Sweep ---")
    layers = ["layer1", "layer2", "layer3", "layer4"]
    ablation_results = []
    
    for merge_type, backbone in [("WA", wa_backbone), ("TA", ta_backbone)]:
        print(f"\nEvaluating Backbone: {merge_type}")
        ntaac_backbone = apply_ntaac(backbone, datasets)
        
        for layer in layers:
            print(f"  Extracting prototypes for {layer}...")
            layer_protos = extract_prototypes_for_layer(ntaac_backbone, datasets, layer)
            ablation_model = AblationMSPRModel(ntaac_backbone, experts, layer_protos, layer)
            
            routing_accs = {}
            for k, name in enumerate(["MNIST", "FMNIST", "CIFAR10"]):
                dataloader = DataLoader(datasets[name]["test"], batch_size=256, shuffle=False)
                correct_routes = 0
                total = 0
                with torch.no_grad():
                    for x, _ in dataloader:
                        x = x.to(device)
                        feat = ablation_model.backbone.conv1(x)
                        feat = ablation_model.backbone.bn1(feat)
                        feat = ablation_model.backbone.relu(feat)
                        feat = ablation_model.backbone.maxpool(feat)
                        
                        if layer == 'layer1':
                            feat_anchor = ablation_model.backbone.layer1(feat)
                        elif layer == 'layer2':
                            feat = ablation_model.backbone.layer1(feat)
                            feat_anchor = ablation_model.backbone.layer2(feat)
                        elif layer == 'layer3':
                            feat = ablation_model.backbone.layer1(feat)
                            feat = ablation_model.backbone.layer2(feat)
                            feat_anchor = ablation_model.backbone.layer3(feat)
                        elif layer == 'layer4':
                            feat = ablation_model.backbone.layer1(feat)
                            feat = ablation_model.backbone.layer2(feat)
                            feat = ablation_model.backbone.layer3(feat)
                            feat_anchor = ablation_model.backbone.layer4(feat)
                            
                        v = torch.mean(feat_anchor, dim=(2, 3))
                        v_norm = v / (torch.norm(v, p=2, dim=1, keepdim=True) + 1e-8)
                        sims = torch.matmul(v_norm, ablation_model.prototypes.t())
                        task_ids = torch.argmax(sims, dim=1)
                        correct_routes += (task_ids == k).sum().item()
                        total += x.size(0)
                routing_accs[name] = correct_routes / total * 100.0
                
            downstream_accs = {}
            for name in ["MNIST", "FMNIST", "CIFAR10"]:
                acc = evaluate_model(ablation_model, datasets[name]["test"])
                downstream_accs[name] = acc
            downstream_avg = np.mean(list(downstream_accs.values()))
            
            print(f"    [{layer}] Routing Accuracy -> MNIST: {routing_accs['MNIST']:.2f}%, FMNIST: {routing_accs['FMNIST']:.2f}%, CIFAR10: {routing_accs['CIFAR10']:.2f}%")
            print(f"    [{layer}] Multi-Task Accuracy -> Avg: {downstream_avg:.2f}%")
            
            ablation_results.append({
                "merge_type": merge_type,
                "layer": layer,
                "routing_mnist": routing_accs['MNIST'],
                "routing_fmnist": routing_accs['FMNIST'],
                "routing_cifar10": routing_accs['CIFAR10'],
                "downstream_avg": downstream_avg,
                "downstream_mnist": downstream_accs['MNIST'],
                "downstream_fmnist": downstream_accs['FMNIST'],
                "downstream_cifar10": downstream_accs['CIFAR10']
            })
            
    print("\n--- Summary of Layer Ablation Sweep ---")
    for r in ablation_results:
        print(f"[{r['merge_type']} - {r['layer']}] Routing (MNIST/FMNIST/CIFAR10): {r['routing_mnist']:.2f}%/{r['routing_fmnist']:.2f}%/{r['routing_cifar10']:.2f}%, Downstream Avg: {r['downstream_avg']:.2f}%")

def run_calibration_sensitivity_sweep():
    print("\n--- Running Calibration Dataset Size Sensitivity Sweep ---")
    sizes = [16, 32, 64, 128, 256]
    sweep_results = []
    
    # We will use the Weight Averaging (WA) backbone as our standard base
    for size in sizes:
        print(f"\nEvaluating with calibration size: {size}")
        
        # Create temporary datasets dict with the specified calibration size
        temp_datasets = {}
        for name in ["MNIST", "FMNIST", "CIFAR10"]:
            temp_datasets[name] = {
                "finetune": datasets[name]["finetune"],
                "calibration": Subset(datasets[name]["calibration"].dataset, list(range(5000, 5000 + size))),
                "test": datasets[name]["test"]
            }
            
        # 1. Apply N-TAAC using this calibration size
        ntaac_backbone = apply_ntaac(wa_backbone, temp_datasets)
        
        # 2. Extract prototypes at Layer 2 using the same calibration size
        protos = extract_prototypes(ntaac_backbone, temp_datasets)
        
        # 3. Create our MSPR model
        mspr_model = MSPRModel(ntaac_backbone, experts, protos)
        
        # 4. Measure Routing Accuracies
        routing_accs = {}
        for k, name in enumerate(["MNIST", "FMNIST", "CIFAR10"]):
            dataloader = DataLoader(temp_datasets[name]["test"], batch_size=256, shuffle=False)
            correct_routes = 0
            total = 0
            with torch.no_grad():
                for x, _ in dataloader:
                    x = x.to(device)
                    feat = mspr_model.backbone.conv1(x)
                    feat = mspr_model.backbone.bn1(feat)
                    feat = mspr_model.backbone.relu(feat)
                    feat = mspr_model.backbone.maxpool(feat)
                    feat = mspr_model.backbone.layer1(feat)
                    feat_anchor = mspr_model.backbone.layer2(feat)
                    
                    v = torch.mean(feat_anchor, dim=(2, 3))
                    v_norm = v / (torch.norm(v, p=2, dim=1, keepdim=True) + 1e-8)
                    sims = torch.matmul(v_norm, mspr_model.prototypes.t())
                    task_ids = torch.argmax(sims, dim=1)
                    correct_routes += (task_ids == k).sum().item()
                    total += x.size(0)
            routing_accs[name] = correct_routes / total * 100.0
            
        # 5. Measure downstream multi-task accuracies
        downstream_accs = {}
        for name in ["MNIST", "FMNIST", "CIFAR10"]:
            acc = evaluate_model(mspr_model, temp_datasets[name]["test"])
            downstream_accs[name] = acc
        downstream_avg = np.mean(list(downstream_accs.values()))
        
        print(f"    [Size {size}] Routing Accuracies -> MNIST: {routing_accs['MNIST']:.2f}%, FMNIST: {routing_accs['FMNIST']:.2f}%, CIFAR10: {routing_accs['CIFAR10']:.2f}%")
        print(f"    [Size {size}] Multi-Task Accuracy -> Avg: {downstream_avg:.2f}%")
        
        sweep_results.append({
            "size": size,
            "routing_mnist": routing_accs['MNIST'],
            "routing_fmnist": routing_accs['FMNIST'],
            "routing_cifar10": routing_accs['CIFAR10'],
            "downstream_avg": downstream_avg,
            "downstream_mnist": downstream_accs['MNIST'],
            "downstream_fmnist": downstream_accs['FMNIST'],
            "downstream_cifar10": downstream_accs['CIFAR10']
        })
        
    print("\n--- Summary of Calibration Sensitivity Sweep ---")
    for r in sweep_results:
        print(f"[Size {r['size']}] Routing (MNIST/FMNIST/CIFAR10): {r['routing_mnist']:.2f}%/{r['routing_fmnist']:.2f}%/{r['routing_cifar10']:.2f}%, Downstream Avg: {r['downstream_avg']:.2f}%")

run_layer_ablation_sweep()
run_calibration_sensitivity_sweep()

