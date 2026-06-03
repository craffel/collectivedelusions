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

# Raw Input Prototype Extraction
def extract_raw_prototypes(datasets):
    raw_prototypes = {}
    for name in ["MNIST", "FMNIST", "CIFAR10"]:
        loader = DataLoader(datasets[name]["calibration"], batch_size=64, shuffle=False)
        flat_acts = []
        for x, _ in loader:
            # Flatten image: [B, 3, 32, 32] -> [B, 3072]
            flat_x = x.view(x.size(0), -1)
            flat_acts.append(flat_x)
        all_flat = torch.cat(flat_acts, dim=0) # [N, 3072]
        mean_proto = all_flat.mean(dim=0) # [3072]
        proto_norm = mean_proto / (torch.norm(mean_proto, p=2) + 1e-8)
        raw_prototypes[name] = proto_norm.to(device)
    return raw_prototypes

# RIPR Model (Raw Input Prototype Routing)
class RIPRModel(nn.Module):
    def __init__(self, backbone, experts, raw_prototypes):
        super().__init__()
        self.backbone = copy.deepcopy(backbone)
        self.experts = nn.ModuleList([experts[name] for name in ["MNIST", "FMNIST", "CIFAR10"]])
        self.raw_prototypes = nn.Parameter(torch.stack([raw_prototypes[name] for name in ["MNIST", "FMNIST", "CIFAR10"]]), requires_grad=False)
        
    def forward(self, x):
        # 1. Flatten inputs to classify task ID (arg max cosine similarity with raw input prototypes)
        flat_x = x.view(x.size(0), -1) # [B, 3072]
        flat_x_norm = flat_x / (torch.norm(flat_x, p=2, dim=1, keepdim=True) + 1e-8)
        sims = torch.matmul(flat_x_norm, self.raw_prototypes.t()) # [B, 3]
        task_ids = torch.argmax(sims, dim=1) # [B]
        
        # 2. Run standard backbone
        feat = self.backbone(x) # [B, 10]
        
        # 3. Route final heads
        logits = torch.zeros(x.size(0), 10, device=x.device)
        for k in range(3):
            mask = (task_ids == k)
            if mask.sum() > 0:
                logits[mask] = self.experts[k].fc(feat[mask]) # Wait, feat here is already logits if backbone runs fully.
                # Let's check: in ResNet-18, backbone is usually model. backbone runs to fc. Wait, ResNet-18 runs fully?
                # Let's run it properly by routing the backbone feature or the expert head.
                # Actually, in ResNet-18, experts have expert.fc(feat). So we can do:
                # feat is feature from backbone (before fc)
        return logits

def run_evaluation():
    wa_backbone = merge_models_wa(experts)
    ntaac_backbone = apply_ntaac(wa_backbone, datasets)
    
    # Extract raw input prototypes
    raw_protos = extract_raw_prototypes(datasets)
    
    # We need a proper RIPR model that routes at the end, but uses raw input for routing.
    # Let's see how MSPR runs:
    # It runs the backbone up to layer4, pools, and then uses the linear expert heads.
    # Let's implement RIPR model exactly matching MSPR, but with raw input routing.
    
    class RIPREvalModel(nn.Module):
        def __init__(self, backbone, experts, raw_prototypes):
            super().__init__()
            self.backbone = copy.deepcopy(backbone)
            self.experts = nn.ModuleList([experts[name] for name in ["MNIST", "FMNIST", "CIFAR10"]])
            self.raw_prototypes = nn.Parameter(torch.stack([raw_prototypes[name] for name in ["MNIST", "FMNIST", "CIFAR10"]]), requires_grad=False)
            
        def forward(self, x):
            # Raw input routing
            flat_x = x.view(x.size(0), -1)
            flat_x_norm = flat_x / (torch.norm(flat_x, p=2, dim=1, keepdim=True) + 1e-8)
            sims = torch.matmul(flat_x_norm, self.raw_prototypes.t())
            task_ids = torch.argmax(sims, dim=1)
            
            # Run backbone
            feat = self.backbone.conv1(x)
            feat = self.backbone.bn1(feat)
            feat = self.backbone.relu(feat)
            feat = self.backbone.maxpool(feat)
            feat = self.backbone.layer1(feat)
            feat = self.backbone.layer2(feat)
            feat = self.backbone.layer3(feat)
            feat = self.backbone.layer4(feat)
            feat = self.backbone.avgpool(feat)
            feat = torch.flatten(feat, 1)
            
            logits = torch.zeros(x.size(0), 10, device=x.device)
            for k in range(3):
                mask = (task_ids == k)
                if mask.sum() > 0:
                    logits[mask] = self.experts[k].fc(feat[mask])
            return logits

    ripr_model = RIPREvalModel(ntaac_backbone, experts, raw_protos)
    
    # Measure Routing Accuracies
    routing_accs = {}
    for k, name in enumerate(["MNIST", "FMNIST", "CIFAR10"]):
        dataloader = DataLoader(datasets[name]["test"], batch_size=256, shuffle=False)
        correct_routes = 0
        total = 0
        with torch.no_grad():
            for x, _ in dataloader:
                x = x.to(device)
                flat_x = x.view(x.size(0), -1)
                flat_x_norm = flat_x / (torch.norm(flat_x, p=2, dim=1, keepdim=True) + 1e-8)
                sims = torch.matmul(flat_x_norm, ripr_model.raw_prototypes.t())
                task_ids = torch.argmax(sims, dim=1)
                correct_routes += (task_ids == k).sum().item()
                total += x.size(0)
        routing_accs[name] = correct_routes / total * 100.0
        
    # Measure downstream accuracies
    downstream_accs = {}
    for name in ["MNIST", "FMNIST", "CIFAR10"]:
        acc = evaluate_model(ripr_model, datasets[name]["test"])
        downstream_accs[name] = acc
    downstream_avg = np.mean(list(downstream_accs.values()))
    
    print("\n--- RIPR (Raw Input Prototype Routing) Results ---")
    print(f"Routing Accuracy -> MNIST: {routing_accs['MNIST']:.2f}%, FMNIST: {routing_accs['FMNIST']:.2f}%, CIFAR10: {routing_accs['CIFAR10']:.2f}%")
    print(f"Downstream Accuracy -> MNIST: {downstream_accs['MNIST']:.2f}%, FMNIST: {downstream_accs['FMNIST']:.2f}%, CIFAR10: {downstream_accs['CIFAR10']:.2f}%, Avg: {downstream_avg:.2f}%")

if __name__ == "__main__":
    run_evaluation()
