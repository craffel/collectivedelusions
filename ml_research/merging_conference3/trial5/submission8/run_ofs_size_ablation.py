import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import timm
from torchvision import datasets, transforms
import copy

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Define dataset loading with identical transforms
def get_transforms(dataset_name):
    if dataset_name in ['MNIST', 'FashionMNIST']:
        return transforms.Compose([
            transforms.Grayscale(3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

def get_dataset(dataset_name, split='train'):
    transform = get_transforms(dataset_name)
    download = True
    root = './data'
    
    if dataset_name == 'MNIST':
        return datasets.MNIST(root=root, train=(split == 'train'), download=download, transform=transform)
    elif dataset_name == 'FashionMNIST':
        return datasets.FashionMNIST(root=root, train=(split == 'train'), download=download, transform=transform)
    elif dataset_name == 'CIFAR10':
        return datasets.CIFAR10(root=root, train=(split == 'train'), download=download, transform=transform)
    elif dataset_name == 'SVHN':
        svhn_split = 'train' if split == 'train' else 'test'
        return datasets.SVHN(root=root, split=svhn_split, download=download, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

# Global model helper
def get_module_by_name(model, name):
    names = name.split('.')
    curr = model
    for n in names:
        curr = getattr(curr, n)
    return curr

class OFSTuneLinear(nn.Module):
    def __init__(self, base_linear, expert_linears, k_tasks, layer_idx, model_ref):
        super().__init__()
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.bias = base_linear.bias
        self.layer_idx = layer_idx

        self.register_buffer('base_weight', base_linear.weight.data.clone())

        task_vectors = []
        for expert in expert_linears:
            task_vectors.append(expert.weight.data.clone() - self.base_weight)
        self.register_buffer('task_vectors', torch.stack(task_vectors, dim=0)) # [K, D_out, D_in]

        object.__setattr__(self, 'model_ref', model_ref)

    def forward(self, x):
        alphas_clamped = torch.clamp(self.model_ref.alphas[self.layer_idx], 0.0, 1.0) # [K]
        delta_W = torch.einsum('k,koi->oi', alphas_clamped, self.task_vectors) # [D_out, D_in]
        W_merged = self.base_weight + delta_W # [D_out, D_in]
        return F.linear(x, W_merged, self.bias)

class OFSTuneModel(nn.Module):
    def __init__(self, base_model, experts, k_tasks):
        super().__init__()
        self.base_model = copy.deepcopy(base_model)
        self.k_tasks = k_tasks

        # Learnable layer-wise ensembling coefficients (48 linear layers in blocks)
        self.alphas = nn.Parameter(torch.ones(48, k_tasks) * 0.3)

        self.replace_linear_layers(self.base_model, experts)

        # Freeze base model parameters recursively so only alphas are optimized!
        for p in self.base_model.parameters():
            p.requires_grad = False

    def replace_linear_layers(self, model, experts, inside_blocks=False, target_count=None):
        if target_count is None:
            target_count = [0]

        for name, child in model.named_children():
            is_blocks = inside_blocks or (name == 'blocks')
            if isinstance(child, nn.Linear) and is_blocks:
                expert_linears = [get_module_by_name(exp, name) for exp in experts]
                new_linear = OFSTuneLinear(child, expert_linears, self.k_tasks, target_count[0], self)
                setattr(model, name, new_linear)
                target_count[0] += 1
            else:
                self.replace_linear_layers(child, [getattr(exp, name) for exp in experts], is_blocks, target_count)

    def forward(self, x):
        return self.base_model(x)

# Multi-task head wrapper
class ExpertHeadsWrapper(nn.Module):
    def __init__(self, model, expert_heads):
        super().__init__()
        self.model = model
        self.heads = nn.ModuleList([copy.deepcopy(h) for h in expert_heads])
        
    def forward(self, x, task_idx):
        features = self.model(x)
        return self.heads[task_idx](features)

# Parameterized Calibration Set Compiler (Dynamic Sizes)
def get_calibration_dataset(samples_per_task=16):
    datasets_list = ['MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN']
    cal_images = []
    cal_labels = []
    cal_task_indices = []
    
    for task_idx, ds in enumerate(datasets_list):
        train_dataset = get_dataset(ds, split='train')
        
        class_buckets = {i: [] for i in range(10)}
        samples_per_class = max(1, (samples_per_task + 9) // 10)
        
        for img, label in train_dataset:
            lbl = int(label)
            if len(class_buckets[lbl]) < samples_per_class:
                class_buckets[lbl].append(img)
            if sum(len(v) for v in class_buckets.values()) >= samples_per_task:
                break
                
        task_imgs = []
        task_lbls = []
        for lbl, imgs in class_buckets.items():
            for img in imgs:
                task_imgs.append(img)
                task_lbls.append(lbl)
                
        task_imgs = task_imgs[:samples_per_task]
        task_lbls = task_lbls[:samples_per_task]
        
        cal_images.extend(task_imgs)
        cal_labels.extend(task_lbls)
        cal_task_indices.extend([task_idx] * samples_per_task)
        
    return TensorDataset(torch.stack(cal_images), torch.tensor(cal_labels), torch.tensor(cal_task_indices))

def evaluate_stream(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels, task_indices in test_loader:
            images, labels = images.to(device), labels.to(device)
            features = model.model(images)
            outputs = []
            for b in range(images.size(0)):
                task_idx = task_indices[b].item()
                out = model.heads[task_idx](features[b:b+1])
                outputs.append(out)
            outputs = torch.cat(outputs, dim=0)
            _, predicted = outputs.max(1)
            total += images.size(0)
            correct += (predicted == labels).sum().item()
    return 100.0 * correct / total

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running OFS-Tune calibration dataset size ablation study on {device}...")
    
    # Load base model
    base_model = timm.create_model('vit_tiny_patch16_224', pretrained=True).to(device)
    base_model.reset_classifier(0)
    
    datasets_list = ['MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN']
    experts = []
    expert_heads = []
    
    for ds in datasets_list:
        model_path = f'checkpoints/{ds.lower()}_expert.pth'
        expert = timm.create_model('vit_tiny_patch16_224', pretrained=True)
        expert.head = nn.Linear(expert.head.in_features, 10)
        expert.load_state_dict(torch.load(model_path, map_location='cpu'))
        expert = expert.to(device)
        
        expert_heads.append(copy.deepcopy(expert.head))
        expert.reset_classifier(0)
        experts.append(expert)
        
    # Load test validation loader
    set_seed(42)
    test_images = []
    test_labels = []
    test_task_indices = []
    for task_idx, ds in enumerate(datasets_list):
        test_ds = get_dataset(ds, split='val')
        indices = list(range(len(test_ds)))
        random.shuffle(indices)
        indices = indices[:500]
        for idx in indices:
            img, label = test_ds[idx]
            test_images.append(img)
            test_labels.append(label)
            test_task_indices.append(task_idx)
            
    test_images_t = torch.stack(test_images)
    test_labels_t = torch.tensor(test_labels)
    test_task_t = torch.tensor(test_task_indices)
    
    shuffled_indices = list(range(len(test_images)))
    random.shuffle(shuffled_indices)
    shuffled_test_loader = DataLoader(
        TensorDataset(test_images_t[shuffled_indices], test_labels_t[shuffled_indices], test_task_t[shuffled_indices]),
        batch_size=16, shuffle=False
    )
    
    criterion = nn.CrossEntropyLoss()
    
    sizes = [64, 128, 256, 512]
    seeds = [42, 100, 2026]
    size_results = {s: [] for s in sizes}
    
    for size in sizes:
        samples_per_task = size // 4
        print(f"\n--- Sweeping OFS-Tune Dataset Size: {size} samples ({samples_per_task}/task) ---")
        for seed in seeds:
            print(f"  Running seed {seed}...")
            set_seed(seed)
            cal_dataset = get_calibration_dataset(samples_per_task=samples_per_task)
            cal_loader = DataLoader(cal_dataset, batch_size=16, shuffle=True)
            
            set_seed(seed)
            ofs_model = OFSTuneModel(base_model, experts, k_tasks=4).to(device)
            wrapped_ofs = ExpertHeadsWrapper(ofs_model, expert_heads).to(device)
            optimizer = torch.optim.Adam(ofs_model.parameters(), lr=1e-3)
            
            ofs_model.train()
            for step in range(100):
                for images, labels, task_indices in cal_loader:
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad()
                    features = ofs_model(images)
                    outputs = []
                    for b in range(images.size(0)):
                        task_idx = task_indices[b].item()
                        out = wrapped_ofs.heads[task_idx](features[b:b+1])
                        outputs.append(out)
                    outputs = torch.cat(outputs, dim=0)
                    loss = criterion(outputs, labels) * images.size(0)
                    loss.backward()
                    optimizer.step()
                    
            acc = evaluate_stream(wrapped_ofs, shuffled_test_loader, device)
            print(f"    Seed {seed} | Accuracy: {acc:.2f}%")
            size_results[size].append(acc)
            
            del ofs_model, wrapped_ofs
            torch.cuda.empty_cache()

    print("\n=== OFS-TUNE SIZE RESULTS (MULTI-SEED) ===")
    print("| Size (Samples) | OFS-Tune Accuracy (%) |")
    print("|----------------|-----------------------|")
    for s in sizes:
        accs = size_results[s]
        print(f"| {s:<14} | {np.mean(accs):.2f}% ± {np.std(accs):.2f}% |")

if __name__ == '__main__':
    main()
