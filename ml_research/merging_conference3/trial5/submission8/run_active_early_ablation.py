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

# EpiMerge Linear Layer Implementation supporting Rank R
class EpiMergeLinear(nn.Module):
    def __init__(self, base_linear, expert_linears, k_tasks, latent_dim, model_ref):
        super().__init__()
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.bias = base_linear.bias
        
        self.register_buffer('base_weight', base_linear.weight.data.clone())
        
        task_vectors = []
        for expert in expert_linears:
            task_vectors.append(expert.weight.data.clone() - self.base_weight)
        self.register_buffer('task_vectors', torch.stack(task_vectors, dim=0)) # [K, D_out, D_in]
        
        # Epigenetic Reader Heads (ERH) - low rank row/column masks with rank R
        self.rank = getattr(model_ref, 'rank', 1)
        
        self.U = nn.Parameter(torch.randn(k_tasks, self.out_features, self.rank, latent_dim) * 0.02)
        self.V = nn.Parameter(torch.randn(k_tasks, self.in_features, self.rank, latent_dim) * 0.02)
        
        object.__setattr__(self, 'model_ref', model_ref)
        
    def forward(self, x):
        h = self.model_ref.current_h # [B, d]
        
        r = torch.sigmoid(torch.einsum('kord,bd->kbor', self.U, h)) # [K, B, D_out, R]
        c = torch.sigmoid(torch.einsum('kird,bd->kbir', self.V, h)) # [K, B, D_in, R]
        
        # W_merged_b = W_base + \sum_k \sum_r (r_kr \otimes c_kr) \odot T_k
        delta_W = torch.einsum('kbor,kbir,koi->boi', r, c, self.task_vectors) # [B, D_out, D_in]
        W_merged = self.base_weight.unsqueeze(0) + delta_W # [B, D_out, D_in]
        
        out = torch.einsum('bni,boi->bno', x, W_merged)
        if self.bias is not None:
            out = out + self.bias
        return out

# Dynamic Wrapper Model supporting parameterized l_early
class DynamicMergedModel(nn.Module):
    def __init__(self, base_model, experts, k_tasks, latent_dim, layer_class, sensory_mode='active_early', rank=1, l_early=4):
        super().__init__()
        self.base_model = copy.deepcopy(base_model)
        self.k_tasks = k_tasks
        self.latent_dim = latent_dim
        self.sensory_mode = sensory_mode
        self.rank = rank
        self.l_early = l_early
        
        # Load uniform merged weights first (lambda=0.3) so static parts are high quality
        uniform_state = {}
        base_state = base_model.state_dict()
        expert_states = [exp.state_dict() for exp in experts]
        for name, param in base_state.items():
            if 'blocks' in name and any(p in name for p in ['attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2', 'weight']):
                task_vector_sum = torch.zeros_like(param)
                for k, exp_state in enumerate(expert_states):
                    task_vector_sum += 0.3 * (exp_state[name] - param)
                uniform_state[name] = param + task_vector_sum
            else:
                uniform_state[name] = param
        self.base_model.load_state_dict(uniform_state, strict=False)

        self.sensory_extractor = None
        
        # Frozen projection matrix from embed dim to latent dim
        self.register_buffer('P_proj', torch.randn(latent_dim, 192) * (1.0 / np.sqrt(192)))
        self.current_h = None
        
        # Replace linear layers in Blocks starting from l_early
        self.replace_linear_layers(self.base_model, experts, layer_class)
        
    def replace_linear_layers(self, model, experts, layer_class):
        blocks_module = model.blocks
        for idx in range(self.l_early, len(blocks_module)):
            block = blocks_module[idx]
            exp_blocks = [exp.blocks[idx] for exp in experts]
            self._replace_block_linears(block, exp_blocks, layer_class)

    def _replace_block_linears(self, module, exp_modules, layer_class):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                expert_linears = [get_module_by_name(exp, name) for exp in exp_modules]
                new_linear = layer_class(child, expert_linears, self.k_tasks, self.latent_dim, self)
                setattr(module, name, new_linear)
            else:
                self._replace_block_linears(child, [getattr(exp, name) for exp in exp_modules], layer_class)
                
    def forward(self, images):
        # sensory_mode == 'active_early'
        # 1. Run the first l_early blocks statically
        x = self.base_model.patch_embed(images)
        x = self.base_model._pos_embed(x)
        x = self.base_model.norm_pre(x)
        
        for idx in range(self.l_early):
            x = self.base_model.blocks[idx](x)
            
        # 2. Extract global latent representation from Layer l_early
        pooled = x.mean(dim=1) # [B, 192]
        h = F.linear(pooled, self.P_proj) # [B, latent_dim]
        self.current_h = h
        
        # 3. Run the remaining blocks dynamically
        for idx in range(self.l_early, len(self.base_model.blocks)):
            x = self.base_model.blocks[idx](x)
                
        # Common head forward
        x = self.base_model.norm(x)
        if hasattr(self.base_model, 'forward_head'):
            return self.base_model.forward_head(x, pre_logits=True)
        else:
            return x[:, 0]

# Multi-task head wrapper
class ExpertHeadsWrapper(nn.Module):
    def __init__(self, model, expert_heads):
        super().__init__()
        self.model = model
        self.heads = nn.ModuleList([copy.deepcopy(h) for h in expert_heads])
        
    def forward(self, x, task_idx):
        features = self.model(x)
        return self.heads[task_idx](features)

# Calibration Set Compiler
def get_calibration_dataset():
    datasets_list = ['MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN']
    cal_images = []
    cal_labels = []
    cal_task_indices = []
    
    for task_idx, ds in enumerate(datasets_list):
        train_dataset = get_dataset(ds, split='train')
        
        class_buckets = {i: [] for i in range(10)}
        for img, label in train_dataset:
            lbl = int(label)
            if len(class_buckets[lbl]) < 2:
                class_buckets[lbl].append(img)
            if sum(len(v) for v in class_buckets.values()) >= 16:
                break
                
        task_imgs = []
        task_lbls = []
        for lbl, imgs in class_buckets.items():
            for img in imgs:
                task_imgs.append(img)
                task_lbls.append(lbl)
                
        task_imgs = task_imgs[:16]
        task_lbls = task_lbls[:16]
        
        cal_images.extend(task_imgs)
        cal_labels.extend(task_lbls)
        cal_task_indices.extend([task_idx] * 16)
        
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
    print(f"Running active early partition boundary ablation study on {device}...")
    
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
    
    # Ablation values: 2, 4, 6, 8, 10
    l_early_values = [2, 4, 6, 8, 10]
    seeds = [42, 100, 2026]
    ablation_results = {l_early: [] for l_early in l_early_values}
    
    for l_early in l_early_values:
        print(f"\n--- Training EpiMerge-Active with L_early = {l_early} ---")
        for seed in seeds:
            print(f"  Running seed {seed}...")
            set_seed(seed)
            cal_dataset = get_calibration_dataset()
            cal_loader = DataLoader(cal_dataset, batch_size=16, shuffle=True)
            
            epimerge_model = DynamicMergedModel(
                base_model, experts, k_tasks=4, latent_dim=4, layer_class=EpiMergeLinear, 
                sensory_mode='active_early', rank=1, l_early=l_early
            ).to(device)
            wrapped_epimerge = ExpertHeadsWrapper(epimerge_model, expert_heads).to(device)
            
            optimizer = torch.optim.Adam(epimerge_model.parameters(), lr=1e-3)
            
            epimerge_model.train()
            for step in range(100):
                for images, labels, task_indices in cal_loader:
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad()
                    features = epimerge_model(images)
                    outputs = []
                    for b in range(images.size(0)):
                        task_idx = task_indices[b].item()
                        out = wrapped_epimerge.heads[task_idx](features[b:b+1])
                        outputs.append(out)
                    outputs = torch.cat(outputs, dim=0)
                    loss = criterion(outputs, labels) * images.size(0)
                    loss.backward()
                    optimizer.step()
                    
            acc = evaluate_stream(wrapped_epimerge, shuffled_test_loader, device)
            print(f"    Seed {seed} | Accuracy: {acc:.2f}%")
            ablation_results[l_early].append(acc)
            
            # Clean up model to avoid any CUDA OOM
            del epimerge_model, wrapped_epimerge
            torch.cuda.empty_cache()
        
    print("\n=== ACTIVE EARLY PARTITION ABLATION RESULTS (MULTI-SEED) ===")
    print("| L_early | Multi-Task Accuracy (%) |")
    print("|---------|-------------------------|")
    for l_early in l_early_values:
        accs = ablation_results[l_early]
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        print(f"| {l_early:<7} | {mean_acc:.2f}% ± {std_acc:.2f}% |")

if __name__ == '__main__':
    main()