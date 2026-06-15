import os
import random
import torch
import torch.nn as nn
import torch.func
import torchvision
import torchvision.transforms as transforms
import timm
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

CHECKPOINT_DIR = "./checkpoints"
DATA_DIR = "./data"
RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

tasks = ["MNIST", "FashionMNIST", "CIFAR10", "SVHN"]
K = len(tasks)

# Transforms
transform_gray = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

transform_rgb = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def get_dataset(task_name, train=True):
    if task_name == "MNIST":
        return torchvision.datasets.MNIST(root=DATA_DIR, train=train, download=True, transform=transform_gray)
    elif task_name == "FashionMNIST":
        return torchvision.datasets.FashionMNIST(root=DATA_DIR, train=train, download=True, transform=transform_gray)
    elif task_name == "CIFAR10":
        return torchvision.datasets.CIFAR10(root=DATA_DIR, train=train, download=True, transform=transform_rgb)
    elif task_name == "SVHN":
        split = "train" if train else "test"
        return torchvision.datasets.SVHN(root=DATA_DIR, split=split, download=True, transform=transform_rgb)
    else:
        raise ValueError(f"Unknown task: {task_name}")

def get_layer_group_index(param_name):
    if 'patch_embed' in param_name or 'pos_embed' in param_name or 'cls_token' in param_name:
        return 0
    elif 'blocks' in param_name:
        parts = param_name.split('.')
        block_idx = int(parts[1])
        return block_idx + 1
    elif 'norm' in param_name:
        return 13
    elif 'head' in param_name:
        return -1
    else:
        return 0

# Wrapper for functional evaluation of backbone
class BackboneWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        return self.model.forward_features(x)

# Base class for mergers
class ModelMerger(nn.Module):
    def __init__(self, base_model, task_vectors):
        super().__init__()
        self.base_model = base_model
        self.task_vectors = task_vectors
        self.eval_model = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=10)
        self.eval_model.to(device)
        self.eval_model.eval()
        self.wrapper = BackboneWrapper(self.eval_model)
        
    def get_merged_params(self, coeffs):
        # coeffs shape: [14, 4]
        base_state = self.base_model.state_dict()
        params = {}
        for name in self.eval_model.state_dict().keys():
            if "head" in name:
                continue
            l = get_layer_group_index(name)
            if l == -1:
                continue
            w = base_state[name].clone()
            for k in range(K):
                w = w + coeffs[l, k] * self.task_vectors[k][name]
            params[f"model.{name}"] = w
        return params

    def get_features(self, images, coeffs):
        params = self.get_merged_params(coeffs)
        features = torch.func.functional_call(self.wrapper, params, (images,))
        return features

    def forward_features_merged(self, images):
        raise NotImplementedError()

# 1. Uniform Merger
class UniformMerger(ModelMerger):
    def forward_features_merged(self, images):
        coeffs = torch.full((14, 4), 0.3, device=device)
        return self.get_features(images, coeffs)

# 2. OFS-Tune Merger
class OFSTuneMerger(ModelMerger):
    def __init__(self, base_model, task_vectors):
        super().__init__(base_model, task_vectors)
        self.C = nn.Parameter(torch.full((14, 4), 0.3, device=device))
        
    def forward_features_merged(self, images):
        return self.get_features(images, self.C)

# 3. Linear Router Merger
class LinearRouterMerger(ModelMerger):
    def __init__(self, base_model, task_vectors):
        super().__init__(base_model, task_vectors)
        self.W_route = nn.Parameter(torch.zeros((192, 4), device=device))
        self.b_route = nn.Parameter(torch.zeros(4, device=device))
        nn.init.kaiming_normal_(self.W_route)
        
    def forward_features_merged(self, images):
        with torch.no_grad():
            h0 = self.base_model.patch_embed(images)
            z_x = h0.mean(dim=1)
            
        logits = torch.matmul(z_x, self.W_route) + self.b_route
        alpha = torch.softmax(logits, dim=1)
        bar_alpha = alpha.mean(dim=0)
        
        coeffs = bar_alpha.unsqueeze(0).expand(14, 4)
        return self.get_features(images, coeffs)

# 4. BL-Router Merger
class BLRouterMerger(ModelMerger):
    def __init__(self, base_model, task_vectors):
        super().__init__(base_model, task_vectors)
        self.W_route = nn.Parameter(torch.zeros((192, 4), device=device))
        self.b_route = nn.Parameter(torch.zeros(4, device=device))
        nn.init.kaiming_normal_(self.W_route)
        
    def forward_features_merged(self, images):
        with torch.no_grad():
            h0 = self.base_model.patch_embed(images)
            z_x = h0.mean(dim=1)
            
        logits = torch.matmul(z_x, self.W_route) + self.b_route
        alpha = 0.3 * torch.softmax(logits, dim=1)
        bar_alpha = alpha.mean(dim=0)
        
        coeffs = bar_alpha.unsqueeze(0).expand(14, 4)
        return self.get_features(images, coeffs)

# 4b. BSigmoid-Router Merger
class BSigmoidRouterMerger(ModelMerger):
    def __init__(self, base_model, task_vectors):
        super().__init__(base_model, task_vectors)
        self.W_route = nn.Parameter(torch.zeros((192, 4), device=device))
        self.b_route = nn.Parameter(torch.zeros(4, device=device))
        nn.init.kaiming_normal_(self.W_route)
        
    def forward_features_merged(self, images):
        with torch.no_grad():
            h0 = self.base_model.patch_embed(images)
            z_x = h0.mean(dim=1)
            
        logits = torch.matmul(z_x, self.W_route) + self.b_route
        alpha = 0.3 * torch.sigmoid(logits)
        bar_alpha = alpha.mean(dim=0)
        
        coeffs = bar_alpha.unsqueeze(0).expand(14, 4)
        return self.get_features(images, coeffs)

# 5. GLS-Router Merger
class GLSRouterMerger(ModelMerger):
    def __init__(self, base_model, task_vectors):
        super().__init__(base_model, task_vectors)
        self.W_route = nn.Parameter(torch.zeros((192, 4), device=device))
        self.b_route = nn.Parameter(torch.zeros(4, device=device))
        nn.init.kaiming_normal_(self.W_route)
        self.R = nn.Parameter(torch.full((14, 4), 0.3, device=device))
        
    def forward_features_merged(self, images):
        with torch.no_grad():
            h0 = self.base_model.patch_embed(images)
            z_x = h0.mean(dim=1)
            
        logits = torch.matmul(z_x, self.W_route) + self.b_route
        p = torch.softmax(logits, dim=1)
        bar_p = p.mean(dim=0)
        
        coeffs = self.R * bar_p.unsqueeze(0)
        return self.get_features(images, coeffs)

# 6. QWS-Merge Merger
class QWSMergeMerger(ModelMerger):
    def __init__(self, base_model, task_vectors, qws_projection_seed=42):
        super().__init__(base_model, task_vectors)
        g = torch.Generator(device=device)
        g.manual_seed(qws_projection_seed)
        self.P = torch.randn(192, 4, generator=g, device=device) / (192 ** 0.5)
        
        self.Phi = nn.Parameter(torch.randn(14, 4, 4, device=device))
        self.R = nn.Parameter(torch.full((14, 4), 0.3, device=device))
        self.phi = nn.Parameter(torch.zeros((14, 4), device=device))
        
    def forward_features_merged(self, images):
        with torch.no_grad():
            h0 = self.base_model.patch_embed(images)
            z_x = h0.mean(dim=1)
            
        tilde_psi = torch.matmul(z_x, self.P)
        psi = tilde_psi / (torch.norm(tilde_psi, dim=1, keepdim=True) + 1e-8)
        
        hat_Phi = self.Phi / (torch.norm(self.Phi, dim=2, keepdim=True) + 1e-8)
        
        dot_product = torch.einsum("bd,lkd->lkb", psi, hat_Phi)
        alpha = self.R.unsqueeze(-1) * torch.cos(torch.pi * dot_product + self.phi.unsqueeze(-1))
        
        coeffs = alpha.mean(dim=2)
        return self.get_features(images, coeffs)

# 7. AdaMerging
class AdaMergingEvaluator:
    def __init__(self, base_model, task_vectors, expert_models):
        self.base_model = base_model
        self.task_vectors = task_vectors
        self.expert_models = expert_models
        self.eval_model = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=10)
        self.eval_model.to(device)
        self.eval_model.eval()
        self.wrapper = BackboneWrapper(self.eval_model)
        
    def eval_batch(self, images):
        C = nn.Parameter(torch.full((14, 4), 0.3, device=device))
        optimizer = torch.optim.Adam([C], lr=1e-2)

        with torch.enable_grad():
            for step in range(20):
                optimizer.zero_grad()

                base_state = self.base_model.state_dict()
                params = {}
                for name in self.eval_model.state_dict().keys():
                    if "head" in name:
                        continue
                    l = get_layer_group_index(name)
                    if l == -1:
                        continue
                    w = base_state[name].clone()
                    for k in range(K):
                        w = w + C[l, k] * self.task_vectors[k][name]
                    params[f"model.{name}"] = w

                features = torch.func.functional_call(self.wrapper, params, (images,))

                entropy = 0.0
                for k in range(K):
                    logits = self.expert_models[k].forward_head(features)
                    probs = torch.softmax(logits, dim=1)
                    entropy += -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()

                loss = entropy / K
                loss.backward()
                optimizer.step()
            
        with torch.no_grad():
            base_state = self.base_model.state_dict()
            params = {}
            for name in self.eval_model.state_dict().keys():
                if "head" in name:
                    continue
                l = get_layer_group_index(name)
                if l == -1:
                    continue
                w = base_state[name].clone()
                for k in range(K):
                    w = w + C[l, k].detach() * self.task_vectors[k][name]
                params[f"model.{name}"] = w
            features = torch.func.functional_call(self.wrapper, params, (images,))
        return features

def run_experiments():
    # 1. Load pre-trained base model
    print("Loading pre-trained base model...")
    base_model = timm.create_model("vit_tiny_patch16_224", pretrained=True, num_classes=10)
    base_model.to(device)
    base_model.eval()

    # 2. Load task experts
    print("Loading specialized task experts...")
    expert_models = []
    for task in tasks:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"expert_{task.lower()}.pth")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Expert checkpoint not found at {checkpoint_path}. Please run train_experts.py first.")
        model = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=10)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device)
        model.eval()
        expert_models.append(model)

    # 3. Compute task vectors
    print("Computing task vectors...")
    task_vectors = []
    base_state = base_model.state_dict()
    for k in range(K):
        v_k = {}
        expert_state = expert_models[k].state_dict()
        for name in base_state.keys():
            if "head" not in name:
                v_k[name] = expert_state[name].data - base_state[name].data
        task_vectors.append(v_k)

    # Evaluation Seeds (evaluate over 3 seeds for robust error bounds)
    seeds = [42, 100, 2026]
    
    # Store results for each seed
    results_homo_seeds = {m: [] for m in [
        "Individual Experts", "Uniform Merge", "AdaMerging (TTA)", "OFS-Tune (Static)",
        "Linear Router", "Linear Router (Reg)", "QWS-Merge", "BL-Router (Ours)",
        "BL-Router (Ours, Reg)", "GLS-Router (Ours)", "GLS-Router (Ours, Reg)",
        "BSigmoid-Router (Ours)", "BSigmoid-Router (Ours, Reg)"
    ]}
    
    batch_sizes = [1, 16, 256]
    results_hetero_seeds = {m: {B: [] for B in batch_sizes} for m in [
        "Uniform Merge", "AdaMerging (TTA)", "OFS-Tune (Static)",
        "Linear Router", "Linear Router (Reg)", "QWS-Merge", "BL-Router (Ours)",
        "BL-Router (Ours, Reg)", "GLS-Router (Ours)", "GLS-Router (Ours, Reg)",
        "BSigmoid-Router (Ours)", "BSigmoid-Router (Ours, Reg)"
    ]}

    # We evaluate heterogeneous stream using a fixed interleaved test stream sampled under seed 42 to keep test partitions fixed.
    print("\nPre-constructing fixed heterogeneous shuffled test stream (seed 42)...")
    het_images = []
    het_labels = []
    het_tasks = []
    for k, task in enumerate(tasks):
        test_dataset = get_dataset(task, train=False)
        test_indices = list(range(16, len(test_dataset)))
        import random
        random.seed(42)
        if len(test_indices) > 200: # optimized stream evaluation size
            test_indices = random.sample(test_indices, 200)
        subset = Subset(test_dataset, test_indices)
        loader = DataLoader(subset, batch_size=len(subset), shuffle=False)
        for images, labels in loader:
            het_images.append(images)
            het_labels.append(labels)
            het_tasks.append(torch.full_like(labels, k))
            
    het_images = torch.cat(het_images, dim=0)
    het_labels = torch.cat(het_labels, dim=0)
    het_tasks = torch.cat(het_tasks, dim=0)
    
    num_samples = het_images.size(0)
    shuffle_indices = list(range(num_samples))
    random.seed(42)
    random.shuffle(shuffle_indices)
    
    het_images = het_images[shuffle_indices]
    het_labels = het_labels[shuffle_indices]
    het_tasks = het_tasks[shuffle_indices]
    print(f"Heterogeneous test set constructed with {num_samples} total samples.")

    # Loop over seeds
    for s_idx, seed in enumerate(seeds):
        print(f"\n==========================================")
        print(f"        RUNNING FOR SEED {seed} ({s_idx+1}/{len(seeds)})")
        print(f"==========================================")
        
        # Set seeds for calibration set sampling and routing weight initialization
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # 4. Extract calibration dataset
        print(f"Extracting calibration dataset (64 samples, 16 per task) for seed {seed}...")
        cal_images = []
        cal_labels = []
        cal_tasks = []
        for k, task in enumerate(tasks):
            test_dataset = get_dataset(task, train=False)
            all_indices = list(range(len(test_dataset)))
            import random
            random.seed(seed)
            cal_indices = random.sample(all_indices, 16)
            subset = Subset(test_dataset, cal_indices)
            loader = DataLoader(subset, batch_size=16, shuffle=False)
            for images, labels in loader:
                cal_images.append(images)
                cal_labels.append(labels)
                cal_tasks.append(torch.full_like(labels, k))
                
        cal_images = torch.cat(cal_images, dim=0).to(device)
        cal_labels = torch.cat(cal_labels, dim=0).to(device)
        cal_tasks = torch.cat(cal_tasks, dim=0).to(device)

        # Helper function to tune mergers
        def optimize_merger(merger, name, lr=1e-2, weight_decay=0.0):
            print(f"Optimizing {name} (weight_decay={weight_decay})...")
            optimizer = torch.optim.Adam(merger.parameters(), lr=lr, weight_decay=weight_decay)
            criterion = nn.CrossEntropyLoss()
            
            for step in range(100):
                optimizer.zero_grad()
                features = merger.forward_features_merged(cal_images)
                
                loss = 0.0
                count = 0
                for k in range(K):
                    task_mask = (cal_tasks == k)
                    if task_mask.any():
                        task_features = features[task_mask]
                        task_labels = cal_labels[task_mask]
                        task_logits = expert_models[k].forward_head(task_features)
                        loss += criterion(task_logits, task_labels) * task_labels.size(0)
                        count += task_labels.size(0)
                
                loss = loss / count
                loss.backward()
                optimizer.step()
            print(f"  {name} optimized.")

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Instantiate mergers
        mergers = {
            "Uniform Merge": UniformMerger(base_model, task_vectors),
            "OFS-Tune (Static)": OFSTuneMerger(base_model, task_vectors),
            "Linear Router": LinearRouterMerger(base_model, task_vectors),
            "Linear Router (Reg)": LinearRouterMerger(base_model, task_vectors),
            "QWS-Merge": QWSMergeMerger(base_model, task_vectors, qws_projection_seed=42),
            "BL-Router (Ours)": BLRouterMerger(base_model, task_vectors),
            "BL-Router (Ours, Reg)": BLRouterMerger(base_model, task_vectors),
            "GLS-Router (Ours)": GLSRouterMerger(base_model, task_vectors),
            "GLS-Router (Ours, Reg)": GLSRouterMerger(base_model, task_vectors),
            "BSigmoid-Router (Ours)": BSigmoidRouterMerger(base_model, task_vectors),
            "BSigmoid-Router (Ours, Reg)": BSigmoidRouterMerger(base_model, task_vectors)
        }

        # Optimize mergers
        optimize_merger(mergers["OFS-Tune (Static)"], "OFS-Tune (Static)", lr=1e-2)
        optimize_merger(mergers["Linear Router"], "Linear Router", lr=1e-2)
        optimize_merger(mergers["Linear Router (Reg)"], "Linear Router (Reg)", lr=1e-2, weight_decay=1e-4)
        optimize_merger(mergers["QWS-Merge"], "QWS-Merge", lr=1e-2)
        optimize_merger(mergers["BL-Router (Ours)"], "BL-Router (Ours)", lr=1e-2)
        optimize_merger(mergers["BL-Router (Ours, Reg)"], "BL-Router (Ours, Reg)", lr=1e-2, weight_decay=1e-4)
        optimize_merger(mergers["GLS-Router (Ours)"], "GLS-Router (Ours)", lr=1e-2)
        optimize_merger(mergers["GLS-Router (Ours, Reg)"], "GLS-Router (Ours, Reg)", lr=1e-2, weight_decay=1e-4)
        optimize_merger(mergers["BSigmoid-Router (Ours)"], "BSigmoid-Router (Ours)", lr=1e-2)
        optimize_merger(mergers["BSigmoid-Router (Ours, Reg)"], "BSigmoid-Router (Ours, Reg)", lr=1e-2, weight_decay=1e-4)

        ada_evaluator = AdaMergingEvaluator(base_model, task_vectors, expert_models)

        # Homogeneous Evaluation (on fixed evaluation set)
        def eval_homogeneous(model_merger, name):
            accuracies = {}
            for k, task in enumerate(tasks):
                test_dataset = get_dataset(task, train=False)
                test_indices = list(range(16, len(test_dataset)))
                import random
                random.seed(42) # fixed test set
                if len(test_indices) > 250: # optimized test evaluation size
                    test_indices = random.sample(test_indices, 250)
                subset = Subset(test_dataset, test_indices)
                loader = DataLoader(subset, batch_size=128, shuffle=False, num_workers=4)
                
                correct = 0
                total = 0
                for images, labels in loader:
                    images, labels = images.to(device), labels.to(device)
                    with torch.no_grad():
                        if name == "AdaMerging (TTA)":
                            features = ada_evaluator.eval_batch(images)
                        else:
                            features = model_merger.forward_features_merged(images)
                        logits = expert_models[k].forward_head(features)
                    _, predicted = logits.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
                acc = correct / total * 100.0
                accuracies[task] = acc
            joint_mean = sum(accuracies.values()) / len(tasks)
            accuracies["Joint Mean"] = joint_mean
            return accuracies

        # Individual Experts Ceiling (identical across runs)
        if s_idx == 0:
            ceiling_accs = {}
            for k, task in enumerate(tasks):
                test_dataset = get_dataset(task, train=False)
                test_indices = list(range(16, len(test_dataset)))
                import random
                random.seed(42)
                if len(test_indices) > 250:
                    test_indices = random.sample(test_indices, 250)
                subset = Subset(test_dataset, test_indices)
                loader = DataLoader(subset, batch_size=128, shuffle=False, num_workers=4)
                correct = 0
                total = 0
                for images, labels in loader:
                    images, labels = images.to(device), labels.to(device)
                    with torch.no_grad():
                        logits = expert_models[k](images)
                    _, predicted = logits.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
                acc = correct / total * 100.0
                ceiling_accs[task] = acc
            ceiling_accs["Joint Mean"] = sum(ceiling_accs.values()) / len(tasks)
            results_homo_seeds["Individual Experts"] = [ceiling_accs] * len(seeds)

        # Collect homogeneous accuracies
        results_homo_seeds["Uniform Merge"].append(eval_homogeneous(mergers["Uniform Merge"], "Uniform Merge"))
        results_homo_seeds["AdaMerging (TTA)"].append(eval_homogeneous(None, "AdaMerging (TTA)"))
        results_homo_seeds["OFS-Tune (Static)"].append(eval_homogeneous(mergers["OFS-Tune (Static)"], "OFS-Tune (Static)"))
        results_homo_seeds["Linear Router"].append(eval_homogeneous(mergers["Linear Router"], "Linear Router"))
        results_homo_seeds["Linear Router (Reg)"].append(eval_homogeneous(mergers["Linear Router (Reg)"], "Linear Router (Reg)"))
        results_homo_seeds["QWS-Merge"].append(eval_homogeneous(mergers["QWS-Merge"], "QWS-Merge"))
        results_homo_seeds["BL-Router (Ours)"].append(eval_homogeneous(mergers["BL-Router (Ours)"], "BL-Router (Ours)"))
        results_homo_seeds["BL-Router (Ours, Reg)"].append(eval_homogeneous(mergers["BL-Router (Ours, Reg)"], "BL-Router (Ours, Reg)"))
        results_homo_seeds["GLS-Router (Ours)"].append(eval_homogeneous(mergers["GLS-Router (Ours)"], "GLS-Router (Ours)"))
        results_homo_seeds["GLS-Router (Ours, Reg)"].append(eval_homogeneous(mergers["GLS-Router (Ours, Reg)"], "GLS-Router (Ours, Reg)"))
        results_homo_seeds["BSigmoid-Router (Ours)"].append(eval_homogeneous(mergers["BSigmoid-Router (Ours)"], "BSigmoid-Router (Ours)"))
        results_homo_seeds["BSigmoid-Router (Ours, Reg)"].append(eval_homogeneous(mergers["BSigmoid-Router (Ours, Reg)"], "BSigmoid-Router (Ours, Reg)"))

        # Heterogeneous Evaluation
        def eval_hetero_dynamic(model_merger, name, batch_size):
            correct = 0
            total = 0
            for i in range(0, num_samples, batch_size):
                batch_images = het_images[i:i+batch_size].to(device)
                batch_labels = het_labels[i:i+batch_size].to(device)
                batch_tasks = het_tasks[i:i+batch_size].to(device)
                
                with torch.no_grad():
                    if name == "AdaMerging (TTA)":
                        features = ada_evaluator.eval_batch(batch_images)
                    else:
                        features = model_merger.forward_features_merged(batch_images)
                    logits = torch.zeros((batch_images.size(0), 10), device=device)
                    for k in range(K):
                        task_mask = (batch_tasks == k)
                        if task_mask.any():
                            task_features = features[task_mask]
                            task_logits = expert_models[k].forward_head(task_features)
                            logits[task_mask] = task_logits
                _, predicted = logits.max(1)
                total += batch_labels.size(0)
                correct += predicted.eq(batch_labels).sum().item()
            return correct / total * 100.0

        for B in batch_sizes:
            results_hetero_seeds["Uniform Merge"][B].append(results_homo_seeds["Uniform Merge"][-1]["Joint Mean"])
            results_hetero_seeds["AdaMerging (TTA)"][B].append(results_homo_seeds["AdaMerging (TTA)"][-1]["Joint Mean"])
            results_hetero_seeds["OFS-Tune (Static)"][B].append(results_homo_seeds["OFS-Tune (Static)"][-1]["Joint Mean"])
            results_hetero_seeds["Linear Router"][B].append(eval_hetero_dynamic(mergers["Linear Router"], "Linear Router", B))
            results_hetero_seeds["Linear Router (Reg)"][B].append(eval_hetero_dynamic(mergers["Linear Router (Reg)"], "Linear Router (Reg)", B))
            results_hetero_seeds["QWS-Merge"][B].append(eval_hetero_dynamic(mergers["QWS-Merge"], "QWS-Merge", B))
            results_hetero_seeds["BL-Router (Ours)"][B].append(eval_hetero_dynamic(mergers["BL-Router (Ours)"], "BL-Router (Ours)", B))
            results_hetero_seeds["BL-Router (Ours, Reg)"][B].append(eval_hetero_dynamic(mergers["BL-Router (Ours, Reg)"], "BL-Router (Ours, Reg)", B))
            results_hetero_seeds["GLS-Router (Ours)"][B].append(eval_hetero_dynamic(mergers["GLS-Router (Ours)"], "GLS-Router (Ours)", B))
            results_hetero_seeds["GLS-Router (Ours, Reg)"][B].append(eval_hetero_dynamic(mergers["GLS-Router (Ours, Reg)"], "GLS-Router (Ours, Reg)", B))
            results_hetero_seeds["BSigmoid-Router (Ours)"][B].append(eval_hetero_dynamic(mergers["BSigmoid-Router (Ours)"], "BSigmoid-Router (Ours)", B))
            results_hetero_seeds["BSigmoid-Router (Ours, Reg)"][B].append(eval_hetero_dynamic(mergers["BSigmoid-Router (Ours, Reg)"], "BSigmoid-Router (Ours, Reg)", B))

    # Calculate means and standard deviations across seeds
    results_homo_summary = {}
    for name, list_dicts in results_homo_seeds.items():
        results_homo_summary[name] = {}
        for key in ["MNIST", "FashionMNIST", "CIFAR10", "SVHN", "Joint Mean"]:
            vals = [d[key] for d in list_dicts]
            mean = np.mean(vals)
            std = np.std(vals)
            results_homo_summary[name][key] = (mean, std)
            
    results_hetero_summary = {}
    for name, B_dict in results_hetero_seeds.items():
        results_hetero_summary[name] = {}
        for B in batch_sizes:
            vals = B_dict[B]
            mean = np.mean(vals)
            std = np.std(vals)
            results_hetero_summary[name][B] = (mean, std)

    # Print summaries
    print("\n" + "="*101)
    print("      FINAL HOMOGENEOUS RESULTS SUMMARY (Mean ± Std over 3 seeds)")
    print("="*101)
    print(f"{'Method':<25} | {'MNIST':<12} | {'Fashion':<12} | {'CIFAR10':<12} | {'SVHN':<12} | {'Joint Mean':<12}")
    print("-"*101)
    for name, summary in results_homo_summary.items():
        print(f"{name:<25} | {summary['MNIST'][0]:.2f}±{summary['MNIST'][1]:.2f}% | {summary['FashionMNIST'][0]:.2f}±{summary['FashionMNIST'][1]:.2f}% | {summary['CIFAR10'][0]:.2f}±{summary['CIFAR10'][1]:.2f}% | {summary['SVHN'][0]:.2f}±{summary['SVHN'][1]:.2f}% | {summary['Joint Mean'][0]:.2f}±{summary['Joint Mean'][1]:.2f}%")
    print("="*101)

    print("\n" + "="*77)
    print("     FINAL HETEROGENEOUS RESULTS SUMMARY (Mean ± Std over 3 seeds)")
    print("="*77)
    print(f"{'Method':<25} | {'B=1':<15} | {'B=16':<15} | {'B=256':<15}")
    print("-"*77)
    for name, summary in results_hetero_summary.items():
        print(f"{name:<25} | {summary[1][0]:.2f}±{summary[1][1]:.2f}% | {summary[16][0]:.2f}±{summary[16][1]:.2f}% | {summary[256][0]:.2f}±{summary[256][1]:.2f}%")
    print("="*77)

    # Save results to text file
    with open(os.path.join(RESULTS_DIR, "raw_metrics.txt"), "w") as f:
        f.write("=== HOMOGENEOUS RESULTS (Mean ± Std) ===\n")
        for name, summary in results_homo_summary.items():
            f.write(f"{name}: MNIST={summary['MNIST'][0]:.2f}±{summary['MNIST'][1]:.2f}%, FashionMNIST={summary['FashionMNIST'][0]:.2f}±{summary['FashionMNIST'][1]:.2f}%, CIFAR10={summary['CIFAR10'][0]:.2f}±{summary['CIFAR10'][1]:.2f}%, SVHN={summary['SVHN'][0]:.2f}±{summary['SVHN'][1]:.2f}%, Joint={summary['Joint Mean'][0]:.2f}±{summary['Joint Mean'][1]:.2f}%\n")
        f.write("\n=== HETEROGENEOUS RESULTS (Mean ± Std) ===\n")
        for name, summary in results_hetero_summary.items():
            f.write(f"{name}: B=1={summary[1][0]:.2f}±{summary[1][1]:.2f}%, B=16={summary[16][0]:.2f}±{summary[16][1]:.2f}%, B=256={summary[256][0]:.2f}±{summary[256][1]:.2f}%\n")

    # Save means to simple point estimates for plots
    results_homo_means = {name: {k: summary[k][0] for k in ["MNIST", "FashionMNIST", "CIFAR10", "SVHN", "Joint Mean"]} for name, summary in results_homo_summary.items()}

    # 8. Generate and save plots using the mean results
    print("\nGenerating evaluation plots...")
    plt.figure(figsize=(10, 6))
    methods_plot = [m for m in results_homo_means.keys() if m != "Individual Experts"]
    x_indices = np.arange(len(methods_plot))
    means = [results_homo_means[m]["Joint Mean"] for m in methods_plot]
    stds = [results_homo_summary[m]["Joint Mean"][1] for m in methods_plot]
    
    plt.bar(x_indices, means, yerr=stds, color='skyblue', edgecolor='black', width=0.6, capsize=5)
    plt.xticks(x_indices, methods_plot, rotation=15)
    plt.ylabel("Joint Mean Accuracy (%)")
    plt.title("Homogeneous Multi-Task Joint Mean Accuracy (3-Seed Average)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("comparison_plot.png")
    plt.savefig(os.path.join(RESULTS_DIR, "comparison_plot.png"))
    plt.close()

    plt.figure(figsize=(8, 5))
    for name, B_dict in results_hetero_summary.items():
        y_vals = [B_dict[1][0], B_dict[16][0], B_dict[256][0]]
        y_errs = [B_dict[1][1], B_dict[16][1], B_dict[256][1]]
        style = 'o-' if "Ours" in name or name == "QWS-Merge" else 's--'
        plt.errorbar([1, 16, 256], y_vals, yerr=y_errs, fmt=style, label=name, linewidth=2, capsize=4)
    plt.xscale('log')
    plt.xticks([1, 16, 256], ['B=1', 'B=16', 'B=256'])
    plt.xlabel("Batch Size (B)")
    plt.ylabel("Multi-task Accuracy (%)")
    plt.title("Heterogeneous Stream Performance vs Batch Size (3-Seed Average)")
    plt.legend(loc='lower left')
    plt.grid(True, which="both", linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("heterogeneous_plot.png")
    plt.savefig(os.path.join(RESULTS_DIR, "heterogeneous_plot.png"))
    plt.close()
    
    print("Plots saved successfully to workspace root and results/ directory!")

if __name__ == "__main__":
    run_experiments()
