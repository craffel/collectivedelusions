import os
import sys
import copy
import time
import random
import json
import tqdm
import numpy as np
import torch

torch.set_num_threads(8)

# Setup sys.path to load local_datasets
sys.path.insert(0, os.path.abspath('AdaMerging/src'))

# Dummy module setup to allow unpickling the ClassificationHead objects
import types
from modeling import ClassificationHead, ImageClassifier
sys.modules['src'] = types.ModuleType('src')
sys.modules['src.models'] = types.ModuleType('src.models')
sys.modules['src.models.modeling'] = types.ModuleType('src.models.modeling')
sys.modules['src.models.modeling'].ClassificationHead = ClassificationHead

import open_clip
from local_datasets.registry import get_dataset

def get_group_idx(key):
    if 'model.visual' not in key:
        return None
    if 'visual.proj' in key:
        return 12
    for i in range(12):
        if f'resblocks.{i}.' in key:
            return i
    # Other visual keys (conv1, ln_pre, class_embedding, positional_embedding, etc.) -> Group 0
    return 0

def del_attr(obj, names):
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(getattr(obj, names[0]), names[1:])

def set_attr(obj, names, val):
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        set_attr(getattr(obj, names[0]), names[1:], val)

def make_functional(mod):
    orig_params = tuple(mod.parameters())
    names = []
    for name, p in list(mod.named_parameters()):
        del_attr(mod, name.split("."))
        names.append(name)
    return orig_params, names

def load_weights(mod, names, params):
    for name, p in zip(names, params):
        set_attr(mod, name.split("."), p)

class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model
        if hasattr(self.model, 'transformer'):
            delattr(self.model, 'transformer')

    def forward(self, images):
        return self.model.encode_image(images)

class AdaMergingOptimizer(torch.nn.Module):
    def __init__(self, pretrained_sd, task_vectors, model, names, mode='task-wise'):
        super(AdaMergingOptimizer, self).__init__()
        self.model = model
        self.names = names
        self.mode = mode
        
        # Convert state dicts to tuples matching self.names EXACTLY on correct device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pretrained_params = tuple(pretrained_sd[k].detach().clone().to(device) for k in self.names)
        self.task_vectors_params = []
        for tv in task_vectors:
            self.task_vectors_params.append(tuple(tv[k].detach().clone().to(device) for k in self.names))
            
        K = len(task_vectors)
        if mode == 'task-wise':
            # 1 scaling coefficient per task (K parameters)
            self.lambdas_raw = torch.nn.Parameter(torch.ones(1, K) * 0.3)
        elif mode == 'layer-wise':
            # 13 layers, so 13 scaling coefficients per task (13 * K parameters)
            self.lambdas_raw = torch.nn.Parameter(torch.ones(13, K) * 0.3)
            
        # Map each parameter in self.names to its group index (0-12)
        self.param_group_indices = []
        for k in self.names:
            idx = get_group_idx(k)
            if idx is None:
                idx = 0  # non-visual parameters mapped to group 0
            self.param_group_indices.append(idx)

    def lambdas(self):
        return torch.clamp(self.lambdas_raw, min=0.0, max=1.0)

    def get_image_encoder(self):
        alphas = self.lambdas() # shape (1, K) or (13, K)
        K = len(self.task_vectors_params)
        
        merged_params = []
        for j in range(len(self.pretrained_params)):
            g_idx = self.param_group_indices[j]
            # Sum scaled task vectors
            tv_sum = 0.0
            for k in range(K):
                if self.mode == 'task-wise':
                    coef = alphas[0, k]
                else:
                    coef = alphas[g_idx, k]
                tv_sum += coef * self.task_vectors_params[k][j]
            merged_params.append(self.pretrained_params[j] + tv_sum)
            
        merged_params = tuple(merged_params)
        load_weights(self.model, self.names, merged_params)
        return self.model

def get_subsets(dataset_name, preprocess, seed):
    # Set PyTorch seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Load dataset
    ds = get_dataset(dataset_name, preprocess, location='data')
    train_full = ds.train_dataset
    test_full = ds.test_dataset
    
    # Deterministically select 512 indices from test_full
    test_indices = list(range(len(test_full)))
    random.seed(seed)
    random.shuffle(test_indices)
    eval_indices = test_indices[:512]
    eval_subset = torch.utils.data.Subset(test_full, eval_indices)
    
    # Deterministically select 64 indices from train_full for calibration
    train_indices = list(range(len(train_full)))
    random.seed(seed + 1000)
    random.shuffle(train_indices)
    calib_indices = train_indices[:64]
    calib_subset = torch.utils.data.Subset(train_full, calib_indices)
    
    return calib_subset, eval_subset

def softmax_entropy(x):
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def evaluate_model(model, eval_subsets, heads, device):
    model.eval()
    results = {}
    for dataset_name, subset in eval_subsets.items():
        dataloader = torch.utils.data.DataLoader(subset, batch_size=128, shuffle=False)
        head = heads[dataset_name].to(device)
        head.eval()
        
        correct = 0
        total = 0
        with torch.no_grad():
            for data in dataloader:
                if isinstance(data, dict):
                    images = data['images'].to(device)
                    labels = data['labels'].to(device)
                else:
                    images, labels = data
                    images = images.to(device)
                    labels = labels.to(device)
                
                features = model(images)
                logits = head(features)
                preds = logits.argmax(dim=1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)
                
        acc = correct / total
        results[dataset_name] = acc
    return results

def merge_ta(pretrained_sd, task_vectors_sd, lambda_0=0.3):
    merged_sd = {}
    for k in pretrained_sd.keys():
        if k in task_vectors_sd[0]:
            merged_sd[k] = pretrained_sd[k] + lambda_0 * sum(tv[k] for tv in task_vectors_sd)
        else:
            merged_sd[k] = pretrained_sd[k]
    return merged_sd

def merge_neta(pretrained_sd, task_vectors_sd, lambda_0=0.3):
    visual_keys = [k for k in pretrained_sd.keys() if 'visual' in k]
    non_visual_keys = [k for k in pretrained_sd.keys() if k not in visual_keys]
    
    groups = {i: [] for i in range(13)}
    for k in visual_keys:
        if 'visual.proj' in k:
            groups[12].append(k)
        else:
            found = False
            for i in range(12):
                if f'resblocks.{i}.' in k:
                    groups[i].append(k)
                    found = True
                    break
            if not found:
                groups[0].append(k)
                
    scaled_vectors = [{} for _ in task_vectors_sd]
    for i, tv in enumerate(task_vectors_sd):
        for k in non_visual_keys:
            scaled_vectors[i][k] = tv[k]
            
    for g in range(13):
        g_keys = groups[g]
        if not g_keys:
            continue
        norms = []
        for tv in task_vectors_sd:
            val = sum(torch.sum(tv[k]**2).item() for k in g_keys)
            norms.append(val ** 0.5)
        mean_norm = sum(norms) / len(norms)
        for i, tv in enumerate(task_vectors_sd):
            norm_k = norms[i]
            w_k = mean_norm / (norm_k + 1e-6)
            for k in g_keys:
                scaled_vectors[i][k] = w_k * tv[k]
                
    merged_sd = {}
    for k in pretrained_sd.keys():
        if k in scaled_vectors[0]:
            merged_sd[k] = pretrained_sd[k] + lambda_0 * sum(sv[k] for sv in scaled_vectors)
        else:
            merged_sd[k] = pretrained_sd[k]
    return merged_sd

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")
    
    seeds = [42, 100, 2026]
    datasets_list = ['MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN']
    
    # 1. Load pre-trained model and preprocessing
    clip_model, _, val_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    
    # 2. Load our raw checkpoints (which are state_dicts on CPU)
    pretrained_sd = torch.load('checkpoints/ViT-B-32/zeroshot.pt', map_location='cpu')
    
    finetuned_sds = {}
    for dataset in datasets_list:
        finetuned_sds[dataset] = torch.load(f'checkpoints/ViT-B-32/{dataset}/finetuned.pt', map_location='cpu')
        
    # Extract task vectors
    task_vectors_sd = []
    for dataset in datasets_list:
        tv = {}
        for k in pretrained_sd.keys():
            tv[k] = finetuned_sds[dataset][k] - pretrained_sd[k]
        task_vectors_sd.append(tv)
        
    # Load classification heads
    heads = {}
    for dataset in datasets_list:
        heads[dataset] = torch.load(f'checkpoints/ViT-B-32/{dataset}/head_{dataset}.pt', map_location='cpu', weights_only=False)
        
    # We will record metrics for each method
    all_results = {
        'task_arithmetic': {d: [] for d in datasets_list},
        'neta': {d: [] for d in datasets_list},
        'task_wise_adamerging': {d: [] for d in datasets_list},
        'layer_wise_adamerging': {d: [] for d in datasets_list},
    }
    
    for seed in seeds:
        print(f"\n=================== Running Seed {seed} ===================")
        
        # Extract evaluation and calibration subsets
        eval_subsets = {}
        calib_subsets = {}
        for dataset in datasets_list:
            calib_sub, eval_sub = get_subsets(dataset, val_preprocess, seed)
            eval_subsets[dataset] = eval_sub
            calib_subsets[dataset] = calib_sub
            print(f"Dataset {dataset}: Calibration size = {len(calib_sub)}, Evaluation size = {len(eval_sub)}")
            
        # ---------------- 1. Task Arithmetic Baseline ----------------
        print("\nEvaluating Task Arithmetic Baseline...")
        ta_sd = merge_ta(pretrained_sd, task_vectors_sd, lambda_0=0.3)
        # Load onto visual encoder of clip model
        # Strip model. prefix to match open_clip
        ta_sd_clean = {k.replace('model.', ''): v for k, v in ta_sd.items()}
        
        # Create a fresh copy of model to prevent weight contamination
        seed_model = ModelWrapper(copy.deepcopy(clip_model)).to(device)
        seed_model.model.load_state_dict(ta_sd_clean, strict=False)
        
        ta_res = evaluate_model(seed_model, eval_subsets, heads, device)
        for dataset in datasets_list:
            all_results['task_arithmetic'][dataset].append(ta_res[dataset])
            print(f"Task Arithmetic - {dataset}: {ta_res[dataset] * 100:.2f}%")
            
        # ---------------- 2. NETA Proposed Method ----------------
        print("\nEvaluating NETA (Proposed Zero-Shot Balanced)...")
        neta_sd = merge_neta(pretrained_sd, task_vectors_sd, lambda_0=0.3)
        neta_sd_clean = {k.replace('model.', ''): v for k, v in neta_sd.items()}
        
        seed_model = ModelWrapper(copy.deepcopy(clip_model)).to(device)
        seed_model.model.load_state_dict(neta_sd_clean, strict=False)
        
        neta_res = evaluate_model(seed_model, eval_subsets, heads, device)
        for dataset in datasets_list:
            all_results['neta'][dataset].append(neta_res[dataset])
            print(f"NETA - {dataset}: {neta_res[dataset] * 100:.2f}%")
            
        # ---------------- 3. Task-Wise AdaMerging (Continuous TTA) ----------------
        print("\nOptimizing Task-Wise AdaMerging...")
        seed_clip = copy.deepcopy(clip_model)
        seed_model = ModelWrapper(seed_clip).to(device)
        _, names = make_functional(seed_model)
        
        # We only pass visual parameters of pretrained_sd
        opt_model = AdaMergingOptimizer(pretrained_sd, task_vectors_sd, seed_model, names, mode='task-wise').to(device)
        optimizer = torch.optim.Adam(opt_model.parameters(), lr=5e-3, weight_decay=0.)
        
        # Optimize over 20 epochs on calibration set
        epochs = 20
        for epoch in range(epochs):
            losses = 0.
            for dataset in datasets_list:
                dataloader = torch.utils.data.DataLoader(calib_subsets[dataset], batch_size=32, shuffle=True)
                # Take exactly 1 batch (32 images) per dataset to make 128 total calibration images per epoch
                data = next(iter(dataloader))
                if isinstance(data, dict):
                    x = data['images'].to(device)
                else:
                    x = data[0].to(device)
                
                # Dynamic model weights update and forward pass
                img_enc = opt_model.get_image_encoder()
                head = heads[dataset].to(device)
                
                features = img_enc(x)
                logits = head(features)
                loss = softmax_entropy(logits).mean(0)
                losses += loss
                
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
        print(f"Optimized task-wise lambdas: {opt_model.lambdas().data.cpu().numpy().tolist()}")
        
        # Evaluate optimized task-wise model
        final_img_enc = opt_model.get_image_encoder()
        tw_res = evaluate_model(final_img_enc, eval_subsets, heads, device)
        for dataset in datasets_list:
            all_results['task_wise_adamerging'][dataset].append(tw_res[dataset])
            print(f"Task-Wise AdaMerging - {dataset}: {tw_res[dataset] * 100:.2f}%")
            
        # ---------------- 4. Layer-Wise AdaMerging (Continuous TTA) ----------------
        print("\nOptimizing Layer-Wise AdaMerging (52 parameters)...")
        seed_clip = copy.deepcopy(clip_model)
        seed_model = ModelWrapper(seed_clip).to(device)
        _, names = make_functional(seed_model)
        
        opt_model = AdaMergingOptimizer(pretrained_sd, task_vectors_sd, seed_model, names, mode='layer-wise').to(device)
        optimizer = torch.optim.Adam(opt_model.parameters(), lr=5e-3, weight_decay=0.)
        
        for epoch in range(epochs):
            losses = 0.
            for dataset in datasets_list:
                dataloader = torch.utils.data.DataLoader(calib_subsets[dataset], batch_size=32, shuffle=True)
                data = next(iter(dataloader))
                if isinstance(data, dict):
                    x = data['images'].to(device)
                else:
                    x = data[0].to(device)
                
                img_enc = opt_model.get_image_encoder()
                head = heads[dataset].to(device)
                
                features = img_enc(x)
                logits = head(features)
                loss = softmax_entropy(logits).mean(0)
                losses += loss
                
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
        print("Layer-wise AdaMerging optimization complete.")
        
        # Evaluate optimized layer-wise model
        final_img_enc = opt_model.get_image_encoder()
        lw_res = evaluate_model(final_img_enc, eval_subsets, heads, device)
        for dataset in datasets_list:
            all_results['layer_wise_adamerging'][dataset].append(lw_res[dataset])
            print(f"Layer-Wise AdaMerging - {dataset}: {lw_res[dataset] * 100:.2f}%")
            
    # Compute and print final summary
    print("\n\n=================== FINAL SUMMARY ===================")
    final_metrics = {}
    for method, results in all_results.items():
        print(f"\n--- Method: {method} ---")
        final_metrics[method] = {}
        for dataset in datasets_list:
            accs = np.array(results[dataset])
            mean_acc = np.mean(accs)
            std_acc = np.std(accs)
            final_metrics[method][dataset] = {
                'mean': float(mean_acc),
                'std': float(std_acc)
            }
            print(f"{dataset}: Mean = {mean_acc * 100:.2f}%, Std = {std_acc * 100:.4f}%")
            
    # Save results to a json file
    with open('experiment_metrics.json', 'w') as f:
        json.dump(final_metrics, f, indent=2)
    print("Saved final metrics to experiment_metrics.json!")
