import os
import sys
import copy
import time
import random
import json
import numpy as np
import torch

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
    return 0

class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model
        if hasattr(self.model, 'transformer'):
            delattr(self.model, 'transformer')

    def forward(self, images):
        return self.model.encode_image(images)

def get_subsets(dataset_name, preprocess, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    ds = get_dataset(dataset_name, preprocess, location='data')
    test_full = ds.test_dataset
    
    test_indices = list(range(len(test_full)))
    random.seed(seed)
    random.shuffle(test_indices)
    eval_indices = test_indices[:512]
    eval_subset = torch.utils.data.Subset(test_full, eval_indices)
    
    return eval_subset

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
    device = "cpu"
    print("Running FAST local evaluation for TA and NETA on CPU...")
    
    seeds = [42, 100, 2026]
    datasets_list = ['MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN']
    
    clip_model, _, val_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    pretrained_sd = torch.load('checkpoints/ViT-B-32/zeroshot.pt', map_location='cpu')
    
    finetuned_sds = {}
    for dataset in datasets_list:
        finetuned_sds[dataset] = torch.load(f'checkpoints/ViT-B-32/{dataset}/finetuned.pt', map_location='cpu')
        
    task_vectors_sd = []
    for dataset in datasets_list:
        tv = {}
        for k in pretrained_sd.keys():
            tv[k] = finetuned_sds[dataset][k] - pretrained_sd[k]
        task_vectors_sd.append(tv)
        
    heads = {}
    for dataset in datasets_list:
        heads[dataset] = torch.load(f'checkpoints/ViT-B-32/{dataset}/head_{dataset}.pt', map_location='cpu', weights_only=False)
        
    results_ta = {d: [] for d in datasets_list}
    results_neta = {d: [] for d in datasets_list}
    
    for seed in seeds:
        print(f"\nSeed {seed}:")
        eval_subsets = {}
        for dataset in datasets_list:
            eval_sub = get_subsets(dataset, val_preprocess, seed)
            eval_subsets[dataset] = eval_sub
            
        # 1. TA
        ta_sd = merge_ta(pretrained_sd, task_vectors_sd, lambda_0=0.3)
        ta_sd_clean = {k.replace('model.', ''): v for k, v in ta_sd.items()}
        m_ta = ModelWrapper(copy.deepcopy(clip_model)).to(device)
        m_ta.model.load_state_dict(ta_sd_clean, strict=False)
        ta_res = evaluate_model(m_ta, eval_subsets, heads, device)
        for d in datasets_list:
            results_ta[d].append(ta_res[d])
            print(f"  TA - {d}: {ta_res[d]*100:.2f}%")
            
        # 2. NETA
        neta_sd = merge_neta(pretrained_sd, task_vectors_sd, lambda_0=0.3)
        neta_sd_clean = {k.replace('model.', ''): v for k, v in neta_sd.items()}
        m_neta = ModelWrapper(copy.deepcopy(clip_model)).to(device)
        m_neta.model.load_state_dict(neta_sd_clean, strict=False)
        neta_res = evaluate_model(m_neta, eval_subsets, heads, device)
        for d in datasets_list:
            results_neta[d].append(neta_res[d])
            print(f"  NETA - {d}: {neta_res[d]*100:.2f}%")
            
    print("\n--- FINAL SUMMARY (FAST LOCAL) ---")
    for d in datasets_list:
        print(f"{d} TA: {np.mean(results_ta[d])*100:.2f}% ± {np.std(results_ta[d])*100:.4f}%")
        print(f"{d} NETA: {np.mean(results_neta[d])*100:.2f}% ± {np.std(results_neta[d])*100:.4f}%")
