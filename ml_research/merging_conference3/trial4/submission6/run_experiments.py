import os
import sys
import time
import torch
import numpy as np
import copy
from collections import OrderedDict
from types import ModuleType

# Add AdaMerging/src to python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'AdaMerging', 'src'))

from eval import eval_single_dataset
from args import parse_arguments
import modeling
from modeling import ImageEncoder
from ties_merging_utils import state_dict_to_vector, vector_to_state_dict, ties_merging

# Create dummy modules so that src.models.modeling.ClassificationHead resolves correctly during unpickling
src = ModuleType('src')
sys.modules['src'] = src

models = ModuleType('models')
src.models = models
sys.modules['src.models'] = models

models.modeling = modeling
sys.modules['src.models.modeling'] = modeling

def dare_drop(v_k, p):
    """Applies DARE-Merging dropout to a task vector v_k with drop probability p."""
    if p <= 0.0:
        return v_k
    v_dare = {}
    for key, tensor in v_k.items():
        if tensor.dim() == 0 or tensor.dtype not in [torch.float16, torch.float32, torch.bfloat16]:
            v_dare[key] = tensor
            continue
        mask = (torch.rand_like(tensor) >= p).float()
        v_dare[key] = (tensor * mask) / (1.0 - p)
    return v_dare

def sta_prune(v_k, s):
    """Applies Sparse Task Arithmetic magnitude pruning layer-wise keeping top-s%."""
    if s >= 100.0:
        return v_k
    v_sparse = {}
    for key, tensor in v_k.items():
        if tensor.dim() == 0 or tensor.dtype not in [torch.float16, torch.float32, torch.bfloat16]:
            v_sparse[key] = tensor
            continue
        flat_tensor = tensor.abs().view(-1)
        k = int(flat_tensor.numel() * (s / 100.0))
        if k == 0:
            v_sparse[key] = torch.zeros_like(tensor)
        elif k >= flat_tensor.numel():
            v_sparse[key] = tensor
        else:
            threshold, _ = torch.topk(flat_tensor, k)
            thresh_val = threshold[-1]
            mask = (tensor.abs() >= thresh_val).float()
            v_sparse[key] = tensor * mask
    return v_sparse

def evaluate_merged_model(merged_state_dict, datasets, args):
    # Instantiate clean ImageEncoder model
    image_encoder = ImageEncoder(args, keep_lang=False)
    image_encoder.load_state_dict(merged_state_dict, strict=True)
    image_encoder = image_encoder.to(args.device)
    
    results = {}
    for dataset in datasets:
        print(f"Evaluating on {dataset}...")
        metrics = eval_single_dataset(image_encoder, dataset, args)
        results[dataset] = metrics.get('top1') * 100
    
    avg_acc = np.mean(list(results.values()))
    results['Avg'] = avg_acc
    return results

def main():
    args = parse_arguments()
    args.model = 'ViT-B-32'
    args.max_eval_batches = 16
    
    # Paths relative to submission6 folder
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    args.data_location = os.path.join(BASE_DIR, 'data')
    args.save = os.path.join(BASE_DIR, 'checkpoints', args.model)
    args.logs_path = os.path.join(BASE_DIR, 'logs', args.model)
    args.openclip_cachedir = os.path.join(BASE_DIR, 'openclip_cache')
    pretrained_checkpoint = os.path.join(args.save, 'zeroshot.pt')
    
    os.makedirs(args.logs_path, exist_ok=True)
    
    # 4 datasets specified in final_idea.md
    datasets = ['MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN']
    
    print("Loading checkpoints...")
    base_sd = torch.load(pretrained_checkpoint, map_location='cpu', weights_only=False)
    
    # Load task experts and extract task vectors
    task_vectors = {}
    for dataset in datasets:
        print(f"Loading {dataset} expert...")
        expert_sd = torch.load(os.path.join(args.save, dataset, 'finetuned.pt'), map_location='cpu', weights_only=False)
        
        # Compute task vector (delta updates)
        v_k = {}
        for key in base_sd:
            if base_sd[key].dtype in [torch.int64, torch.uint8]:
                continue
            v_k[key] = expert_sd[key] - base_sd[key]
        task_vectors[dataset] = v_k

    scaling_coef = 0.3
    
    all_results = {}
    
    # --- 1. Task Arithmetic (Uniform s=100%) ---
    print("\n" + "="*30)
    print("Running Baseline: Task Arithmetic (TA)")
    print("="*30)
    v_merged_ta = {}
    for key in base_sd:
        if base_sd[key].dtype in [torch.int64, torch.uint8]:
            continue
        v_merged_ta[key] = sum(task_vectors[d][key] for d in datasets)
        
    merged_state_dict_ta = {}
    for key in base_sd:
        if key in v_merged_ta:
            merged_state_dict_ta[key] = base_sd[key] + scaling_coef * v_merged_ta[key]
        else:
            merged_state_dict_ta[key] = base_sd[key]
            
    all_results['Task Arithmetic'] = evaluate_merged_model(merged_state_dict_ta, datasets, args)
    
    # --- 2. DARE-Merging (p=0.8) ---
    print("\n" + "="*30)
    print("Running Baseline: DARE-Merging (p=0.8)")
    print("="*30)
    v_dare_tasks = [dare_drop(task_vectors[d], 0.8) for d in datasets]
    v_merged_dare = {}
    for key in base_sd:
        if base_sd[key].dtype in [torch.int64, torch.uint8]:
            continue
        v_merged_dare[key] = sum(v_dare_tasks[i][key] for i in range(len(datasets)))
        
    merged_state_dict_dare = {}
    for key in base_sd:
        if key in v_merged_dare:
            merged_state_dict_dare[key] = base_sd[key] + scaling_coef * v_merged_dare[key]
        else:
            merged_state_dict_dare[key] = base_sd[key]
            
    all_results['DARE (p=0.8)'] = evaluate_merged_model(merged_state_dict_dare, datasets, args)

    # --- 3. TIES-Merging (reset_thresh=20) ---
    print("\n" + "="*30)
    print("Running Baseline: TIES-Merging (reset_thresh=20)")
    print("="*30)
    ft_checks = [torch.load(os.path.join(args.save, d, 'finetuned.pt'), map_location='cpu', weights_only=False) for d in datasets]
    ptm_check = torch.load(pretrained_checkpoint, map_location='cpu', weights_only=False)
    
    remove_keys = []
    flat_ft = torch.vstack([state_dict_to_vector(check, remove_keys) for check in ft_checks])
    flat_ptm = state_dict_to_vector(ptm_check, remove_keys)
    tv_flat_checks = flat_ft - flat_ptm
    
    merged_tv_ties = ties_merging(tv_flat_checks, reset_thresh=20, merge_func="dis-sum")
    merged_check_ties = flat_ptm + scaling_coef * merged_tv_ties
    merged_state_dict_ties = vector_to_state_dict(merged_check_ties, ptm_check, remove_keys=remove_keys)
    
    all_results['TIES-Merging'] = evaluate_merged_model(merged_state_dict_ties, datasets, args)

    # --- 4. Sparse Task Arithmetic (STA) (Sweeping s across 5%, 10%, 20%, 50%) ---
    for s in [5, 10, 20, 50]:
        print("\n" + "="*30)
        print(f"Running Proposed: Sparse Task Arithmetic (STA) (s={s}%)")
        print("="*30)
        v_sta_tasks = [sta_prune(task_vectors[d], s) for d in datasets]
        v_merged_sta = {}
        for key in base_sd:
            if base_sd[key].dtype in [torch.int64, torch.uint8]:
                continue
            v_merged_sta[key] = sum(v_sta_tasks[i][key] for i in range(len(datasets)))
            
        merged_state_dict_sta = {}
        for key in base_sd:
            if key in v_merged_sta:
                merged_state_dict_sta[key] = base_sd[key] + scaling_coef * v_merged_sta[key]
            else:
                merged_state_dict_sta[key] = base_sd[key]
                
        all_results[f'STA (s={s}%)'] = evaluate_merged_model(merged_state_dict_sta, datasets, args)

    # --- Print Final Results Table ---
    print("\n" + "="*50)
    print("FINAL MODEL MERGING RESULTS SUMMARY")
    print("="*50)
    header = f"{'Method':<20} | " + " | ".join([f"{d:<10}" for d in datasets]) + " | Avg"
    print(header)
    print("-" * len(header))
    for method, results in all_results.items():
        row = f"{method:<20} | " + " | ".join([f"{results[d]:.2f}%" for d in datasets]) + f" | {results['Avg']:.2f}%"
        print(row)
    print("="*50)

if __name__ == '__main__':
    main()
