import os
import sys
import torch
import numpy as np
from types import ModuleType

# Add AdaMerging/src to python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'AdaMerging', 'src'))

import modeling
from eval import eval_single_dataset
from args import parse_arguments
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
    image_encoder = ImageEncoder(args, keep_lang=False)
    image_encoder.load_state_dict(merged_state_dict, strict=True)
    image_encoder = image_encoder.to(args.device)
    
    results = {}
    for dataset in datasets:
        metrics = eval_single_dataset(image_encoder, dataset, args)
        results[dataset] = metrics.get('top1') * 100
    avg_acc = np.mean(list(results.values()))
    results['Avg'] = avg_acc
    return results

def main():
    args = parse_arguments()
    args.model = 'ViT-B-32'
    args.max_eval_batches = 16
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    args.data_location = os.path.join(BASE_DIR, 'data')
    args.save = os.path.join(BASE_DIR, 'checkpoints', args.model)
    args.openclip_cachedir = os.path.join(BASE_DIR, 'openclip_cache')
    pretrained_checkpoint = os.path.join(args.save, 'zeroshot.pt')
    
    datasets = ['MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN']
    base_sd = torch.load(pretrained_checkpoint, map_location='cpu', weights_only=False)
    
    task_vectors = {}
    for dataset in datasets:
        expert_sd = torch.load(os.path.join(args.save, dataset, 'finetuned.pt'), map_location='cpu', weights_only=False)
        v_k = {}
        for key in base_sd:
            if base_sd[key].dtype in [torch.int64, torch.uint8]:
                continue
            v_k[key] = expert_sd[key] - base_sd[key]
        task_vectors[dataset] = v_k

    # Let's sweep lambda for ALL methods from 0.1 to 1.0!
    lambdas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    # 1. Sweep Task Arithmetic (Full s=100%)
    print("\nSweeping lambda for Task Arithmetic...")
    for l in lambdas:
        v_merged_ta = {}
        for key in base_sd:
            if base_sd[key].dtype in [torch.int64, torch.uint8]:
                continue
            v_merged_ta[key] = sum(task_vectors[d][key] for d in datasets)
        merged_sd = {k: (base_sd[k] + l * v_merged_ta[k] if k in v_merged_ta else base_sd[k]) for k in base_sd}
        res = evaluate_merged_model(merged_sd, datasets, args)
        print(f"TA lambda={l:.1f}: Avg={res['Avg']:.2f}%")

    # 2. Sweep DARE (p=0.8)
    print("\nSweeping lambda for DARE (p=0.8)...")
    v_dare_tasks = [dare_drop(task_vectors[d], 0.8) for d in datasets]
    for l in lambdas:
        v_merged_dare = {}
        for key in base_sd:
            if base_sd[key].dtype in [torch.int64, torch.uint8]:
                continue
            v_merged_dare[key] = sum(v_dare_tasks[i][key] for i in range(len(datasets)))
        merged_sd = {k: (base_sd[k] + l * v_merged_dare[k] if k in v_merged_dare else base_sd[k]) for k in base_sd}
        res = evaluate_merged_model(merged_sd, datasets, args)
        print(f"DARE lambda={l:.1f}: Avg={res['Avg']:.2f}%")

    # 3. Sweep TIES-Merging (reset_thresh=20)
    print("\nSweeping lambda for TIES-Merging...")
    ft_checks = [torch.load(os.path.join(args.save, d, 'finetuned.pt'), map_location='cpu', weights_only=False) for d in datasets]
    ptm_check = torch.load(pretrained_checkpoint, map_location='cpu', weights_only=False)
    remove_keys = []
    flat_ft = torch.vstack([state_dict_to_vector(check, remove_keys) for check in ft_checks])
    flat_ptm = state_dict_to_vector(ptm_check, remove_keys)
    tv_flat_checks = flat_ft - flat_ptm
    merged_tv_ties = ties_merging(tv_flat_checks, reset_thresh=20, merge_func="dis-sum")
    
    for l in lambdas:
        merged_check_ties = flat_ptm + l * merged_tv_ties
        merged_sd = vector_to_state_dict(merged_check_ties, ptm_check, remove_keys=remove_keys)
        res = evaluate_merged_model(merged_sd, datasets, args)
        print(f"TIES lambda={l:.1f}: Avg={res['Avg']:.2f}%")

if __name__ == '__main__':
    main()
