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

# Create dummy modules so that src.models.modeling.ClassificationHead resolves correctly during unpickling
src = ModuleType('src')
sys.modules['src'] = src

models = ModuleType('models')
src.models = models
sys.modules['src.models'] = models

models.modeling = modeling
sys.modules['src.models.modeling'] = modeling

def sta_prune_rescaled(v_k, s, rescale_factor=1.0):
    """Applies Sparse Task Arithmetic magnitude pruning layer-wise keeping top-s% with optional rescaling."""
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
            val = tensor * mask
            v_sparse[key] = val * rescale_factor
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

    # Let's sweep lambda for standard STA (no rescaling) at s=20%
    # For s=20%, without rescaling, the effective update is small, so maybe a larger scaling_coef lambda helps!
    print("Sweeping lambda for Standard STA at s=20% (No Rescaling)")
    for l_coef in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        v_sta_tasks = [sta_prune_rescaled(task_vectors[d], 20, rescale_factor=1.0) for d in datasets]
        v_merged_sta = {}
        for key in base_sd:
            if base_sd[key].dtype in [torch.int64, torch.uint8]:
                continue
            v_merged_sta[key] = sum(v_sta_tasks[i][key] for i in range(len(datasets)))
            
        merged_state_dict_sta = {}
        for key in base_sd:
            if key in v_merged_sta:
                merged_state_dict_sta[key] = base_sd[key] + l_coef * v_merged_sta[key]
            else:
                merged_state_dict_sta[key] = base_sd[key]
                
        results = evaluate_merged_model(merged_state_dict_sta, datasets, args)
        print(f"Standard STA (s=20%) with lambda={l_coef:.1f}: Avg={results['Avg']:.2f}% (MNIST={results['MNIST']:.1f}%, Fashion={results['FashionMNIST']:.1f}%, CIFAR={results['CIFAR10']:.1f}%, SVHN={results['SVHN']:.1f}%)")

if __name__ == '__main__':
    main()
