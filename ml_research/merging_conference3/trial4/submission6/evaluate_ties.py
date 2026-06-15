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

def evaluate_merged_model(image_encoder, merged_state_dict, datasets, args):
    image_encoder.load_state_dict(merged_state_dict, strict=True)
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
    
    # Instantiate the encoder ONCE
    image_encoder = ImageEncoder(args, keep_lang=False)
    image_encoder = image_encoder.to(args.device)

    # Load checkpoints and compute TIES-Merging vector
    ft_checks = [torch.load(os.path.join(args.save, d, 'finetuned.pt'), map_location='cpu', weights_only=False) for d in datasets]
    ptm_check = torch.load(pretrained_checkpoint, map_location='cpu', weights_only=False)
    remove_keys = []
    flat_ft = torch.vstack([state_dict_to_vector(check, remove_keys) for check in ft_checks])
    flat_ptm = state_dict_to_vector(ptm_check, remove_keys)
    tv_flat_checks = flat_ft - flat_ptm
    merged_tv_ties = ties_merging(tv_flat_checks, reset_thresh=20, merge_func="dis-sum")
    
    print("\nSweeping lambda for TIES-Merging...")
    for l in [0.3, 0.4, 0.5]:
        merged_check_ties = flat_ptm + l * merged_tv_ties
        merged_sd = vector_to_state_dict(merged_check_ties, ptm_check, remove_keys=remove_keys)
        res = evaluate_merged_model(image_encoder, merged_sd, datasets, args)
        print(f"TIES lambda={l:.2f}: Avg={res['Avg']:.2f}% (MNIST={res['MNIST']:.1f}%, Fashion={res['FashionMNIST']:.1f}%, CIFAR={res['CIFAR10']:.1f}%, SVHN={res['SVHN']:.1f}%)")

if __name__ == '__main__':
    main()
