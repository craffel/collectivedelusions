import sys
sys.path.append('SyMerge')
sys.path.append('SyMerge/src')
sys.path.append('local_packages')

import os
import time
import argparse
import torch
import numpy as np

from task_vectors import TaskVector
from dataset.registry import get_dataset
from dataset.common import get_dataloader, maybe_dictionarize, get_dataloader_shuffle
from heads import get_classification_head
from modeling import ImageClassifier

def patch_model(model):
    for m in model.modules():
        if m.__class__.__name__ == 'Transformer':
            m.batch_first = False

def run_calibration(pretrained_model, base_state_dict, tv_state_dicts, proj_key, calibration_images, exam_datasets):
    salience_vectors = []
    
    W_base = base_state_dict[proj_key]
    W_experts = [tv_state_dicts[k][proj_key] + W_base for k in range(len(exam_datasets))]
    
    original_proj = pretrained_model.model.visual.proj
    pretrained_model.model.visual.proj = None
    
    with torch.no_grad():
        for k, name in enumerate(exam_datasets):
            X_k = pretrained_model.model.visual(calibration_images[name])
            
            H_base = X_k @ W_base
            H_k = X_k @ W_experts[k]
            
            delta_H = H_k - H_base
            
            frob_norm = torch.norm(delta_H, p='fro')
            delta_H_tilde = delta_H / (frob_norm + 1e-8)
            
            S_k = torch.mean(torch.abs(delta_H_tilde), dim=0)
            salience_vectors.append(S_k)
            
    pretrained_model.model.visual.proj = original_proj
    return torch.stack(salience_vectors, dim=0)

def evaluate_model(pretrained_model, classification_heads, test_loaders, exam_datasets, args):
    accs = []
    for name in exam_datasets:
        classifier = ImageClassifier(pretrained_model, classification_heads[name])
        classifier.eval()
        
        correct = 0
        total = 0
        with torch.no_grad():
            for b_idx, batch in enumerate(test_loaders[name]):
                if args.max_eval_batches is not None and b_idx >= args.max_eval_batches:
                    break
                batch = maybe_dictionarize(batch)
                x = batch['images'].to(args.device)
                y = batch['labels'].to(args.device)
                logits = classifier(x)
                pred = logits.argmax(dim=1, keepdim=True)
                correct += pred.eq(y.view_as(pred)).sum().item()
                total += y.size(0)
        accs.append(correct / total)
    return sum(accs) / len(accs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-location', type=str, default='datasets')
    parser.add_argument('--checkpoints-root', type=str, default='checkpoints_tint')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--max-eval-batches', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    print(f"Using device: {args.device}")
    
    exam_datasets = ['SUN397', 'Cars', 'RESISC45', 'EuroSAT', 'SVHN', 'GTSRB', 'MNIST', 'DTD']
    model_name = 'ViT-B-32'
    
    checkpoints_root_path = os.path.abspath(args.checkpoints_root)
    model_root = os.path.join(checkpoints_root_path, model_name)
    pretrained_checkpoint = os.path.join(model_root, 'zeroshot.pt')
    save_dir = model_root
    
    class CustomArgs:
        model = 'ViT-B-32'
        data_location = args.data_location
        checkpoints_root = checkpoints_root_path
        save = save_dir
        pretrained_checkpoint = 'zeroshot.pt'
        batch_size = args.batch_size
        device = args.device
        openclip_cachedir = os.path.expanduser('~/.cache/open_clip')
        cache_dir = None
    
    c_args = CustomArgs()

    print("Loading pretrained base model...")
    pretrained_model = torch.load(pretrained_checkpoint, weights_only=False)
    patch_model(pretrained_model)
    pretrained_model = pretrained_model.to(args.device)
    pretrained_model.eval()

    print("Loading classification heads...")
    classification_heads = {}
    for name in exam_datasets:
        classification_heads[name] = get_classification_head(c_args, name).to(args.device)
        classification_heads[name].eval()

    print("Loading task vectors...")
    task_vectors = []
    for name in exam_datasets:
        finetuned_path = os.path.join(model_root, name, 'finetuned.pt')
        tv = TaskVector(pretrained_checkpoint, finetuned_path)
        task_vectors.append(tv)

    base_state_dict = {k: v.detach().clone().to(args.device) for k, v in pretrained_model.state_dict().items()}
    tv_state_dicts = [{k: v.detach().clone().to(args.device) for k, v in tv.vector.items()} for tv in task_vectors]

    proj_key = 'model.visual.proj'
    if proj_key not in base_state_dict:
        for k in base_state_dict:
            if 'visual.proj' in k:
                proj_key = k
                break

    print("Loading validation datasets for evaluation...")
    test_loaders = {}
    for name in exam_datasets:
        dataset = get_dataset(name, pretrained_model.val_preprocess, location=args.data_location, batch_size=args.batch_size)
        test_loaders[name] = get_dataloader(dataset, is_train=False, args=c_args)

    # We will run a comparison of physical calibration vs data-free synthetic calibration
    seeds = [42, 100, 2026]
    B = 32
    
    results = {
        'physical': [],
        'synthetic_gaussian': [],
        'synthetic_zeros': []
    }
    
    print("\n--- Running Evaluation with Physical Calibration Data ---")
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        calibration_images = {}
        for name in exam_datasets:
            dataset = get_dataset(name, pretrained_model.val_preprocess, location=args.data_location, batch_size=B)
            loader = get_dataloader_shuffle(dataset)
            for batch in loader:
                batch = maybe_dictionarize(batch)
                calibration_images[name] = batch['images'].to(args.device)
                break
                
        S_stacked = run_calibration(pretrained_model, base_state_dict, tv_state_dicts, proj_key, calibration_images, exam_datasets)
        alpha = torch.softmax(S_stacked / 0.5, dim=0)
        
        merged_sd = {}
        for k in base_state_dict:
            if k == proj_key:
                gated_proj_tv = sum(
                    tv_state_dicts[idx][proj_key] * alpha[idx].unsqueeze(0)
                    for idx in range(len(exam_datasets))
                )
                merged_sd[k] = base_state_dict[k] + 0.3 * gated_proj_tv
            else:
                total_tv = sum(tv_sd[k] for tv_sd in tv_state_dicts if k in tv_sd)
                merged_sd[k] = base_state_dict[k] + 0.3 * total_tv
                
        pretrained_model.load_state_dict(merged_sd)
        pretrained_model.eval()
        
        avg_acc = evaluate_model(pretrained_model, classification_heads, test_loaders, exam_datasets, args)
        results['physical'].append(avg_acc)
        print(f"Physical Accuracy (Seed {seed}): {avg_acc*100:.4f}%")

    print("\n--- Running Evaluation with Synthetic Gaussian Noise Calibration ---")
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        calibration_images = {}
        for name in exam_datasets:
            # Generate random Gaussian noise with same shape as standard CLIP batch
            calibration_images[name] = torch.randn(B, 3, 224, 224).to(args.device)
            
        S_stacked = run_calibration(pretrained_model, base_state_dict, tv_state_dicts, proj_key, calibration_images, exam_datasets)
        alpha = torch.softmax(S_stacked / 0.5, dim=0)
        
        merged_sd = {}
        for k in base_state_dict:
            if k == proj_key:
                gated_proj_tv = sum(
                    tv_state_dicts[idx][proj_key] * alpha[idx].unsqueeze(0)
                    for idx in range(len(exam_datasets))
                )
                merged_sd[k] = base_state_dict[k] + 0.3 * gated_proj_tv
            else:
                total_tv = sum(tv_sd[k] for tv_sd in tv_state_dicts if k in tv_sd)
                merged_sd[k] = base_state_dict[k] + 0.3 * total_tv
                
        pretrained_model.load_state_dict(merged_sd)
        pretrained_model.eval()
        
        avg_acc = evaluate_model(pretrained_model, classification_heads, test_loaders, exam_datasets, args)
        results['synthetic_gaussian'].append(avg_acc)
        print(f"Synthetic Gaussian Accuracy (Seed {seed}): {avg_acc*100:.4f}%")

    print("\n--- Running Evaluation with Synthetic Zero Tensors (Purely Zero-shot/Data-free) ---")
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        calibration_images = {}
        for name in exam_datasets:
            calibration_images[name] = torch.zeros(B, 3, 224, 224).to(args.device)
            
        S_stacked = run_calibration(pretrained_model, base_state_dict, tv_state_dicts, proj_key, calibration_images, exam_datasets)
        alpha = torch.softmax(S_stacked / 0.5, dim=0)
        
        merged_sd = {}
        for k in base_state_dict:
            if k == proj_key:
                gated_proj_tv = sum(
                    tv_state_dicts[idx][proj_key] * alpha[idx].unsqueeze(0)
                    for idx in range(len(exam_datasets))
                )
                merged_sd[k] = base_state_dict[k] + 0.3 * gated_proj_tv
            else:
                total_tv = sum(tv_sd[k] for tv_sd in tv_state_dicts if k in tv_sd)
                merged_sd[k] = base_state_dict[k] + 0.3 * total_tv
                
        pretrained_model.load_state_dict(merged_sd)
        pretrained_model.eval()
        
        avg_acc = evaluate_model(pretrained_model, classification_heads, test_loaders, exam_datasets, args)
        results['synthetic_zeros'].append(avg_acc)
        print(f"Synthetic Zeros Accuracy (Seed {seed}): {avg_acc*100:.4f}%")

    # Save results
    import json
    with open('results/data_free_results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print("\n=== Data-free Evaluation Summary ===")
    for key, val in results.items():
        accs = np.array(val) * 100
        print(f"{key:20s} | Mean Accuracy: {np.mean(accs):.4f}% +/- {np.std(accs):.4f}%")

if __name__ == '__main__':
    main()
