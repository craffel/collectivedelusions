import sys
sys.path.append('SyMerge')
sys.path.append('SyMerge/src')
sys.path.append('local_packages')

import os
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

def run_calibration(pretrained_model, base_state_dict, tv_state_dicts, proj_key, calibration_images, exam_datasets, use_sndas, device):
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
            
            if use_sndas:
                frob_norm = torch.norm(delta_H, p='fro')
                delta_H_tilde = delta_H / (frob_norm + 1e-8)
            else:
                delta_H_tilde = delta_H
                
            S_k = torch.mean(torch.abs(delta_H_tilde), dim=0)
            salience_vectors.append(S_k)
            
    pretrained_model.model.visual.proj = original_proj
    return torch.stack(salience_vectors, dim=0)

def evaluate_model(pretrained_model, classification_heads, test_loaders, exam_datasets, max_eval_batches, device):
    accs = []
    for name in exam_datasets:
        classifier = ImageClassifier(pretrained_model, classification_heads[name])
        classifier.eval()
        
        correct = 0
        total = 0
        with torch.no_grad():
            for b_idx, batch in enumerate(test_loaders[name]):
                if max_eval_batches is not None and b_idx >= max_eval_batches:
                    break
                batch = maybe_dictionarize(batch)
                x = batch['images'].to(device)
                y = batch['labels'].to(device)
                logits = classifier(x)
                pred = logits.argmax(dim=1, keepdim=True)
                correct += pred.eq(y.view_as(pred)).sum().item()
                total += y.size(0)
        accs.append(correct / total)
    return sum(accs) / len(accs)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    exam_datasets = ['SUN397', 'Cars', 'RESISC45', 'EuroSAT', 'SVHN', 'GTSRB', 'MNIST', 'DTD']
    model_name = 'ViT-B-32'
    
    checkpoints_root = 'checkpoints_tint'
    checkpoints_root_path = os.path.abspath(checkpoints_root)
    model_root = os.path.join(checkpoints_root_path, model_name)
    pretrained_checkpoint = os.path.join(model_root, 'zeroshot.pt')
    
    class CustomArgs:
        pass
    
    c_args = CustomArgs()
    c_args.model = 'ViT-B-32'
    c_args.data_location = 'datasets'
    c_args.checkpoints_root = checkpoints_root_path
    c_args.save = model_root
    c_args.pretrained_checkpoint = 'zeroshot.pt'
    c_args.batch_size = 128
    c_args.device = device
    c_args.openclip_cachedir = os.path.expanduser('~/.cache/open_clip')
    c_args.cache_dir = None

    print("Loading pretrained base model...")
    pretrained_model = torch.load(pretrained_checkpoint, weights_only=False)
    patch_model(pretrained_model)
    pretrained_model = pretrained_model.to(device)
    pretrained_model.eval()

    print("Loading classification heads...")
    classification_heads = {}
    for name in exam_datasets:
        classification_heads[name] = get_classification_head(c_args, name).to(device)
        classification_heads[name].eval()

    print("Loading task vectors...")
    task_vectors = []
    for name in exam_datasets:
        finetuned_path = os.path.join(model_root, name, 'finetuned.pt')
        tv = TaskVector(pretrained_checkpoint, finetuned_path)
        task_vectors.append(tv)

    base_state_dict = {k: v.detach().clone().to(device) for k, v in pretrained_model.state_dict().items()}
    tv_state_dicts = [{k: v.detach().clone().to(device) for k, v in tv.vector.items()} for tv in task_vectors]

    print("Loading validation datasets...")
    test_loaders = {}
    for name in exam_datasets:
        dataset = get_dataset(name, pretrained_model.val_preprocess, location='datasets', batch_size=128)
        test_loaders[name] = get_dataloader(dataset, is_train=False, args=c_args)

    proj_key = 'model.visual.proj'
    if proj_key not in base_state_dict:
        for k in base_state_dict:
            if 'visual.proj' in k:
                proj_key = k
                break

    # Get calibration batches
    print("Loading calibration images...")
    calibration_images = {}
    for name in exam_datasets:
        dataset = get_dataset(name, pretrained_model.val_preprocess, location='datasets', batch_size=32)
        loader = get_dataloader_shuffle(dataset)
        for batch in loader:
            batch = maybe_dictionarize(batch)
            calibration_images[name] = batch['images'].to(device)
            break

    # We use optimal DSR scales as backdrops: l_static=0.25, l_proj=0.20, temp=0.10
    l_static = 0.25
    l_proj = 0.20
    temp = 0.10

    # 1. Ablation: No SNDAS (No Frobenius Normalization)
    print("\n=== Ablation 1: No SNDAS (No scale-normalization) ===")
    S_stacked_no_sndas = run_calibration(pretrained_model, base_state_dict, tv_state_dicts, proj_key, calibration_images, exam_datasets, use_sndas=False, device=device)
    alpha_no_sndas = torch.softmax(S_stacked_no_sndas / temp, dim=0)
    gated_proj_tv_no_sndas = sum(tv_state_dicts[idx][proj_key] * alpha_no_sndas[idx].unsqueeze(0) for idx in range(len(exam_datasets)))
    merged_sd = {}
    for k in base_state_dict:
        if k == proj_key:
            merged_sd[k] = base_state_dict[k] + l_proj * gated_proj_tv_no_sndas
        else:
            total_tv = sum(tv_sd[k] for tv_sd in tv_state_dicts if k in tv_sd)
            merged_sd[k] = base_state_dict[k] + l_static * total_tv
    pretrained_model.load_state_dict(merged_sd)
    pretrained_model.eval()
    acc_no_sndas = evaluate_model(pretrained_model, classification_heads, test_loaders, exam_datasets, max_eval_batches=8, device=device)
    print(f"Accuracy without SNDAS: {acc_no_sndas*100:.4f}%")

    # 2. Ablation: Layer-wise Gating (LWG)
    print("\n=== Ablation 2: Layer-wise Gating (LWG) ===")
    S_stacked_sndas = run_calibration(pretrained_model, base_state_dict, tv_state_dicts, proj_key, calibration_images, exam_datasets, use_sndas=True, device=device)
    # average salience over channels to form layer salience
    S_layer = torch.mean(S_stacked_sndas, dim=1) # Shape: [K]
    alpha_layer = torch.softmax(S_layer / temp, dim=0) # Shape: [K]
    print(f"Layer-gating coefficients (alpha_layer): {alpha_layer.cpu().numpy()}")
    # Apply global scalar per expert
    gated_proj_tv_lwg = sum(tv_state_dicts[idx][proj_key] * alpha_layer[idx] for idx in range(len(exam_datasets)))
    merged_sd = {}
    for k in base_state_dict:
        if k == proj_key:
            merged_sd[k] = base_state_dict[k] + l_proj * gated_proj_tv_lwg
        else:
            total_tv = sum(tv_sd[k] for tv_sd in tv_state_dicts if k in tv_sd)
            merged_sd[k] = base_state_dict[k] + l_static * total_tv
    pretrained_model.load_state_dict(merged_sd)
    pretrained_model.eval()
    acc_lwg = evaluate_model(pretrained_model, classification_heads, test_loaders, exam_datasets, max_eval_batches=8, device=device)
    print(f"Accuracy with Layer-wise Gating (LWG): {acc_lwg*100:.4f}%")

    # 3. Ablation: Uniform Gating (Uniform)
    print("\n=== Ablation 3: Uniform Gating ===")
    alpha_uniform = torch.full((len(exam_datasets),), 1.0 / len(exam_datasets), device=device)
    gated_proj_tv_uniform = sum(tv_state_dicts[idx][proj_key] * alpha_uniform[idx] for idx in range(len(exam_datasets)))
    merged_sd = {}
    for k in base_state_dict:
        if k == proj_key:
            merged_sd[k] = base_state_dict[k] + l_proj * gated_proj_tv_uniform
        else:
            total_tv = sum(tv_sd[k] for tv_sd in tv_state_dicts if k in tv_sd)
            merged_sd[k] = base_state_dict[k] + l_static * total_tv
    pretrained_model.load_state_dict(merged_sd)
    pretrained_model.eval()
    acc_uniform = evaluate_model(pretrained_model, classification_heads, test_loaders, exam_datasets, max_eval_batches=8, device=device)
    print(f"Accuracy with Uniform Gating: {acc_uniform*100:.4f}%")

    # Full Decoupled EdgeMerge (DSR, Ours) as reference
    print("\n=== Reference: Decoupled EdgeMerge (DSR, Ours) ===")
    alpha_dsr = torch.softmax(S_stacked_sndas / temp, dim=0)
    gated_proj_tv_dsr = sum(tv_state_dicts[idx][proj_key] * alpha_dsr[idx].unsqueeze(0) for idx in range(len(exam_datasets)))
    merged_sd = {}
    for k in base_state_dict:
        if k == proj_key:
            merged_sd[k] = base_state_dict[k] + l_proj * gated_proj_tv_dsr
        else:
            total_tv = sum(tv_sd[k] for tv_sd in tv_state_dicts if k in tv_sd)
            merged_sd[k] = base_state_dict[k] + l_static * total_tv
    pretrained_model.load_state_dict(merged_sd)
    pretrained_model.eval()
    acc_dsr = evaluate_model(pretrained_model, classification_heads, test_loaders, exam_datasets, max_eval_batches=8, device=device)
    print(f"Reference Accuracy: {acc_dsr*100:.4f}%")

if __name__ == '__main__':
    main()
