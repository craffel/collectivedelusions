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

def run_calibration_mismatch(pretrained_model, base_state_dict, tv_state_dicts, proj_key, calibration_images, exam_datasets, device):
    # Old method (with feature mismatch)
    salience_vectors = []
    W_base = base_state_dict[proj_key]
    W_experts = [tv_state_dicts[k][proj_key] + W_base for k in range(len(exam_datasets))]
    
    # Load base state dict first
    pretrained_model.load_state_dict(base_state_dict)
    pretrained_model.eval()
    
    # Base model visual encoder features
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

def run_calibration_correct(pretrained_model, base_state_dict, tv_state_dicts, proj_key, calibration_images, exam_datasets, device):
    # New mathematically sound method (no feature mismatch)
    salience_vectors = []
    W_base = base_state_dict[proj_key]
    
    with torch.no_grad():
        for k, name in enumerate(exam_datasets):
            # Load expert state dict first
            expert_sd = {}
            for key in base_state_dict:
                if key in tv_state_dicts[k]:
                    expert_sd[key] = base_state_dict[key] + tv_state_dicts[k][key]
                else:
                    expert_sd[key] = base_state_dict[key]
            pretrained_model.load_state_dict(expert_sd)
            pretrained_model.eval()
            
            # Temporarily remove proj after loading
            original_proj = pretrained_model.model.visual.proj
            pretrained_model.model.visual.proj = None
            
            # Extract expert features
            X_k_expert = pretrained_model.model.visual(calibration_images[name])
            
            # Restore proj immediately
            pretrained_model.model.visual.proj = original_proj
            W_expert = expert_sd[proj_key]
            
            # Correct delta activation on expert's drifted features
            H_no_proj_update = X_k_expert @ W_base
            H_expert = X_k_expert @ W_expert
            delta_H = H_expert - H_no_proj_update
            
            frob_norm = torch.norm(delta_H, p='fro')
            delta_H_tilde = delta_H / (frob_norm + 1e-8)
            S_k = torch.mean(torch.abs(delta_H_tilde), dim=0)
            salience_vectors.append(S_k)
            
    # Restore base state dict
    pretrained_model.load_state_dict(base_state_dict)
    pretrained_model.eval()
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

    # Calibration 1: Mismatched (Old)
    print("Running mismatched calibration...")
    S_mismatched = run_calibration_mismatch(pretrained_model, base_state_dict, tv_state_dicts, proj_key, calibration_images, exam_datasets, device)

    # Calibration 2: Correct (New)
    print("Running correct calibration (no feature mismatch)...")
    S_correct = run_calibration_correct(pretrained_model, base_state_dict, tv_state_dicts, proj_key, calibration_images, exam_datasets, device)

    # Let's run a comparative evaluation over a small grid of hyperparams
    lmbda_statics = [0.25]
    lmbda_projs = [0.20, 0.40, 0.60]
    temps = [0.1, 0.5, 1.0]

    print("\n=== Evaluating with Gating from Mismatched (Old) Calibration ===")
    for temp in temps:
        alpha_m = torch.softmax(S_mismatched / temp, dim=0)
        gated_proj_tv = sum(tv_state_dicts[idx][proj_key] * alpha_m[idx].unsqueeze(0) for idx in range(len(exam_datasets)))
        for l_static in lmbda_statics:
            for l_proj in lmbda_projs:
                merged_sd = {}
                for k in base_state_dict:
                    if k == proj_key:
                        merged_sd[k] = base_state_dict[k] + l_proj * gated_proj_tv
                    else:
                        total_tv = sum(tv_sd[k] for tv_sd in tv_state_dicts if k in tv_sd)
                        merged_sd[k] = base_state_dict[k] + l_static * total_tv
                pretrained_model.load_state_dict(merged_sd)
                acc = evaluate_model(pretrained_model, classification_heads, test_loaders, exam_datasets, max_eval_batches=8, device=device)
                print(f"Mismatched EM (temp={temp:.2f}, l_static={l_static:.2f}, l_proj={l_proj:.2f}): {acc*100:.4f}%")

    print("\n=== Evaluating with Gating from Correct (New) Calibration ===")
    for temp in temps:
        alpha_c = torch.softmax(S_correct / temp, dim=0)
        gated_proj_tv = sum(tv_state_dicts[idx][proj_key] * alpha_c[idx].unsqueeze(0) for idx in range(len(exam_datasets)))
        for l_static in lmbda_statics:
            for l_proj in lmbda_projs:
                merged_sd = {}
                for k in base_state_dict:
                    if k == proj_key:
                        merged_sd[k] = base_state_dict[k] + l_proj * gated_proj_tv
                    else:
                        total_tv = sum(tv_sd[k] for tv_sd in tv_state_dicts if k in tv_sd)
                        merged_sd[k] = base_state_dict[k] + l_static * total_tv
                pretrained_model.load_state_dict(merged_sd)
                acc = evaluate_model(pretrained_model, classification_heads, test_loaders, exam_datasets, max_eval_batches=8, device=device)
                print(f"Correct EM (temp={temp:.2f}, l_static={l_static:.2f}, l_proj={l_proj:.2f}): {acc*100:.4f}%")

if __name__ == '__main__':
    main()
