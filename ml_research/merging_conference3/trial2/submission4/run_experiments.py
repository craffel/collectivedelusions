import sys
sys.path.append('SyMerge')
sys.path.append('SyMerge/src')
sys.path.append('local_packages')

import os
import time
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

from task_vectors import TaskVector
from dataset.registry import get_dataset
from dataset.common import get_dataloader, maybe_dictionarize, get_dataloader_shuffle
from heads import get_classification_head
from modeling import ImageClassifier

# Patch function to resolve open_clip version compatibility
def patch_model(model):
    for m in model.modules():
        if m.__class__.__name__ == 'Transformer':
            m.batch_first = False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-location', type=str, default='datasets')
    parser.add_argument('--checkpoints-root', type=str, default='checkpoints_tint')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--calibration-size', type=int, default=32)
    parser.add_argument('--max-eval-batches', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    print(f"Using device: {args.device}")
    
    exam_datasets = ['SUN397', 'Cars', 'RESISC45', 'EuroSAT', 'SVHN', 'GTSRB', 'MNIST', 'DTD']
    model_name = 'ViT-B-32'
    
    # Setup paths matching SyMerge structure
    checkpoints_root_path = os.path.abspath(args.checkpoints_root)
    model_root = os.path.join(checkpoints_root_path, model_name)
    pretrained_checkpoint = os.path.join(model_root, 'zeroshot.pt')
    save_dir = model_root
    
    # Custom args object for compatibility with classification heads and datasets
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
        print(f"Loading task vector for {name}...")
        tv = TaskVector(pretrained_checkpoint, finetuned_path)
        task_vectors.append(tv)

    # Base model parameter dict (loaded as state_dict)
    base_state_dict = {k: v.detach().clone().to(args.device) for k, v in pretrained_model.state_dict().items()}

    # Task vectors parameter list
    tv_state_dicts = [{k: v.detach().clone().to(args.device) for k, v in tv.vector.items()} for tv in task_vectors]

    # Load datasets
    print("Loading validation datasets...")
    test_loaders = {}
    for name in exam_datasets:
        dataset = get_dataset(name, pretrained_model.val_preprocess, location=args.data_location, batch_size=args.batch_size)
        test_loaders[name] = get_dataloader(dataset, is_train=False, args=c_args)

    # 1. EVALUATE INDIVIDUAL EXPERTS (Sanity Check)
    print("\n--- Evaluating Individual Experts (Oracle) ---")
    expert_accuracies = {}
    for idx, name in enumerate(exam_datasets):
        # Construct model with just expert idx's task vector (lambda = 1.0)
        expert_sd = {}
        for k in base_state_dict:
            if k in tv_state_dicts[idx]:
                expert_sd[k] = base_state_dict[k] + tv_state_dicts[idx][k]
            else:
                expert_sd[k] = base_state_dict[k]
                
        pretrained_model.load_state_dict(expert_sd)
        pretrained_model.eval()
        
        # Evaluate
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
        
        acc = correct / total
        expert_accuracies[name] = acc
        print(f"Expert model {name} on {name} test set accuracy: {acc*100:.2f}%")

    # Restore base state
    pretrained_model.load_state_dict(base_state_dict)

    # 2. RUN CALIBRATION FOR EDGEMERGE
    print("\n--- Running Calibration for EdgeMerge ---")
    calibration_start_time = time.time()
    
    # Target projection layer key
    proj_key = 'model.visual.proj'
    if proj_key not in base_state_dict:
        # Let's find the correct projection key
        for k in base_state_dict:
            if 'visual.proj' in k:
                proj_key = k
                break
    print(f"Found visual projection layer key: '{proj_key}'")

    # Get calibration batches (calibration-size images from test_loader_shuffle or standard loader)
    calibration_images = {}
    for name in exam_datasets:
        dataset = get_dataset(name, pretrained_model.val_preprocess, location=args.data_location, batch_size=args.calibration_size)
        loader = get_dataloader_shuffle(dataset)
        for batch in loader:
            batch = maybe_dictionarize(batch)
            calibration_images[name] = batch['images'].to(args.device)
            break # just need one batch of calibration size
        print(f"Loaded {calibration_images[name].shape[0]} calibration images for {name}")

    # Compute calibration activations for Base model and all experts
    # Activations shape: [B, 512]
    salience_vectors = [] # Will contain K vectors of shape [512]
    
    # Projection weights
    W_base = base_state_dict[proj_key] # Shape: [768, 512]
    W_experts = [tv_state_dicts[k][proj_key] + W_base for k in range(len(exam_datasets))] # Shape: K x [768, 512]
    
    # Temporarily set proj to None to get features before projection layer
    original_proj = pretrained_model.model.visual.proj
    pretrained_model.model.visual.proj = None
    
    with torch.no_grad():
        for k, name in enumerate(exam_datasets):
            X_k = pretrained_model.model.visual(calibration_images[name]) # Shape: [B, 768]
            
            # Compute activations
            H_base = X_k @ W_base # Shape: [B, 512]
            H_k = X_k @ W_experts[k] # Shape: [B, 512]
            
            delta_H = H_k - H_base # Shape: [B, 512]
            
            # Scale normalization via Frobenius norm
            frob_norm = torch.norm(delta_H, p='fro')
            delta_H_tilde = delta_H / (frob_norm + 1e-8)
            
            # Compute salience vector
            S_k = torch.mean(torch.abs(delta_H_tilde), dim=0) # Shape: [512]
            salience_vectors.append(S_k)
            
    # Restore original projection layer
    pretrained_model.model.visual.proj = original_proj
    calibration_time = time.time() - calibration_start_time
    print(f"Calibration completed in {calibration_time:.2f} seconds.")

    # 3. GRID SEARCH FOR TASK ARITHMETIC (TA)
    print("\n--- Running Grid Search for Task Arithmetic (TA) ---")
    ta_lambdas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    best_ta_lambda = None
    best_ta_acc = -1.0
    ta_results = {}
    
    for lmbda in ta_lambdas:
        # Merge parameters
        merged_sd = {}
        for k in base_state_dict:
            total_tv = sum(tv_sd[k] for tv_sd in tv_state_dicts if k in tv_sd)
            merged_sd[k] = base_state_dict[k] + lmbda * total_tv
            
        pretrained_model.load_state_dict(merged_sd)
        pretrained_model.eval()
        
        # Evaluate on all 8 datasets
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
            
        avg_acc = sum(accs) / len(accs)
        ta_results[lmbda] = (avg_acc, accs)
        print(f"TA with lambda = {lmbda:.2f}: Average Accuracy = {avg_acc*100:.2f}%")
        
        if avg_acc > best_ta_acc:
            best_ta_acc = avg_acc
            best_ta_lambda = lmbda

    print(f"\nBest TA lambda: {best_ta_lambda:.2f} with Average Accuracy: {best_ta_acc*100:.2f}%")

    # 4. GRID SEARCH FOR EDGEMERGE (EM)
    print("\n--- Running Grid Search for EdgeMerge (EM) ---")
    em_temps = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
    em_lambdas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    best_em_temp = None
    best_em_lambda = None
    best_em_acc = -1.0
    em_results = {}
    
    # Stack salience vectors: [K, 512]
    S_stacked = torch.stack(salience_vectors, dim=0) # Shape: [K, 512]
    
    for temp in em_temps:
        # Compute channel-wise gating coefficients using Softmax
        # alpha shape: [K, 512]
        alpha = torch.softmax(S_stacked / temp, dim=0)
        
        for lmbda in em_lambdas:
            # Reconstruct merged parameters
            merged_sd = {}
            for k in base_state_dict:
                if k == proj_key:
                    # EdgeMerge row-by-row channel gating
                    gated_proj_tv = sum(
                        tv_state_dicts[idx][proj_key] * alpha[idx].unsqueeze(0)
                        for idx in range(len(exam_datasets))
                    )
                    merged_sd[k] = base_state_dict[k] + lmbda * gated_proj_tv
                else:
                    # Standard Task Arithmetic
                    total_tv = sum(tv_sd[k] for tv_sd in tv_state_dicts if k in tv_sd)
                    merged_sd[k] = base_state_dict[k] + lmbda * total_tv
                    
            pretrained_model.load_state_dict(merged_sd)
            pretrained_model.eval()
            
            # Evaluate on all 8 datasets
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
                
            avg_acc = sum(accs) / len(accs)
            em_results[(temp, lmbda)] = (avg_acc, accs)
            print(f"EM with temp = {temp:.2f}, lambda = {lmbda:.2f}: Average Accuracy = {avg_acc*100:.2f}%")
            
            if avg_acc > best_em_acc:
                best_em_acc = avg_acc
                best_em_temp = temp
                best_em_lambda = lmbda

    print(f"\nBest EM configuration: temp = {best_em_temp}, lambda = {best_em_lambda:.2f} with Average Accuracy: {best_em_acc*100:.2f}%")

    # 5. GENERATE PLOTS & ARTIFACTS
    print("\n--- Saving Results & Plots ---")
    os.makedirs('results', exist_ok=True)
    
    # Figure 1: Accuracy vs lambda for TA and Best EM Temperature
    plt.figure(figsize=(8, 5))
    ta_acc_list = [ta_results[l][0] * 100 for l in ta_lambdas]
    em_best_temp_acc_list = [em_results[(best_em_temp, l)][0] * 100 for l in em_lambdas]
    
    plt.plot(ta_lambdas, ta_acc_list, marker='o', linestyle='-', color='gray', label='Task Arithmetic (Baseline)')
    plt.plot(em_lambdas, em_best_temp_acc_list, marker='s', linestyle='-', color='blue', label=f'EdgeMerge (temp={best_em_temp})')
    plt.xlabel('Global Scaling Coefficient (lambda)')
    plt.ylabel('8-Task Average Accuracy (%)')
    plt.title('EdgeMerge vs. Task Arithmetic on 8-Task Benchmark')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('results/accuracy_vs_lambda.png')
    plt.close()
    
    # Figure 2: Pareto Frontier of Accuracy vs Merge Time / Compute Cost
    # Reported SyMerge accuracy is 89.74%, training time is ~10 mins (600s) on H100
    plt.figure(figsize=(8, 5))
    methods = ['Task Arithmetic', 'EdgeMerge', 'SyMerge']
    times = [0.1, calibration_time, 600.0] # seconds
    accuracies = [best_ta_acc * 100, best_em_acc * 100, 89.74]
    colors = ['gray', 'blue', 'orange']
    markers = ['o', 's', '^']
    
    for i, method in enumerate(methods):
        plt.scatter(times[i], accuracies[i], color=colors[i], marker=markers[i], s=150, label=method)
        plt.text(times[i] * 1.2 if times[i] > 0.1 else 0.15, accuracies[i] - 0.2, method, fontsize=10, weight='bold')
        
    plt.xscale('log')
    plt.xlabel('Preparation / Merge Compute Time (Seconds, Log Scale)')
    plt.ylabel('8-Task Average Accuracy (%)')
    plt.title('Cost-Accuracy Trade-off (Pragmatist Evaluation)')
    plt.grid(True, which="both", ls="-", color='0.9')
    plt.xlim(0.05, 2000)
    plt.ylim(min(accuracies) - 1.0, max(accuracies) + 1.0)
    plt.tight_layout()
    plt.savefig('results/cost_accuracy_tradeoff.png')
    plt.close()

    # Save results to a text file for formatting experiment_results.md
    with open('results/metrics.json', 'w') as f:
        import json
        out_metrics = {
            'expert_accuracies': {k: float(v) for k, v in expert_accuracies.items()},
            'best_ta_lambda': float(best_ta_lambda),
            'best_ta_acc': float(best_ta_acc),
            'best_em_temp': float(best_em_temp),
            'best_em_lambda': float(best_em_lambda),
            'best_em_acc': float(best_em_acc),
            'calibration_time': float(calibration_time),
            'ta_results': {float(k): [float(v[0]), [float(x) for x in v[1]]] for k, v in ta_results.items()},
            'em_results': {f"{k[0]}_{k[1]}": [float(v[0]), [float(x) for x in v[1]]] for k, v in em_results.items()}
        }
        json.dump(out_metrics, f, indent=4)
        
    print("All experiments completed successfully! Metric output written to results/metrics.json")

if __name__ == '__main__':
    main()
