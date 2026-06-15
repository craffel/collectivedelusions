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

def run_calibration(pretrained_model, base_state_dict, tv_state_dicts, proj_key, calibration_images, exam_datasets):
    salience_vectors = []
    
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
    return torch.stack(salience_vectors, dim=0) # Shape: [K, 512]

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

    # Target projection layer key
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

    # We will run a sensitivity analysis for B in {4, 8, 16, 32, 64} across 3 random seeds
    seeds = [42, 100, 2026]
    batch_sizes = [4, 8, 16, 32, 64]
    
    results_by_b = {}
    
    print("\n--- Running Sensitivity Analysis across Calibration Batch Sizes & Seeds ---")
    for B in batch_sizes:
        results_by_b[B] = []
        for seed in seeds:
            print(f"Evaluating Calibration Size B={B}, Seed={seed}...")
            # Set random seed
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            # Load calibration images
            calibration_images = {}
            for name in exam_datasets:
                dataset = get_dataset(name, pretrained_model.val_preprocess, location=args.data_location, batch_size=B)
                loader = get_dataloader_shuffle(dataset)
                for batch in loader:
                    batch = maybe_dictionarize(batch)
                    calibration_images[name] = batch['images'].to(args.device)
                    break
            
            # Run calibration
            S_stacked = run_calibration(pretrained_model, base_state_dict, tv_state_dicts, proj_key, calibration_images, exam_datasets)
            
            # Get alpha with optimal temp = 0.5
            alpha = torch.softmax(S_stacked / 0.5, dim=0)
            
            # Reconstruct model with optimal lambda = 0.3
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
            results_by_b[B].append(avg_acc)
            print(f"Accuracy for B={B}, Seed={seed}: {avg_acc*100:.4f}%")

    # Print summary statistics
    print("\n=== Summary of Calibration Batch Size Sensitivity Analysis ===")
    for B in batch_sizes:
        accs = np.array(results_by_b[B]) * 100
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        print(f"B = {B:2d} | Accuracy: {mean_acc:.3f}% +/- {std_acc:.3f}%")

    # RESTORE base model and run default calibration for plotting
    pretrained_model.load_state_dict(base_state_dict)
    torch.manual_seed(42)
    np.random.seed(42)
    calibration_images = {}
    for name in exam_datasets:
        dataset = get_dataset(name, pretrained_model.val_preprocess, location=args.data_location, batch_size=32)
        loader = get_dataloader_shuffle(dataset)
        for batch in loader:
            batch = maybe_dictionarize(batch)
            calibration_images[name] = batch['images'].to(args.device)
            break
            
    S_stacked = run_calibration(pretrained_model, base_state_dict, tv_state_dicts, proj_key, calibration_images, exam_datasets)
    alpha = torch.softmax(S_stacked / 0.5, dim=0) # Shape: [8, 512]
    alpha_np = alpha.cpu().numpy()
    
    # Let's generate a beautiful visualization!
    print("\n--- Generating Gating Distribution Plot ---")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left Panel: Histogram of Max Routing Coefficient per Channel
    max_alpha = np.max(alpha_np, axis=0)
    ax1.hist(max_alpha, bins=20, color='royalblue', edgecolor='black', alpha=0.8)
    ax1.axvline(x=0.5, color='red', linestyle='--', linewidth=1.5, label='Specialization Threshold (0.50)')
    ax1.set_xlabel('Maximum Gating Coefficient ($\max_k \\alpha_k[j]$)', fontsize=11)
    ax1.set_ylabel('Number of Bottleneck Channels', fontsize=11)
    ax1.set_title('Distribution of Channel Specialization', fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend()
    
    # Right Panel: Stacked Bar Plot of Gating Coefficients for the first 30 Channels
    num_channels_to_plot = 30
    channels = np.arange(num_channels_to_plot)
    bottom = np.zeros(num_channels_to_plot)
    colors = plt.cm.tab10(np.linspace(0, 1, len(exam_datasets)))
    
    for idx, name in enumerate(exam_datasets):
        ax2.bar(channels, alpha_np[idx, :num_channels_to_plot], bottom=bottom, color=colors[idx], label=name, width=0.8)
        bottom += alpha_np[idx, :num_channels_to_plot]
        
    ax2.set_xlabel('Bottleneck Channel Index ($j$)', fontsize=11)
    ax2.set_ylabel('Gating Coefficient (\\alpha_k[j])', fontsize=11)
    ax2.set_title(f'Task Routing for First {num_channels_to_plot} Channels', fontsize=12, fontweight='bold')
    ax2.set_xlim(-1, num_channels_to_plot)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, linestyle='--', alpha=0.3)
    # Put legend outside or nicely formatted
    ax2.legend(bbox_to_anchor=(1.04, 1), loc="upper left", title="Task Experts")
    
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/gating_analysis.png', bbox_inches='tight', dpi=300)
    plt.savefig('submission/gating_analysis.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Gating plot successfully saved to results/gating_analysis.png and copied to submission/gating_analysis.png!")
    
    # Let's save the statistics to a json file
    import json
    stats = {
        'batch_size_sensitivity': {
            str(B): {
                'accuracies': [float(x) for x in results_by_b[B]],
                'mean': float(np.mean(results_by_b[B])),
                'std': float(np.std(results_by_b[B]))
            } for B in batch_sizes
        }
    }
    with open('results/gating_stats.json', 'w') as f:
        json.dump(stats, f, indent=4)
    print("Stats written to results/gating_stats.json")

if __name__ == '__main__':
    main()
