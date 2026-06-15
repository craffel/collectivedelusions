import sys
sys.path.append('SyMerge')
sys.path.append('SyMerge/src')
sys.path.append('local_packages')

import os
import torch
import numpy as np
from scipy.stats import spearmanr

from task_vectors import TaskVector
from dataset.registry import get_dataset
from dataset.common import get_dataloader_shuffle, maybe_dictionarize

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

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    exam_datasets = ['SUN397', 'Cars', 'RESISC45', 'EuroSAT', 'SVHN', 'GTSRB', 'MNIST', 'DTD']
    model_name = 'ViT-B-32'
    
    checkpoints_root_path = os.path.abspath('checkpoints_tint')
    model_root = os.path.join(checkpoints_root_path, model_name)
    pretrained_checkpoint = os.path.join(model_root, 'zeroshot.pt')
    save_dir = model_root
    
    class CustomArgs:
        pass
    
    c_args = CustomArgs()
    c_args.model = 'ViT-B-32'
    c_args.data_location = 'datasets'
    c_args.checkpoints_root = checkpoints_root_path
    c_args.save = save_dir
    c_args.pretrained_checkpoint = 'zeroshot.pt'
    c_args.batch_size = 32
    c_args.device = device
    c_args.openclip_cachedir = os.path.expanduser('~/.cache/open_clip')
    c_args.cache_dir = None

    print("Loading pretrained base model...")
    pretrained_model = torch.load(pretrained_checkpoint, weights_only=False)
    patch_model(pretrained_model)
    pretrained_model = pretrained_model.to(device)
    pretrained_model.eval()

    print("Loading task vectors...")
    task_vectors = []
    for name in exam_datasets:
        finetuned_path = os.path.join(model_root, name, 'finetuned.pt')
        tv = TaskVector(pretrained_checkpoint, finetuned_path)
        task_vectors.append(tv)

    base_state_dict = {k: v.detach().clone().to(device) for k, v in pretrained_model.state_dict().items()}
    tv_state_dicts = [{k: v.detach().clone().to(device) for k, v in tv.vector.items()} for tv in task_vectors]

    proj_key = 'model.visual.proj'
    if proj_key not in base_state_dict:
        for k in base_state_dict:
            if 'visual.proj' in k:
                proj_key = k
                break

    # Seed 42
    torch.manual_seed(42)
    np.random.seed(42)
    B = 32

    # 1. Physical Calibration
    print("\nExtracting physical calibration activations...")
    physical_images = {}
    for name in exam_datasets:
        dataset = get_dataset(name, pretrained_model.val_preprocess, location='datasets', batch_size=B)
        loader = get_dataloader_shuffle(dataset)
        for batch in loader:
            batch = maybe_dictionarize(batch)
            physical_images[name] = batch['images'].to(device)
            break
    S_physical = run_calibration(pretrained_model, base_state_dict, tv_state_dicts, proj_key, physical_images, exam_datasets)

    # 2. Synthetic Gaussian Calibration
    print("Extracting synthetic Gaussian activations...")
    gaussian_images = {}
    for name in exam_datasets:
        gaussian_images[name] = torch.randn(B, 3, 224, 224).to(device)
    S_gaussian = run_calibration(pretrained_model, base_state_dict, tv_state_dicts, proj_key, gaussian_images, exam_datasets)

    # 3. Synthetic Zero Calibration
    print("Extracting synthetic Zero activations...")
    zero_images = {}
    for name in exam_datasets:
        zero_images[name] = torch.zeros(B, 3, 224, 224).to(device)
    S_zero = run_calibration(pretrained_model, base_state_dict, tv_state_dicts, proj_key, zero_images, exam_datasets)

    # Computations
    print("\n--- Correlation and Similarity Analysis ---")
    
    # We will report metrics task by task
    tasks_metrics = {}
    for k, name in enumerate(exam_datasets):
        phys = S_physical[k]
        gauss = S_gaussian[k]
        zero = S_zero[k]
        
        # Physical vs Gaussian
        cos_phys_gauss = torch.nn.functional.cosine_similarity(phys, gauss, dim=0).item()
        spear_phys_gauss, _ = spearmanr(phys.cpu().numpy(), gauss.cpu().numpy())
        
        # Physical vs Zeros
        cos_phys_zero = torch.nn.functional.cosine_similarity(phys, zero, dim=0).item()
        spear_phys_zero, _ = spearmanr(phys.cpu().numpy(), zero.cpu().numpy())
        
        # Gaussian vs Zeros
        cos_gauss_zero = torch.nn.functional.cosine_similarity(gauss, zero, dim=0).item()
        spear_gauss_zero, _ = spearmanr(gauss.cpu().numpy(), zero.cpu().numpy())
        
        tasks_metrics[name] = {
            'cos_phys_gauss': cos_phys_gauss,
            'spear_phys_gauss': spear_phys_gauss,
            'cos_phys_zero': cos_phys_zero,
            'spear_phys_zero': spear_phys_zero,
            'cos_gauss_zero': cos_gauss_zero,
            'spear_gauss_zero': spear_gauss_zero
        }
        
        print(f"\nTask: {name}")
        print(f"  Physical vs. Gaussian | Cosine Sim: {cos_phys_gauss:.6f} | Spearman: {spear_phys_gauss:.6f}")
        print(f"  Physical vs. Zeros    | Cosine Sim: {cos_phys_zero:.6f}  | Spearman: {spear_phys_zero:.6f}")
        print(f"  Gaussian vs. Zeros    | Cosine Sim: {cos_gauss_zero:.6f} | Spearman: {spear_gauss_zero:.6f}")

    # Averages
    avg_cos_phys_gauss = np.mean([metrics['cos_phys_gauss'] for metrics in tasks_metrics.values()])
    avg_spear_phys_gauss = np.mean([metrics['spear_phys_gauss'] for metrics in tasks_metrics.values()])
    avg_cos_phys_zero = np.mean([metrics['cos_phys_zero'] for metrics in tasks_metrics.values()])
    avg_spear_phys_zero = np.mean([metrics['spear_phys_zero'] for metrics in tasks_metrics.values()])
    avg_cos_gauss_zero = np.mean([metrics['cos_gauss_zero'] for metrics in tasks_metrics.values()])
    avg_spear_gauss_zero = np.mean([metrics['spear_gauss_zero'] for metrics in tasks_metrics.values()])

    print("\n" + "="*50)
    print("AVERAGE METRICS ACROSS ALL 8 TASKS:")
    print(f"Physical vs. Gaussian | Avg Cosine Sim: {avg_cos_phys_gauss:.6f} | Avg Spearman: {avg_spear_phys_gauss:.6f}")
    print(f"Physical vs. Zeros    | Avg Cosine Sim: {avg_cos_phys_zero:.6f}  | Avg Spearman: {avg_spear_phys_zero:.6f}")
    print(f"Gaussian vs. Zeros    | Avg Cosine Sim: {avg_cos_gauss_zero:.6f} | Avg Spearman: {avg_spear_gauss_zero:.6f}")
    print("="*50)

    # Save to a JSON
    import json
    with open('results/calibration_correlation.json', 'w') as f:
        json.dump(tasks_metrics, f, indent=4)

if __name__ == '__main__':
    main()
