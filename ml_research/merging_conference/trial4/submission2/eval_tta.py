import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from torch.func import functional_call
import os
import copy
import numpy as np

# Import the BaseEncoder and ClassHead from train_experts
from train_experts import BaseEncoder, ClassHead, ExpertModel, get_dataloader

# Set random seed for reproducibility before running evaluations
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.enabled = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Unnormalize and normalize helper functions
def unnormalize(tensor):
    return tensor * 0.3081 + 0.1307

def normalize(tensor):
    return (tensor - 0.1307) / 0.3081

# Image corruptions
def apply_gaussian_noise(x, sigma=0.4):
    unnorm_x = unnormalize(x)
    noise = torch.randn_like(unnorm_x) * sigma
    corrupted_x = torch.clamp(unnorm_x + noise, 0.0, 1.0)
    return normalize(corrupted_x)

def apply_gaussian_blur(x, sigma=2.0):
    unnorm_x = unnormalize(x)
    # Ensure kernel size is odd
    corrupted_x = TF.gaussian_blur(unnorm_x, kernel_size=[5, 5], sigma=[sigma, sigma])
    return normalize(corrupted_x)

def apply_contrast_reduction(x, alpha=0.15):
    unnorm_x = unnormalize(x)
    corrupted_x = torch.clamp(0.5 + alpha * (unnorm_x - 0.5), 0.0, 1.0)
    return normalize(corrupted_x)

# Augmentations for consistency loss (Task-Aware)
def translate_image(x, max_translation=2):
    B, C, H, W = x.shape
    dx = torch.randint(-max_translation, max_translation + 1, (B,), device=x.device)
    dy = torch.randint(-max_translation, max_translation + 1, (B,), device=x.device)
    x_aug = x.clone()
    for i in range(B):
        x_aug[i] = TF.affine(
            x[i], angle=0, translate=[int(dx[i]), int(dy[i])], scale=1.0, shear=0.0,
            interpolation=transforms.InterpolationMode.NEAREST,
            fill=0
        )
    return x_aug

def augment_image(x, task_name, max_translation=2):
    x_aug = translate_image(x, max_translation)
    if task_name == 'fashion':
        # Apply random horizontal flip for objects
        B = x.shape[0]
        flip_mask = torch.rand(B) > 0.5
        for i in range(B):
            if flip_mask[i]:
                x_aug[i] = TF.hflip(x_aug[i])
    return x_aug

# Entropy calculation
def compute_entropy(logits, eps=1e-12):
    probs = torch.softmax(logits, dim=1)
    entropy = -torch.sum(probs * torch.log(probs + eps), dim=1)
    return torch.mean(entropy)

# Consistency loss calculation (KL divergence)
def compute_consistency_loss(logits_orig, logits_aug):
    p_orig = torch.softmax(logits_orig, dim=1).detach() # Stop gradient on target
    p_aug = torch.log_softmax(logits_aug, dim=1)
    kl_loss = nn.KLDivLoss(reduction='batchmean')(p_aug, p_orig)
    return kl_loss

# Precompute Fisher Information for EWC
def precompute_fisher(experts, dataloaders, N=200, epsilon=1e-8):
    print("\n--- Pre-computing Fisher Information for Expert Classification Heads ---")
    fisher_dict = {}
    for task_idx, (task_name, model) in enumerate(experts.items()):
        model.eval()
        fisher_dict[task_name] = {}
        for name, param in model.head.named_parameters():
            fisher_dict[task_name][name] = torch.zeros_like(param)
            
        samples_count = 0
        train_loader = dataloaders[task_name]['train']
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            for i in range(images.size(0)):
                img = images[i:i+1]
                lbl = labels[i:i+1]
                outputs = model(img)
                log_probs = torch.log_softmax(outputs, dim=1)
                loss = log_probs[0, lbl[0]]
                
                grads = torch.autograd.grad(loss, model.head.parameters(), retain_graph=False)
                for (name, param), grad in zip(model.head.named_parameters(), grads):
                    fisher_dict[task_name][name] += (grad.data ** 2)
                    
                samples_count += 1
                if samples_count >= N:
                    break
            if samples_count >= N:
                break
                
        # Normalize and add small epsilon for stability
        for name in fisher_dict[task_name]:
            fisher_dict[task_name][name] = fisher_dict[task_name][name] / N + epsilon
        print(f"Computed FIM for {task_name.upper()} classification head.")
        
    return fisher_dict

# Main TTA Evaluation Loop
def evaluate_tta(method_name, experts, base_params, task_vectors, test_loaders, fisher_dict=None, corruption_type='clean', num_cycles=100, ewc_lambda=50.0):
    torch.manual_seed(42) # Ensure reproducible test streams
    
    # Initialize task classification heads for TTA
    # We create a deepcopy of the expert heads to adapt them during TTA
    adapted_heads = {
        name: copy.deepcopy(model.head).to(device) for name, model in experts.items()
    }
    
    # Keep copies of the initial expert heads for EWC calculation
    init_heads = {
        name: copy.deepcopy(model.head).to(device) for name, model in experts.items()
    }
    
    # Initialize merging coefficients raw_lambdas
    # raw_lambdas has shape [8, 3] for 8 layer-wise parameter tensors and 3 tasks
    raw_lambdas = torch.zeros(8, 3, device=device, requires_grad=True)
    
    # Optimizer setup
    # If the method adapts classification heads, we optimize raw_lambdas and the active head
    # Let's map parameter parameters for optimization
    if method_name == 'static_merged':
        # No optimization
        optimizer = None
    elif method_name == 's2c_merge':
        # Teacher-free: adapts only merging coefficients, heads are frozen
        optimizer = optim.Adam([raw_lambdas], lr=0.005)
    elif method_name in ['standard_tta_tf', 'uewc_merge']:
        # Teacher-free: adapts both coefficients and active heads
        # We will dynamically re-instantiate the optimizer for the active head + coefficients
        # or update only the active parameters
        optimizer = None
    elif method_name in ['standard_tta_tg', 'ewc_tta']:
        # Teacher-guided: adapts both coefficients and active heads
        optimizer = None
        
    # Set base encoder statelessly using BaseEncoder
    base_encoder = BaseEncoder().to(device)
    base_encoder.eval()
    
    # Create iterators for test loaders
    iterators = {
        name: iter(loader) for name, loader in test_loaders.items()
    }
    
    task_names = list(experts.keys())
    correct_predictions = 0
    total_samples = 0
    
    # TTA Optimization loop
    for cycle in range(num_cycles):
        for task_idx, task_name in enumerate(task_names):
            try:
                images, labels = next(iterators[task_name])
            except StopIteration:
                # Re-initialize iterator if it runs out
                iterators[task_name] = iter(test_loaders[task_name])
                images, labels = next(iterators[task_name])
                
            images, labels = images.to(device), labels.to(device)
            
            # Apply chosen corruption
            if corruption_type == 'gaussian_noise':
                images_corrupted = apply_gaussian_noise(images, sigma=0.4)
            elif corruption_type == 'gaussian_blur':
                images_corrupted = apply_gaussian_blur(images, sigma=2.0)
            elif corruption_type == 'contrast':
                images_corrupted = apply_contrast_reduction(images, alpha=0.15)
            else:
                images_corrupted = images.clone()
                
            active_head = adapted_heads[task_name]
            
            # Execute adaptation step if method is dynamic
            if method_name != 'static_merged':
                # Dynamically set up optimizer for the active head + raw_lambdas
                # We do this because classification heads are task-specific, so only the active head is updated
                if method_name in ['standard_tta_tf', 'standard_tta_tg', 'ewc_tta', 'uewc_merge']:
                    opt = optim.Adam([
                        {'params': [raw_lambdas], 'lr': 0.005},
                        {'params': active_head.parameters(), 'lr': 0.05}
                    ])
                else: # s2c_merge (heads are frozen)
                    opt = optimizer
                    
                opt.zero_grad()
                
                # Reconstruct merged encoder parameters
                # UEWC-Merge and Static-Merged use Softmax constraint (C-TMM)
                # Others can use clamped unconstrained linear coefficients
                use_softmax = (method_name in ['uewc_merge', 'static_merged', 'ewc_tta'])
                
                merged_params = {}
                if use_softmax:
                    lambdas = torch.softmax(raw_lambdas, dim=1)
                else:
                    lambdas = torch.clamp(raw_lambdas + 1.0/3.0, 0.0, 1.0) # centered at 1/3
                    
                for p_idx, (name, base_p) in enumerate(base_params.items()):
                    w_merged = base_p.to(device).clone()
                    for k in range(3):
                        tv_val = task_vectors[k][name].to(device)
                        w_merged = w_merged + lambdas[p_idx, k] * tv_val
                    merged_params[name] = w_merged
                    
                # Stateless forward pass of the encoder
                features = functional_call(base_encoder, merged_params, images_corrupted)
                logits = active_head(features)
                
                loss = 0.0
                
                # Loss Calculation
                if method_name == 'standard_tta_tf':
                    # Teacher-free Standard TTA: prediction entropy minimization
                    loss = compute_entropy(logits)
                    
                elif method_name == 'standard_tta_tg':
                    # Teacher-guided Standard TTA: KL against original expert predictions
                    with torch.no_grad():
                        expert_logits = experts[task_name](images_corrupted)
                        expert_probs = torch.softmax(expert_logits, dim=1)
                    loss = nn.KLDivLoss(reduction='batchmean')(torch.log_softmax(logits, dim=1), expert_probs)
                    
                elif method_name == 's2c_merge':
                    # S2C-Merge: Entropy + Task-Aware Consistency
                    loss_ent = compute_entropy(logits)
                    
                    # Compute augmented features & predictions
                    images_aug = augment_image(images_corrupted, task_name)
                    features_aug = functional_call(base_encoder, merged_params, images_aug)
                    logits_aug = active_head(features_aug)
                    
                    loss_const = compute_consistency_loss(logits, logits_aug)
                    loss = loss_ent + 1.0 * loss_const
                    
                elif method_name == 'ewc_tta':
                    # EWC-TTA (Teacher-Guided): KL against experts + EWC penalty on heads
                    with torch.no_grad():
                        expert_logits = experts[task_name](images_corrupted)
                        expert_probs = torch.softmax(expert_logits, dim=1)
                    loss_kl = nn.KLDivLoss(reduction='batchmean')(torch.log_softmax(logits, dim=1), expert_probs)
                    
                    # EWC Penalty
                    loss_ewc = 0.0
                    for name, param in active_head.named_parameters():
                        fim = fisher_dict[task_name][name]
                        init_param = init_heads[task_name].state_dict()[name]
                        loss_ewc += 0.5 * torch.sum(fim * (param - init_param) ** 2)
                        
                    loss = loss_kl + ewc_lambda * loss_ewc
                    
                elif method_name == 'uewc_merge':
                    # UEWC-Merge (Teacher-Free Ours!): Entropy + Task-Aware Consistency + EWC on heads
                    loss_ent = compute_entropy(logits)
                    
                    # Consistency
                    images_aug = augment_image(images_corrupted, task_name)
                    features_aug = functional_call(base_encoder, merged_params, images_aug)
                    logits_aug = active_head(features_aug)
                    loss_const = compute_consistency_loss(logits, logits_aug)
                    
                    # EWC Penalty
                    loss_ewc = 0.0
                    for name, param in active_head.named_parameters():
                        fim = fisher_dict[task_name][name]
                        init_param = init_heads[task_name].state_dict()[name]
                        loss_ewc += 0.5 * torch.sum(fim * (param - init_param) ** 2)
                        
                    loss = loss_ent + 1.0 * loss_const + ewc_lambda * loss_ewc
                    
                # Backprop and Step
                loss.backward()
                opt.step()
                
            # Evaluation with final parameters for this step
            with torch.no_grad():
                # Reconstruct merged encoder parameters
                use_softmax = (method_name in ['uewc_merge', 'static_merged', 'ewc_tta'])
                merged_params_eval = {}
                if use_softmax:
                    lambdas_eval = torch.softmax(raw_lambdas, dim=1)
                else:
                    lambdas_eval = torch.clamp(raw_lambdas + 1.0/3.0, 0.0, 1.0)
                    
                for p_idx, (name, base_p) in enumerate(base_params.items()):
                    w_merged = base_p.to(device).clone()
                    for k in range(3):
                        tv_val = task_vectors[k][name].to(device)
                        w_merged = w_merged + lambdas_eval[p_idx, k] * tv_val
                    merged_params_eval[name] = w_merged
                    
                features_eval = functional_call(base_encoder, merged_params_eval, images_corrupted)
                logits_eval = active_head(features_eval)
                
                _, predicted = logits_eval.max(1)
                correct_predictions += predicted.eq(labels).sum().item()
                total_samples += labels.size(0)
                
    accuracy = 100. * correct_predictions / total_samples
    return accuracy

if __name__ == '__main__':
    print("\nStarting Test-Time Model Merging Evaluation pipeline...")
    
    # 1. Load experts and base initialization weights
    expert_names = ['mnist', 'fashion', 'kmnist']
    experts = {}
    base_encoder = BaseEncoder().to(device)
    
    # Load base encoder init weights
    base_encoder.load_state_dict(torch.load('./experts/base_encoder_init.pt', map_location=device, weights_only=True))
    base_params = {name: param.cpu() for name, param in base_encoder.named_parameters()}
    
    # Load each trained expert model
    task_vectors = []
    for k, name in enumerate(expert_names):
        ckpt_path = f'./experts/{name}_expert.pt'
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Expert checkpoint not found: {ckpt_path}. Run train_experts.py first!")
            
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        
        # Expert encoder
        expert_encoder = BaseEncoder().to(device)
        expert_encoder.load_state_dict(ckpt['encoder_state_dict'])
        
        # Expert head
        expert_head = ClassHead().to(device)
        expert_head.load_state_dict(ckpt['head_state_dict'])
        
        experts[name] = ExpertModel(expert_encoder, expert_head).to(device)
        experts[name].eval()
        
        # Calculate task vectors: \tau = \Theta_expert - \Theta_pre
        tv = {}
        for p_name, p_val in expert_encoder.named_parameters():
            base_val = base_encoder.state_dict()[p_name]
            tv[p_name] = (p_val - base_val).cpu()
        task_vectors.append(tv)
        
    print("Loaded all experts and computed task vectors successfully.")
    
    # 2. Get dataloaders for pre-computing Fisher and test evaluation
    dataloaders = {}
    test_loaders = {}
    for name in expert_names:
        dataloaders[name] = {
            'train': get_dataloader(name, batch_size=64, train=True),
            'test': get_dataloader(name, batch_size=64, train=False)
        }
        test_loaders[name] = dataloaders[name]['test']
        
    # 3. Pre-compute Fisher Information for active heads
    fisher_dict = precompute_fisher(experts, dataloaders, N=200)
    
    # 4. Evaluate across all environments
    corruptions = ['clean', 'gaussian_noise', 'gaussian_blur', 'contrast']
    methods = [
        'static_merged',
        'standard_tta_tf',
        'standard_tta_tg',
        's2c_merge',
        'ewc_tta',
        'uewc_merge'
    ]
    
    # Save results to a report file
    results = {}
    
    print("\n" + "="*50)
    print("  RUNNING DETERMINISTIC TEST-TIME ADAPTATION EVALUATION")
    print("="*50)
    
    for corr in corruptions:
        print(f"\nEvaluating under corruption: {corr.upper()}")
        results[corr] = {}
        for method in methods:
            # We run 100 cycles of interleaved batches (300 batches in total)
            acc = evaluate_tta(
                method_name=method,
                experts=experts,
                base_params=base_params,
                task_vectors=task_vectors,
                test_loaders=test_loaders,
                fisher_dict=fisher_dict,
                corruption_type=corr,
                num_cycles=100,
                ewc_lambda=50.0
            )
            results[corr][method] = acc
            print(f"  [{method.upper()}] Accuracy: {acc:.2f}%")
            
    # Save the evaluation results to a python file or report
    import json
    with open('./experts/evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("\nSaved evaluation results to ./experts/evaluation_results.json")
    
    # Print clean Markdown Table
    print("\n### Evaluation Results Summary Table ###\n")
    headers = ["Method", "Clean", "Gaussian Noise", "Gaussian Blur", "Contrast"]
    print(f"| {' | '.join(headers)} |")
    print(f"| {' | '.join(['---']*len(headers))} |")
    for method in methods:
        row = [method.upper()]
        for corr in corruptions:
            row.append(f"{results[corr][method]:.2f}%")
        print(f"| {' | '.join(row)} |")
