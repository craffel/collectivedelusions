import argparse
import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from transformers import ViTForImageClassification
from peft import PeftModel
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import copy
from torch.func import functional_call

# Stateless weight reconstruction helper
def reconstruct_weights(base_model, lora_weights, lambdas):
    """
    reconstruct_weights computes W_merged = W_0 + (alpha/r) * (lambda_1 * B_1 A_1 + lambda_2 * B_2 A_2)
    and returns a parameter dictionary for functional_call.
    """
    param_dict = {}
    alpha = 16
    r = 8
    scale = alpha / r
    
    # We only need to override the q_proj and v_proj weights in each of the 12 attention layers.
    for l in range(12):
        for proj in ["q_proj", "v_proj"]:
            weight_name = f"vit.layers.{l}.attention.{proj}.weight"
            
            # Base frozen weight W_0
            W_0 = base_model.state_dict()[f"vit.layers.{l}.attention.{proj}.weight"]
            
            # Expert 1 LoRA matrices (cifar10)
            A1 = lora_weights["cifar10"][f"base_model.model.vit.layers.{l}.attention.{proj}.lora_A.weight"]
            B1 = lora_weights["cifar10"][f"base_model.model.vit.layers.{l}.attention.{proj}.lora_B.weight"]
            
            # Expert 2 LoRA matrices (svhn)
            A2 = lora_weights["svhn"][f"base_model.model.vit.layers.{l}.attention.{proj}.lora_A.weight"]
            B2 = lora_weights["svhn"][f"base_model.model.vit.layers.{l}.attention.{proj}.lora_B.weight"]
            
            # Compute merged update
            # lora_B is [out_features, r] and lora_A is [r, in_features]
            # B @ A yields [out_features, in_features]
            update1 = torch.matmul(B1, A1)
            update2 = torch.matmul(B2, A2)
            
            # Merge coefficients for query/value projections of this layer
            # There are 12 layers, 2 projections per layer, and 2 experts.
            # So lambdas is of shape [12, 2, 2].
            proj_idx = 0 if proj == "q_proj" else 1
            l1 = lambdas[l, proj_idx, 0]
            l2 = lambdas[l, proj_idx, 1]
            
            W_merged = W_0 + scale * (l1 * update1 + l2 * update2)
            param_dict[weight_name] = W_merged
            
    return param_dict

def translate_edge(x, shift_h, shift_w):
    b, c, h, w = x.shape
    x_aug = x
    if shift_h > 0:
        x_aug = torch.cat([x_aug[:, :, :1, :].expand(-1, -1, shift_h, -1), x_aug[:, :, :-shift_h, :]], dim=2)
    elif shift_h < 0:
        sh = -shift_h
        x_aug = torch.cat([x_aug[:, :, sh:, :], x_aug[:, :, -1:, :].expand(-1, -1, sh, -1)], dim=2)
        
    if shift_w > 0:
        x_aug = torch.cat([x_aug[:, :, :, :1].expand(-1, -1, -1, shift_w), x_aug[:, :, :, :-shift_w]], dim=3)
    elif shift_w < 0:
        sw = -shift_w
        x_aug = torch.cat([x_aug[:, :, :, sw:], x_aug[:, :, :, -1:].expand(-1, -1, -1, sw)], dim=3)
    return x_aug

def augment(x, task):
    """
    On-device self-supervised augmentations: task-aware to prevent breaking semantic digit identities.
    Uses edge-padded translations to avoid unnatural wrap-around artifacts.
    """
    b, c, h, w = x.shape
    if task == "cifar10":
        # Random horizontal flip for CIFAR-10 objects
        flip = torch.rand(b, 1, 1, 1, device=x.device) > 0.5
        x_aug = torch.where(flip, torch.flip(x, [3]), x)
    else:
        # Digit classes in SVHN are NOT horizontally symmetric! Flipped digits are semantically broken.
        x_aug = x.clone()
        
    # Random translation shift with edge padding
    shift_h = torch.randint(-2, 3, (1,)).item()
    shift_w = torch.randint(-2, 3, (1,)).item()
    x_aug = translate_edge(x_aug, shift_h, shift_w)
    return x_aug

def get_dataloaders(batch_size=64):
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    cifar10_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    svhn_test = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform_test)
    
    # Dataloaders for full evaluation
    cifar10_loader = DataLoader(cifar10_test, batch_size=batch_size, shuffle=False, num_workers=4)
    svhn_loader = DataLoader(svhn_test, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return cifar10_test, svhn_test, cifar10_loader, svhn_loader

def main():
    parser = argparse.ArgumentParser(description="Test-Time Adaptation for Model Merging")
    parser.add_argument("--method", type=str, default="s2c_merge", 
                        choices=["static", "adamerging", "standard_tta", "sbf_sat_symerge", "s2c_merge", "s2c_sam"],
                        help="TTA Method")
    parser.add_argument("--rho", type=float, default=0.05, help="SAM perturbation scale")
    parser.add_argument("--lr", type=float, default=0.02, help="TTA learning rate")
    parser.add_argument("--steps", type=int, default=100, help="Number of TTA adaptation steps")
    parser.add_argument("--batch_size", type=int, default=64, help="TTA batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gamma", type=float, default=1.0, help="S2C-Merge consistency weight")
    parser.add_argument("--corruption", type=str, default="none",
                        choices=["none", "gaussian_noise", "brightness", "blur"],
                        help="Corruption to apply to test stream")
    parser.add_argument("--corruption_severity", type=float, default=0.1,
                        help="Severity of corruption")
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running TTA on device: {device} | Method: {args.method}")
    
    # 1. Load data
    cifar10_test, svhn_test, cifar10_loader, svhn_loader = get_dataloaders(args.batch_size)
    
    # 2. Load base model and expert state dicts
    print("Loading base pre-trained model...")
    base_model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=10,
        ignore_mismatched_sizes=True
    ).to(device)
    base_model.eval()
    
    print("Loading expert LoRA state dicts...")
    lora_weights = {}
    classifiers = {}
    
    for task in ["cifar10", "svhn"]:
        checkpoint_path = f"checkpoints/{task}_best"
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Expert model checkpoint for {task} not found at {checkpoint_path}. Please train experts first.")
        
        # Load weights
        weights_file = os.path.join(checkpoint_path, "adapter_model.bin")
        if not os.path.exists(weights_file):
            # SafeTensors format
            from safetensors.torch import load_file
            weights_file = os.path.join(checkpoint_path, "adapter_model.safetensors")
            state_dict = load_file(weights_file)
        else:
            state_dict = torch.load(weights_file, map_location="cpu")
            
        # Extract LoRA weights for q_proj and v_proj
        lora_weights[task] = {k: v.to(device) for k, v in state_dict.items() if "lora" in k}
        
        # Extract classifier weights
        cls_w = state_dict["base_model.model.classifier.weight"].to(device)
        cls_b = state_dict["base_model.model.classifier.bias"].to(device)
        classifiers[task] = {"weight": cls_w, "bias": cls_b}
        
    print("Successfully loaded expert checkpoints.")
    
    # 3. Initialize TTA parameters
    # Lambdas: [12 layers, 2 projections (q, v), 2 experts]
    # Initialized to equal merge 0.5
    lambdas = torch.full((12, 2, 2), 0.5, device=device, requires_grad=True)
    
    # Classifier heads to adapt
    # We clone the original classifiers
    class_heads = {
        "cifar10": {
            "weight": classifiers["cifar10"]["weight"].clone().detach().requires_grad_(True),
            "bias": classifiers["cifar10"]["bias"].clone().detach().requires_grad_(True)
        },
        "svhn": {
            "weight": classifiers["svhn"]["weight"].clone().detach().requires_grad_(True),
            "bias": classifiers["svhn"]["bias"].clone().detach().requires_grad_(True)
        }
    }
    
    # We construct unmerged expert models if needed as teachers
    expert_models = {}
    if args.method in ["standard_tta", "sbf_sat_symerge"]:
        print("Constructing frozen unmerged expert teacher models...")
        for task in ["cifar10", "svhn"]:
            # Reconstruct the exact expert parameters
            teacher_lambdas = torch.zeros((12, 2, 2), device=device)
            task_idx = 0 if task == "cifar10" else 1
            teacher_lambdas[:, :, task_idx] = 1.0
            teacher_params = reconstruct_weights(base_model, lora_weights, teacher_lambdas)
            teacher_params["classifier.weight"] = classifiers[task]["weight"]
            teacher_params["classifier.bias"] = classifiers[task]["bias"]
            expert_models[task] = teacher_params
            
    # Optimizer
    params_to_opt = [lambdas]
    # For s2c_merge, we ONLY optimize the 24 layer-wise merging coefficients 'lambdas' to prevent decision boundary collapse!
    if args.method in ["standard_tta", "sbf_sat_symerge"]:
        params_to_opt.extend([
            class_heads["cifar10"]["weight"], class_heads["cifar10"]["bias"],
            class_heads["svhn"]["weight"], class_heads["svhn"]["bias"]
        ])
        
    # We use the user-specified learning rate directly
    lr = args.lr
    optimizer = optim.Adam(params_to_opt, lr=lr)
    
    # 4. Construct test stream (alternating CIFAR-10 and SVHN batches)
    print("Generating mixed test-time stream...")
    cifar_iter = iter(DataLoader(cifar10_test, batch_size=args.batch_size, shuffle=True))
    svhn_iter = iter(DataLoader(svhn_test, batch_size=args.batch_size, shuffle=True))
    
    # SBF-SAT-SyMerge parameters
    running_fisher = {}
    fisher_beta = 0.9  # Fisher momentum
    
    # TTA adaptation loop
    print(f"Adapting model for {args.steps} steps...")
    for step in range(args.steps):
        # Alternate tasks
        task = "cifar10" if step % 2 == 0 else "svhn"
        
        # Get next batch
        try:
            inputs, _ = next(cifar_iter) if task == "cifar10" else next(svhn_iter)
        except StopIteration:
            # Reinitialize generator
            if task == "cifar10":
                cifar_iter = iter(DataLoader(cifar10_test, batch_size=args.batch_size, shuffle=True))
                inputs, _ = next(cifar_iter)
            else:
                svhn_iter = iter(DataLoader(svhn_test, batch_size=args.batch_size, shuffle=True))
                inputs, _ = next(svhn_iter)
                
        inputs = inputs.to(device)
        
        # Apply corruption to inputs if requested
        if args.corruption != "none":
            if args.corruption == "gaussian_noise":
                inputs = inputs + torch.randn_like(inputs) * args.corruption_severity
            elif args.corruption == "brightness":
                inputs = inputs + args.corruption_severity * 2.0
            elif args.corruption == "blur":
                import torchvision.transforms.functional as TF
                sigma = args.corruption_severity * 3.0 + 0.1
                inputs = TF.gaussian_blur(inputs, [5, 5], [sigma, sigma])
        
        # We define a function to compute loss for standard or perturbed parameters
        def compute_loss(curr_lambdas, curr_head, x):
            # Reconstruct merged weights
            merged_params = reconstruct_weights(base_model, lora_weights, curr_lambdas)
            merged_params["classifier.weight"] = curr_head["weight"]
            merged_params["classifier.bias"] = curr_head["bias"]
            
            # Forward pass
            outputs = functional_call(base_model, merged_params, (x,)).logits
            
            if args.method == "adamerging":
                # Entropy minimization on prediction
                probs = F.softmax(outputs, dim=-1)
                loss = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1).mean()
                return loss, outputs
                
            elif args.method in ["standard_tta", "sbf_sat_symerge"]:
                # Expert-guided self-labeling: compute teacher predictions
                with torch.no_grad():
                    teacher_out = functional_call(base_model, expert_models[task], (x,)).logits
                    teacher_probs = F.softmax(teacher_out, dim=-1)
                
                # KL Divergence from teacher to merged
                probs = F.log_softmax(outputs, dim=-1)
                loss = F.kl_div(probs, teacher_probs, reduction="batchmean")
                return loss, outputs
                
            elif args.method in ["s2c_merge", "s2c_sam"]:
                # S2C-Merge (Our Proposed Method)
                # Prediction entropy of original view
                probs = F.softmax(outputs, dim=-1)
                loss_ent = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1).mean()
                
                # Augmentation consistency (KL divergence on augmented view)
                x_aug = augment(x, task)
                outputs_aug = functional_call(base_model, merged_params, (x_aug,)).logits
                probs_aug = F.log_softmax(outputs_aug, dim=-1)
                loss_const = F.kl_div(probs_aug, probs.detach(), reduction="batchmean")
                
                # Total loss
                gamma = args.gamma
                loss = loss_ent + gamma * loss_const
                return loss, outputs
                
            else:
                return torch.tensor(0.0, device=device), outputs

        # Execute optimization step
        if args.method == "static":
            continue
            
        elif args.method in ["adamerging", "standard_tta", "s2c_merge"]:
            optimizer.zero_grad()
            curr_head = class_heads[task]
            loss, _ = compute_loss(lambdas, curr_head, inputs)
            loss.backward()
            optimizer.step()
            
            # Project lambdas to stay within reasonable bounds (e.g. [0, 1])
            with torch.no_grad():
                lambdas.clamp_(0.0, 1.0)
                
        elif args.method == "s2c_sam":
            optimizer.zero_grad()
            curr_head = class_heads[task]
            
            # 1. Unperturbed Forward & Backward
            loss, _ = compute_loss(lambdas, curr_head, inputs)
            loss.backward()
            
            # Save unperturbed grad
            grad_unperturbed = lambdas.grad.clone().detach()
            
            # 2. Compute SAM Perturbation
            grad_norm = torch.norm(grad_unperturbed) + 1e-12
            epsilon = args.rho * grad_unperturbed / grad_norm
            
            # Perturb lambdas
            with torch.no_grad():
                lambdas_perturbed = lambdas.clone().detach() + epsilon
                lambdas_perturbed.requires_grad = True
                
            # 3. Perturbed Forward & Backward
            optimizer.zero_grad()
            loss_perturbed, _ = compute_loss(lambdas_perturbed, curr_head, inputs)
            loss_perturbed.backward()
            
            # 4. Restore gradient of original lambdas
            lambdas.grad = lambdas_perturbed.grad.clone().detach()
            
            # 5. Optimizer step
            optimizer.step()
            
            # Project lambdas
            with torch.no_grad():
                lambdas.clamp_(0.0, 1.0)
                
        elif args.method == "sbf_sat_symerge":
            # Soft-Bounded Fisher-Guided SAM TTA
            optimizer.zero_grad()
            curr_head = class_heads[task]
            
            # 1. Unperturbed Forward & Backward to get unperturbed gradients
            loss, outputs = compute_loss(lambdas, curr_head, inputs)
            loss.backward()
            
            # Extract active parameters and their gradients
            active_params = [lambdas, curr_head["weight"], curr_head["bias"]]
            unperturbed_grads = [p.grad.clone().detach() if p.grad is not None else torch.zeros_like(p) for p in active_params]
            
            # 2. Update Running Fisher Information
            for idx, p in enumerate(active_params):
                param_id = f"{task}_p_{idx}"
                grad_sq = unperturbed_grads[idx] ** 2
                if param_id not in running_fisher:
                    running_fisher[param_id] = grad_sq
                else:
                    running_fisher[param_id] = fisher_beta * running_fisher[param_id] + (1.0 - fisher_beta) * grad_sq
            
            # 3. Compute Fisher-Guided Perturbations
            perturbed_params = []
            for idx, p in enumerate(active_params):
                param_id = f"{task}_p_{idx}"
                F_i = running_fisher[param_id]
                F_mean = F_i.mean()
                
                # Soft-bounded scaling factor t_i = exp(-F_i / (F_mean + 1e-8))
                t_i = torch.exp(-F_i / (F_mean + 1e-8))
                
                # Scaled unperturbed gradient g_i_perturbed = t_i^2 * g_i
                g_i = unperturbed_grads[idx]
                g_perturbed = (t_i ** 2) * g_i
                
                # Parameter-level norm of scaled gradients
                g_norm = torch.norm(g_perturbed) + 1e-12
                
                # Perturbation epsilon = rho * g_perturbed / g_norm
                epsilon = args.rho * g_perturbed / g_norm
                
                # Apply perturbation and track
                p_perturbed = p.clone().detach() + epsilon
                p_perturbed.requires_grad = True
                perturbed_params.append(p_perturbed)
                
            # 4. Perturbed Forward & Backward to get perturbed gradients
            optimizer.zero_grad()
            lambdas_p = perturbed_params[0]
            curr_head_p = {"weight": perturbed_params[1], "bias": perturbed_params[2]}
            
            loss_p, _ = compute_loss(lambdas_p, curr_head_p, inputs)
            loss_p.backward()
            
            # 5. Restore gradients of original parameters to match the perturbed gradients
            lambdas.grad = lambdas_p.grad.clone().detach()
            curr_head["weight"].grad = curr_head_p["weight"].grad.clone().detach()
            curr_head["bias"].grad = curr_head_p["bias"].grad.clone().detach()
            
            # 6. Optimizer update step
            optimizer.step()
            
            # Project lambdas
            with torch.no_grad():
                lambdas.clamp_(0.0, 1.0)
                
        if (step + 1) % 10 == 0:
            print(f"Step {step+1}/{args.steps} | Loss: {loss.item():.4f}")
            print(f"  Current average lambdas: CIFAR10: {lambdas[:, :, 0].mean().item():.3f} | SVHN: {lambdas[:, :, 1].mean().item():.3f}")

    # 5. Final Evaluation on Full Test Sets
    print("Adaptation complete. Evaluating on full test sets...")
    
    results = {}
    with torch.no_grad():
        for task in ["cifar10", "svhn"]:
            loader = cifar10_loader if task == "cifar10" else svhn_loader
            correct = 0
            total = 0
            
            # Reconstruct final merged model for this task
            merged_params = reconstruct_weights(base_model, lora_weights, lambdas)
            merged_params["classifier.weight"] = class_heads[task]["weight"]
            merged_params["classifier.bias"] = class_heads[task]["bias"]
            
            for inputs, targets in tqdm(loader, desc=f"Evaluating {task}"):
                inputs, targets = inputs.to(device), targets.to(device)
                if args.corruption != "none":
                    if args.corruption == "gaussian_noise":
                        inputs = inputs + torch.randn_like(inputs) * args.corruption_severity
                    elif args.corruption == "brightness":
                        inputs = inputs + args.corruption_severity * 2.0
                    elif args.corruption == "blur":
                        import torchvision.transforms.functional as TF
                        sigma = args.corruption_severity * 3.0 + 0.1
                        inputs = TF.gaussian_blur(inputs, [5, 5], [sigma, sigma])
                outputs = functional_call(base_model, merged_params, (inputs,)).logits
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
            acc = 100. * correct / total
            results[task] = acc
            print(f"Accuracy on {task}: {acc:.2f}%")
            
    avg_acc = (results["cifar10"] + results["svhn"]) / 2.0
    print(f"\nFinal Results | CIFAR10: {results['cifar10']:.2f}% | SVHN: {results['svhn']:.2f}% | Average: {avg_acc:.2f}%")

if __name__ == "__main__":
    main()
