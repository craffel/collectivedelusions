import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from evaluation import load_test_data, build_stream, compute_joint_fisher
from model import CNNEncoder, ClassifierHead, MergedModel

def evaluate_method_ad(method_name, test_batches, experts_sds, base_sd, expert_encoders, expert_heads, joint_fisher, device, lr=0.10, alpha_lfwa=0.5):
    # Initialize merged model
    merged_model = MergedModel(experts_sds, base_sd, num_experts=3).to(device)
    
    # Active expert models (for self-labeling targets)
    experts = []
    heads = []
    for k in range(3):
        enc = CNNEncoder().to(device)
        enc.load_state_dict(expert_encoders[k])
        enc.eval()
        experts.append(enc)
        
        hd = ClassifierHead().to(device)
        hd.load_state_dict(expert_heads[k])
        hd.eval()
        heads.append(hd)
        
    optimizer = optim.SGD([merged_model.raw_lambdas], lr=lr, momentum=0.9)
    
    fisher_scalars = []
    if joint_fisher is not None:
        for name in merged_model.tensor_names:
            fisher_scalars.append(joint_fisher[name])
    fisher_scalars = torch.tensor(fisher_scalars, device=device)
    fisher_multipliers = (fisher_scalars + 1e-8).pow(-alpha_lfwa)
    
    # OPR and AD parameters
    running_ema_loss = 0.0
    beta_ema = 0.90
    
    # Tracking accuracy
    total_samples = 0
    correct_predictions = 0
    
    # Reset history to detect high-frequency switching
    recent_resets = []
    
    for t, (imgs, lbls, active_task_idx) in enumerate(test_batches):
        imgs, lbls = imgs.to(device), lbls.to(device)
        
        # 1. Unmerged, frozen expert k targets
        with torch.no_grad():
            expert_features = experts[active_task_idx](imgs)
            expert_logits = heads[active_task_idx](expert_features)
            expert_probs = F.softmax(expert_logits, dim=1)
            
        # 2. Compute current self-labeling loss
        merged_logits = merged_model(imgs, heads[active_task_idx])
        loss_val = F.kl_div(F.log_softmax(merged_logits, dim=1), expert_probs, reduction='batchmean').item()
        
        # Evaluate on this batch before updating
        with torch.no_grad():
            _, preds = merged_logits.max(1)
            correct_predictions += preds.eq(lbls).sum().item()
            total_samples += imgs.size(0)
            
        if method_name == "static":
            continue
            
        # OPR Reset Check
        has_reset = False
        threshold = 4.0 if "clean" in method_name else 2.5
        if t > 0:
            if loss_val > threshold * running_ema_loss:
                # Reset raw_lambdas to 0 (uniform)
                with torch.no_grad():
                    merged_model.raw_lambdas.zero_()
                optimizer = optim.SGD([merged_model.raw_lambdas], lr=lr, momentum=0.9)
                merged_logits = merged_model(imgs, heads[active_task_idx])
                loss_val = F.kl_div(F.log_softmax(merged_logits, dim=1), expert_probs, reduction='batchmean').item()
                has_reset = True
                if "AD-Merge" in method_name:
                    print(f"DEBUG Reset triggered at t={t}: loss_val={loss_val:.4f}, ema={running_ema_loss:.4f}")
                
        # Update EMA loss
        if t == 0 or has_reset:
            running_ema_loss = loss_val
        else:
            running_ema_loss = beta_ema * running_ema_loss + (1.0 - beta_ema) * loss_val
            
        # Log reset history
        recent_resets.append(1 if has_reset else 0)
        if len(recent_resets) > 5:
            recent_resets.pop(0)
            
        # Dynamic Dampening (AD-Merge)
        current_lr = lr
        if "AD-Merge" in method_name and sum(recent_resets) >= 3:
            current_lr = 0.0
            # If we detect high-frequency switching, keep lambdas at uniform
            with torch.no_grad():
                merged_model.raw_lambdas.zero_()
                
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
            
        if current_lr == 0.0:
            continue
            
        # Zero gradients
        optimizer.zero_grad()
        
        # Class-Specific Gradient Projection
        expert_preds = expert_logits.argmax(dim=1)
        unique_classes = expert_preds.unique()
        
        class_grads = {}
        for cls in unique_classes:
            mask = expert_preds.eq(cls)
            if mask.sum() == 0:
                continue
            cls_merged_logits = merged_model(imgs[mask], heads[active_task_idx])
            cls_expert_probs = expert_probs[mask]
            cls_kl_loss = F.kl_div(F.log_softmax(cls_merged_logits, dim=1), cls_expert_probs, reduction='batchmean')
            
            merged_model.raw_lambdas.grad = None
            cls_kl_loss.backward(retain_graph=True)
            
            if merged_model.raw_lambdas.grad is not None:
                class_grads[cls.item()] = merged_model.raw_lambdas.grad.clone()
                
        # Pairwise projection
        projected_grads = {}
        active_clses = list(class_grads.keys())
        for idx_a, cls_a in enumerate(active_clses):
            g_a = class_grads[cls_a].clone()
            for idx_b, cls_b in enumerate(active_clses):
                if cls_a == cls_b:
                    continue
                g_b = class_grads[cls_b]
                dot_prod = torch.sum(g_a * g_b)
                if dot_prod < 0:
                    norm_b_sq = torch.sum(g_b * g_b) + 1e-8
                    g_a = g_a - (dot_prod / norm_b_sq) * g_b
            projected_grads[cls_a] = g_a
            
        if len(projected_grads) > 0:
            g_final = torch.stack(list(projected_grads.values())).sum(dim=0)
        else:
            g_final = torch.zeros_like(merged_model.raw_lambdas)
            
        # Apply Fisher preconditioning
        if "Fisher" in method_name:
            with torch.no_grad():
                g_final = g_final * fisher_multipliers.unsqueeze(1)
                
        merged_model.raw_lambdas.grad = g_final
        optimizer.step()
        
        with torch.no_grad():
            merged_model.raw_lambdas.clamp_(-10.0, 10.0)
            
    return 100. * correct_predictions / total_samples

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    encoder_paths = [
        "checkpoints/mnist_encoder.pth",
        "checkpoints/fmnist_encoder.pth",
        "checkpoints/kmnist_encoder.pth"
    ]
    head_paths = [
        "checkpoints/mnist_head.pth",
        "checkpoints/fmnist_head.pth",
        "checkpoints/kmnist_head.pth"
    ]
    base_path = "checkpoints/base_encoder.pth"
    
    expert_encoders = []
    expert_heads = []
    experts_sds = []
    for k in range(3):
        enc_sd = torch.load(encoder_paths[k], map_location=device)
        hd_sd = torch.load(head_paths[k], map_location=device)
        expert_encoders.append(enc_sd)
        expert_heads.append(hd_sd)
        experts_sds.append(enc_sd)
    base_sd = torch.load(base_path, map_location=device)
    
    joint_fisher = torch.load("checkpoints/joint_fisher.pth", map_location=device)
    test_data = load_test_data()
    
    streams = ["sequential", "alternating"]
    domains = ["clean", "noise", "blur", "contrast"]
    methods = [
        "static",
        "Fisher-PC-Merge",
        "Fisher-PC-Merge + AD-Merge"
    ]
    
    results = {}
    for stream in streams:
        results[stream] = {}
        for domain in domains:
            results[stream][domain] = {}
            batches = build_stream(test_data, stream_type=stream, corruption_type=domain)
            for method in methods:
                fish = joint_fisher if "Fisher" in method else None
                acc = evaluate_method_ad(
                    method_name=method,
                    test_batches=batches,
                    experts_sds=experts_sds,
                    base_sd=base_sd,
                    expert_encoders=expert_encoders,
                    expert_heads=expert_heads,
                    joint_fisher=fish,
                    device=device,
                    lr=0.10,
                    alpha_lfwa=0.5
                )
                results[stream][domain][method] = acc
                print(f"{stream.upper()} - {domain.upper()} - {method}: {acc:.2f}%")
                
    print("\n--- COMPARATIVE RESULTS ---")
    for stream in streams:
        print(f"\n{stream.upper()} STREAM:")
        for method in methods:
            row = []
            for domain in domains:
                row.append(f"{results[stream][domain][method]:.2f}%")
            avg = sum(results[stream][domain][method] for domain in domains) / 4.0
            print(f"{method:30s} | {' | '.join(row)} | Avg: {avg:.2f}%")

if __name__ == "__main__":
    main()
