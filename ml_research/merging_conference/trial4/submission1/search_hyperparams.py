import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.transforms.functional import gaussian_blur
from torch.func import functional_call
import copy
import numpy as np

from models import SharedEncoder, ClassificationHead
from eval_tta import get_test_streams, precompute_fisher_priors, apply_corruption, augment_batch, set_seed

def run_eval_custom(lr_logits, lr_heads, durgp_weight, stream, fisher_priors, device):
    set_seed(42)
    
    encoder_mnist = torch.load("encoder_mnist.pth", map_location=device, weights_only=True)
    encoder_fmnist = torch.load("encoder_fmnist.pth", map_location=device, weights_only=True)
    encoder_kmnist = torch.load("encoder_kmnist.pth", map_location=device, weights_only=True)
    expert_encoders = [encoder_mnist, encoder_fmnist, encoder_kmnist]
    param_names = list(encoder_mnist.keys())
    
    heads = {
        0: ClassificationHead().to(device),
        1: ClassificationHead().to(device),
        2: ClassificationHead().to(device)
    }
    heads[0].load_state_dict(torch.load("head_mnist.pth", map_location=device, weights_only=True))
    heads[1].load_state_dict(torch.load("head_fmnist.pth", map_location=device, weights_only=True))
    heads[2].load_state_dict(torch.load("head_kmnist.pth", map_location=device, weights_only=True))
    
    base_encoder = SharedEncoder().to(device)
    num_layers = len(param_names)
    merging_logits = torch.zeros((num_layers, 3), device=device, requires_grad=True)
    
    params_to_opt = []
    params_to_opt.append({"params": [merging_logits], "lr": lr_logits})
    if lr_heads > 0:
        for k in [0, 1, 2]:
            params_to_opt.append({"params": heads[k].parameters(), "lr": lr_heads})
            
    optimizer = torch.optim.Adam(params_to_opt)
    
    total_correct = 0
    total_samples = 0
    
    for step, (x, y, task_idx) in enumerate(stream):
        x = x.to(device)
        y = y.to(device)
        x_corrupted = apply_corruption(x, "noise") # Evaluate on noise corruption as a representative test
        
        head = heads[task_idx]
        
        weights = torch.softmax(merging_logits, dim=1)
        
        merged_params = {}
        for l_idx, name in enumerate(param_names):
            merged_params[name] = (
                weights[l_idx, 0] * expert_encoders[0][name].to(device) +
                weights[l_idx, 1] * expert_encoders[1][name].to(device) +
                weights[l_idx, 2] * expert_encoders[2][name].to(device)
            )
            
        base_encoder.train()
        head.train() if lr_heads > 0 else head.eval()
        
        features = functional_call(base_encoder, merged_params, x_corrupted)
        outputs = head(features)
        probs = F.softmax(outputs, dim=-1)
        
        with torch.no_grad():
            _, predicted = outputs.max(1)
            total_correct += predicted.eq(y).sum().item()
            total_samples += y.size(0)
            
        optimizer.zero_grad()
        ent_loss = -torch.mean(torch.sum(probs * torch.log(probs + 1e-12), dim=-1))
        loss = ent_loss
        
        # S2C consistency loss (KL on augmentations)
        x_aug = augment_batch(x_corrupted, task_idx)
        features_aug = functional_call(base_encoder, merged_params, x_aug)
        outputs_aug = head(features_aug)
        probs_aug = F.softmax(outputs_aug, dim=-1)
        kl_loss = F.kl_div(torch.log(probs_aug + 1e-12), probs.detach(), reduction="batchmean")
        loss += 1.0 * kl_loss
        
        # DURGP regularization
        if lr_heads > 0:
            w_norm = F.normalize(head.linear.weight, p=2, dim=1)
            w0_norm = F.normalize(fisher_priors[task_idx]["init_weight"], p=2, dim=1)
            G = torch.mm(w_norm, w_norm.t())
            G0 = torch.mm(w0_norm, w0_norm.t())
            q = torch.mean(probs, dim=0)
            Q = torch.outer(q, q)
            durgp_loss = torch.sum(Q * (G - G0) ** 2)
            
            H = ent_loss.item()
            gamma_k = max(0, 1.0 - H / np.log(10.0))
            loss += durgp_weight * gamma_k * durgp_loss
            
        loss.backward()
        optimizer.step()
        
    return 100.0 * total_correct / total_samples

def main():
    device = torch.device("cpu")
    seq_stream, alt_stream = get_test_streams(batch_size=64, num_batches_per_task=50)
    fisher_priors = precompute_fisher_priors(device)
    
    print("\n--- Hyperparameter Search for DURGP on Noise Sequential Stream ---")
    
    # Grid search
    lrs_heads = [0.0, 0.001, 0.005, 0.01, 0.05]
    durgp_weights = [0.1, 1.0, 10.0, 100.0]
    
    for lr_h in lrs_heads:
        if lr_h == 0.0:
            # S2C baseline style (frozen heads)
            acc = run_eval_custom(0.005, 0.0, 0.0, seq_stream, fisher_priors, device)
            print(f"LR Heads: {lr_h:.4f} | DURGP Weight: N/A   | Accuracy: {acc:.2f}%")
        else:
            for dw in durgp_weights:
                acc = run_eval_custom(0.005, lr_h, dw, seq_stream, fisher_priors, device)
                print(f"LR Heads: {lr_h:.4f} | DURGP Weight: {dw:5.1f} | Accuracy: {acc:.2f}%")

if __name__ == "__main__":
    main()
