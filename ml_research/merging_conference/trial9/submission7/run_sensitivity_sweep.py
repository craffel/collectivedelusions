import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
from train_experts import SimpleCNN
from eval_stream import (
    generate_stream, 
    estimate_noise_level, 
    merge_weights, 
    fuse_bn_statistics,
    set_seed
)

class SweepEvaluator:
    def __init__(self, denoise_coeff, sparsity_limit):
        self.w_global = 0.0
        self.deltas = {}
        self.running_g2 = {}
        self.smoothed_gap = None
        self.denoise_coeff = denoise_coeff
        self.sparsity_limit = sparsity_limit

    def evaluate_batch(self, images, labels, batch_idx, expert0, expert1, reset=False, device="cpu"):
        if reset:
            self.w_global = 0.0
            self.deltas = {k: torch.tensor(0.0, device=device, requires_grad=True) for k, p in expert0.named_parameters() if p.requires_grad}
            self.running_g2 = {k: torch.tensor(1.0, device=device) for k, p in expert0.named_parameters() if p.requires_grad}
            self.smoothed_gap = None
            return None
            
        # 1. Estimate noise level of the batch
        sigma_est = estimate_noise_level(images)
        
        # 2. Extract feature activations and measure sparsity
        expert0.eval()
        expert1.eval()
        with torch.no_grad():
            f0 = expert0.get_features(images)
            f1 = expert1.get_features(images)
            sparsity0 = (f0.abs() < 0.1).float().mean().item()
            sparsity1 = (f1.abs() < 0.1).float().mean().item()
            avg_sparsity = 0.5 * (sparsity0 + sparsity1)
            
        # 3. Sparsity-Calibrated Denoised Soft-Routing
        theta_thresh = self.denoise_coeff * sigma_est if avg_sparsity > self.sparsity_limit else 0.0
        
        with torch.no_grad():
            if theta_thresh > 0:
                f0_denoised = torch.sign(f0) * torch.clamp(f0.abs() - theta_thresh, min=0.0)
                f1_denoised = torch.sign(f1) * torch.clamp(f1.abs() - theta_thresh, min=0.0)
                if expert0.use_cosface:
                    out0 = F.linear(F.normalize(f0_denoised), F.normalize(expert0.weight)) * expert0.s
                    out1 = F.linear(F.normalize(f1_denoised), F.normalize(expert1.weight)) * expert1.s
                else:
                    out0 = expert0.fc2(f0_denoised)
                    out1 = expert1.fc2(f1_denoised)
            else:
                out0 = expert0(images)
                out1 = expert1(images)
                
            p0 = F.softmax(out0, dim=-1)
            p1 = F.softmax(out1, dim=-1)
            h0 = -torch.sum(p0 * F.log_softmax(out0, dim=-1), dim=-1).mean().item()
            h1 = -torch.sum(p1 * F.log_softmax(out1, dim=-1), dim=-1).mean().item()
            
        gap = abs(h0 - h1)
        if self.smoothed_gap is None:
            self.smoothed_gap = gap
        else:
            self.smoothed_gap = 0.9 * self.smoothed_gap + 0.1 * gap
            
        tau = self.smoothed_gap / 3.0 + 150.0 * (1.0 + 2.0 * sigma_est)
        
        w0_raw = np.exp(-h0 / tau)
        w1_raw = np.exp(-h1 / tau)
        w0 = w0_raw / (w0_raw + w1_raw)
        w1 = 1.0 - w0
        
        w_global_tensor = torch.tensor(self.w_global, device=device, requires_grad=True)
        deltas = {k: torch.tensor(self.deltas[k].item(), device=device, requires_grad=True) for k in self.deltas.keys()}
        
        optimizer_w = torch.optim.SGD([w_global_tensor], lr=0.05)
        
        for step in range(5):
            lambda_dict = {k: torch.sigmoid(w_global_tensor + deltas[k]) for k in deltas.keys()}
            model = SimpleCNN(use_cosface=expert0.use_cosface).to(device)
            merged_weights = merge_weights(expert0.state_dict(), expert1.state_dict(), lambda_dict)
            model.load_state_dict(merged_weights)
            
            mean_lam = torch.stack(list(lambda_dict.values())).mean().item()
            fuse_bn_statistics(model, expert0, expert1, mean_lam, 1.0 - mean_lam)
            
            if theta_thresh > 0:
                features = model.get_features(images)
                features_denoised = torch.sign(features) * torch.clamp(features.abs() - theta_thresh, min=0.0)
                if model.use_cosface:
                    outputs = F.linear(F.normalize(features_denoised), F.normalize(model.weight)) * model.s
                else:
                    outputs = model.fc2(features_denoised)
            else:
                outputs = model(images)
                
            probs = F.softmax(outputs, dim=-1)
            l_entropy = -torch.sum(probs * F.log_softmax(outputs, dim=-1), dim=-1).mean()
            l_kl = 1.5 * (mean_lam - w0)**2
            l_coherence = 0.02 * sum((self.running_g2[k] * (deltas[k]**2)).sum() for k in deltas.keys())
            
            loss = l_entropy + l_kl + l_coherence
            
            optimizer_w.zero_grad()
            for k in deltas.keys():
                if deltas[k].grad is not None:
                    deltas[k].grad.zero_()
                    
            loss.backward()
            
            with torch.no_grad():
                for k in deltas.keys():
                    if deltas[k].grad is not None:
                        self.running_g2[k] = 0.9 * self.running_g2[k] + 0.1 * (deltas[k].grad.data ** 2)
                        
            optimizer_w.step()
            with torch.no_grad():
                for k in deltas.keys():
                    grad_d = deltas[k].grad
                    if grad_d is not None:
                        precond = self.running_g2[k] + 150.0
                        deltas[k] -= 0.05 / precond * grad_d
                        
        self.w_global = w_global_tensor.item()
        self.deltas = {k: d.detach() for k, d in deltas.items()}
        
        with torch.no_grad():
            lambda_dict = {k: torch.sigmoid(w_global_tensor + deltas[k]) for k in deltas.keys()}
            model = SimpleCNN(use_cosface=expert0.use_cosface).to(device)
            merged_weights = merge_weights(expert0.state_dict(), expert1.state_dict(), lambda_dict)
            model.load_state_dict(merged_weights)
            mean_lam = torch.stack(list(lambda_dict.values())).mean().item()
            fuse_bn_statistics(model, expert0, expert1, mean_lam, 1.0 - mean_lam)
            
            if theta_thresh > 0:
                features = model.get_features(images)
                features_denoised = torch.sign(features) * torch.clamp(features.abs() - theta_thresh, min=0.0)
                if model.use_cosface:
                    outputs = F.linear(F.normalize(features_denoised), F.normalize(model.weight)) * model.s
                else:
                    outputs = model.fc2(features_denoised)
            else:
                outputs = model(images)
                
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).sum().item()
            total = labels.size(0)
        return correct, total

def run_eval(evaluator, stream_batches, expert0, expert1, device="cpu"):
    evaluator.evaluate_batch(None, None, None, expert0, expert1, reset=True, device=device)
    segment_accs = {
        "Clean MNIST": [],
        "Noisy MNIST": [],
        "Clean Fashion": [],
        "Noisy Fashion": [],
        "Novel KMNIST": []
    }
    
    for idx, (images, labels, segment_name) in enumerate(stream_batches):
        correct, total = evaluator.evaluate_batch(images, labels, idx, expert0, expert1, reset=False, device=device)
        acc = 100.0 * correct / total
        segment_accs[segment_name].append(acc)
        
    overall = []
    for seg, accs in segment_accs.items():
        overall.append(np.mean(accs))
    return np.mean(overall)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running hyperparameter sweep on device: {device}")
    
    stream_batches = generate_stream(device=device)
    
    expert0_std = SimpleCNN(use_cosface=False).to(device)
    expert1_std = SimpleCNN(use_cosface=False).to(device)
    expert0_std.load_state_dict(torch.load("models/mnist_standard.pt", map_location=device))
    expert1_std.load_state_dict(torch.load("models/fashionmnist_standard.pt", map_location=device))
    
    denoise_coeffs = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    sparsity_limits = [0.2, 0.3, 0.4, 0.5, 0.6]
    
    sweep_grid = {}
    
    for dc in denoise_coeffs:
        sweep_grid[dc] = {}
        for sl in sparsity_limits:
            evaluator = SweepEvaluator(dc, sl)
            acc = run_eval(evaluator, stream_batches, expert0_std, expert1_std, device=device)
            sweep_grid[dc][sl] = acc
            print(f"Coeff: {dc:.2f}, Sparsity: {sl:.1f} -> Overall Acc: {acc:.2f}%")
            
    # Save sweep results to json
    with open("sensitivity_results.json", "w") as f:
        json.dump(sweep_grid, f, indent=4)
    print("\nSensitivity results saved to sensitivity_results.json")

if __name__ == "__main__":
    main()
