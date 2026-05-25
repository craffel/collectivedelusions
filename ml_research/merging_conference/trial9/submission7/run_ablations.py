import torch
import numpy as np
import json
from eval_stream import Evaluators, generate_stream, run_evaluation, SimpleCNN

class AblationEvaluators(Evaluators):
    def adasim_no_denoising(self, images, labels, batch_idx, expert0, expert1, reset=False, device="cpu"):
        if reset:
            self.w_global = 0.0
            self.deltas = {k: torch.tensor(0.0, device=device, requires_grad=True) for k, p in expert0.named_parameters() if p.requires_grad}
            self.running_g2 = {k: torch.tensor(1.0, device=device) for k, p in expert0.named_parameters() if p.requires_grad}
            self.smoothed_gap = None
            return None
            
        # Estimate noise level
        sigma_est = estimate_noise_level_images(images)
        
        # 2. Extract feature activations and measure sparsity (but force denoising threshold to 0)
        theta_thresh = 0.0
        
        expert0.eval()
        expert1.eval()
        with torch.no_grad():
            out0 = expert0(images)
            out1 = expert1(images)
            p0 = torch.softmax(out0, dim=-1)
            p1 = torch.softmax(out1, dim=-1)
            h0 = -torch.sum(p0 * torch.log_softmax(out0, dim=-1), dim=-1).mean().item()
            h1 = -torch.sum(p1 * torch.log_softmax(out1, dim=-1), dim=-1).mean().item()
            
        gap = abs(h0 - h1)
        if self.smoothed_gap is None:
            self.smoothed_gap = gap
        else:
            self.smoothed_gap = 0.9 * self.smoothed_gap + 0.1 * gap
            
        # Keep adaptive temperature scaling
        tau = self.smoothed_gap / 3.0 + 150.0 * (1.0 + 2.0 * sigma_est)
        
        w0_raw = np.exp(-h0 / tau)
        w1_raw = np.exp(-h1 / tau)
        w0 = w0_raw / (w0_raw + w1_raw)
        
        w_global_tensor = torch.tensor(self.w_global, device=device, requires_grad=True)
        deltas = {k: torch.tensor(self.deltas[k].item(), device=device, requires_grad=True) for k in self.deltas.keys()}
        
        optimizer_w = torch.optim.SGD([w_global_tensor], lr=0.05)
        
        for step in range(5):
            lambda_dict = {k: torch.sigmoid(w_global_tensor + deltas[k]) for k in deltas.keys()}
            model = SimpleCNN(use_cosface=expert0.use_cosface).to(device)
            merged_weights = merge_weights_local(expert0.state_dict(), expert1.state_dict(), lambda_dict)
            model.load_state_dict(merged_weights)
            
            mean_lam = torch.stack(list(lambda_dict.values())).mean().item()
            fuse_bn_statistics_local(model, expert0, expert1, mean_lam, 1.0 - mean_lam)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=-1)
            l_entropy = -torch.sum(probs * torch.log_softmax(outputs, dim=-1), dim=-1).mean()
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
            merged_weights = merge_weights_local(expert0.state_dict(), expert1.state_dict(), lambda_dict)
            model.load_state_dict(merged_weights)
            mean_lam = torch.stack(list(lambda_dict.values())).mean().item()
            fuse_bn_statistics_local(model, expert0, expert1, mean_lam, 1.0 - mean_lam)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).sum().item()
            total = labels.size(0)
        return correct, total

    def adasim_no_adaptive_temp(self, images, labels, batch_idx, expert0, expert1, reset=False, device="cpu"):
        if reset:
            self.w_global = 0.0
            self.deltas = {k: torch.tensor(0.0, device=device, requires_grad=True) for k, p in expert0.named_parameters() if p.requires_grad}
            self.running_g2 = {k: torch.tensor(1.0, device=device) for k, p in expert0.named_parameters() if p.requires_grad}
            self.smoothed_gap = None
            return None
            
        # Estimate noise level
        sigma_est = estimate_noise_level_images(images)
        
        # Measure feature sparsity
        expert0.eval()
        expert1.eval()
        with torch.no_grad():
            f0 = expert0.get_features(images)
            f1 = expert1.get_features(images)
            sparsity0 = (f0.abs() < 0.1).float().mean().item()
            sparsity1 = (f1.abs() < 0.1).float().mean().item()
            avg_sparsity = 0.5 * (sparsity0 + sparsity1)
            
        # Apply soft-thresholding feature denoising
        theta_thresh = 0.5 * sigma_est if avg_sparsity > 0.4 else 0.0
        
        with torch.no_grad():
            if theta_thresh > 0:
                f0_denoised = torch.sign(f0) * torch.clamp(f0.abs() - theta_thresh, min=0.0)
                f1_denoised = torch.sign(f1) * torch.clamp(f1.abs() - theta_thresh, min=0.0)
                if expert0.use_cosface:
                    out0 = torch.nn.functional.linear(torch.nn.functional.normalize(f0_denoised), torch.nn.functional.normalize(expert0.weight)) * expert0.s
                    out1 = torch.nn.functional.linear(torch.nn.functional.normalize(f1_denoised), torch.nn.functional.normalize(expert1.weight)) * expert1.s
                else:
                    out0 = expert0.fc2(f0_denoised)
                    out1 = expert1.fc2(f1_denoised)
            else:
                out0 = expert0(images)
                out1 = expert1(images)
                
            p0 = torch.softmax(out0, dim=-1)
            p1 = torch.softmax(out1, dim=-1)
            h0 = -torch.sum(p0 * torch.log_softmax(out0, dim=-1), dim=-1).mean().item()
            h1 = -torch.sum(p1 * torch.log_softmax(out1, dim=-1), dim=-1).mean().item()
            
        gap = abs(h0 - h1)
        if self.smoothed_gap is None:
            self.smoothed_gap = gap
        else:
            self.smoothed_gap = 0.9 * self.smoothed_gap + 0.1 * gap
            
        # Use standard non-adaptive temperature scaling (no noise scaling)
        tau = self.smoothed_gap / 3.0 + 150.0
        
        w0_raw = np.exp(-h0 / tau)
        w1_raw = np.exp(-h1 / tau)
        w0 = w0_raw / (w0_raw + w1_raw)
        
        w_global_tensor = torch.tensor(self.w_global, device=device, requires_grad=True)
        deltas = {k: torch.tensor(self.deltas[k].item(), device=device, requires_grad=True) for k in self.deltas.keys()}
        
        optimizer_w = torch.optim.SGD([w_global_tensor], lr=0.05)
        
        for step in range(5):
            lambda_dict = {k: torch.sigmoid(w_global_tensor + deltas[k]) for k in deltas.keys()}
            model = SimpleCNN(use_cosface=expert0.use_cosface).to(device)
            merged_weights = merge_weights_local(expert0.state_dict(), expert1.state_dict(), lambda_dict)
            model.load_state_dict(merged_weights)
            
            mean_lam = torch.stack(list(lambda_dict.values())).mean().item()
            fuse_bn_statistics_local(model, expert0, expert1, mean_lam, 1.0 - mean_lam)
            
            if theta_thresh > 0:
                features = model.get_features(images)
                features_denoised = torch.sign(features) * torch.clamp(features.abs() - theta_thresh, min=0.0)
                if model.use_cosface:
                    outputs = torch.nn.functional.linear(torch.nn.functional.normalize(features_denoised), torch.nn.functional.normalize(model.weight)) * model.s
                else:
                    outputs = model.fc2(features_denoised)
            else:
                outputs = model(images)
                
            probs = torch.softmax(outputs, dim=-1)
            l_entropy = -torch.sum(probs * torch.log_softmax(outputs, dim=-1), dim=-1).mean()
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
            merged_weights = merge_weights_local(expert0.state_dict(), expert1.state_dict(), lambda_dict)
            model.load_state_dict(merged_weights)
            mean_lam = torch.stack(list(lambda_dict.values())).mean().item()
            fuse_bn_statistics_local(model, expert0, expert1, mean_lam, 1.0 - mean_lam)
            
            if theta_thresh > 0:
                features = model.get_features(images)
                features_denoised = torch.sign(features) * torch.clamp(features.abs() - theta_thresh, min=0.0)
                if model.use_cosface:
                    outputs = torch.nn.functional.linear(torch.nn.functional.normalize(features_denoised), torch.nn.functional.normalize(model.weight)) * model.s
                else:
                    outputs = model.fc2(features_denoised)
            else:
                outputs = model(images)
                
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).sum().item()
            total = labels.size(0)
        return correct, total

def estimate_noise_level_images(images):
    kernel = torch.tensor([[-1., -1., -1.],
                           [-1.,  8., -1.],
                           [-1., -1., -1.]], device=images.device).view(1, 1, 3, 3)
    kernel = kernel / 8.0
    residual = torch.nn.functional.conv2d(images, kernel, padding=1)
    return residual.std().item()

def merge_weights_local(m0, m1, lambdas):
    merged = {}
    for k in m0.keys():
        if k in lambdas:
            merged[k] = lambdas[k] * m0[k] + (1.0 - lambdas[k]) * m1[k]
        else:
            merged[k] = 0.5 * m0[k] + 0.5 * m1[k]
    return merged

def fuse_bn_statistics_local(model, exp0, exp1, w0, w1):
    with torch.no_grad():
        for (name, module), (_, m0), (_, m1) in zip(model.named_modules(), exp0.named_modules(), exp1.named_modules()):
            if isinstance(module, torch.nn.BatchNorm2d) or isinstance(module, torch.nn.BatchNorm1d):
                module.running_mean.copy_(w0 * m0.running_mean + w1 * m1.running_mean)
                mean_fused = module.running_mean
                var_fused = w0 * (m0.running_var + (m0.running_mean - mean_fused)**2) + \
                            w1 * (m1.running_var + (m1.running_mean - mean_fused)**2)
                module.running_var.copy_(var_fused)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running ablations on device: {device}")
    stream_batches = generate_stream(device=device)
    
    expert0_std = SimpleCNN(use_cosface=False).to(device)
    expert1_std = SimpleCNN(use_cosface=False).to(device)
    expert0_std.load_state_dict(torch.load("models/mnist_standard.pt", map_location=device))
    expert1_std.load_state_dict(torch.load("models/fashionmnist_standard.pt", map_location=device))
    
    evals = AblationEvaluators()
    
    print("\nEvaluating: AdaSim-CoMerge w/o Denoising")
    res_no_denoising = run_evaluation("AdaSim-CoMerge w/o Denoising", evals.adasim_no_denoising, stream_batches, expert0_std, expert1_std, device=device)
    
    print("\nEvaluating: AdaSim-CoMerge w/o Adaptive Temp")
    res_no_temp = run_evaluation("AdaSim-CoMerge w/o Adaptive Temp", evals.adasim_no_adaptive_temp, stream_batches, expert0_std, expert1_std, device=device)
    
    with open("ablations.json", "w") as f:
        json.dump({"no_denoising": res_no_denoising, "no_temp": res_no_temp}, f, indent=4)
    print("\nAblation results saved to ablations.json")
