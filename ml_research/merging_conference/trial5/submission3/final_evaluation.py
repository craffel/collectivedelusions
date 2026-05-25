import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms
import numpy as np
import copy
import os

from models import ResNetEncoder, ClassificationHead
from merge_eval import TestTimeModelMerger, get_test_streams

class FinalEvaluator(TestTimeModelMerger):
    def run_eval(self, stream, method, lr_base, beta1=0.9, beta2=0.99, eps=1e-8):
        # Initialize merging coefficients
        lambdas_raw = torch.ones(len(self.parameter_names), 2, device=self.device) * 0.5
        lambdas_raw.requires_grad = True
        
        # Standard moving averages
        m = torch.zeros_like(lambdas_raw)
        v = torch.zeros_like(lambdas_raw)
        
        # Task-conditioned moving averages (2 tasks)
        m_tc = torch.zeros(2, len(self.parameter_names), 2, device=self.device)
        v_tc = torch.zeros(2, len(self.parameter_names), 2, device=self.device)
        task_steps = [0, 0]
        
        pre_accuracies = []
        task_accuracies = {0: [], 1: []}
        
        for step, (images, labels, task_id) in enumerate(stream):
            images, labels = images.to(self.device), labels.to(self.device)
            head = self.heads[task_id]
            
            # 1. EVALUATION (PRE-ADAPTATION ACCURACY)
            with torch.no_grad():
                merged_params = {}
                for i, name in enumerate(self.parameter_names):
                    l1 = torch.clamp(lambdas_raw[i, 0], 0.0, 1.0)
                    l2 = torch.clamp(lambdas_raw[i, 1], 0.0, 1.0)
                    merged_params[name] = self.base_params[name] + l1 * self.task_vectors[0][name] + l2 * self.task_vectors[1][name]
                
                features = torch.func.functional_call(self.base_encoder, merged_params, images)
                outputs = head(features)
                _, predicted = outputs.max(1)
                correct = predicted.eq(labels).sum().item()
                acc = 100.0 * correct / len(labels)
                pre_accuracies.append(acc)
                task_accuracies[task_id].append(acc)
                
            # 2. ADAPTATION (IF NOT STATIC)
            if method != 'Static':
                merged_params = {}
                for i, name in enumerate(self.parameter_names):
                    l1 = torch.clamp(lambdas_raw[i, 0], 0.0, 1.0)
                    l2 = torch.clamp(lambdas_raw[i, 1], 0.0, 1.0)
                    merged_params[name] = self.base_params[name] + l1 * self.task_vectors[0][name] + l2 * self.task_vectors[1][name]
                
                features = torch.func.functional_call(self.base_encoder, merged_params, images)
                outputs = head(features)
                
                probs = F.softmax(outputs, dim=1)
                log_probs = F.log_softmax(outputs, dim=1)
                entropy_per_sample = -(probs * log_probs).sum(dim=1)
                
                if method == 'AdaSNR_Adam_TC_CG':
                    mask = entropy_per_sample < 0.40
                    if mask.sum() > 0:
                        entropy_loss = entropy_per_sample[mask].mean()
                    else:
                        entropy_loss = None
                else:
                    entropy_loss = entropy_per_sample.mean()
                
                if entropy_loss is not None:
                    if lambdas_raw.grad is not None:
                        lambdas_raw.grad.zero_()
                    entropy_loss.backward()
                    grad = lambdas_raw.grad.data.clone().detach()
                else:
                    grad = None
                
                t_step = step + 1
                
                if grad is not None:
                    if method == 'Uniform':
                        lambdas_raw.data -= lr_base * grad
                        
                    elif method == 'LFWA':
                        scaled_grad = torch.zeros_like(grad)
                        for i, name in enumerate(self.parameter_names):
                            f_val = self.layer_fisher[name]
                            scale_factor = 1.0 / (f_val + 1e-5)
                            scaled_grad[i] = scale_factor * grad[i]
                        lambdas_raw.data -= lr_base * scaled_grad
                        
                    elif method == 'AdaSNR_Standard':
                        m = beta1 * m + (1.0 - beta1) * grad
                        v = beta2 * v + (1.0 - beta2) * grad.pow(2)
                        m_hat = m / (1.0 - beta1 ** t_step)
                        v_hat = v / (1.0 - beta2 ** t_step)
                        snr = m_hat.pow(2) / (v_hat + eps)
                        alpha0 = 0.1
                        scale = alpha0 + (1.0 - alpha0) * (1.0 - torch.exp(-snr))
                        lambdas_raw.data -= lr_base * scale * grad
                        
                    elif method in ['AdaSNR_Adam_TC', 'AdaSNR_Adam_TC_CG']:
                        # This is our Task-Conditioned Adam (which represents Task-Conditioned SNR-Adam)
                        task_steps[task_id] += 1
                        curr_t = task_steps[task_id]
                        
                        m_tc[task_id] = beta1 * m_tc[task_id] + (1.0 - beta1) * grad
                        v_tc[task_id] = beta2 * v_tc[task_id] + (1.0 - beta2) * grad.pow(2)
                        
                        m_hat = m_tc[task_id] / (1.0 - beta1 ** curr_t)
                        v_hat = v_tc[task_id] / (1.0 - beta2 ** curr_t)
                        
                        # Gradient SNR / Adam update
                        update = m_hat / (torch.sqrt(v_hat) + eps)
                        lambdas_raw.data -= lr_base * update
                    
                # Clamp merging coefficients back to [0, 1]
                lambdas_raw.data.clamp_(0.0, 1.0)
                
        avg_acc = np.mean(pre_accuracies)
        cifar_acc = np.mean(task_accuracies[0])
        svhn_acc = np.mean(task_accuracies[1])
        
        return avg_acc, cifar_acc, svhn_acc

def get_noisy_stream(stream, noise_std=0.15, seed=42):
    torch.manual_seed(seed)
    noisy_stream = []
    for images, labels, task_id in stream:
        noise = torch.randn_like(images) * noise_std
        noisy_images = torch.clamp(images + noise, -3.0, 3.0)
        noisy_stream.append((noisy_images, labels, task_id))
    return noisy_stream

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running final evaluation on device: {device}")
    
    torch.backends.cudnn.enabled = False
    alt_stream, seq_stream = get_test_streams(batch_size=64)
    noisy_stream = get_noisy_stream(alt_stream, noise_std=0.15)
    
    evaluator = FinalEvaluator(device)
    
    # Compute and normalize Fisher Information for LFWA
    layer_fisher = evaluator.compute_fisher_information(num_samples=256)
    fisher_vals = np.array(list(layer_fisher.values()))
    mean_fisher = fisher_vals.mean()
    for name in evaluator.parameter_names:
        evaluator.layer_fisher[name] /= mean_fisher
        
    # Standard Best Configurations
    configs = {
        'Static (Task Arithmetic)': {
            'method': 'Static',
            'lr_alt': 0.0,
            'lr_seq': 0.0,
            'lr_noise': 0.0
        },
        'Uniform TTA (SGD)': {
            'method': 'Uniform',
            'lr_alt': 0.10,
            'lr_seq': 0.20,
            'lr_noise': 0.10
        },
        'LFWA TTA (SGD)': {
            'method': 'LFWA',
            'lr_alt': 0.01,
            'lr_seq': 0.05,
            'lr_noise': 0.01
        },
        'AdaSNR TTA (SGD-Standard)': {
            'method': 'AdaSNR_Standard',
            'lr_alt': 1.00,
            'lr_seq': 0.50,
            'lr_noise': 1.00
        },
        'AdaSNR-Adam-TC (Ours)': {
            'method': 'AdaSNR_Adam_TC',
            'lr_alt': 0.02,
            'lr_seq': 0.02,
            'lr_noise': 0.02
        },
        'AdaSNR-Adam-TC-CG (Ours)': {
            'method': 'AdaSNR_Adam_TC_CG',
            'lr_alt': 0.02,
            'lr_seq': 0.02,
            'lr_noise': 0.02
        }
    }
    
    results = {}
    
    print("\nEvaluating all methods...")
    for name, cfg in configs.items():
        print(f"Evaluating {name}...")
        avg_alt, cifar_alt, svhn_alt = evaluator.run_eval(alt_stream, cfg['method'], cfg['lr_alt'])
        avg_seq, cifar_seq, svhn_seq = evaluator.run_eval(seq_stream, cfg['method'], cfg['lr_seq'])
        avg_noise, cifar_noise, svhn_noise = evaluator.run_eval(noisy_stream, cfg['method'], cfg['lr_noise'])
        
        results[name] = {
            'alt': (avg_alt, cifar_alt, svhn_alt, cfg['lr_alt']),
            'seq': (avg_seq, cifar_seq, svhn_seq, cfg['lr_seq']),
            'noise': (avg_noise, cifar_noise, svhn_noise, cfg['lr_noise'])
        }
        
    print("\n" + "="*140)
    print(f"{'Method':<30} | {'Alternating Stream (Best)':<34} | {'Block-Sequential Stream (Best)':<34} | {'Noisy OOD Stream (Best)':<34}")
    print(f"{'':<30} | {'Avg Acc (LR)':<14} | {'CIFAR / SVHN':<16} | {'Avg Acc (LR)':<14} | {'CIFAR / SVHN':<16} | {'Avg Acc (LR)':<14} | {'CIFAR / SVHN':<16}")
    print("="*140)
    
    for name in configs.keys():
        alt_avg, alt_c, alt_s, alt_lr = results[name]['alt']
        seq_avg, seq_c, seq_s, seq_lr = results[name]['seq']
        noise_avg, noise_c, noise_s, noise_lr = results[name]['noise']
        
        alt_lr_str = f"({alt_lr:.2f})" if name != 'Static (Task Arithmetic)' else "(-)"
        seq_lr_str = f"({seq_lr:.2f})" if name != 'Static (Task Arithmetic)' else "(-)"
        noise_lr_str = f"({noise_lr:.2f})" if name != 'Static (Task Arithmetic)' else "(-)"
        
        print(f"{name:<30} | {alt_avg:.2f}% {alt_lr_str:<8} | {alt_c:.1f}% / {alt_s:.1f}% | {seq_avg:.2f}% {seq_lr_str:<8} | {seq_c:.1f}% / {seq_s:.1f}% | {noise_avg:.2f}% {noise_lr_str:<8} | {noise_c:.1f}% / {noise_s:.1f}%")
    print("="*140)
    
    with open("results_final.txt", "w") as f:
        f.write("Final Comparative Evaluation Results:\n\n")
        f.write("="*140 + "\n")
        f.write(f"{'Method':<30} | {'Alternating Stream (Best)':<34} | {'Block-Sequential Stream (Best)':<34} | {'Noisy OOD Stream (Best)':<34}\n")
        f.write(f"{'':<30} | {'Avg Acc (LR)':<14} | {'CIFAR / SVHN':<16} | {'Avg Acc (LR)':<14} | {'CIFAR / SVHN':<16} | {'Avg Acc (LR)':<14} | {'CIFAR / SVHN':<16}\n")
        f.write("="*140 + "\n")
        for name in configs.keys():
            alt_avg, alt_c, alt_s, alt_lr = results[name]['alt']
            seq_avg, seq_c, seq_s, seq_lr = results[name]['seq']
            noise_avg, noise_c, noise_s, noise_lr = results[name]['noise']
            
            alt_lr_str = f"({alt_lr:.2f})" if name != 'Static (Task Arithmetic)' else "(-)"
            seq_lr_str = f"({seq_lr:.2f})" if name != 'Static (Task Arithmetic)' else "(-)"
            noise_lr_str = f"({noise_lr:.2f})" if name != 'Static (Task Arithmetic)' else "(-)"
            
            f.write(f"{name:<30} | {alt_avg:.2f}% {alt_lr_str:<8} | {alt_c:.1f}% / {alt_s:.1f}% | {seq_avg:.2f}% {seq_lr_str:<8} | {seq_c:.1f}% / {seq_s:.1f}% | {noise_avg:.2f}% {noise_lr_str:<8} | {noise_c:.1f}% / {noise_s:.1f}%\n")
        f.write("="*140 + "\n")
        
    print("\nResults saved to results_final.txt!")

if __name__ == "__main__":
    main()
