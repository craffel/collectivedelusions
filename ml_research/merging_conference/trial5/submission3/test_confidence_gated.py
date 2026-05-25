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

class TestTimeModelMergerConfidenceGated(TestTimeModelMerger):
    def evaluate_stream_cg(self, stream, lr_base=0.02, entropy_threshold=None, beta1=0.9, beta2=0.99, eps=1e-8):
        # Initialize merging coefficients
        lambdas_raw = torch.ones(len(self.parameter_names), 2, device=self.device) * 0.5
        lambdas_raw.requires_grad = True
        
        # Task-conditioned Adam moving averages
        m_tc = torch.zeros(2, len(self.parameter_names), 2, device=self.device)
        v_tc = torch.zeros(2, len(self.parameter_names), 2, device=self.device)
        task_steps = [0, 0]
        
        pre_accuracies = []
        task_accuracies = {0: [], 1: []}
        
        for step, (images, labels, task_id) in enumerate(stream):
            images, labels = images.to(self.device), labels.to(self.device)
            head = self.cifar_head if task_id == 0 else self.svhn_head
            
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
                
            # 2. ADAPTATION
            merged_params = {}
            for i, name in enumerate(self.parameter_names):
                l1 = torch.clamp(lambdas_raw[i, 0], 0.0, 1.0)
                l2 = torch.clamp(lambdas_raw[i, 1], 0.0, 1.0)
                merged_params[name] = self.base_params[name] + l1 * self.task_vectors[0][name] + l2 * self.task_vectors[1][name]
            
            features = torch.func.functional_call(self.base_encoder, merged_params, images)
            outputs = head(features)
            
            probs = F.softmax(outputs, dim=1)
            log_probs = F.log_softmax(outputs, dim=1)
            
            # Compute entropy per sample
            entropy = -(probs * log_probs).sum(dim=1)
            
            if entropy_threshold is not None:
                # Mask out samples with entropy >= threshold (i.e. keep low-entropy, confident samples)
                mask = entropy < entropy_threshold
                num_confident_samples = mask.sum().item()
                if num_confident_samples > 0:
                    entropy_loss = entropy[mask].mean()
                else:
                    entropy_loss = None
            else:
                entropy_loss = entropy.mean()
                num_confident_samples = len(entropy)
                
            if entropy_loss is not None:
                if lambdas_raw.grad is not None:
                    lambdas_raw.grad.zero_()
                entropy_loss.backward()
                
                grad = lambdas_raw.grad.data.clone().detach()
                
                # Perform task-conditioned update
                task_steps[task_id] += 1
                curr_t = task_steps[task_id]
                
                m_tc[task_id] = beta1 * m_tc[task_id] + (1.0 - beta1) * grad
                v_tc[task_id] = beta2 * v_tc[task_id] + (1.0 - beta2) * grad.pow(2)
                
                m_hat = m_tc[task_id] / (1.0 - beta1 ** curr_t)
                v_hat = v_tc[task_id] / (1.0 - beta2 ** curr_t)
                
                update = m_hat / (torch.sqrt(v_hat) + eps)
                lambdas_raw.data -= lr_base * update
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
    print(f"Running Confidence-Gated Sweep on device: {device}")
    
    torch.backends.cudnn.enabled = False
    alt_stream, seq_stream = get_test_streams(batch_size=64)
    noisy_stream = get_noisy_stream(alt_stream, noise_std=0.15)
    
    evaluator = TestTimeModelMergerConfidenceGated(device)
    
    # Let's sweep a set of thresholds.
    # Max entropy for 10 classes is ln(10) ≈ 2.3025.
    thresholds = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, None]
    
    print("\nSweep over entropy thresholds for AdaSNR-Adam-TC (Ours)...")
    print(f"{'Threshold':<12} | {'Alternating Stream':<26} | {'Block-Sequential Stream':<26} | {'Noisy OOD Stream':<26}")
    print(f"{'':<12} | {'Avg Acc':<8} | {'CIFAR/SVHN':<15} | {'Avg Acc':<8} | {'CIFAR/SVHN':<15} | {'Avg Acc':<8} | {'CIFAR/SVHN':<15}")
    print("-"*106)
    
    for thr in thresholds:
        thr_str = f"{thr:.2f}" if thr is not None else "None (Std)"
        
        avg_alt, cifar_alt, svhn_alt = evaluator.evaluate_stream_cg(alt_stream, lr_base=0.02, entropy_threshold=thr)
        avg_seq, cifar_seq, svhn_seq = evaluator.evaluate_stream_cg(seq_stream, lr_base=0.02, entropy_threshold=thr)
        avg_noise, cifar_noise, svhn_noise = evaluator.evaluate_stream_cg(noisy_stream, lr_base=0.02, entropy_threshold=thr)
        
        print(f"{thr_str:<12} | {avg_alt:.2f}%   | {cifar_alt:.1f}%/{svhn_alt:.1f}% | {avg_seq:.2f}%   | {cifar_seq:.1f}%/{svhn_seq:.1f}% | {avg_noise:.2f}%   | {cifar_noise:.1f}%/{svhn_noise:.1f}%")
        
    print("-"*106)

if __name__ == "__main__":
    main()
