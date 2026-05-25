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

class TestTimeModelMergerAdam(TestTimeModelMerger):
    def evaluate_stream_variant(self, stream, variant_name, lr_base=0.1, beta1=0.9, beta2=0.99, eps=1e-8):
        lambdas_raw = torch.ones(len(self.parameter_names), 2, device=self.device) * 0.5
        lambdas_raw.requires_grad = True
        
        # Standard Adam moving averages
        m = torch.zeros_like(lambdas_raw)
        v = torch.zeros_like(lambdas_raw)
        
        # Task-conditioned Adam moving averages
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
            entropy_loss = -(probs * log_probs).sum(dim=1).mean()
            
            if lambdas_raw.grad is not None:
                lambdas_raw.grad.zero_()
            entropy_loss.backward()
            
            grad = lambdas_raw.grad.data.clone().detach()
            
            t_step = step + 1
            
            if variant_name == 'Adam_Standard':
                m = beta1 * m + (1.0 - beta1) * grad
                v = beta2 * v + (1.0 - beta2) * grad.pow(2)
                m_hat = m / (1.0 - beta1 ** t_step)
                v_hat = v / (1.0 - beta2 ** t_step)
                
                update = m_hat / (torch.sqrt(v_hat) + eps)
                lambdas_raw.data -= lr_base * update
                
            elif variant_name == 'Adam_TC':
                task_steps[task_id] += 1
                curr_t = task_steps[task_id]
                
                m_tc[task_id] = beta1 * m_tc[task_id] + (1.0 - beta1) * grad
                v_tc[task_id] = beta2 * v_tc[task_id] + (1.0 - beta2) * grad.pow(2)
                
                m_hat = m_tc[task_id] / (1.0 - beta1 ** curr_t)
                v_hat = v_tc[task_id] / (1.0 - beta2 ** curr_t)
                
                update = m_hat / (torch.sqrt(v_hat) + eps)
                lambdas_raw.data -= lr_base * update
                
            elif variant_name == 'AdaSNR_Adam_Standard':
                # Adam update scaled by standard SNR
                m = beta1 * m + (1.0 - beta1) * grad
                v = beta2 * v + (1.0 - beta2) * grad.pow(2)
                m_hat = m / (1.0 - beta1 ** t_step)
                v_hat = v / (1.0 - beta2 ** t_step)
                
                # Standard SNR
                snr = m_hat.pow(2) / (v_hat + eps)
                alpha0 = 0.1
                scale = alpha0 + (1.0 - alpha0) * (1.0 - torch.exp(-snr))
                
                update = m_hat / (torch.sqrt(v_hat) + eps)
                lambdas_raw.data -= lr_base * scale * update
                
            elif variant_name == 'AdaSNR_Adam_TC':
                # Adam update scaled by Task-conditioned SNR
                task_steps[task_id] += 1
                curr_t = task_steps[task_id]
                
                m_tc[task_id] = beta1 * m_tc[task_id] + (1.0 - beta1) * grad
                v_tc[task_id] = beta2 * v_tc[task_id] + (1.0 - beta2) * grad.pow(2)
                
                m_hat = m_tc[task_id] / (1.0 - beta1 ** curr_t)
                v_hat = v_tc[task_id] / (1.0 - beta2 ** curr_t)
                
                # TC SNR
                snr = m_hat.pow(2) / (v_hat + eps)
                alpha0 = 0.1
                scale = alpha0 + (1.0 - alpha0) * (1.0 - torch.exp(-snr))
                
                update = m_hat / (torch.sqrt(v_hat) + eps)
                lambdas_raw.data -= lr_base * scale * update
                
            elif variant_name == 'AdaSNR_Adam_TC_Wide':
                # Adam update scaled by Task-conditioned Wide SNR
                task_steps[task_id] += 1
                curr_t = task_steps[task_id]
                
                m_tc[task_id] = beta1 * m_tc[task_id] + (1.0 - beta1) * grad
                v_tc[task_id] = beta2 * v_tc[task_id] + (1.0 - beta2) * grad.pow(2)
                
                m_hat = m_tc[task_id] / (1.0 - beta1 ** curr_t)
                v_hat = v_tc[task_id] / (1.0 - beta2 ** curr_t)
                
                std = torch.sqrt(torch.clamp(v_hat - m_hat.pow(2), min=1e-8))
                snr = m_hat.abs() / (std + eps)
                
                scale = 2.0 * torch.sigmoid(snr) - 1.0
                alpha0 = 0.1
                scale = alpha0 + (1.0 - alpha0) * scale
                
                update = m_hat / (torch.sqrt(v_hat) + eps)
                lambdas_raw.data -= lr_base * scale * update
                
            lambdas_raw.data.clamp_(0.0, 1.0)
            
        avg_acc = np.mean(pre_accuracies)
        cifar_acc = np.mean(task_accuracies[0])
        svhn_acc = np.mean(task_accuracies[1])
        
        return avg_acc, cifar_acc, svhn_acc

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Adam variants on device: {device}")
    
    torch.backends.cudnn.enabled = False
    alt_stream, seq_stream = get_test_streams(batch_size=64)
    merger = TestTimeModelMergerAdam(device)
    
    variants = ['Adam_Standard', 'Adam_TC', 'AdaSNR_Adam_Standard', 'AdaSNR_Adam_TC', 'AdaSNR_Adam_TC_Wide']
    lr_grid = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2] # Typically Adam uses smaller learning rates
    
    results = {}
    for var in variants:
        results[var] = {'alt': [], 'seq': []}
        
    for var in variants:
        print(f"\nEvaluating Variant: {var}")
        for lr in lr_grid:
            avg_alt, cifar_alt, svhn_alt = merger.evaluate_stream_variant(alt_stream, var, lr_base=lr)
            avg_seq, cifar_seq, svhn_seq = merger.evaluate_stream_variant(seq_stream, var, lr_base=lr)
            print(f"  LR: {lr:<5} | Alt: {avg_alt:.2f}% | Seq: {avg_seq:.2f}%")
            results[var]['alt'].append((avg_alt, lr))
            results[var]['seq'].append((avg_seq, lr))
            
    print("\n" + "="*80)
    print("BEST RESULTS FOR EACH ADAM VARIANT")
    print("="*80)
    for var in variants:
        best_alt = max(results[var]['alt'], key=lambda x: x[0])
        best_seq = max(results[var]['seq'], key=lambda x: x[0])
        print(f"Variant: {var:<24} | Alt Best: {best_alt[0]:.2f}% (LR={best_alt[1]}) | Seq Best: {best_seq[0]:.2f}% (LR={best_seq[1]})")
    print("="*80)

if __name__ == "__main__":
    main()
