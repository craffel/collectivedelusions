import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models import ResNetEncoder, ClassificationHead
from merge_eval import TestTimeModelMerger, get_test_streams

class CoefficientAnalyst(TestTimeModelMerger):
    def run_analysis(self, stream, method, lr_base, beta1=0.9, beta2=0.99, eps=1e-8):
        # Initialize merging coefficients
        lambdas_raw = torch.ones(len(self.parameter_names), 2, device=self.device) * 0.5
        lambdas_raw.requires_grad = True
        
        # Task-conditioned moving averages (2 tasks)
        m_tc = torch.zeros(2, len(self.parameter_names), 2, device=self.device)
        v_tc = torch.zeros(2, len(self.parameter_names), 2, device=self.device)
        task_steps = [0, 0]
        
        # Track evolution of lambdas over time
        evolution = []
        
        for step, (images, labels, task_id) in enumerate(stream):
            images, labels = images.to(self.device), labels.to(self.device)
            head = self.heads[task_id]
            
            # Store current lambdas
            with torch.no_grad():
                l_current = torch.clamp(lambdas_raw, 0.0, 1.0).clone().cpu().numpy()
                evolution.append(l_current)
            
            # Adaptation
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
            
            # Confidence gating
            mask = entropy_per_sample < 0.40
            if mask.sum() > 0:
                entropy_loss = entropy_per_sample[mask].mean()
            else:
                entropy_loss = None
            
            if entropy_loss is not None:
                if lambdas_raw.grad is not None:
                    lambdas_raw.grad.zero_()
                entropy_loss.backward()
                grad = lambdas_raw.grad.data.clone().detach()
            else:
                grad = None
            
            if grad is not None:
                task_steps[task_id] += 1
                curr_t = task_steps[task_id]
                
                m_tc[task_id] = beta1 * m_tc[task_id] + (1.0 - beta1) * grad
                v_tc[task_id] = beta2 * v_tc[task_id] + (1.0 - beta2) * grad.pow(2)
                
                m_hat = m_tc[task_id] / (1.0 - beta1 ** curr_t)
                v_hat = v_tc[task_id] / (1.0 - beta2 ** curr_t)
                
                update = m_hat / (torch.sqrt(v_hat) + eps)
                lambdas_raw.data -= lr_base * update
                
            lambdas_raw.data.clamp_(0.0, 1.0)
            
        final_lambdas = torch.clamp(lambdas_raw, 0.0, 1.0).detach().cpu().numpy()
        return final_lambdas, np.array(evolution)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running coefficient analysis on device: {device}")
    
    torch.backends.cudnn.enabled = False
    alt_stream, seq_stream = get_test_streams(batch_size=64)
    
    analyst = CoefficientAnalyst(device)
    final_lambdas, evolution = analyst.run_analysis(alt_stream, 'AdaSNR_Adam_TC_CG', lr_base=0.02)
    
    # Categorize parameters
    groups = {
        'Group 1 (Conv1/BN1/Layer1)': [],
        'Group 2 (Layer2)': [],
        'Group 3 (Layer3)': [],
        'Group 4 (Layer4)': []
    }
    
    for i, name in enumerate(analyst.parameter_names):
        if 'resnet.conv1' in name or 'resnet.bn1' in name or 'resnet.layer1' in name:
            groups['Group 1 (Conv1/BN1/Layer1)'].append(i)
        elif 'resnet.layer2' in name:
            groups['Group 2 (Layer2)'].append(i)
        elif 'resnet.layer3' in name:
            groups['Group 3 (Layer3)'].append(i)
        elif 'resnet.layer4' in name:
            groups['Group 4 (Layer4)'].append(i)
            
    print("\n" + "="*80)
    print(f"{'Layer Group':<35} | {'CIFAR-10 Coeff (Mean ± Std)':<25} | {'SVHN Coeff (Mean ± Std)':<25}")
    print("="*80)
    
    for group_name, idxs in groups.items():
        if len(idxs) == 0:
            continue
        cifar_vals = final_lambdas[idxs, 0]
        svhn_vals = final_lambdas[idxs, 1]
        
        cifar_mean, cifar_std = np.mean(cifar_vals), np.std(cifar_vals)
        svhn_mean, svhn_std = np.mean(svhn_vals), np.std(svhn_vals)
        
        print(f"{group_name:<35} | {cifar_mean:.4f} ± {cifar_std:.4f} | {svhn_mean:.4f} ± {svhn_std:.4f}")
    print("="*80)
    
    # Let's also print absolute changes from initialization (0.5)
    print("\nMean absolute change from initialization (0.5):")
    print("="*80)
    for group_name, idxs in groups.items():
        if len(idxs) == 0:
            continue
        cifar_diffs = np.abs(final_lambdas[idxs, 0] - 0.5)
        svhn_diffs = np.abs(final_lambdas[idxs, 1] - 0.5)
        mean_diff = np.mean(np.concatenate([cifar_diffs, svhn_diffs]))
        print(f"{group_name:<35} | {mean_diff:.4f}")
    print("="*80)

if __name__ == '__main__':
    main()
