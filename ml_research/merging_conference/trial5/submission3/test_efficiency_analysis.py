import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from models import ResNetEncoder, ClassificationHead
from merge_eval import TestTimeModelMerger, get_test_streams

class EfficiencyProfiler(TestTimeModelMerger):
    def profile_method(self, stream, method, lr_base, beta1=0.9, beta2=0.99, eps=1e-8, num_warmup=5, num_eval=15):
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
        
        latencies = []
        
        for step, (images, labels, task_id) in enumerate(stream):
            if step >= num_warmup + num_eval:
                break
                
            images, labels = images.to(self.device), labels.to(self.device)
            head = self.heads[task_id]
            
            # Start timer for adaptation step
            t_start = time.perf_counter()
            
            # 1. Forward pass
            merged_params = {}
            for i, name in enumerate(self.parameter_names):
                l1 = torch.clamp(lambdas_raw[i, 0], 0.0, 1.0)
                l2 = torch.clamp(lambdas_raw[i, 1], 0.0, 1.0)
                merged_params[name] = self.base_params[name] + l1 * self.task_vectors[0][name] + l2 * self.task_vectors[1][name]
            
            features = torch.func.functional_call(self.base_encoder, merged_params, images)
            outputs = head(features)
            
            # Static doesn't optimize/backprop
            if method != 'Static':
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
                        task_id_act = task_id
                        task_steps[task_id_act] += 1
                        curr_t = task_steps[task_id_act]
                        
                        m_tc[task_id_act] = beta1 * m_tc[task_id_act] + (1.0 - beta1) * grad
                        v_tc[task_id_act] = beta2 * v_tc[task_id_act] + (1.0 - beta2) * grad.pow(2)
                        
                        m_hat = m_tc[task_id_act] / (1.0 - beta1 ** curr_t)
                        v_hat = v_tc[task_id_act] / (1.0 - beta2 ** curr_t)
                        
                        update = m_hat / (torch.sqrt(v_hat) + eps)
                        lambdas_raw.data -= lr_base * update
                    
                lambdas_raw.data.clamp_(0.0, 1.0)
            
            t_end = time.perf_counter()
            
            if step >= num_warmup:
                latencies.append((t_end - t_start) * 1000.0) # milliseconds
                
        return np.mean(latencies)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running efficiency profiling on device: {device}")
    
    torch.backends.cudnn.enabled = False
    alt_stream, _ = get_test_streams(batch_size=64)
    
    profiler = EfficiencyProfiler(device)
    
    # Initialize Fisher information as in final_evaluation.py
    layer_fisher = profiler.compute_fisher_information(num_samples=256)
    fisher_vals = np.array(list(layer_fisher.values()))
    mean_fisher = fisher_vals.mean()
    for name in profiler.parameter_names:
        profiler.layer_fisher[name] /= mean_fisher
        
    methods = {
        'Static (Task Arithmetic)': {'method': 'Static', 'lr': 0.0},
        'Uniform TTA (SGD)': {'method': 'Uniform', 'lr': 0.1},
        'LFWA TTA (SGD)': {'method': 'LFWA', 'lr': 0.05},
        'AdaSNR TTA (SGD-Standard)': {'method': 'AdaSNR_Standard', 'lr': 1.0},
        'AdaSNR-Adam-TC (Ours)': {'method': 'AdaSNR_Adam_TC', 'lr': 0.02},
        'AdaSNR-Adam-TC-CG (Ours)': {'method': 'AdaSNR_Adam_TC_CG', 'lr': 0.02}
    }
    
    num_params = len(profiler.parameter_names)
    num_experts = 2
    
    # Track extra variables and footprint (bytes)
    # float32 = 4 bytes
    print("\n" + "="*110)
    print(f"{'Method':<30} | {'Step Latency (ms)':<20} | {'Extra State Variables':<24} | {'Memory Overhead':<18} | {'Dataless?'}")
    print("="*110)
    
    for name, cfg in methods.items():
        avg_latency = profiler.profile_method(alt_stream, cfg['method'], cfg['lr'])
        
        # Determine extra states
        if cfg['method'] == 'Static':
            vars_tracked = 0
            dataless = "Yes"
        elif cfg['method'] == 'Uniform':
            vars_tracked = num_params * num_experts # lambdas
            dataless = "Yes"
        elif cfg['method'] == 'LFWA':
            vars_tracked = num_params * num_experts + num_params # lambdas + layer fisher
            dataless = "No (Needs Calibration Set)"
        elif cfg['method'] == 'AdaSNR_Standard':
            # lambdas + standard m + standard v
            vars_tracked = num_params * num_experts + 2 * (num_params * num_experts)
            dataless = "Yes"
        elif cfg['method'] in ['AdaSNR_Adam_TC', 'AdaSNR_Adam_TC_CG']:
            # lambdas + tc m + tc v for both tasks
            vars_tracked = num_params * num_experts + 2 * num_experts * (num_params * num_experts)
            dataless = "Yes"
            
        mem_bytes = vars_tracked * 4
        if mem_bytes >= 1024 * 1024:
            mem_str = f"{mem_bytes / (1024*1024):.2f} MB"
        elif mem_bytes >= 1024:
            mem_str = f"{mem_bytes / 1024:.2f} KB"
        else:
            mem_str = f"{mem_bytes} B"
            
        print(f"{name:<30} | {avg_latency:>16.2f} ms | {vars_tracked:>24d} | {mem_str:>18} | {dataless}")
    print("="*110)

if __name__ == '__main__':
    main()
