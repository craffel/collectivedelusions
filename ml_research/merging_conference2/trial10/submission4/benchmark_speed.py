import os
import time
import json
import torch
import torch.nn as nn
from torchvision.models import resnet18

# Disable cuDNN
torch.backends.cudnn.enabled = False

# Import functions from evaluate_merging
from evaluate_merging import apply_qcot, apply_qwc, apply_emqc, apply_cwss, apply_cwss_qc

def main():
    print("Initializing benchmark...")
    device = torch.device("cpu") # Benchmark on CPU first to simulate edge or controller nodes
    
    # Initialize mock model checkpoints with ResNet-18 architecture
    # This ensures consistent benchmarking even if some checkpoints are on different devices
    m_init = resnet18()
    m_init.fc = nn.Identity()
    w_init = m_init.state_dict()
    
    w_experts = []
    for i in range(3):
        m_exp = resnet18()
        m_exp.fc = nn.Identity()
        # Add small random perturbations to mock fine-tuned experts
        for param in m_exp.parameters():
            param.data += 0.01 * torch.randn_like(param.data)
        w_experts.append(m_exp.state_dict())
        
    # Standard linear weight average
    w_merged = {}
    for name in w_init.keys():
        if w_init[name].dtype.is_floating_point:
            w_merged[name] = torch.mean(torch.stack([we[name] for we in w_experts]), dim=0)
        else:
            w_merged[name] = w_init[name].clone()
        
    # Benchmark each method
    methods = {
        "Weight Averaging": lambda: {name: w_merged[name].clone() for name in w_init.keys()},
        "QWC (q=0.99)": lambda: apply_qwc(w_init, w_merged, q=0.99),
        "EMQC": lambda: apply_emqc(w_init, w_merged, q_candidates=[0.95, 0.99, 0.999], bits=4),
        "CWSS": lambda: apply_cwss(w_init, w_experts, w_merged),
        "CWSS-QC (q=0.9999)": lambda: apply_cwss_qc(w_init, w_experts, w_merged, q=0.9999),
        "QCOT (C=0.1)": lambda: apply_qcot(w_init, w_experts, w_merged, C=0.1)
    }
    
    results = {}
    # Warm up
    print("Warming up...")
    for name, fn in methods.items():
        if name != "QCOT (C=0.1)": # Skip QCOT warm up if it takes too long
            _ = fn()
            
    print("Running benchmark...")
    for name, fn in methods.items():
        start = time.time()
        _ = fn()
        elapsed = time.time() - start
        results[name] = elapsed
        print(f"{name:20s}: {elapsed*1000:10.2f} ms ({elapsed:8.4f} s)")
        
    # Let's save benchmark results to a json file
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
if __name__ == "__main__":
    main()
