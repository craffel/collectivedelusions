import torch
import torch.nn as nn
import time
import numpy as np
from torchvision.models import resnet18
from cpos_merging import CPOSResNet, QCPOSResNet, HCPOSResNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Benchmarking on device: {device}")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def profile_latency(model, x, num_warmup=10, num_runs=50):
    model.eval()
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(x)
            
    if device.type == "cuda":
        torch.cuda.synchronize()
        
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(x)
            
    if device.type == "cuda":
        torch.cuda.synchronize()
        
    end_time = time.time()
    avg_latency = (end_time - start_time) / num_runs * 1000.0 # in milliseconds
    return avg_latency

def main():
    print("\n=======================================================")
    print("CPOS INFERENCE LATENCY & PARAMETER BENCHMARK")
    print("=======================================================")
    
    # Input batch (size 128, 3 channels, 32x32)
    x = torch.randn(128, 3, 32, 32).to(device)
    
    # 1. Base / Standard Model (WA, TA, TIES, DARE)
    base_model = resnet18(weights=None)
    base_model.fc = nn.Linear(512, 10)
    base_model = base_model.to(device)
    
    base_params = count_parameters(base_model)
    base_latency = profile_latency(base_model, x)
    print(f"Standard ResNet-18 (1 Task/Baseline):")
    print(f"  - Parameters: {base_params:,}")
    print(f"  - Latency: {base_latency:.2f} ms")
    
    # 2. CPOS (2 Tasks)
    m_A = resnet18(weights=None)
    m_A.fc = nn.Linear(512, 10)
    m_B = resnet18(weights=None)
    m_B.fc = nn.Linear(512, 10)
    
    cpos_model = CPOSResNet(m_A, m_B).to(device)
    cpos_params = count_parameters(cpos_model)
    cpos_latency = profile_latency(cpos_model, x)
    print(f"CPOS ResNet-18 (2 Tasks):")
    print(f"  - Parameters: {cpos_params:,} ({cpos_params/base_params:.1f}x)")
    print(f"  - Latency: {cpos_latency:.2f} ms ({cpos_latency/base_latency:.1f}x)")
    
    # 3. Q-CPOS (3 Tasks)
    m_C = resnet18(weights=None)
    m_C.fc = nn.Linear(512, 10)
    
    qcpos_model = QCPOSResNet(m_A, m_B, m_C).to(device)
    qcpos_params = count_parameters(qcpos_model)
    qcpos_latency = profile_latency(qcpos_model, x)
    print(f"Q-CPOS ResNet-18 (3 Tasks):")
    print(f"  - Parameters: {qcpos_params:,} ({qcpos_params/base_params:.1f}x)")
    print(f"  - Latency: {qcpos_latency:.2f} ms ({qcpos_latency/base_latency:.1f}x)")
    
    # 4. H-CPOS (4 Tasks)
    m_D = resnet18(weights=None)
    m_D.fc = nn.Linear(512, 10)
    
    hcpos_model = HCPOSResNet([m_A, m_B, m_C, m_D]).to(device)
    hcpos_params = count_parameters(hcpos_model)
    hcpos_latency = profile_latency(hcpos_model, x)
    print(f"H-CPOS ResNet-18 (4 Tasks):")
    print(f"  - Parameters: {hcpos_params:,} ({hcpos_params/base_params:.1f}x)")
    print(f"  - Latency: {hcpos_latency:.2f} ms ({hcpos_latency/base_latency:.1f}x)")
    
    print("\nResults Markdown Table:")
    print("| Configuration | Active Parameters | Parameter Scaling | Inference Latency (ms) | Latency Scaling |")
    print("|---|---|---|---|---|")
    print(f"| Standard Model (WA/TA/TIES/DARE) | {base_params:,} | 1.0x | {base_latency:.2f} ms | 1.0x |")
    print(f"| CPOS (Ours, 2 Tasks) | {cpos_params:,} | {cpos_params/base_params:.1f}x | {cpos_latency:.2f} ms | {cpos_latency/base_latency:.1f}x |")
    print(f"| Q-CPOS (Ours, 3 Tasks) | {qcpos_params:,} | {qcpos_params/base_params:.1f}x | {qcpos_latency:.2f} ms | {qcpos_latency/base_latency:.1f}x |")
    print(f"| H-CPOS (Ours, 4 Tasks) | {hcpos_params:,} | {hcpos_params/base_params:.1f}x | {hcpos_latency:.2f} ms | {hcpos_latency/base_latency:.1f}x |")

if __name__ == "__main__":
    main()
