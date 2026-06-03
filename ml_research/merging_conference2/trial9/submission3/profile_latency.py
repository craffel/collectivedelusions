import time
import torch
import torch.nn as nn
from models import ResNet18Backbone, MLPBackbone
from merging import weight_averaging, wcpr_merging, qr_sp_wcpr_merging

def profile_architecture(backbone_class, progenitor_path, expert_paths, arch_name):
    print(f"\nProfiling {arch_name}...")
    progenitor = backbone_class()
    progenitor.load_state_dict(torch.load(progenitor_path, map_location='cpu'))
    
    experts = []
    for path in expert_paths:
        expert = backbone_class()
        expert.load_state_dict(torch.load(path, map_location='cpu'))
        experts.append(expert)
        
    # Profile Weight Averaging
    start_time = time.time()
    _ = weight_averaging(experts)
    wa_time = (time.time() - start_time) * 1000.0
    print(f"  Weight Averaging: {wa_time:.2f} ms")
    
    # Profile WCPR
    start_time = time.time()
    _ = wcpr_merging(experts, progenitor)
    wcpr_time = (time.time() - start_time) * 1000.0
    print(f"  WCPR Merging: {wcpr_time:.2f} ms")
    
    # Profile QR-SP-WCPR (Ours)
    start_time = time.time()
    _ = qr_sp_wcpr_merging(experts, progenitor)
    our_time = (time.time() - start_time) * 1000.0
    print(f"  QR-SP-WCPR Merging (Ours): {our_time:.2f} ms")
    
    return wa_time, wcpr_time, our_time

def main():
    # ResNet-18 experts
    resnet_experts = [
        'checkpoints/resnet18_mnist_backbone.pt',
        'checkpoints/resnet18_fmnist_backbone.pt',
        'checkpoints/resnet18_cifar10_backbone.pt'
    ]
    resnet_prog = 'checkpoints/resnet18_progenitor.pt'
    
    # MLP experts
    mlp_experts = [
        'checkpoints/mlp_mnist_backbone.pt',
        'checkpoints/mlp_fmnist_backbone.pt',
        'checkpoints/mlp_cifar10_backbone.pt'
    ]
    mlp_prog = 'checkpoints/mlp_progenitor.pt'
    
    try:
        r_wa, r_wcpr, r_ours = profile_architecture(ResNet18Backbone, resnet_prog, resnet_experts, "ResNet-18")
    except Exception as e:
        print(f"Error profiling ResNet-18: {e}")
        r_wa, r_wcpr, r_ours = None, None, None
        
    try:
        m_wa, m_wcpr, m_ours = profile_architecture(MLPBackbone, mlp_prog, mlp_experts, "MLP")
    except Exception as e:
        print(f"Error profiling MLP: {e}")
        m_wa, m_wcpr, m_ours = None, None, None
        
    print("\n--- Summary ---")
    if r_wa is not None:
        print(f"ResNet-18 WA: {r_wa:.2f} ms | WCPR: {r_wcpr:.2f} ms | QR-SP-WCPR: {r_ours:.2f} ms")
    if m_wa is not None:
        print(f"MLP       WA: {m_wa:.2f} ms | WCPR: {m_wcpr:.2f} ms | QR-SP-WCPR: {m_ours:.2f} ms")

if __name__ == '__main__':
    main()
