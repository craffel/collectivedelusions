import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import resnet18
from merge_and_evaluate import (
    get_backbone_and_head,
    merge_weight_averaging,
    merge_task_arithmetic,
    merge_u_ipr,
    merge_hns,
    merge_ucpr,
    merge_depth_adaptive_hpr,
    create_full_model,
    calibrate_batchnorm
)

def main():
    print("Initializing dummy models to test merge logic...")
    device = torch.device("cpu")
    
    # Initialize progenitor and two experts with random weights
    prog = resnet18()
    prog.fc = nn.Linear(prog.fc.in_features, 10)
    prog_sd = prog.state_dict()
    
    exp1 = resnet18()
    exp1.fc = nn.Linear(exp1.fc.in_features, 10)
    exp1_sd = exp1.state_dict()
    
    exp2 = resnet18()
    exp2.fc = nn.Linear(exp2.fc.in_features, 10)
    exp2_sd = exp2.state_dict()
    
    # Split
    prog_backbone, _ = get_backbone_and_head(prog_sd)
    exp1_backbone, exp1_head = get_backbone_and_head(exp1_sd)
    exp2_backbone, exp2_head = get_backbone_and_head(exp2_sd)
    
    expert_backbones = [exp1_backbone, exp2_backbone]
    
    print("Testing Weight Averaging (WA)...")
    wa = merge_weight_averaging(expert_backbones)
    assert wa.keys() == prog_backbone.keys(), "WA keys mismatch!"
    
    print("Testing Task Arithmetic (TA)...")
    ta = merge_task_arithmetic(prog_backbone, expert_backbones, 0.5)
    assert ta.keys() == prog_backbone.keys(), "TA keys mismatch!"
    
    print("Testing Isotropic Parameter Resonance (U-IPR)...")
    u_ipr = merge_u_ipr(prog_backbone, expert_backbones)
    assert u_ipr.keys() == prog_backbone.keys(), "U-IPR keys mismatch!"
    
    print("Testing Holographic Norm Scaling (HNS)...")
    hns = merge_hns(prog_backbone, expert_backbones, 0)
    assert hns.keys() == prog_backbone.keys(), "HNS keys mismatch!"
    
    print("Testing Unified Channel-wise Parameter Resonance (UCPR)...")
    ucpr = merge_ucpr(prog_backbone, expert_backbones)
    assert ucpr.keys() == prog_backbone.keys(), "UCPR keys mismatch!"
    
    print("Testing Depth-Adaptive Hybrid Parameter Resonance (DA-HPR)...")
    da_hpr = merge_depth_adaptive_hpr(prog_backbone, expert_backbones, 1.0, 0.1)
    assert da_hpr.keys() == prog_backbone.keys(), "DA-HPR keys mismatch!"
    
    print("Testing model reconstruction...")
    model = create_full_model(ucpr, exp1_head, device)
    
    print("Testing BatchNorm Calibration (BNC)...")
    dummy_data = torch.randn(10, 3, 32, 32)
    dummy_labels = torch.zeros(10, dtype=torch.long)
    dummy_dataset = TensorDataset(dummy_data, dummy_labels)
    dummy_loader = DataLoader(dummy_dataset, batch_size=2)
    
    train_loaders = {"task1": dummy_loader}
    heads = {"task1": exp1_head}
    
    calibrated = calibrate_batchnorm(da_hpr, heads, train_loaders, device, num_samples_per_task=4, batch_size=2)
    assert calibrated.keys() == prog_backbone.keys(), "BNC calibrated backbone keys mismatch!"
    
    print("All merge logic checks passed successfully!")

if __name__ == "__main__":
    main()
