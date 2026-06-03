import torch
import torch.nn as nn
import numpy as np
import os
import random

from data_utils import get_dataloader, get_calibration_subset, get_dataset
from merge_and_evaluate import quantize_weight, quantize_weight_channelwise, merge_wcpr, merge_qr_wcpr, merge_ties, merge_dare
from models import ResNet18CIFAR

def test_quantization_equivalence():
    print("Testing channel-wise quantization equivalence...")
    # Create a dummy weight tensor of shape [128, 64, 3, 3]
    torch.manual_seed(42)
    W = torch.randn(128, 64, 3, 3) * 5.0
    
    # 1. Manual channel-wise quantization (old method)
    num_bits = 4
    qmax = 2**(num_bits - 1) - 1
    W_manual = torch.zeros_like(W)
    for c in range(W.shape[0]):
        channel_w = W[c]
        max_val = torch.max(torch.abs(channel_w))
        if max_val == 0:
            W_manual[c] = channel_w
            continue
        delta = max_val / qmax
        W_manual[c] = torch.clamp(torch.round(channel_w / delta), -qmax, qmax) * delta
        
    # 2. Vectorized channel-wise quantization (our optimized method)
    W_vectorized = quantize_weight_channelwise(W, num_bits=num_bits)
    
    # Verify they are identical
    diff = torch.abs(W_manual - W_vectorized).max().item()
    print(f"  Max absolute difference: {diff}")
    assert diff < 1e-6, f"Quantization results differ! Max diff: {diff}"
    print("  Quantization equivalence test passed!")

def test_calibration_subset():
    print("Testing calibration subset sampling...")
    # Get calibration subset for MNIST
    num_samples = 20
    loader = get_calibration_subset("mnist", num_samples=num_samples, seed=42)
    
    # Check loader size and batch content
    images, labels = next(iter(loader))
    print(f"  Loaded batch shape: {images.shape}, Labels shape: {labels.shape}")
    assert images.shape[0] == num_samples, f"Expected {num_samples} samples, got {images.shape[0]}"
    assert labels.shape[0] == num_samples, f"Expected {num_samples} labels, got {labels.shape[0]}"
    
    # Verify balance (it should have exactly num_samples // 10 per class)
    unique, counts = torch.unique(labels, return_counts=True)
    print(f"  Class distribution: {dict(zip(unique.tolist(), counts.tolist()))}")
    for count in counts:
        assert count == num_samples // 10, f"Expected balanced classes of size {num_samples // 10}, got {count}"
        
    print("  Calibration subset sampling test passed!")

def test_merging_correctness():
    print("Testing vectorized model merging correctness...")
    progenitor = ResNet18CIFAR(num_classes=10)
    experts = [ResNet18CIFAR(num_classes=10) for _ in range(3)]
    
    # Randomize weights
    torch.manual_seed(42)
    for p in progenitor.parameters():
        p.data.copy_(torch.randn_like(p) * 2.0)
    for exp in experts:
        for p in exp.parameters():
            p.data.copy_(torch.randn_like(p) * 2.0)
            
    # Compute using vectorized merging
    merged_wcpr = merge_wcpr(progenitor, experts, lambd=0.5)
    merged_qr_wcpr = merge_qr_wcpr(progenitor, experts, lambd=0.5, gamma=1.5)
    merged_ties = merge_ties(progenitor, experts, lambd=0.5, fraction=0.2)
    merged_dare = merge_dare(progenitor, experts, lambd=0.5, p=0.2)
    
    # Verify shapes match progenitor
    for k, v in progenitor.state_dict().items():
        assert merged_wcpr.state_dict()[k].shape == v.shape, f"WCPR Shape mismatch for key {k}!"
        assert merged_qr_wcpr.state_dict()[k].shape == v.shape, f"QR-WCPR Shape mismatch for key {k}!"
        assert merged_ties.state_dict()[k].shape == v.shape, f"TIES Shape mismatch for key {k}!"
        assert merged_dare.state_dict()[k].shape == v.shape, f"DARE Shape mismatch for key {k}!"
        assert not torch.isnan(merged_wcpr.state_dict()[k]).any(), f"NaN found in WCPR {k}!"
        assert not torch.isnan(merged_qr_wcpr.state_dict()[k]).any(), f"NaN found in QR-WCPR {k}!"
        assert not torch.isnan(merged_ties.state_dict()[k]).any(), f"NaN found in TIES {k}!"
        assert not torch.isnan(merged_dare.state_dict()[k]).any(), f"NaN found in DARE {k}!"
        
    print("  Vectorized model merging correctness test passed!")

if __name__ == "__main__":
    print("="*40)
    print("RUNNING OPTIMIZATION UNIT TESTS")
    print("="*40)
    test_quantization_equivalence()
    test_calibration_subset()
    test_merging_correctness()
    print("="*40)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("="*40)
