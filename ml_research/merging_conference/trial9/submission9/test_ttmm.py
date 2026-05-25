import torch
import torch.nn as nn
import numpy as np
from run_experiments import (
    SimpleCNN,
    KroneckerTraceTracker,
    make_merged_parameters,
    fuse_bn_buffers,
    add_gaussian_noise,
    kl_bernoulli
)

def test_simple_cnn_creation():
    model = SimpleCNN()
    assert isinstance(model, nn.Module)
    
    # Test forward pass with a dummy batch of size 4
    x = torch.randn(4, 1, 28, 28)
    out = model(x)
    assert out.shape == (4, 10)
    print("test_simple_cnn_creation: PASSED")

def test_kronecker_trace_tracker():
    model = SimpleCNN()
    tracker = KroneckerTraceTracker(model)
    
    # Run a dummy forward and backward pass to populate activations and gradients
    x = torch.randn(4, 1, 28, 28)
    out = model(x)
    loss = out.sum()
    loss.backward()
    
    sensitivities = tracker.compute_sensitivities()
    assert isinstance(sensitivities, dict)
    
    # Check that sensitivities are normalized (sum to 1)
    total_sens = sum(sensitivities.values())
    assert abs(total_sens - 1.0) < 1e-5
    
    # Check that crucial layers have sensitivities
    assert 'conv1' in sensitivities
    assert 'conv2' in sensitivities
    assert 'fc1' in sensitivities
    assert 'fc2' in sensitivities
    
    # Remove hooks
    tracker.remove()
    print("test_kronecker_trace_tracker: PASSED")

def test_make_merged_parameters():
    expert_0 = SimpleCNN()
    expert_1 = SimpleCNN()
    
    # Merge with a static coefficient of 0.3
    coeff = 0.3
    merged_params = make_merged_parameters(expert_0, expert_1, coeff)
    
    # Check that all parameters are present and correctly merged
    for name, p in expert_0.named_parameters():
        assert name in merged_params
        p_expected = (1.0 - coeff) * p + coeff * dict(expert_1.named_parameters())[name]
        assert torch.allclose(merged_params[name], p_expected, atol=1e-6)
    print("test_make_merged_parameters: PASSED")

def test_fuse_bn_buffers():
    expert_0 = SimpleCNN()
    expert_1 = SimpleCNN()
    merged_model = SimpleCNN()
    
    # Set known buffers in expert_0 and expert_1
    for m0, m1 in zip(expert_0.modules(), expert_1.modules()):
        if isinstance(m0, nn.BatchNorm2d) and isinstance(m1, nn.BatchNorm2d):
            m0.running_mean.fill_(1.0)
            m0.running_var.fill_(2.0)
            m1.running_mean.fill_(3.0)
            m1.running_var.fill_(4.0)
            
    fuse_bn_buffers(expert_0, expert_1, merged_model, 0.4)
    
    # Check that fused mean and var are mathematically correct
    # w_expert_1 = 0.4, w_expert_0 = 0.6
    # mean_fused = 0.6 * 1.0 + 0.4 * 3.0 = 1.8
    # var_fused = 0.6 * (2.0 + (1.0 - 1.8)**2) + 0.4 * (4.0 + (3.0 - 1.8)**2)
    # var_fused = 0.6 * (2.0 + 0.64) + 0.4 * (4.0 + 1.44) = 0.6 * 2.64 + 0.4 * 5.44 = 1.584 + 2.176 = 3.76
    
    for m in merged_model.modules():
        if isinstance(m, nn.BatchNorm2d):
            assert torch.allclose(m.running_mean, torch.tensor(1.8), atol=1e-5)
            assert torch.allclose(m.running_var, torch.tensor(3.76), atol=1e-5)
    print("test_fuse_bn_buffers: PASSED")

def test_kl_bernoulli():
    # Test with standard values
    p = torch.tensor([0.3, 0.5])
    q = torch.tensor([0.7, 0.5])
    kl = kl_bernoulli(p, q)
    assert abs(kl[1].item()) < 1e-6  # KL(0.5 || 0.5) must be 0
    assert kl[0].item() > 0.0        # KL(0.3 || 0.7) must be > 0
    print("test_kl_bernoulli: PASSED")

def test_add_gaussian_noise():
    x = torch.zeros(100, 100)
    noisy_x = add_gaussian_noise(x, sigma=0.6)
    assert noisy_x.shape == x.shape
    # Check standard deviation of noise
    assert abs(noisy_x.std().item() - 0.6) < 0.05
    print("test_add_gaussian_noise: PASSED")

if __name__ == "__main__":
    print("Running TTMM Unit Tests...")
    test_simple_cnn_creation()
    test_kronecker_trace_tracker()
    test_make_merged_parameters()
    test_fuse_bn_buffers()
    test_kl_bernoulli()
    test_add_gaussian_noise()
    print("ALL TESTS PASSED SUCCESSFULLY!")
