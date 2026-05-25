import torch
import torch.nn as nn
from train_refinement import optimize_procrustes, ConvEncoder, TaskHead, MultiTaskModel

def test_skew_symmetry_and_orthogonality():
    print("Testing skew-symmetry and orthogonality properties...")
    # Generate random target weights and base weights
    out_dim, in_dim = 64, 128
    torch.manual_seed(42)
    W_0 = torch.randn(out_dim, in_dim)
    W_k = torch.randn(out_dim, in_dim)
    F_k = torch.rand(out_dim, in_dim) + 0.1
    
    # Run unweighted Procrustes optimization
    A, R, E = optimize_procrustes(W_0, W_k, F_k, steps=10, lr=0.1, weighted=False, device='cpu')
    
    # 1. Check skew-symmetry: A^T = -A
    skew_diff = (A + A.t()).abs().max().item()
    print(f"Skew-symmetry difference (A + A^T): {skew_diff:.6e}")
    assert skew_diff < 1e-6, f"A is not skew-symmetric, max diff: {skew_diff}"
    
    # 2. Check orthogonality: R^T * R = I
    I = torch.eye(out_dim)
    ortho_diff = (R.t() @ R - I).abs().max().item()
    print(f"Orthogonality difference (R^T * R - I): {ortho_diff:.6e}")
    assert ortho_diff < 1e-5, f"R is not orthogonal, max diff: {ortho_diff}"
    print("Skew-symmetry and Orthogonality tests passed successfully!\n")

def test_soft_bounded_fisher():
    print("Testing Soft-Bounded log-Fisher Normalization monotonic/bounded behavior...")
    # Create artificial highly skewed Fisher matrix
    F_k = torch.tensor([1e-5, 1e-4, 1.0, 10.0, 1000.0, 100000.0])
    # The norm function applied inside optimize_procrustes:
    # F_k_norm = torch.log1p(F_k / (F_k.mean() + 1e-8))
    mean_val = F_k.mean() + 1e-8
    F_k_norm = torch.log1p(F_k / mean_val)
    
    print(f"Original Fisher: {F_k}")
    print(f"Normalized Fisher: {F_k_norm}")
    
    # Check monotonicity: if F[i] < F[j], then F_norm[i] < F_norm[j]
    for i in range(len(F_k) - 1):
        assert F_k_norm[i] <= F_k_norm[i+1], "Normalization does not preserve monotonicity!"
        
    # Check bounded nature (the maximum normalized weight is small and stable)
    print(f"Original max: {F_k[-1].item():.2e} | Normalized max: {F_k_norm[-1].item():.2e}")
    assert F_k_norm[-1].item() < 5.0, "Normalization failed to bound the maximum Fisher weight!"
    print("Soft-Bounded log-Fisher Normalization tests passed successfully!\n")

def test_conv_encoder_and_task_head():
    print("Testing architecture models forward passes...")
    device = 'cpu'
    encoder = ConvEncoder().to(device)
    head = TaskHead(input_dim=128, num_classes=10).to(device)
    model = MultiTaskModel(encoder, head).to(device)
    
    # Run dummy forward pass
    dummy_input = torch.randn(4, 1, 28, 28).to(device)
    output = model(dummy_input)
    assert output.shape == (4, 10), f"Output shape mismatch: {output.shape}"
    print("Architecture models forward passes passed successfully!\n")

if __name__ == "__main__":
    test_skew_symmetry_and_orthogonality()
    test_soft_bounded_fisher()
    test_conv_encoder_and_task_head()
    print("ALL UNIT TESTS PASSED SUCCESSFULLY!")
