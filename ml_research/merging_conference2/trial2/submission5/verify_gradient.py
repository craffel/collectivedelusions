import torch
import numpy as np

def verify_odc_gradient():
    print("====================================================")
    print("Numerical Verification of ODC Analytical Gradient")
    print("====================================================")
    
    # Set seed for reproducibility
    torch.manual_seed(1234)
    np.random.seed(1234)
    
    # Choose dimension C (number of channels)
    C = 64
    print(f"Dimension (C): {C}")
    
    # Generate random positive definite covariance matrices
    # Sigma_M (Merged)
    A = torch.randn(C, C, dtype=torch.float64)
    Sigma_M = A @ A.T + torch.eye(C, dtype=torch.float64) * 0.1
    
    # Sigma_k (Expert/Target)
    B = torch.randn(C, C, dtype=torch.float64)
    Sigma_k = B @ B.T + torch.eye(C, dtype=torch.float64) * 0.1
    
    # Define diagonal scaling vector d as a leaf node with gradients enabled
    d = torch.randn(C, dtype=torch.float64, requires_grad=True)
    
    # Compute ODC objective function
    # f(d) = || D * Sigma_M * D - Sigma_k ||_F^2
    D = torch.diag(d)
    error_matrix = D @ Sigma_M @ D - Sigma_k
    f_val = torch.sum(error_matrix ** 2)
    
    # Compute autograd gradient
    f_val.backward()
    grad_autograd = d.grad.clone()
    
    # Compute analytical gradient
    # grad_analytical = 4 * (E * Sigma_M) * d (where * in E * Sigma_M is element-wise product)
    # E = D @ Sigma_M @ D - Sigma_k
    E = D @ Sigma_M @ D - Sigma_k
    grad_analytical = 4.0 * (E * Sigma_M) @ d
    
    # Compare gradients
    abs_diff = torch.abs(grad_autograd - grad_analytical)
    max_abs_diff = torch.max(abs_diff).item()
    mean_abs_diff = torch.mean(abs_diff).item()
    
    # Relative difference
    denom = torch.norm(grad_autograd)
    rel_diff = torch.norm(grad_autograd - grad_analytical) / denom
    rel_diff_val = rel_diff.item()
    
    print(f"Objective value f(d): {f_val.item():.6f}")
    print(f"Maximum absolute difference: {max_abs_diff:.2e}")
    print(f"Mean absolute difference: {mean_abs_diff:.2e}")
    print(f"Relative gradient difference: {rel_diff_val:.2e}")
    
    assert max_abs_diff < 1e-10, f"Gradient verification failed! Max abs diff: {max_abs_diff}"
    print("\nSUCCESS: The analytical gradient matches PyTorch's autograd with double precision accuracy (< 1e-10)!")
    print("====================================================")

if __name__ == "__main__":
    verify_odc_gradient()
