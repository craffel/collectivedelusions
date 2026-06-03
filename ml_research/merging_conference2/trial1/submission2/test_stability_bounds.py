import torch
import numpy as np

# Helper function to compute matrix power of symmetric positive-definite matrices (float64 version)
def sym_matrix_power_64(M, power, eps=1e-12):
    eigenvalues, eigenvectors = torch.linalg.eigh(M)
    eigenvalues = torch.clamp(eigenvalues, min=eps)
    return eigenvectors @ torch.diag(torch.pow(eigenvalues, power)) @ eigenvectors.T

# Bures-Wasserstein Barycenter for symmetric positive-definite matrices (float64 version)
def bures_wasserstein_barycenter_64(covs, weights, max_iter=50, tol=1e-9, eps=1e-12):
    d = covs[0].shape[0]
    device = covs[0].device
    dtype = covs[0].dtype
    
    # Initialize with arithmetic mean
    sigma = torch.zeros((d, d), device=device, dtype=dtype)
    for w, cov in zip(weights, covs):
        sigma += w * cov
        
    for iteration in range(max_iter):
        sigma_sqrt = sym_matrix_power_64(sigma, 0.5, eps)
        sigma_inv_sqrt = sym_matrix_power_64(sigma, -0.5, eps)
        
        sum_term = torch.zeros((d, d), device=device, dtype=dtype)
        for w, cov in zip(weights, covs):
            inner = sigma_sqrt @ cov @ sigma_sqrt
            inner_sqrt = sym_matrix_power_64(inner, 0.5, eps)
            sum_term += w * inner_sqrt
            
        next_sigma = sigma_inv_sqrt @ (sum_term @ sum_term) @ sigma_inv_sqrt
        next_sigma = next_sigma + eps * torch.eye(d, device=device, dtype=dtype)
        
        diff = torch.norm(next_sigma - sigma) / torch.norm(sigma)
        sigma = next_sigma
        if diff < tol:
            break
            
    return sigma

def bures_wasserstein_distance_64(cov1, cov2, eps=1e-12):
    cov1_sqrt = sym_matrix_power_64(cov1, 0.5, eps)
    inner = cov1_sqrt @ cov2 @ cov1_sqrt
    inner_sqrt = sym_matrix_power_64(inner, 0.5, eps)
    dist2 = torch.trace(cov1 + cov2 - 2 * inner_sqrt)
    return torch.sqrt(torch.clamp(dist2, min=0.0))

def test_perturbation_bounds():
    print("=== Testing Theorem 4.7: Robustness & Perturbation Bounds ===")
    torch.manual_seed(42)
    d = 8
    K = 3
    weights = [0.4, 0.3, 0.3]
    
    # Generate K positive-definite matrices in float64
    covs = []
    for _ in range(K):
        A = torch.randn(d, d, dtype=torch.float64)
        cov = A @ A.T + 1.0 * torch.eye(d, dtype=torch.float64)  # Ensure well-conditioned
        covs.append(cov)
        
    # Compute true barycenter
    bary = bures_wasserstein_barycenter_64(covs, weights, max_iter=100, eps=1e-12)
    
    # Apply perturbations of different scales
    for delta in [1e-5, 1e-4, 1e-3, 1e-2]:
        perturbed_covs = []
        dists_inputs = []
        for cov in covs:
            # Generate symmetric perturbation
            E = torch.randn(d, d, dtype=torch.float64)
            E = 0.5 * (E + E.T)
            E = E / torch.norm(E) * delta
            
            perturbed_cov = cov + E
            # Ensure positive definite
            eigenvalues, eigenvectors = torch.linalg.eigh(perturbed_cov)
            eigenvalues = torch.clamp(eigenvalues, min=1e-10)
            perturbed_cov = eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.T
            
            perturbed_covs.append(perturbed_cov)
            
            # Compute distance between input and perturbed input
            dists_inputs.append(bures_wasserstein_distance_64(cov, perturbed_cov))
            
        # Compute perturbed barycenter
        perturbed_bary = bures_wasserstein_barycenter_64(perturbed_covs, weights, max_iter=100, eps=1e-12)
        
        # Compute distance between true and perturbed barycenter
        dist_bary = bures_wasserstein_distance_64(bary, perturbed_bary)
        
        # Weighted sum of input distances
        weighted_dist_inputs = sum(w * d for w, d in zip(weights, dists_inputs))
        
        print(f"Perturbation delta = {delta:.1e}:")
        print(f"  - Bures-Wasserstein distance d_BW(Sigma*, tilde_Sigma*) = {dist_bary.item():.8f}")
        print(f"  - Weighted sum of input distances sum(lambda_i * d_BW(Sigma_i, tilde_Sigma_i)) = {weighted_dist_inputs.item():.8f}")
        
        # Verify the Wasserstein Barycenter 1-Lipschitz (convexity) metric inequality:
        # d_BW(Sigma*, tilde_Sigma*) <= sum_i lambda_i d_BW(Sigma_i, tilde_Sigma_i)
        is_valid = dist_bary <= weighted_dist_inputs + 1e-7
        print(f"  - Theorem 4.7 Bound Valid: {is_valid} (diff: {(weighted_dist_inputs - dist_bary).item():.8e})")
        assert is_valid, "Perturbation bound violated!"

def test_low_rank_preservation():
    print("\n=== Testing Theorem 4.8: Low-Rank Preservation ===")
    torch.manual_seed(42)
    d = 8
    K = 3
    r = 2  # Low-rank dimension
    weights = [1.0/K] * K
    
    # Generate low-rank covariances (rank r) in float64
    covs = []
    for _ in range(K):
        # Generate low-rank updates
        A = torch.randn(d, r, dtype=torch.float64)
        cov = A @ A.T + 1e-12 * torch.eye(d, dtype=torch.float64)
        covs.append(cov)
        
    # Compute barycenter
    bary = bures_wasserstein_barycenter_64(covs, weights, max_iter=100, eps=1e-12)
    
    # Compute eigenvalues of original and barycenter
    bary_eigenvalues, _ = torch.linalg.eigh(bary)
    bary_eigenvalues = torch.sort(bary_eigenvalues, descending=True).values
    
    print("Eigenvalues of the Bures-Wasserstein barycenter:")
    for idx, val in enumerate(bary_eigenvalues):
        print(f"  - Eigenvalue {idx+1}: {val.item():.2e}")
        
    # Since inputs are rank r + epsilon, the barycenter's eigenvalues should be extremely small for indices > r
    r_sum = torch.sum(bary_eigenvalues[:r]).item()
    total_sum = torch.sum(bary_eigenvalues).item()
    ratio = r_sum / total_sum
    print(f"Ratio of top {r} eigenvalues to total trace: {ratio * 100:.6f}%")
    
    # The ratio should be extremely close to 100%
    is_valid = ratio > 0.999
    print(f"  - Theorem 4.8 Low-Rank Preservation Valid: {is_valid}")
    assert is_valid, "Low-rank preservation failed!"

if __name__ == '__main__':
    test_perturbation_bounds()
    test_low_rank_preservation()
