import torch

def bures_wasserstein_barycenter(covs, weights, max_iter=20, tol=1e-6):
    d = covs[0].shape[0]
    device = covs[0].device
    dtype = covs[0].dtype
    
    # Initialize with arithmetic mean
    sigma = torch.zeros((d, d), device=device, dtype=dtype)
    for w, cov in zip(weights, covs):
        sigma += w * cov
        
    for iteration in range(max_iter):
        eigenvalues, eigenvectors = torch.linalg.eigh(sigma)
        eigenvalues = torch.clamp(eigenvalues, min=1e-8)
        
        sigma_sqrt = eigenvectors @ torch.diag(torch.sqrt(eigenvalues)) @ eigenvectors.T
        sigma_inv_sqrt = eigenvectors @ torch.diag(1.0 / torch.sqrt(eigenvalues)) @ eigenvectors.T
        
        sum_term = torch.zeros((d, d), device=device, dtype=dtype)
        for w, cov in zip(weights, covs):
            inner = sigma_sqrt @ cov @ sigma_sqrt
            val_in, vec_in = torch.linalg.eigh(inner)
            val_in = torch.clamp(val_in, min=1e-8)
            inner_sqrt = vec_in @ torch.diag(torch.sqrt(val_in)) @ vec_in.T
            sum_term += w * inner_sqrt
            
        next_sigma = sigma_inv_sqrt @ (sum_term @ sum_term) @ sigma_inv_sqrt
        
        diff = torch.norm(next_sigma - sigma) / torch.norm(sigma)
        print(f"Iter {iteration+1}: relative diff = {diff.item():.2e}")
        sigma = next_sigma
        if diff < tol:
            print("Converged!")
            break
            
    return sigma

if __name__ == '__main__':
    print("Testing Bures-Wasserstein Barycenter...")
    torch.manual_seed(42)
    d = 4
    # Create 3 random positive-definite matrices
    covs = []
    for _ in range(3):
        A = torch.randn(d, d)
        cov = A @ A.T + 0.1 * torch.eye(d)
        covs.append(cov)
        
    weights = [0.33, 0.33, 0.34]
    barycenter = bures_wasserstein_barycenter(covs, weights)
    print("Barycenter matrix:")
    print(barycenter)
