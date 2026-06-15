import torch
import numpy as np

def run_simulation():
    torch.manual_seed(42)
    np.random.seed(42)

    D = 10
    K = 2
    N = 1000

    # 1. Define centroids with some overlap (overlap = 0.5)
    v1 = torch.zeros(D)
    v1[0] = 1.0
    v2 = torch.zeros(D)
    v2[0] = 0.5
    v2[1] = np.sqrt(0.75)

    v = torch.stack([v1, v2], dim=0) # K x D
    
    # 2. Define highly anisotropic noise covariance (huge noise on dimension 0)
    cov_diag = torch.ones(D) * 0.01
    cov_diag[0] = 1.5 # Highly anisotropic noise along dimension 0!
    Sigma = torch.diag(cov_diag)
    Sigma_sqrt = torch.diag(torch.sqrt(cov_diag))

    # 3. Generate samples for Task 0 and Task 1
    # Clean representations
    z_clean_0 = v1.unsqueeze(0).repeat(N, 1)
    z_clean_1 = v2.unsqueeze(0).repeat(N, 1)
    z_clean = torch.cat([z_clean_0, z_clean_1], dim=0)
    task_labels = torch.cat([torch.zeros(N, dtype=torch.long), torch.ones(N, dtype=torch.long)], dim=0)

    # Corrupted representations with anisotropic noise
    noise = torch.randn(2 * N, D) @ Sigma_sqrt
    z_corrupted = z_clean + noise

    # Define simple OTSP router function
    def compute_otsp_routing(X, centroids, tau=0.01):
        # Normalize centroids
        centroids_norm = centroids / (torch.norm(centroids, dim=1, keepdim=True) + 1e-8)
        # Compute Gram overlap matrix S
        S = torch.matmul(centroids_norm, centroids_norm.t())
        # Löwdin symmetric orthogonalization
        eigenvalues, eigenvectors = torch.linalg.eigh(S)
        inv_sqrt_eigenvalues = 1.0 / torch.sqrt(eigenvalues + 1e-6)
        S_inv_sqrt = torch.matmul(eigenvectors, torch.matmul(torch.diag(inv_sqrt_eigenvalues), eigenvectors.t()))
        Q = torch.matmul(S_inv_sqrt, centroids_norm)
        
        # Normalize X
        X_norm = X / (torch.norm(X, dim=-1, keepdim=True) + 1e-8)
        # Project onto Q
        u = torch.matmul(X_norm, Q.t())
        # Absolute coordinates
        u_abs = torch.abs(u)
        # Softmax dynamic routing
        alpha = torch.softmax(u_abs / tau, dim=-1)
        preds = torch.argmax(alpha, dim=1)
        return preds

    # A. Clean Evaluation
    preds_clean = compute_otsp_routing(z_clean, v)
    acc_clean = (preds_clean == task_labels).float().mean().item()

    # B. Corrupted Evaluation (No Whitening)
    preds_corrupted = compute_otsp_routing(z_corrupted, v)
    acc_corrupted = (preds_corrupted == task_labels).float().mean().item()

    # C. Second-Moment (Origin-Centered) Whitening
    # Estimate empirical second moment of corrupted representations
    empirical_R = torch.matmul(z_corrupted.t(), z_corrupted) / (2 * N) + 1e-5 * torch.eye(D)
    
    # Compute R^{-1/2}
    evals, evecs = torch.linalg.eigh(empirical_R)
    inv_sqrt_evals = 1.0 / torch.sqrt(evals + 1e-6)
    R_inv_sqrt = torch.matmul(evecs, torch.matmul(torch.diag(inv_sqrt_evals), evecs.t()))

    # Whiten both centroids and representations without centering
    v_whitened = torch.matmul(v, R_inv_sqrt)
    z_whitened = torch.matmul(z_corrupted, R_inv_sqrt)

    preds_whitened = compute_otsp_routing(z_whitened, v_whitened)
    acc_whitened = (preds_whitened == task_labels).float().mean().item()

    print(f"OTSP Routing Accuracy (Clean): {acc_clean * 100:.2f}%")
    print(f"OTSP Routing Accuracy (Anisotropic Noise, No Whitening): {acc_corrupted * 100:.2f}%")
    print(f"OTSP Routing Accuracy (Anisotropic Noise, with Origin-Centered Whitening): {acc_whitened * 100:.2f}%")

if __name__ == '__main__':
    run_simulation()
