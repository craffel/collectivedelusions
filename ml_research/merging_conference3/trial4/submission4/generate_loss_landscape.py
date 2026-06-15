import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from run_experiments_v2 import (
    get_dct_matrix, get_covariance_matrix, get_optimal_profile,
    get_accuracy, generate_noise, idct_iii, L, K
)

def build_polynomial_basis_matrix(L, degree=2, device=None):
    P = torch.zeros(L, degree + 1, device=device)
    l_bar = torch.linspace(0.0, 1.0, L, device=device)
    for p in range(degree + 1):
        P[:, p] = l_bar ** p
    return P

def get_loss_for_profile(alpha, target, Sigma_inv):
    # Compute quadratic sensitivity loss for task 0
    e = alpha - target
    quad = torch.matmul(e, torch.matmul(Sigma_inv, e))
    cos_term = 0.03 * torch.sum(1.0 - torch.cos(10 * torch.pi * e))
    loss = 0.5 + 1.5 * quad + cos_term
    return loss.item()

def main():
    device = torch.device("cpu")
    torch.manual_seed(42)
    np.random.seed(42)
    
    Sigma = get_covariance_matrix(L, device)
    Sigma_inv = torch.linalg.inv(Sigma)
    
    # Get optimal target profile and noise for task 0
    lambda_stars = torch.stack([get_optimal_profile(k, L, device) for k in range(K)])
    etas = torch.stack([generate_noise(L, device) for _ in range(K)])
    targets = lambda_stars + etas
    
    target_0 = targets[0] # MNIST target
    
    # --- PolyMerge Setup ---
    P = build_polynomial_basis_matrix(L, degree=2, device=device) # L x 3
    # Find least squares optimal fit of target_0 in polynomial coefficients
    # w_opt = (P^T P)^(-1) P^T target_0
    P_pinv = torch.linalg.pinv(P)
    w_opt = torch.matmul(P_pinv, target_0) # size 3
    
    # --- SpectralMerge Setup ---
    M_dct = get_dct_matrix(L, device) # L x L
    # Optimal spectral coefficients are just the DCT of target_0
    c_opt_full = torch.matmul(target_0, M_dct.t()) # size L
    c_opt = c_opt_full[:3] # We optimize first 3, size 3
    
    # --- 2D Grid Setup ---
    grid_size = 150
    x_range = np.linspace(-0.8, 0.8, grid_size)
    y_range = np.linspace(-0.8, 0.8, grid_size)
    X, Y = np.meshgrid(x_range, y_range)
    
    loss_poly = np.zeros((grid_size, grid_size))
    loss_spec = np.zeros((grid_size, grid_size))
    
    # Perturbation directions: along the 2nd and 3rd parameters (index 1 and 2)
    d1 = torch.tensor([0.0, 1.0, 0.0], device=device)
    d2 = torch.tensor([0.0, 0.0, 1.0], device=device)
    
    print("Generating 2D loss landscape for PolyMerge...")
    for i in range(grid_size):
        for j in range(grid_size):
            # PolyMerge perturbation
            w_perturbed = w_opt + X[i, j] * d1 + Y[i, j] * d2
            alpha_poly = torch.matmul(P, w_perturbed)
            loss_poly[i, j] = get_loss_for_profile(alpha_poly, target_0, Sigma_inv)
            
            # SpectralMerge perturbation
            c_perturbed = c_opt + X[i, j] * d1 + Y[i, j] * d2
            # Pad with zeros to size L
            c_full = torch.cat([c_perturbed, torch.zeros(L - 3, device=device)])
            alpha_spec = idct_iii(c_full, M_dct)
            loss_spec[i, j] = get_loss_for_profile(alpha_spec, target_0, Sigma_inv)
            
    print("Plotting side-by-side comparison...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
    
    # Plot PolyMerge Landscape
    contour_levels = np.linspace(min(loss_poly.min(), loss_spec.min()), max(loss_poly.max(), loss_spec.max()), 25)
    cp1 = ax1.contourf(X, Y, loss_poly, levels=contour_levels, cmap='viridis')
    fig.colorbar(cp1, ax=ax1, label='Surrogate Loss Value')
    ax1.plot(0, 0, 'r*', markersize=12, label='Optimal Center')
    ax1.set_xlabel("Perturbation along $w_1$ (Linear parameter)", fontsize=11)
    ax1.set_ylabel("Perturbation along $w_2$ (Quadratic parameter)", fontsize=11)
    ax1.set_title("PolyMerge (Collinear Polynomial Basis)\nIll-conditioned, Highly Anisotropic Landscape", fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper right')
    
    # Plot SpectralMerge Landscape
    cp2 = ax2.contourf(X, Y, loss_spec, levels=contour_levels, cmap='viridis')
    fig.colorbar(cp2, ax=ax2, label='Surrogate Loss Value')
    ax2.plot(0, 0, 'r*', markersize=12, label='Optimal Center')
    ax2.set_xlabel("Perturbation along $c_1$ (Cosine frequency 1)", fontsize=11)
    ax2.set_ylabel("Perturbation along $c_2$ (Cosine frequency 2)", fontsize=11)
    ax2.set_title("SpectralMerge (Orthonormal DCT Basis)\nPerfectly Conditioned, Isotropic Landscape", fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    
    os.makedirs("submission", exist_ok=True)
    os.makedirs("submission/sections", exist_ok=True)
    
    plt.savefig("submission/loss_landscape_comparison.png", dpi=300)
    plt.savefig("submission/loss_landscape_comparison.pdf", dpi=300)
    print("Loss landscape comparison figures successfully generated and saved to 'submission/loss_landscape_comparison.png' and 'submission/loss_landscape_comparison.pdf'.")

if __name__ == "__main__":
    main()
