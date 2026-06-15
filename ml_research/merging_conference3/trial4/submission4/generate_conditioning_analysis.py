import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def build_vandermonde_matrix(L, degree):
    # Normalized depth coordinates in [0, 1]
    l_bar = torch.linspace(0.0, 1.0, L)
    V = torch.zeros(L, degree + 1)
    for p in range(degree + 1):
        V[:, p] = l_bar ** p
    return V

def build_chebyshev_matrix(L, degree):
    # Chebyshev polynomials evaluated on uniform grid [0, 1]
    l_bar = torch.linspace(-1.0, 1.0, L)
    T = torch.zeros(L, degree + 1)
    T[:, 0] = 1.0
    if degree >= 1:
        T[:, 1] = l_bar
    for p in range(2, degree + 1):
        T[:, p] = 2.0 * l_bar * T[:, p - 1] - T[:, p - 2]
    return T

def build_dct_matrix(L, F):
    # DCT-II matrix truncated to first F components
    M = torch.zeros(L, F)
    for j in range(F):
        w = 1.0 / (L ** 0.5) if j == 0 else (2.0 / L) ** 0.5
        for l in range(L):
            M[l, j] = w * torch.cos(torch.tensor(torch.pi * j * (l + 0.5) / L))
    return M

def main():
    depths = [12, 24, 48, 96, 192]
    degrees = [1, 2, 3, 4, 5]
    
    # Store condition numbers
    cond_vander = {d: [] for d in degrees}
    cond_cheb = {d: [] for d in degrees}
    cond_dct = {d: [] for d in degrees}
    
    for L in depths:
        for d in degrees:
            # PolyMerge basis (Vandermonde)
            V = build_vandermonde_matrix(L, d)
            cond_v = torch.linalg.cond(V).item()
            cond_vander[d].append(cond_v)
            
            # Chebyshev basis
            T = build_chebyshev_matrix(L, d)
            cond_t = torch.linalg.cond(T).item()
            cond_cheb[d].append(cond_t)
            
            # DCT basis
            D = build_dct_matrix(L, d + 1) # F = d + 1 coordinates
            cond_d = torch.linalg.cond(D).item()
            cond_dct[d].append(cond_d)
            
    print("Vandermonde Condition Numbers:", cond_vander)
    print("Chebyshev Condition Numbers:", cond_cheb)
    print("DCT Condition Numbers:", cond_dct)
    
    # --- Plotting ---
    plt.figure(figsize=(12, 6))
    
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    markers = ['o', 's', '^', 'D', 'v']
    
    for i, d in enumerate(degrees):
        # Vandermonde
        plt.plot(depths, cond_vander[d], label=f"Vandermonde (PolyMerge d={d})", 
                 color=colors[i], linestyle='-', marker=markers[i], linewidth=2)
        # Chebyshev
        plt.plot(depths, cond_cheb[d], label=f"Chebyshev (d={d})", 
                 color=colors[i], linestyle='--', marker=markers[i], linewidth=1.5, alpha=0.7)
                 
    # DCT-II (is always 1.0, show single flat line)
    plt.axhline(y=1.0, color='black', linestyle=':', linewidth=3, label="DCT-II (SpectralMerge - All F)")
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xticks(depths, [str(L) for L in depths])
    plt.xlabel("Network Layer Depth ($L$)", fontsize=12)
    plt.ylabel(r"Condition Number ($\kappa$) [Log Scale]", fontsize=12)
    plt.title("Numerical Conditioning & Scalability of Interpolation Bases", fontsize=14, fontweight='bold')
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=10)
    plt.tight_layout()
    
    os.makedirs("submission", exist_ok=True)
    plt.savefig("submission/conditioning_comparison.png", dpi=150)
    plt.savefig("submission/conditioning_comparison.pdf")
    print("Plots saved to submission/conditioning_comparison.png and submission/conditioning_comparison.pdf", flush=True)

if __name__ == "__main__":
    main()
