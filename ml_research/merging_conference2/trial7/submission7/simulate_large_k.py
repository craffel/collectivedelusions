import torch
import numpy as np

def simulate_scaling(D=10000, K_list=[2, 3, 5, 10, 20, 50, 100], num_seeds=10):
    print(f"=== Running Scaling Simulation (D={D}, Seeds={num_seeds}) ===")
    print(f"{'K':>4} | {'Theory Cos':>10} | {'Emp. Cos (Mean ± SD)':>22} | {'Norm Ratio (Uncal)':>18} | {'Norm Ratio (HNS)':>16} | {'HNS MSE (Mean ± SD)':>20}")
    print("-" * 105)
    
    for K in K_list:
        cos_sims = []
        uncal_norms = []
        hns_norms = []
        hns_mses = []
        
        for seed in range(num_seeds):
            torch.manual_seed(seed)
            # Generate progenitor W_init
            W_init = torch.randn(D)
            
            # Generate K orthogonal expert updates
            # To get truly orthogonal vectors in D dimensions, we can use QR decomposition on a random matrix
            Q, _ = torch.linalg.qr(torch.randn(D, K)) # Shape: (D, K)
            
            # Normalize each column to have norm N_expert = 5.0 (typical norm)
            N_expert = 5.0
            delta_Ws = Q * N_expert # Each column is an orthogonal task vector of norm N_expert
            
            # Transpose to get a list of task vectors
            delta_Ws = delta_Ws.T # Shape: (K, D)
            
            # Compute merged task vector
            delta_W_merged = torch.mean(delta_Ws, dim=0) # Shape: (D,)
            
            # Empirical cosine similarity between delta_W_i and delta_W_merged
            for i in range(K):
                cos_val = torch.dot(delta_Ws[i], delta_W_merged) / (torch.norm(delta_Ws[i]) * torch.norm(delta_W_merged))
                cos_sims.append(cos_val.item())
            
            # Norm ratio of uncalibrated merged update vs original expert update
            norm_uncal = torch.norm(delta_W_merged).item()
            uncal_norms.append(norm_uncal / N_expert)
            
            # Apply HNS to reconstruct task vector for each expert i
            for i in range(K):
                # Channel-wise norm scaling in 1D (treating entire vector as one channel for simplicity, 
                # or we can split it into channels. Let's do it as one channel first, which is the most conservative)
                gamma = torch.norm(delta_Ws[i]) / (torch.norm(delta_W_merged) + 1e-8)
                delta_W_hns = gamma * delta_W_merged
                
                hns_norms.append(torch.norm(delta_W_hns).item() / N_expert)
                mse = torch.mean((delta_W_hns - delta_Ws[i])**2).item()
                hns_mses.append(mse)
                
        theory_cos = 1.0 / np.sqrt(K)
        mean_cos = np.mean(cos_sims)
        std_cos = np.std(cos_sims)
        mean_uncal = np.mean(uncal_norms)
        mean_hns = np.mean(hns_norms)
        mean_mse = np.mean(hns_mses)
        std_mse = np.std(hns_mses)
        
        print(f"{K:4d} | {theory_cos:10.4f} | {mean_cos:8.4f} ± {std_cos:6.4f} | {mean_uncal:17.4f} | {mean_hns:16.4f} | {mean_mse:.4e} ± {std_mse:.2e}")

if __name__ == "__main__":
    simulate_scaling()
