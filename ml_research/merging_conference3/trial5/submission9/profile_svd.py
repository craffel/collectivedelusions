import time
import torch

# Limit CPU threads to prevent hangs on cluster logins
torch.set_num_threads(4)

def randomized_svd(M, r, p=10, n_iter=1):
    """
    Randomized SVD algorithm (Halko et al., 2011).
    M: Input matrix of shape (m, n)
    r: Target rank
    p: Oversampling parameter
    n_iter: Number of subspace iterations (1 is usually enough for profiling)
    """
    m, n = M.shape
    k = r + p
    # Step 1: Draw Gaussian random matrix
    Omega = torch.randn(n, k, dtype=M.dtype, device=M.device)
    # Step 2: Form sample matrix
    Y = torch.mm(M, Omega)
    
    # Subspace iteration
    for _ in range(n_iter):
        Q, _ = torch.linalg.qr(Y)
        Y = torch.mm(M, torch.mm(M.t(), Q))
        
    Q, _ = torch.linalg.qr(Y)
    
    # Step 3: Direct SVD on the reduced matrix B = Q^T * M
    B = torch.mm(Q.t(), M)
    U_hat, S, Vh = torch.linalg.svd(B, full_matrices=False)
    
    # Step 4: Reconstruct left singular vectors
    U = torch.mm(Q, U_hat)
    
    return U[:, :r], S[:r], Vh[:r, :]

def profile():
    torch.manual_seed(20260614)
    
    # Configurations (using more moderate sizes for fast profiling on CPU)
    configs = [
        {"name": "ViT-Tiny Layer", "m": 192, "n": 192 * 4},
        {"name": "ViT-Base Layer", "m": 768, "n": 768 * 4},
        {"name": "LLaMA-7B (Reduced)", "m": 2048, "n": 2048 * 4}
    ]
    
    ranks = [0.1, 0.3, 0.5]
    
    print(f"{'Configuration':<20} | {'Rank γ':<8} | {'Exact SVD (s)':<15} | {'Rand SVD (s)':<15} | {'Speedup':<8} | {'Rel Error':<12}")
    print("-" * 88)
    
    for config in configs:
        m, n = config["m"], config["n"]
        name = config["name"]
        
        # Generate random matrix
        M = torch.randn(m, n)
        
        for gamma in ranks:
            r = int(gamma * m)
            
            # 1. Exact SVD profiling
            start = time.time()
            U_ex, S_ex, Vh_ex = torch.linalg.svd(M, full_matrices=False)
            U_ex_r = U_ex[:, :r]
            proj_ex = torch.mm(U_ex_r, torch.mm(U_ex_r.t(), M))
            exact_time = time.time() - start
            
            # 2. Randomized SVD profiling
            start = time.time()
            U_rd, S_rd, Vh_rd = randomized_svd(M, r, p=10, n_iter=1)
            proj_rd = torch.mm(U_rd, torch.mm(U_rd.t(), M))
            rand_time = time.time() - start
            
            # 3. Compute relative reconstruction error
            error_ex = torch.norm(M - proj_ex, p='fro')
            error_rd = torch.norm(M - proj_rd, p='fro')
            
            # Relative error difference (how much worse is Randomized SVD compared to the mathematically optimal exact SVD projection)
            rel_diff = (error_rd - error_ex) / error_ex if error_ex > 0 else 0
            
            speedup = exact_time / rand_time if rand_time > 0 else 0
            
            print(f"{name:<20} | {gamma:<8} | {exact_time:<15.4f} | {rand_time:<15.4f} | {speedup:<8.2f}x | {rel_diff:<12.2e}")

if __name__ == "__main__":
    profile()
