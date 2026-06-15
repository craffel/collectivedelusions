import numpy as np
import scipy.linalg
import torch
import torch.nn as nn
import torch.optim as optim
import time

# Set parameters
D = 192          # Dimensionality
K = 4            # Number of task experts
N_sub = 64       # Calibration size for projection extraction
N_test = 200     # Test samples per task
num_seeds = 5

def apply_nonlinear_distortion(x):
    """
    Applies an extremely severe non-linear, curving distortion to the representations
    by scaling and mixing features via high-amplitude sin and cos functions.
    """
    x_scaled = x * 12.0
    x_distorted = np.zeros_like(x)
    for i in range(D):
        next_i = (i + 1) % D
        prev_i = (i - 1) % D
        x_distorted[:, i] = x_scaled[:, i] + 6.5 * np.sin(x_scaled[:, next_i]) + 5.0 * np.cos(x_scaled[:, prev_i])
    x_distorted = x_distorted / 12.0
    return x_distorted

class ContrastiveProjectionHead(nn.Module):
    def __init__(self, input_dim=192, hidden_dim=64, output_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

def run_evaluation_on_seed(seed, rbf_gamma):
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Generate orthogonal centroids
    v_orth = np.zeros((K, D))
    for k in range(K):
        start = 48 * k
        end = 48 * (k + 1)
        v_orth[k, start:end] = np.random.normal(size=48)
        v_orth[k] /= np.linalg.norm(v_orth[k])
        
    # We introduce entanglement (rho = 0.5) to make task separation harder
    rho = 0.5
    v_shared = np.random.normal(size=D)
    v_shared /= np.linalg.norm(v_shared)
    
    v_centroids = np.zeros((K, D))
    for k in range(K):
        v_centroids[k] = np.sqrt(1 - rho) * v_orth[k] + np.sqrt(rho) * v_shared
        v_centroids[k] /= np.linalg.norm(v_centroids[k])
        
    # Generate calibration and test sets with noise scale 0.20
    noise_scale = 0.20
    
    sub_data = []
    sub_labels = []
    for k in range(K):
        eps = np.random.normal(scale=noise_scale, size=(N_sub, D))
        sub_data.append(v_centroids[k] + eps)
        sub_labels.append(np.ones(N_sub, dtype=int) * k)
    sub_data = np.concatenate(sub_data, axis=0)
    sub_labels = np.concatenate(sub_labels, axis=0)
    
    test_data = []
    test_labels = []
    for k in range(K):
        eps = np.random.normal(scale=noise_scale, size=(N_test, D))
        test_data.append(v_centroids[k] + eps)
        test_labels.append(np.ones(N_test, dtype=int) * k)
    test_data = np.concatenate(test_data, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)
    
    # Apply severe non-linear distortion to calibration and test features
    sub_distorted = apply_nonlinear_distortion(sub_data)
    test_distorted = apply_nonlinear_distortion(test_data)
    
    # ----------------------------------------------------
    # Method 1: Linear PCA Projection (UN-PCA-SEP)
    # ----------------------------------------------------
    V_k_d = []
    for k in range(K):
        idx = (sub_labels == k)
        z = sub_distorted[idx]
        z_norm = z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-5)
        cov = np.dot(z_norm.T, z_norm) / N_sub
        U, S, Vt = np.linalg.svd(cov)
        V_k_d.append(U[:, 0:1]) # Top 1 PC
        
    def get_linear_coords(z):
        z_norm = z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-5)
        coords = np.zeros((z.shape[0], K))
        for k in range(K):
            coords[:, k] = np.abs(np.dot(z_norm, V_k_d[k]).squeeze())
        return coords
        
    linear_coords = get_linear_coords(test_distorted)
    linear_preds = np.argmax(linear_coords, axis=-1)
    linear_acc = np.mean(linear_preds == test_labels)
    
    # ----------------------------------------------------
    # Method 2: Uncentered Kernel PCA Projection (UN-KPCA-SEP)
    # ----------------------------------------------------
    def compute_rbf_kernel(x1, x2):
        sq_dist = np.sum(x1**2, axis=-1, keepdims=True) + np.sum(x2**2, axis=-1) - 2 * np.dot(x1, x2.T)
        return np.exp(-rbf_gamma * sq_dist)
        
    kpca_cal_samples = []
    kpca_eigenvectors = []
    
    for k in range(K):
        idx = (sub_labels == k)
        z = sub_distorted[idx]
        z_norm = z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-5)
        kpca_cal_samples.append(z_norm)
        
        # Compute Uncentered Kernel Matrix
        K_mat = compute_rbf_kernel(z_norm, z_norm)
        
        # Solve Eigenvalue Problem on Uncentered Kernel Matrix
        eigvals, eigvecs = scipy.linalg.eigh(K_mat)
        idx_sort = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx_sort]
        eigvecs = eigvecs[:, idx_sort]
        
        # Extract top 1 eigenvector
        alpha_1 = eigvecs[:, 0]
        norm_factor = np.sqrt(np.maximum(1e-5, eigvals[0] * np.sum(alpha_1**2)))
        alpha_1_normalized = alpha_1 / norm_factor
        kpca_eigenvectors.append(alpha_1_normalized)
        
    def get_kernel_coords(z):
        z_norm = z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-5)
        coords = np.zeros((z.shape[0], K))
        for k in range(K):
            cal_pts = kpca_cal_samples[k]
            k_x = compute_rbf_kernel(z_norm, cal_pts) # [B, N_sub]
            alpha = kpca_eigenvectors[k]
            coords[:, k] = np.abs(np.dot(k_x, alpha))
        return coords
        
    t0_kpca = time.time()
    kernel_coords = get_kernel_coords(test_distorted)
    t1_kpca = time.time()
    kpca_latency_ms = (t1_kpca - t0_kpca) * 1000 / len(test_distorted)
    
    kernel_preds = np.argmax(kernel_coords, axis=-1)
    kernel_acc = np.mean(kernel_preds == test_labels)
    
    # ----------------------------------------------------
    # Method 3: Parameterized Contrastive Projection Head (UN-CPH-SEP)
    # ----------------------------------------------------
    X_train = torch.tensor(sub_distorted, dtype=torch.float32)
    X_train_norm = X_train / (torch.norm(X_train, p=2, dim=-1, keepdim=True) + 1e-5)
    y_train = torch.tensor(sub_labels, dtype=torch.long)
    
    X_test = torch.tensor(test_distorted, dtype=torch.float32)
    X_test_norm = X_test / (torch.norm(X_test, p=2, dim=-1, keepdim=True) + 1e-5)
    y_test = torch.tensor(test_labels, dtype=torch.long)
    
    head = ContrastiveProjectionHead(input_dim=D, hidden_dim=64, output_dim=K)
    optimizer = optim.Adam(head.parameters(), lr=0.01, weight_decay=1e-4)
    tau = 0.1
    criterion = nn.CrossEntropyLoss()
    
    head.train()
    for epoch in range(150):
        optimizer.zero_grad()
        logits = head(X_train_norm)
        loss = criterion(logits / tau, y_train)
        loss.backward()
        optimizer.step()
        
    head.eval()
    with torch.no_grad():
        t0_cph = time.time()
        logits_test = head(X_test_norm)
        t1_cph = time.time()
        cph_latency_ms = (t1_cph - t0_cph) * 1000 / len(test_distorted)
        
        cph_preds = torch.argmax(logits_test, dim=-1)
        cph_acc = (cph_preds == y_test).float().mean().item()
        
    return linear_acc, kernel_acc, cph_acc, kpca_latency_ms, cph_latency_ms

# Run evaluation over 5 seeds
best_gamma = 2.0
linear_accs = []
kernel_accs = []
cph_accs = []
kpca_lats = []
cph_lats = []

print("="*60)
print(f"RUNNING FINAL 5-SEED EVALUATION WITH OPTIMAL rbf_gamma = {best_gamma}")
print("="*60)
for s_idx in range(num_seeds):
    seed = s_idx + 42
    lin_a, kern_a, cph_a, kpca_lat, cph_lat = run_evaluation_on_seed(seed, best_gamma)
    linear_accs.append(lin_a)
    kernel_accs.append(kern_a)
    cph_accs.append(cph_a)
    kpca_lats.append(kpca_lat)
    cph_lats.append(cph_lat)
    print(f"Seed {seed} | UN-PCA-SEP: {lin_a*100:.2f}% | UN-KPCA-SEP: {kern_a*100:.2f}% | UN-CPH-SEP: {cph_a*100:.2f}%")

print("="*60)
print("FINAL RESULTS SYNTHESIS (Mean +/- Std Across 5 Seeds)")
print("="*60)
print(f"UN-PCA-SEP (Linear PCA) Accuracy : {np.mean(linear_accs)*100:.2f}% +/- {np.std(linear_accs)*100:.2f}%")
print(f"UN-KPCA-SEP (Kernel PCA) Accuracy: {np.mean(kernel_accs)*100:.2f}% +/- {np.std(kernel_accs)*100:.2f}%")
print(f"UN-CPH-SEP (Contrastive Head) Acc: {np.mean(cph_accs)*100:.2f}% +/- {np.std(cph_accs)*100:.2f}%")
print("-"*60)
print(f"UN-KPCA-SEP Inference Latency: {np.mean(kpca_lats):.6f} ms per sample")
print(f"UN-CPH-SEP Inference Latency: {np.mean(cph_lats):.6f} ms per sample")
print(f"Latency Speedup Factor       : {np.mean(kpca_lats) / np.mean(cph_lats):.2f}x speedup")
print("="*60)

# Save result to a file for easy reading
with open("nonlinear_projection_results.txt", "w") as f:
    f.write("UN-PCA-SEP VS. UN-KPCA-SEP VS. UN-CPH-SEP COORDINATE PROJECTION UNDER SEVERE REPRESENTATION NON-LINEARITY\n")
    f.write("="*100 + "\n")
    for s_idx in range(num_seeds):
        seed = s_idx + 42
        f.write(f"Seed {seed} | UN-PCA-SEP: {linear_accs[s_idx]*100:.2f}% | UN-KPCA-SEP: {kernel_accs[s_idx]*100:.2f}% | UN-CPH-SEP: {cph_accs[s_idx]*100:.2f}%\n")
    f.write("="*100 + "\n")
    f.write(f"UN-PCA-SEP (Linear PCA) Mean Accuracy : {np.mean(linear_accs)*100:.2f}% +/- {np.std(linear_accs)*100:.2f}%\n")
    f.write(f"UN-KPCA-SEP (Kernel PCA) Mean Accuracy: {np.mean(kernel_accs)*100:.2f}% +/- {np.std(kernel_accs)*100:.2f}%\n")
    f.write(f"UN-CPH-SEP (Contrastive Head) Mean Acc: {np.mean(cph_accs)*100:.2f}% +/- {np.std(cph_accs)*100:.2f}%\n")
    f.write("-" * 100 + "\n")
    f.write(f"UN-KPCA-SEP Inference Latency: {np.mean(kpca_lats):.6f} ms per sample\n")
    f.write(f"UN-CPH-SEP Inference Latency: {np.mean(cph_lats):.6f} ms per sample\n")
    f.write(f"Latency Speedup Factor       : {np.mean(kpca_lats) / np.mean(cph_lats):.2f}x speedup\n")
    f.write("="*100 + "\n")
