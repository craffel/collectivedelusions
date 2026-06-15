import os
import time
import numpy as np
import tracemalloc
import warnings
import sys
from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky
from run_experiments import ShrunkGMM

warnings.filterwarnings('ignore')

class ShrunkFullGMM(GaussianMixture):
    def __init__(self, n_components=2, random_state=42, **kwargs):
        super().__init__(n_components=n_components, covariance_type='full', random_state=random_state, **kwargs)
        
    def fit(self, X, y=None):
        super().fit(X, y)
        resp = self.predict_proba(X)
        n_samples, n_features = X.shape
        shrunk_covariances = np.zeros_like(self.covariances_)
        
        for m in range(self.n_components):
            sigma = self.covariances_[m]
            mean = self.means_[m]
            w = resp[:, m]
            W = np.sum(w)
            if W < 1e-5:
                shrunk_covariances[m] = sigma
                continue
                
            Z = X - mean
            nu = np.trace(sigma) / n_features
            T = nu * np.identity(n_features)
            
            sum_var = 0.0
            for i in range(n_features):
                for j in range(n_features):
                    dev = Z[:, i] * Z[:, j] - sigma[i, j]
                    sum_var += np.sum(w**2 * dev**2)
                    
            sum_var = sum_var / (W**2 + 1e-8)
            sum_diff = np.sum((sigma - T)**2)
            
            if sum_diff < 1e-8:
                alpha_opt = 1.0
            else:
                alpha_opt = sum_var / sum_diff
            alpha_opt = np.clip(alpha_opt, 0.0, 1.0)
            
            shrunk_covariances[m] = (1.0 - alpha_opt) * sigma + alpha_opt * T
            
        self.covariances_ = shrunk_covariances
        self.precisions_cholesky_ = _compute_precision_cholesky(shrunk_covariances, 'full')
        return self

def profile_model(covariance_type, K, N=64, M=2, num_runs=50):
    # Generate dummy data
    X_train = np.random.normal(loc=0.5, scale=0.1, size=(N, K))
    X_test_single = np.random.normal(loc=0.5, scale=0.1, size=(1, K))
    
    # 1. Parameter storage size calculation
    # Storage parameters: weights (M), means (M * K), covariances
    # Diagonal covariance: M * K elements
    # Full covariance: M * K * (K+1)/2 elements
    if covariance_type == 'diag':
        num_params = M + M * K + M * K
    else:
        num_params = M + M * K + M * (K * (K + 1) // 2)
    storage_bytes = num_params * 4 # 32-bit floats
    
    # 2. Profile Calibration (Fit)
    tracemalloc.start()
    start_time = time.perf_counter()
    for _ in range(num_runs):
        if covariance_type == 'diag':
            model = ShrunkGMM(n_components=M, target_type='global_diagonal', random_state=42)
        else:
            model = ShrunkFullGMM(n_components=M, random_state=42)
        model.fit(X_train)
    end_time = time.perf_counter()
    _, peak_calib_ram = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    calib_latency_ms = ((end_time - start_time) / num_runs) * 1000
    
    # 3. Profile Inference (Score_samples for 1 sample)
    model.score_samples(X_test_single) # Warmup
    tracemalloc.start()
    start_time = time.perf_counter()
    for _ in range(num_runs * 10):
        model.score_samples(X_test_single)
    end_time = time.perf_counter()
    _, peak_inf_ram = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    inf_latency_ms = ((end_time - start_time) / (num_runs * 10)) * 1000
    
    return {
        "storage_bytes": storage_bytes,
        "calib_latency_ms": calib_latency_ms,
        "calib_peak_ram_kb": peak_calib_ram / 1024.0,
        "inf_latency_ms": inf_latency_ms,
        "inf_peak_ram_kb": peak_inf_ram / 1024.0
    }

def main():
    K_values = [4, 8, 16]
    print("Running Emulated On-Device Resource Profiling Benchmark...")
    print("=" * 110)
    
    print("DIAGONAL COVARIANCE SHRUNK GMM (SRC-DE):")
    print("-" * 110)
    print(f"{'K':4s} | {'Storage (B)':12s} | {'Calib Latency (ms)':20s} | {'Calib Peak RAM (KB)':20s} | {'Inference Latency (ms)':23s} | {'Inf Peak RAM (KB)':20s}")
    print("-" * 110)
    for K in K_values:
        res = profile_model('diag', K)
        print(f"{K:<4d} | {res['storage_bytes']:<12d} | {res['calib_latency_ms']:<20.4f} | {res['calib_peak_ram_kb']:<20.4f} | {res['inf_latency_ms']:<23.4f} | {res['inf_peak_ram_kb']:<20.4f}")
    print("-" * 110)
    
    print("\nFULL COVARIANCE SHRUNK GMM:")
    print("-" * 110)
    print(f"{'K':4s} | {'Storage (B)':12s} | {'Calib Latency (ms)':20s} | {'Calib Peak RAM (KB)':20s} | {'Inference Latency (ms)':23s} | {'Inf Peak RAM (KB)':20s}")
    print("-" * 110)
    for K in K_values:
        res = profile_model('full', K)
        print(f"{K:<4d} | {res['storage_bytes']:<12d} | {res['calib_latency_ms']:<20.4f} | {res['calib_peak_ram_kb']:<20.4f} | {res['inf_latency_ms']:<23.4f} | {res['inf_peak_ram_kb']:<20.4f}")
    print("-" * 110)

if __name__ == "__main__":
    main()
