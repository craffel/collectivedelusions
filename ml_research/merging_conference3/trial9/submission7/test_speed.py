import numpy as np
import time

D = 192
L = 14
K = 4
batch_size = 1000

h = np.random.normal(size=(batch_size, D))
h /= np.linalg.norm(h, axis=1, keepdims=True)

centroids = np.random.normal(size=(K, D))
centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)

alpha = np.random.uniform(size=(batch_size, K))
alpha /= np.sum(alpha, axis=1, keepdims=True)

C = np.random.uniform(size=(batch_size, K))

def cos_sim(A, B):
    norm_A = np.linalg.norm(A, axis=1, keepdims=True)
    norm_B = np.linalg.norm(B, axis=1, keepdims=True)
    dot = np.dot(A, B.T)
    return dot / (norm_A * norm_B.T + 1e-9)

# Original method
t0 = time.perf_counter()
for _ in range(500):
    bar_mu = np.dot(alpha, centroids)
    sims_mu_centroids = cos_sim(bar_mu, centroids)
    sims_h_centroids = cos_sim(h, centroids)
    sims_h_bar_mu = np.sum(h * bar_mu, axis=1, keepdims=True) / (np.linalg.norm(h, axis=1, keepdims=True) * np.linalg.norm(bar_mu, axis=1, keepdims=True) + 1e-9)
    A_orig = np.sum(C * (sims_mu_centroids - sims_h_centroids * sims_h_bar_mu), axis=1)
t_orig = (time.perf_counter() - t0) * 1000
print(f"Original method time: {t_orig:.3f} ms")

# Optimized method
t0 = time.perf_counter()
for _ in range(500):
    bar_mu = np.dot(alpha, centroids)
    norm_bar = np.linalg.norm(bar_mu, axis=1, keepdims=True)
    bar_mu_norm = bar_mu / (norm_bar + 1e-9)
    sims_mu_centroids = np.dot(bar_mu_norm, centroids.T)
    sims_h_centroids = np.dot(h, centroids.T)
    sims_h_bar_mu = np.sum(h * bar_mu_norm, axis=1, keepdims=True)
    A_opt = np.sum(C * (sims_mu_centroids - sims_h_centroids * sims_h_bar_mu), axis=1)
t_opt = (time.perf_counter() - t0) * 1000
print(f"Optimized method time: {t_opt:.3f} ms")

# Check if they are identical
assert np.allclose(A_orig, A_opt, atol=1e-5), "Methods are not equivalent!"
print("Equivalence verified!")
