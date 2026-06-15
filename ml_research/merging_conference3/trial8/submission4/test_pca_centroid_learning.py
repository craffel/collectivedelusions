import torch
import torch.nn as nn
import numpy as np
import scipy.stats as stats

torch.manual_seed(42)
np.random.seed(42)

D = 192
K = 4
num_classes = 10
block_size = 48
overlap_size = 0

# Set up orthogonal dimensions
task_dims = {}
for k in range(K):
    task_dims[k] = list(range(k*block_size, (k+1)*block_size))

class_prototypes = {}
for k in range(K):
    subspace_size = len(task_dims[k])
    U, S, V = torch.svd(torch.randn(subspace_size, num_classes))
    prototypes = torch.zeros(num_classes, D)
    for idx, d_idx in enumerate(task_dims[k]):
        prototypes[:, d_idx] = U.t()[:num_classes, idx]
    class_prototypes[k] = prototypes

# Noise levels
noise_levels = [0.01, 0.05, 0.28, 1.35]

# Generate calibration and test samples
calib_x, calib_y, calib_class_y = [], [], []
for k in range(K):
    for i in range(16):
        c = np.random.randint(0, num_classes)
        x = class_prototypes[k][c] + noise_levels[k] * torch.randn(D)
        calib_x.append(x)
        calib_y.append(k)
        calib_class_y.append(c)
calib_x = torch.stack(calib_x)
calib_y = torch.tensor(calib_y)
calib_class_y = torch.tensor(calib_class_y)

test_x, test_y, test_class_y = [], [], []
for k in range(K):
    for i in range(250):
        c = np.random.randint(0, num_classes)
        x = class_prototypes[k][c] + noise_levels[k] * torch.randn(D)
        test_x.append(x)
        test_y.append(k)
        test_class_y.append(c)
test_x = torch.stack(test_x)
test_y = torch.tensor(test_y)

# Extract Layer 3 features
W_base = {}
for l in range(1, 14):
    W_base[l] = 0.05 * torch.eye(D)

h_calib = calib_x.clone()
for l in range(1, 4):
    h_calib = h_calib + torch.relu(h_calib @ W_base[l])
z_calib = h_calib.clone()

h_test = test_x.clone()
for l in range(1, 4):
    h_test = h_test + torch.relu(h_test @ W_base[l])
z_test = h_test.clone()

# SVD on class centroids
V_pca_cent = {}
for k in range(K):
    class_centroids = []
    for c in range(num_classes):
        mask = (calib_y == k) & (calib_class_y == c)
        if mask.sum() > 0:
            class_centroids.append(z_calib[mask].mean(dim=0))
        else:
            class_centroids.append(torch.zeros(D))
    class_centroids = torch.stack(class_centroids)
    U_k, S_k, V_k = torch.svd(class_centroids)
    V_pca_cent[k] = V_k[:, :10]

# Compute PCA norms
calib_pca_norms = torch.zeros(64, K)
test_pca_norms = torch.zeros(1000, K)
for b in range(K):
    calib_pca_norms[:, b] = (z_calib @ V_pca_cent[b]).norm(dim=1)
    test_pca_norms[:, b] = (z_test @ V_pca_cent[b]).norm(dim=1)

# PAC Router
class PACRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_tau = nn.Parameter(torch.zeros(K))
    def forward(self, x):
        return x / torch.exp(self.log_tau)

router = PACRouter()
optimizer = torch.optim.Adam(router.parameters(), lr=0.05)
N_calib = 64
sigma_0_sq = 1.0

for epoch in range(100):
    optimizer.zero_grad()
    logits = router(calib_pca_norms)
    q = torch.softmax(logits, dim=1)
    risk = 1.0 - q[range(N_calib), calib_y].mean()
    kl = (router.log_tau ** 2).sum() / (2.0 * sigma_0_sq)
    bound = risk + torch.sqrt((kl + np.log(2.0 * np.sqrt(N_calib) / 0.05)) / (2.0 * N_calib))
    bound.backward()
    optimizer.step()

# Evaluate on test set
with torch.no_grad():
    logits_test = router(test_pca_norms)
    preds_test = torch.argmax(logits_test, dim=1)
    acc_test = (preds_test == test_y).float().mean().item() * 100.0
    print(f"PAC-ZCA (Centroid SVD) Test Routing Accuracy: {acc_test:.2f}%")
    for k in range(K):
        mask = (test_y == k)
        task_acc = (preds_test[mask] == k).float().mean().item() * 100.0
        print(f"  Task {k} accuracy: {task_acc:.2f}%")
