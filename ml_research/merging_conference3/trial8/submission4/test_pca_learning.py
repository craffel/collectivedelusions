import torch
import torch.nn as nn
import numpy as np

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

# Generate calibration samples
calib_x = []
calib_y = []
calib_class_y = []
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

# Extract Layer 3 features
W_base = {}
for l in range(1, 14):
    W_base[l] = 0.05 * torch.eye(D)

h_calib = calib_x.clone()
for l in range(1, 4):
    h_calib = h_calib + torch.relu(h_calib @ W_base[l])
z_calib = h_calib.clone()

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
    S_weights = S_k[:10] / (S_k[0] + 1e-8)
    V_pca_cent[k] = V_k[:, :10] * S_weights.unsqueeze(0)

# Compute PCA norms
calib_pca_norms = torch.zeros(64, K)
for b in range(K):
    calib_pca_norms[:, b] = (z_calib @ V_pca_cent[b]).norm(dim=1)

# Let's see raw predictions at tau = 1.0
preds_initial = torch.argmax(calib_pca_norms, dim=1)
acc_initial = (preds_initial == calib_y).float().mean().item() * 100.0
print(f"Initial routing accuracy at tau=1.0: {acc_initial:.2f}%")

# Let's train Temp-Only ERM Router
class TempOnlyERMRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_tau = nn.Parameter(torch.zeros(K))
    def forward(self, x):
        return x / torch.exp(self.log_tau)

router = TempOnlyERMRouter()
optimizer = torch.optim.Adam(router.parameters(), lr=0.05)
criterion = nn.CrossEntropyLoss()

for epoch in range(100):
    optimizer.zero_grad()
    logits = router(calib_pca_norms)
    loss = criterion(logits, calib_y)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0 or epoch == 0:
        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            acc = (preds == calib_y).float().mean().item() * 100.0
            print(f"Epoch {epoch+1:02d} | Loss: {loss.item():.4f} | Accuracy: {acc:.2f}% | Log-Tau: {router.log_tau.detach().numpy()}")
