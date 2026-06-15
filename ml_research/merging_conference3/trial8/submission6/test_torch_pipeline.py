import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score

# Set random seed
torch.manual_seed(42)
np.random.seed(42)

# Dimensions
D_in = 64
D = 192
r = 8
K = 3 # In-distribution tasks
num_classes = 5

# Shared backbone
class SharedBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(D_in, D)
        # Freeze backbone
        for p in self.parameters():
            p.requires_grad = False
    def forward(self, x):
        return torch.relu(self.proj(x))

backbone = SharedBackbone()

# Task-specific domain shift parameters (scales and shifts)
task_scales = []
task_shifts = []
for k in range(K + 1): # 3 in-distribution, 1 OOD
    scale = torch.rand(D) * 1.5 + 0.5
    shift = torch.randn(D) * 2.0
    task_scales.append(scale)
    task_shifts.append(shift)

# Task prototypes and class centers in input space
task_prototypes = []
for k in range(K + 1):
    v = torch.randn(D_in)
    v /= torch.norm(v)
    task_prototypes.append(v)

class_centers = []
for c in range(num_classes):
    v = torch.randn(D_in)
    v /= torch.norm(v)
    class_centers.append(v)

def generate_data(task_idx, num_samples):
    X = []
    y = []
    proto = task_prototypes[task_idx]
    for _ in range(num_samples):
        c = np.random.randint(num_classes)
        sample = proto * 1.0 + class_centers[c] * 1.5 + torch.randn(D_in) * 0.15
        X.append(sample)
        y.append(c)
    return torch.stack(X), torch.tensor(y)

# Get the perturbed activation for task k
def get_perturbed_activation(x, k):
    h_base = backbone(x)
    return h_base * task_scales[k] + task_shifts[k]

# Train task-specific adapters
class AdapterModel(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k
        self.A = nn.Parameter(torch.randn(D, r) * 0.05)
        self.B = nn.Parameter(torch.randn(r, D) * 0.05)
        self.head = nn.Linear(D, num_classes)
        
    def forward(self, x):
        h = get_perturbed_activation(x, self.k)
        h_adapt = h @ self.A @ self.B
        out = h + h_adapt
        logits = self.head(out)
        return logits, h

adapters = []
for k in range(K):
    model = AdapterModel(k)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion_cls = nn.CrossEntropyLoss()
    
    X_train, y_train = generate_data(k, 512)
    
    for epoch in range(150):
        model.train()
        optimizer.zero_grad()
        logits, h = model(X_train)
        loss_cls = criterion_cls(logits, y_train)
        loss_rec = torch.mean((h - h @ model.A @ model.B)**2)
        loss = loss_cls + 1.5 * loss_rec
        loss.backward()
        optimizer.step()
        
    model.eval()
    adapters.append(model)

# Extract Q_k
Qs = []
for k in range(K):
    A_k = adapters[k].A.detach()
    Q_k, _ = torch.linalg.qr(A_k)
    Qs.append(Q_k)

# Calibration and Test Data
cal_data = {k: generate_data(k, 64) for k in range(K)}
test_data = {k: generate_data(k, 256) for k in range(K)}

# Generate OOD test dataset (500 samples)
ood_X, ood_y = generate_data(3, 500) # Task 3 is OOD

# Precompute centroids
centroids = []
for k in range(K):
    X, _ = cal_data[k]
    with torch.no_grad():
        h = get_perturbed_activation(X, k)
        mean_h = h.mean(dim=0)
        centroids.append(mean_h / torch.norm(mean_h))

# Fit GMM for SPS-ZCA
sps_dispersions = []
for k in range(K):
    X, _ = cal_data[k]
    with torch.no_grad():
        h = get_perturbed_activation(X, k)
        sims = h @ centroids[k]
        sps_dispersions.append(sims.std().item() + 1e-5)

def get_sps_coords(h):
    coords = []
    for k in range(K):
        sims = h @ centroids[k]
        coords.append(sims / sps_dispersions[k])
    return torch.stack(coords, dim=1)

cal_coords = []
for k in range(K):
    X, _ = cal_data[k]
    with torch.no_grad():
        h = get_perturbed_activation(X, k)
        cal_coords.append(get_sps_coords(h))
cal_coords = torch.cat(cal_coords, dim=0).cpu().numpy()
gmm = GaussianMixture(n_components=K, covariance_type='diag', random_state=42)
gmm.fit(cal_coords)

# Evaluate OOD AUROC
# We collect scores for In-distribution and Out-of-distribution validation data
in_X, _ = test_data[0] # Let's use test_data[0] as in-distribution
ood_X_samples = ood_X[:256]

# Compute activations
with torch.no_grad():
    h_in = get_perturbed_activation(in_X, 0)
    h_ood = get_perturbed_activation(ood_X_samples, 3) # Task 3 domain shift

# 1. SABLE score (max similarity to centroids)
sable_scores_in = []
sable_scores_ood = []
for k in range(K):
    sable_scores_in.append(h_in @ centroids[k] / (torch.norm(h_in, dim=1) * torch.norm(centroids[k]) + 1e-8))
    sable_scores_ood.append(h_ood @ centroids[k] / (torch.norm(h_ood, dim=1) * torch.norm(centroids[k]) + 1e-8))
sable_in = torch.max(torch.stack(sable_scores_in, dim=1), dim=1)[0].cpu().numpy()
sable_ood = torch.max(torch.stack(sable_scores_ood, dim=1), dim=1)[0].cpu().numpy()

# 2. SPS-ZCA score (GMM density)
coords_in = get_sps_coords(h_in).cpu().numpy()
coords_ood = get_sps_coords(h_ood).cpu().numpy()
sps_in = gmm.score_samples(coords_in)
sps_ood = gmm.score_samples(coords_ood)

# 3. LSPR score (max projection score u_k)
lspr_scores_in = []
lspr_scores_ood = []
for k in range(K):
    lspr_scores_in.append(torch.norm(h_in @ Qs[k], dim=1) / (torch.norm(h_in, dim=1) + 1e-8))
    lspr_scores_ood.append(torch.norm(h_ood @ Qs[k], dim=1) / (torch.norm(h_ood, dim=1) + 1e-8))
lspr_in = torch.max(torch.stack(lspr_scores_in, dim=1), dim=1)[0].cpu().numpy()
lspr_ood = torch.max(torch.stack(lspr_scores_ood, dim=1), dim=1)[0].cpu().numpy()

# Calculate AUROC
y_true = np.concatenate([np.zeros(256), np.ones(256)]) # 0: In, 1: OOD
# Since higher score means In-distribution, OOD score is -score
sable_score = np.concatenate([-sable_in, -sable_ood])
sps_score = np.concatenate([-sps_in, -sps_ood])
lspr_score = np.concatenate([-lspr_in, -lspr_ood])

# Calculate AUROC dynamically
sable_auroc = roc_auc_score(y_true, sable_score)
sps_auroc = roc_auc_score(y_true, sps_score)
lspr_auroc = roc_auc_score(y_true, lspr_score)

print(f"SABLE OOD AUROC:   {sable_auroc:.4f}")
print(f"SPS-ZCA OOD AUROC: {sps_auroc:.4f}")
print(f"LSPR OOD AUROC:    {lspr_auroc:.4f}")
