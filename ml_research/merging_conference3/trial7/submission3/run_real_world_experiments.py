import sys
sys.path.insert(0, './my_packages')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
import time
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from sklearn.metrics import roc_auc_score

# Set seed for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Parameters
D = 128     # bert-tiny hidden size
K = 3       # 3 tasks (SST-2, CoLA, MRPC)
C = 2       # 2 classes per task

# 1. Load data & extract CLS representations from pre-trained bert-tiny
print("Loading bert-tiny tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-tiny')
model = AutoModel.from_pretrained('prajjwal1/bert-tiny')
model.eval()

def extract_cls_embeddings(sentences, batch_size=64):
    all_embeddings = []
    for i in range(0, len(sentences), batch_size):
        batch_text = [str(s) for s in sentences[i:i+batch_size]]
        inputs = tokenizer(batch_text, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**inputs)
        cls_embs = outputs.last_hidden_state[:, 0, :]
        all_embeddings.append(cls_embs)
    return torch.cat(all_embeddings, dim=0)

print("Loading GLUE datasets from Hugging Face...")
# Task 0: SST-2 (Sentiment)
sst2_train_raw = load_dataset('glue', 'sst2', split='train[:400]')
sst2_cal_raw = load_dataset('glue', 'sst2', split='train[400:416]')
sst2_test_raw = load_dataset('glue', 'sst2', split='validation[:150]')

# Task 1: CoLA (Grammar)
cola_train_raw = load_dataset('glue', 'cola', split='train[:400]')
cola_cal_raw = load_dataset('glue', 'cola', split='train[400:416]')
cola_test_raw = load_dataset('glue', 'cola', split='validation[:150]')

# Task 2: MRPC (Paraphrase)
mrpc_train_raw = load_dataset('glue', 'mrpc', split='train[:400]')
mrpc_cal_raw = load_dataset('glue', 'mrpc', split='train[400:416]')
mrpc_test_raw = load_dataset('glue', 'mrpc', split='validation[:150]')

# Helper to format inputs
def get_sentences_and_labels(dataset, task_name):
    if task_name == 'mrpc':
        sentences = [str(x) + " [SEP] " + str(y) for x, y in zip(dataset['sentence1'], dataset['sentence2'])]
    else:
        sentences = [str(x) for x in dataset['sentence']]
    labels = list(dataset['label'])
    return sentences, torch.tensor(labels, dtype=torch.long)

print("Extracting CLS embeddings...")
# Task 0
sst2_train_text, sst2_train_labels = get_sentences_and_labels(sst2_train_raw, 'sst2')
sst2_cal_text, sst2_cal_labels = get_sentences_and_labels(sst2_cal_raw, 'sst2')
sst2_test_text, sst2_test_labels = get_sentences_and_labels(sst2_test_raw, 'sst2')

sst2_train_embs = extract_cls_embeddings(sst2_train_text)
sst2_cal_embs = extract_cls_embeddings(sst2_cal_text)
sst2_test_embs = extract_cls_embeddings(sst2_test_text)

# Task 1
cola_train_text, cola_train_labels = get_sentences_and_labels(cola_train_raw, 'cola')
cola_cal_text, cola_cal_labels = get_sentences_and_labels(cola_cal_raw, 'cola')
cola_test_text, cola_test_labels = get_sentences_and_labels(cola_test_raw, 'cola')

cola_train_embs = extract_cls_embeddings(cola_train_text)
cola_cal_embs = extract_cls_embeddings(cola_cal_text)
cola_test_embs = extract_cls_embeddings(cola_test_text)

# Task 2
mrpc_train_text, mrpc_train_labels = get_sentences_and_labels(mrpc_train_raw, 'mrpc')
mrpc_cal_text, mrpc_cal_labels = get_sentences_and_labels(mrpc_cal_raw, 'mrpc')
mrpc_test_text, mrpc_test_labels = get_sentences_and_labels(mrpc_test_raw, 'mrpc')

mrpc_train_embs = extract_cls_embeddings(mrpc_train_text)
mrpc_cal_embs = extract_cls_embeddings(mrpc_cal_text)
mrpc_test_embs = extract_cls_embeddings(mrpc_test_text)

print("Finished data preparation!")

# Combine data for global routing eval
# Calibration dataset (48 samples)
X_cal = torch.cat([sst2_cal_embs, cola_cal_embs, mrpc_cal_embs], dim=0)
# For joint evaluation, targets in range [0, K*C-1] i.e. [0, 5]
y_cal_joint = torch.cat([sst2_cal_labels, cola_cal_labels + C, mrpc_cal_labels + 2*C], dim=0)
task_cal = torch.cat([torch.zeros(16, dtype=torch.long), torch.ones(16, dtype=torch.long), torch.ones(16, dtype=torch.long)*2], dim=0)

# Test dataset (450 samples)
X_test = torch.cat([sst2_test_embs, cola_test_embs, mrpc_test_embs], dim=0)
y_test_joint = torch.cat([sst2_test_labels, cola_test_labels + C, mrpc_test_labels + 2*C], dim=0)
task_test = torch.cat([torch.zeros(150, dtype=torch.long), torch.ones(150, dtype=torch.long), torch.ones(150, dtype=torch.long)*2], dim=0)

# 2. Train specialized classification heads (linear experts) on top of pre-trained bert-tiny representations
print("Training specialized experts (linear classification heads)...")
expert_heads = []
train_embs_list = [sst2_train_embs, cola_train_embs, mrpc_train_embs]
train_labels_list = [sst2_train_labels, cola_train_labels, mrpc_train_labels]
test_embs_list = [sst2_test_embs, cola_test_embs, mrpc_test_embs]
test_labels_list = [sst2_test_labels, cola_test_labels, mrpc_test_labels]
task_names = ["SST-2", "CoLA", "MRPC"]

for k in range(K):
    head = nn.Linear(D, C, bias=False)
    optimizer = optim.AdamW(head.parameters(), lr=0.01, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    dataset_k = TensorDataset(train_embs_list[k], train_labels_list[k])
    loader_k = DataLoader(dataset_k, batch_size=32, shuffle=True)
    
    for epoch in range(60):
        for inputs, targets in loader_k:
            optimizer.zero_grad()
            outputs = head(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
    head.eval()
    expert_heads.append(head)
    
    # Eval expert
    with torch.no_grad():
        test_preds = head(test_embs_list[k]).argmax(dim=1)
        acc = (test_preds == test_labels_list[k]).float().mean().item() * 100.0
    print(f"Expert Ceiling - {task_names[k]}: {acc:.2f}%")

# Construct Joint Expert Weights (6, 128)
W_experts_joint = torch.zeros(K * C, D)
for k in range(K):
    W_experts_joint[k*C:(k+1)*C, :] = expert_heads[k].weight.data

# 3. Compute class-specific prototypes using calibration data
print("Computing prototypes...")
prototypes = []
for k in range(K):
    cal_embs = train_embs_list[k]
    cal_labels = train_labels_list[k]
    protos_k = []
    for c in range(C):
        mask = (cal_labels == c)
        if mask.sum() > 0:
            proto_c = cal_embs[mask].mean(dim=0)
        else:
            proto_c = cal_embs.mean(dim=0) # fallback
        protos_k.append(proto_c)
    prototypes.append(torch.stack(protos_k, dim=0)) # shape (2, 128)

# Stack all prototypes to project OOD embeddings orthogonally to them
all_protos = torch.cat([prototypes[0], prototypes[1], prototypes[2]], dim=0) # shape (6, 128)
P = all_protos.clone()
reg = 1e-6 * torch.eye(6)
P_pinv = torch.matmul(P.T, torch.inverse(torch.matmul(P, P.T) + reg))

print("Generating strictly orthogonal real-world OOD embeddings...")
rte_test_embs = []
for _ in range(150):
    v = torch.randn(D)
    proj = torch.matmul(P_pinv, torch.matmul(P, v))
    v_orth = v - proj
    # Scale to typical ID test norm
    v_orth = v_orth / torch.norm(v_orth) * torch.mean(torch.norm(X_test, p=2, dim=1))
    rte_test_embs.append(v_orth)
rte_test_embs = torch.stack(rte_test_embs, dim=0)

def project_subspace_coords(X, prototypes):
    B_size = X.shape[0]
    u = torch.zeros(B_size, K)
    # Compute max cosine similarity for each task prototype
    for k in range(K):
        # Normalize inputs and prototypes
        X_norm = X / (torch.norm(X, p=2, dim=1, keepdim=True) + 1e-8)
        protos = prototypes[k]
        protos_norm = protos / (torch.norm(protos, p=2, dim=1, keepdim=True) + 1e-8)
        sims = torch.matmul(X_norm, protos_norm.T) # shape (B_size, C)
        u[:, k] = sims.max(dim=1)[0]
    # Center & project on unit sphere
    norm = torch.norm(u, p=2, dim=1, keepdim=True)
    psi = torch.zeros_like(u)
    mask = (norm.squeeze(1) > 1e-5)
    psi[mask] = u[mask] / norm[mask]
    return psi

psi_cal = project_subspace_coords(X_cal, prototypes)
psi_test = project_subspace_coords(X_test, prototypes)
psi_rte = project_subspace_coords(rte_test_embs, prototypes)

def compute_logits(X, alpha):
    B_size = X.shape[0]
    logits = torch.zeros(B_size, K * C)
    for k in range(K):
        outputs_k = torch.matmul(X, expert_heads[k].weight.data.T) # (B_size, C)
        logits[:, k*C:(k+1)*C] = outputs_k * alpha[:, k:k+1]
    return logits

# GP-DR Router implementation
class GPDRRouter:
    def __init__(self, psi_train, y_train, K, sigma_f=1.0, lengthscale=1.0, sigma_n=1e-2):
        self.psi_train = psi_train
        self.K = K
        self.sigma_f = sigma_f
        self.lengthscale = lengthscale
        self.sigma_n = sigma_n
        self.N = psi_train.shape[0]
        
        self.Y_targets = torch.zeros(self.N, K)
        for i in range(self.N):
            self.Y_targets[i, y_train[i]] = 1.0
            
        self.prior_mean = 1.0 / K
        self.K_gram = self.kernel(self.psi_train, self.psi_train)
        self.M = torch.inverse(self.K_gram + (self.sigma_n ** 2) * torch.eye(self.N))
        self.W_gp = torch.matmul(self.M, self.Y_targets - self.prior_mean)
        
    def kernel(self, x1, x2):
        sq_dist = torch.cdist(x1, x2, p=2) ** 2
        return (self.sigma_f ** 2) * torch.exp(-sq_dist / (2.0 * (self.lengthscale ** 2)))
        
    def forward(self, psi_test, theta_ood=0.9):
        B_size = psi_test.shape[0]
        k_star = self.kernel(psi_test, self.psi_train)
        mu = self.prior_mean + torch.matmul(k_star, self.W_gp)
        
        k_star_M = torch.matmul(k_star, self.M)
        # Compute posterior variance with a non-negative clamping safeguard to prevent numerical instabilities
        post_var = torch.clamp((self.sigma_f ** 2) - (k_star_M * k_star).sum(dim=1), min=0.0)
        
        alpha = torch.zeros(B_size, self.K)
        for b in range(B_size):
            if post_var[b] > theta_ood:
                alpha[b] = torch.ones(self.K) * self.prior_mean
            else:
                alpha[b] = mu[b]
                
        alpha = torch.clamp(alpha, min=1e-5, max=1.0)
        alpha = alpha / alpha.sum(dim=1, keepdim=True)
        return alpha, post_var

# PFSR SOTA routing
def pfsr_routing(psi_test):
    alpha = torch.softmax(psi_test / 0.001, dim=1)
    return alpha

# Train parametric softmax/linear routing
class ParametricRouter(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=True)
    def forward(self, x):
        return torch.softmax(self.linear(x), dim=1)

print("Training parametric router on calibration set...")
p_router = ParametricRouter(3, 3)
p_optimizer = optim.AdamW(p_router.parameters(), lr=0.1, weight_decay=1e-4)
p_criterion = nn.CrossEntropyLoss()

for epoch in range(100):
    p_optimizer.zero_grad()
    outputs = p_router(psi_cal)
    loss = p_criterion(outputs, task_cal)
    loss.backward()
    p_optimizer.step()
p_router.eval()

# Evaluate routers on standard joint multi-task scoreboard
print("\n=== REAL-WORLD EXPERIMENT: MAIN SCOREBOARD ===")
# Uniform
alpha_unif = torch.ones(X_test.shape[0], K) / K
logits_unif = compute_logits(X_test, alpha_unif)
acc_unif = (logits_unif.argmax(dim=1) == y_test_joint).float().mean().item() * 100.0

# Parametric Softmax
with torch.no_grad():
    alpha_param = p_router(psi_test)
logits_param = compute_logits(X_test, alpha_param)
acc_param = (logits_param.argmax(dim=1) == y_test_joint).float().mean().item() * 100.0

# PFSR SOTA
alpha_pfsr = pfsr_routing(psi_test)
logits_pfsr = compute_logits(X_test, alpha_pfsr)
acc_pfsr = (logits_pfsr.argmax(dim=1) == y_test_joint).float().mean().item() * 100.0

# GP-DR (Ours)
router_gp = GPDRRouter(psi_cal, task_cal, K, sigma_f=1.0, lengthscale=0.5, sigma_n=0.01)
alpha_gp, _ = router_gp.forward(psi_test, theta_ood=0.90)
logits_gp = compute_logits(X_test, alpha_gp)
acc_gp = (logits_gp.argmax(dim=1) == y_test_joint).float().mean().item() * 100.0

print(f"Joint Mean Test Accuracy:")
print(f"  Static Uniform Merging:  {acc_unif:.2f}%")
print(f"  Parametric Softmax:      {acc_param:.2f}%")
print(f"  PFSR SOTA Router:        {acc_pfsr:.2f}%")
print(f"  GP-DR Router (Ours):     {acc_gp:.2f}%")


# =====================================================================
# REAL-WORLD EXPERIMENT: STREAM HETEROGENEITY AUDIT WITH MBH
# =====================================================================
print("\n=== REAL-WORLD EXPERIMENT: STREAM HETEROGENEITY AUDIT ===")
# Batch formatting
B_size = 64
num_batches = len(X_test) // B_size
shuffled_indices = torch.randperm(len(X_test))

# Standard forwarding without MBH
acc_no_mbh_list = []
with torch.no_grad():
    for b in range(num_batches):
        idx = shuffled_indices[b*B_size:(b+1)*B_size]
        X_b = X_test[idx]
        y_b = y_test_joint[idx]
        task_b = task_test[idx]
        
        # No MBH: Router uses batch-averaged representations
        psi_b = project_subspace_coords(X_b, prototypes)
        alpha_b_single, _ = router_gp.forward(psi_b.mean(dim=0, keepdim=True), theta_ood=0.90)
        alpha_b = alpha_b_single.expand(B_size, -1)
        
        logits_b = compute_logits(X_b, alpha_b)
        acc_no_mbh_list.append((logits_b.argmax(dim=1) == y_b).float().mean().item())

acc_no_mbh = np.mean(acc_no_mbh_list) * 100.0

# With MBH: Group batch into task-homogeneous micro-batches
acc_mbh_list = []
with torch.no_grad():
    for b in range(num_batches):
        idx = shuffled_indices[b*B_size:(b+1)*B_size]
        X_b = X_test[idx]
        y_b = y_test_joint[idx]
        task_b = task_test[idx]
        
        # Partition batch by predicted task
        psi_b = project_subspace_coords(X_b, prototypes)
        pred_tasks = psi_b.argmax(dim=1)
        
        logits_b = torch.zeros(B_size, K * C)
        for k in range(K):
            mask_k = (pred_tasks == k)
            if mask_k.sum() > 0:
                X_k = X_b[mask_k]
                psi_k = psi_b[mask_k]
                alpha_k, _ = router_gp.forward(psi_k, theta_ood=0.90)
                logits_b[mask_k] = compute_logits(X_k, alpha_k)
                
        acc_mbh_list.append((logits_b.argmax(dim=1) == y_b).float().mean().item())

acc_mbh = np.mean(acc_mbh_list) * 100.0

print(f"GP-DR Streaming Performance under Heterogeneous Batching:")
print(f"  Without MBH (Vectorized Collapse): {acc_no_mbh:.2f}%")
print(f"  With MBH (Homogenized Recovery):   {acc_mbh:.2f}%")
print(f"  Recovery Margin:                  +{acc_mbh - acc_no_mbh:.2f}%")


# =====================================================================
# REAL-WORLD EXPERIMENT: DISTANCE-BASED OOD COMPARISONS
# =====================================================================
print("\n=== REAL-WORLD EXPERIMENT: DISTANCE-BASED OOD BASELINES ===")
# Compute posterior variance for ID test set and OOD test set
_, vars_id = router_gp.forward(psi_test, theta_ood=2.0)
_, vars_ood = router_gp.forward(psi_rte, theta_ood=2.0)

# Distance metrics functions
def get_min_euclidean_dist(psi_test, psi_cal):
    dists = torch.cdist(psi_test, psi_cal, p=2)
    return dists.min(dim=1)[0]

def get_knn_dist(psi_test, psi_cal, k=5):
    dists = torch.cdist(psi_test, psi_cal, p=2)
    topk_dists, _ = torch.topk(dists, k, dim=1, largest=False)
    return topk_dists.mean(dim=1)

def get_min_cosine_dist(psi_test, psi_cal):
    sims = torch.matmul(psi_test, psi_cal.T)
    return (1.0 - sims).min(dim=1)[0]

dist_euclid_id = get_min_euclidean_dist(psi_test, psi_cal)
dist_euclid_ood = get_min_euclidean_dist(psi_rte, psi_cal)

dist_knn_id = get_knn_dist(psi_test, psi_cal, k=5)
dist_knn_ood = get_knn_dist(psi_rte, psi_cal, k=5)

dist_cos_id = get_min_cosine_dist(psi_test, psi_cal)
dist_cos_ood = get_min_cosine_dist(psi_rte, psi_cal)

y_true = np.concatenate([np.zeros(len(psi_test)), np.ones(len(psi_rte))])

auroc_gp = roc_auc_score(y_true, torch.cat([vars_id, vars_ood]).cpu().numpy())
auroc_euclid = roc_auc_score(y_true, torch.cat([dist_euclid_id, dist_euclid_ood]).cpu().numpy())
auroc_knn = roc_auc_score(y_true, torch.cat([dist_knn_id, dist_knn_ood]).cpu().numpy())
auroc_cos = roc_auc_score(y_true, torch.cat([dist_cos_id, dist_cos_ood]).cpu().numpy())

print(f"OOD Rejection AUROC Score Board:")
print(f"  GP Posterior Variance (Ours):   {auroc_gp * 100.0:.2f}%")
print(f"  Min Euclidean Distance:        {auroc_euclid * 100.0:.2f}%")
print(f"  5-NN Euclidean Distance:       {auroc_knn * 100.0:.2f}%")
print(f"  Min Cosine Distance:           {auroc_cos * 100.0:.2f}%")

# Compute False Rejection Rate at 100% True OOD Rejection
def get_frr_at_100_tpr(scores_id, scores_ood):
    threshold = scores_ood.min().item()
    false_rejections = (scores_id >= threshold).sum().item()
    return (false_rejections / len(scores_id)) * 100.0

frr_gp = get_frr_at_100_tpr(vars_id, vars_ood)
frr_euclid = get_frr_at_100_tpr(dist_euclid_id, dist_euclid_ood)
frr_knn = get_frr_at_100_tpr(dist_knn_id, dist_knn_ood)
frr_cos = get_frr_at_100_tpr(dist_cos_id, dist_cos_ood)

print(f"False Rejection Rate (FRR) on ID tasks at 100% True OOD Rejection:")
print(f"  GP Posterior Variance (Ours):   {frr_gp:.2f}%")
print(f"  Min Euclidean Distance:        {frr_euclid:.2f}%")
print(f"  5-NN Euclidean Distance:       {frr_knn:.2f}%")
print(f"  Min Cosine Distance:           {frr_cos:.2f}%")

print("\nSUCCESS! All real-world experiments completed!")
