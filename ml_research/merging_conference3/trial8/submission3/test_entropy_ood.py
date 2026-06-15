import torch
import numpy as np
from run_experiments import model, test_x, test_task, centroids_layer3, test_is_ood

model.eval()

def get_routing_probs(x_batch):
    with torch.no_grad():
        h = x_batch
        for block in model.blocks[:3]:
            W = block.W_base
            max_val = torch.max(torch.abs(W), dim=1, keepdim=True)[0]
            S = max_val / 7.0
            S = torch.clamp(S, min=1e-8)
            Q = torch.round(torch.clamp(W / S, -7, 7))
            W_dequant = Q * S
            h = h @ W_dequant
            
        # Compute dynamic routing coefficients (cosine similarities)
        max_h = torch.max(torch.abs(h), dim=-1, keepdim=True)[0]
        S_h = max_h / 127.0
        S_h = torch.clamp(S_h, min=1e-8)
        Q_h = torch.round(torch.clamp(h / S_h, -127, 127))
        h_q = Q_h * S_h
        
        sims = []
        for j in range(4):
            mu = centroids_layer3[j]
            max_mu = torch.max(torch.abs(mu))
            S_mu = max_mu / 127.0
            S_mu = torch.clamp(S_mu, min=1e-8)
            Q_mu = torch.round(torch.clamp(mu / S_mu, -127, 127))
            mu_q = Q_mu * S_mu
            
            dot_product = torch.sum(h_q * mu_q, dim=-1)
            norm_h = torch.norm(h_q, p=2, dim=-1)
            norm_mu = torch.norm(mu_q, p=2)
            sim = dot_product / (norm_h * norm_mu + 1e-8)
            sims.append(sim)
        sims = torch.stack(sims, dim=1) # (B, 4)
        
        # Softmax to get routing coefficients alpha
        tau = 0.001
        alpha = torch.softmax(sims / tau, dim=1)
        return alpha, sims

# Compute routing probs for ID test set
id_alphas = []
id_sims = []
with torch.no_grad():
    for k in range(4):
        mask = (test_task == k)
        bx = test_x[mask]
        alpha, sims = get_routing_probs(bx)
        id_alphas.append(alpha)
        id_sims.append(sims)
id_alphas = torch.cat(id_alphas, dim=0)
id_sims = torch.cat(id_sims, dim=0)

# Compute routing probs for OOD test set
ood_x = torch.randn(250, 192)
ood_alphas, ood_sims = get_routing_probs(ood_x)

# Evaluate metrics:
# 1. Max routing prob
id_max_prob = id_alphas.max(dim=1)[0]
ood_max_prob = ood_alphas.max(dim=1)[0]
print(f"ID Max Routing Prob: {id_max_prob.mean().item():.4f} | Std: {id_max_prob.std().item():.4f}")
print(f"OOD Max Routing Prob: {ood_max_prob.mean().item():.4f} | Std: {ood_max_prob.std().item():.4f}")

# 2. Entropy of routing probs
def entropy(p):
    return -torch.sum(p * torch.log(p + 1e-10), dim=1)

id_entropy = entropy(id_alphas)
ood_entropy = entropy(ood_alphas)
print(f"ID Routing Entropy: {id_entropy.mean().item():.4f} | Std: {id_entropy.std().item():.4f}")
print(f"OOD Routing Entropy: {ood_entropy.mean().item():.4f} | Std: {ood_entropy.std().item():.4f}")

# 3. Maximum cosine similarity
id_max_sim = id_sims.max(dim=1)[0]
ood_max_sim = ood_sims.max(dim=1)[0]
print(f"ID Max Cosine Sim: {id_max_sim.mean().item():.4f} | Std: {id_max_sim.std().item():.4f}")
print(f"OOD Max Cosine Sim: {ood_max_sim.mean().item():.4f} | Std: {ood_max_sim.std().item():.4f}")

# Let's sweep a threshold on maximum cosine similarity to see if it can reject OOD
# Since OOD should have lower similarity to all centroids than ID.
# Let's check overlap
all_sims = torch.cat([id_max_sim, ood_max_sim])
is_ood = torch.cat([torch.zeros_like(id_max_sim, dtype=torch.bool), torch.ones_like(ood_max_sim, dtype=torch.bool)])

thresholds = np.linspace(0.0, 0.3, 31)
print("\nMax Cosine Similarity Rejection Sweep:")
for thresh in thresholds:
    # If max similarity is LESS than threshold, reject as OOD
    rejected = (all_sims < thresh)
    tpr = (rejected & is_ood).sum().item() / is_ood.sum().item() * 100.0
    fpr = (rejected & (~is_ood)).sum().item() / (~is_ood).sum().item() * 100.0
    print(f"  Thresh: {thresh:.3f} | TPR: {tpr:5.1f}% | FPR: {fpr:5.1f}%")
