import torch
import torch.nn.functional as F
import numpy as np
from merge_eval import MergedCNN, build_test_stream

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
expert_names = ['mnist', 'fmnist', 'kmnist']
expert_sds = [torch.load(f"checkpoints/expert_{name}.pt", map_location=DEVICE) for name in expert_names]
prototypes = torch.load("checkpoints/prototypes.pt", map_location=DEVICE)
fisher_infos = torch.load("checkpoints/fisher_infos.pt", map_location=DEVICE)

stream = build_test_stream()

# Track A-FWPA (beta=2.0)
merged_model = MergedCNN(expert_sds).to(DEVICE)
import torch.optim as optim
optimizer = optim.Adam(merged_model.parameters(), lr=0.05)

layer_domain_coefs = {domain: [] for domain in ['mnist', 'fmnist', 'kmnist']}

epsilon = 1e-5
beta = 2.0

for step, (x, y, domain) in enumerate(stream):
    x, y = x.to(DEVICE), y.to(DEVICE)
    
    # Adapt
    for _ in range(3):
        optimizer.zero_grad()
        out, feat = merged_model(x)
        
        scores_list = [F.softmax(merged_model.scores[k], dim=0) for k in merged_model.scores]
        lambda_avg = torch.stack(scores_list).mean(dim=0)
        
        pi_merged = torch.zeros(10, 128, device=DEVICE)
        for k, exp_name in enumerate(['mnist', 'fmnist', 'kmnist']):
            pi_merged += lambda_avg[k] * prototypes[exp_name].to(DEVICE)
            
        mu_merged = pi_merged.mean(dim=0, keepdim=True)
        z = feat - mu_merged
        pi_centered = pi_merged - mu_merged
        pi_centered_norm = pi_centered / (pi_centered.norm(p=2, dim=1, keepdim=True) + 1e-8)
        
        probs = F.softmax(out, dim=1)
        conf, pseudo_y = probs.max(dim=1)
        mask = conf >= 0.5
        
        if mask.sum() > 0:
            z_conf = z[mask]
            y_conf = pseudo_y[mask]
            proto_conf = pi_centered_norm[y_conf]
            
            fisher_dyn = torch.zeros(128, device=DEVICE)
            for k, exp_name in enumerate(['mnist', 'fmnist', 'kmnist']):
                fisher_dyn += lambda_avg[k] * fisher_infos[exp_name]['fc1.bias'].to(DEVICE)
            fisher_dyn_norm = fisher_dyn / (fisher_dyn.mean() + 1e-8)
            W_feat = 1.0 / (fisher_dyn_norm + epsilon) ** beta
            W_feat = W_feat / W_feat.mean()
            
            dot = torch.sum(W_feat * z_conf * proto_conf, dim=1)
            norm_z = torch.sqrt(torch.sum(W_feat * (z_conf ** 2), dim=1) + 1e-8)
            norm_proto = torch.sqrt(torch.sum(W_feat * (proto_conf ** 2), dim=1) + 1e-8)
            sims = dot / (norm_z * norm_proto)
            loss = -sims.mean()
        else:
            loss = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
            
        loss.backward()
        optimizer.step()
        
    # Read layer-specific coefficients
    with torch.no_grad():
        coef_dict = {}
        for k in merged_model.scores:
            coef_dict[k] = F.softmax(merged_model.scores[k], dim=0).cpu().numpy()
        layer_domain_coefs[domain].append(coef_dict)

print("\nLayer-wise Merging Coefficients per domain:")
for domain in ['mnist', 'fmnist', 'kmnist']:
    print(f"\n--- Domain: {domain.upper()} ---")
    keys = list(layer_domain_coefs[domain][0].keys())
    for key in keys:
        coefs = [d[key] for d in layer_domain_coefs[domain]]
        avg_coef = np.mean(coefs, axis=0)
        print(f"Layer: {key:<15} | MNIST: {avg_coef[0]:.4f} | F-MNIST: {avg_coef[1]:.4f} | K-MNIST: {avg_coef[2]:.4f}")
