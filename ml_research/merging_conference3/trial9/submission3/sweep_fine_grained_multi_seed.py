import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Load real-world features
checkpoint = torch.load("data/real_world_features.pt")
features = checkpoint["features"]
tasks = checkpoint["tasks"]
classes = checkpoint["classes"]

K = 4
D = 192
L = 14
SEEDS = list(range(42, 52))

class ParametricRouter(nn.Module):
    def __init__(self, use_contraction_regularizer=False):
        super().__init__()
        self.use_contraction_regularizer = use_contraction_regularizer
        self.W_route = nn.ParameterList([
            nn.Parameter(torch.randn(K, D) * 0.001) for _ in range(L)
        ])
        self.log_tau = nn.ParameterList([
            nn.Parameter(torch.tensor(np.log(0.05))) for _ in range(L)
        ])

    def get_coefficients(self, h, layer_idx):
        tau = torch.exp(self.log_tau[layer_idx])
        logits = torch.matmul(h, self.W_route[layer_idx].t())
        return torch.softmax(logits / tau, dim=-1)

def propagate_layers(inputs, router_func, prototypes, use_soft_coordinates=True, tau_c=0.05):
    h = inputs.clone()
    alpha_history = []
    gamma_l = [0.1 + 0.9 * (l / L) for l in range(1, L + 1)]
    for l in range(L):
        alpha_l = router_func(h, l)
        alpha_history.append(alpha_l)
        expert_updates_list = []
        for k in range(K):
            scores = torch.matmul(h, prototypes[k].t())
            S_kc = torch.softmax(scores / tau_c, dim=-1)
            best_proto = torch.matmul(S_kc, prototypes[k])
            expert_updates_list.append(gamma_l[l] * (best_proto - h))
        expert_updates = torch.stack(expert_updates_list, dim=1)
        blended_update = torch.sum(alpha_l.unsqueeze(-1) * expert_updates, dim=1)
        h = h + blended_update
    return h, torch.stack(alpha_history, dim=1)

def evaluate_model(final_h, targets_task, targets_class, prototypes):
    B = final_h.shape[0]
    scores = torch.zeros(B, K, 10)
    for k in range(K):
        for c in range(10):
            scores[:, k, c] = torch.cosine_similarity(final_h, prototypes[k, c].unsqueeze(0), dim=-1)
    scores_flat = scores.view(B, K * 10)
    preds_flat = torch.argmax(scores_flat, dim=-1)
    pred_tasks = preds_flat // 10
    pred_classes = preds_flat % 10
    correct_class = torch.sum((pred_tasks == targets_task) & (pred_classes == targets_class)).item()
    correct_routing = torch.sum(pred_tasks == targets_task).item()
    return (correct_class / B) * 100.0, (correct_routing / B) * 100.0

lambdas = [0.000, 0.001, 0.002, 0.004, 0.006, 0.008, 0.010, 0.012, 0.015, 0.020, 0.030, 0.050]

print("Starting 10-seed fine-grained sweep...")
results_dict = {}

for lam in lambdas:
    results_dict[lam] = {
        "class_accs": [],
        "route_accs": [],
        "depth_vars": [],
        "entropies": [],
        "lipschitz_bounds": []
    }
    
    for seed in SEEDS:
        # Set seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Stable class prototypes
        prototypes = torch.zeros(K, 10, D)
        for k in range(K):
            for c in range(10):
                proto = features[(tasks == k) & (classes == c)].mean(dim=0)
                prototypes[k, c] = proto / proto.norm()
                
        # Sample splits
        cal_indices = []
        test_indices = []
        for k in range(K):
            task_indices = torch.where(tasks == k)[0]
            perm = torch.randperm(len(task_indices))
            cal_indices.append(task_indices[perm[:16]])
            test_indices.append(task_indices[perm[16:116]])
        cal_indices = torch.cat(cal_indices)
        test_indices = torch.cat(test_indices)
        
        cal_inputs = features[cal_indices]
        cal_targets_task = tasks[cal_indices]
        cal_targets_class = classes[cal_indices]
        
        test_inputs = features[test_indices]
        test_targets_task = tasks[test_indices]
        test_targets_class = classes[test_indices]
        
        sweep_router = ParametricRouter(use_contraction_regularizer=(lam > 0))
        optimizer_sweep = optim.Adam(sweep_router.parameters(), lr=0.01)
        
        for epoch in range(50):
            optimizer_sweep.zero_grad()
            h_prop = cal_inputs.clone()
            cal_loss = 0.0
            gamma_l = [0.1 + 0.9 * (l / L) for l in range(1, L + 1)]
            for l in range(L):
                alpha_l = sweep_router.get_coefficients(h_prop, l)
                if l < 4:
                    target_dist = torch.ones(alpha_l.shape[0], K) / K
                    cal_loss += -torch.sum(target_dist * torch.log(alpha_l + 1e-10), dim=-1).mean()
                else:
                    cal_loss += nn.CrossEntropyLoss()(alpha_l, cal_targets_task)
                expert_updates = torch.stack([
                    gamma_l[l] * (prototypes[k, cal_targets_class] - h_prop) for k in range(K)
                ], dim=1)
                blended_update = torch.sum(alpha_l.unsqueeze(-1) * expert_updates, dim=1)
                h_prop = h_prop + blended_update
            cal_loss = cal_loss / L
            
            if lam > 0:
                reg_spec = sum(torch.sum(W ** 2) for W in sweep_router.W_route)
                reg_temp = sum(1.0 / (torch.exp(log_tau) ** 2) for log_tau in sweep_router.log_tau)
                total_loss = cal_loss + lam * reg_spec + lam * reg_temp
            else:
                total_loss = cal_loss
                
            total_loss.backward()
            optimizer_sweep.step()
            
        # Evaluate on test inputs
        def sweep_router_func(h, l):
            with torch.no_grad():
                return sweep_router.get_coefficients(h, l)
                
        final_h_sweep, alphas_sweep = propagate_layers(test_inputs, sweep_router_func, prototypes)
        acc_class_sweep, acc_route_sweep = evaluate_model(final_h_sweep, test_targets_task, test_targets_class, prototypes)
        
        # Compute Gating Heuristics
        alphas_mean = alphas_sweep.mean(dim=1, keepdim=True)
        depth_var = torch.mean(torch.sum((alphas_sweep - alphas_mean) ** 2, dim=-1)).item()
        entropy = -torch.mean(torch.sum(alphas_sweep * torch.log(alphas_sweep + 1e-10), dim=-1)).item()
        
        lipschitz_list = []
        with torch.no_grad():
            for l in range(L):
                W = sweep_router.W_route[l]
                spec_norm = torch.linalg.svdvals(W)[0].item()
                tau = torch.exp(sweep_router.log_tau[l]).item()
                L_route = (2.0 / tau) * spec_norm
                L_total = 1.0 + L_route
                lipschitz_list.append(L_total)
        max_lipschitz = max(lipschitz_list)
        
        results_dict[lam]["class_accs"].append(acc_class_sweep)
        results_dict[lam]["route_accs"].append(acc_route_sweep)
        results_dict[lam]["depth_vars"].append(depth_var)
        results_dict[lam]["entropies"].append(entropy)
        results_dict[lam]["lipschitz_bounds"].append(max_lipschitz)

print("\n================ 10-SEED SWEEP RESULTS ================")
print(f"{'Lambda':<8} | {'Classification (%)':<22} | {'Routing (%)':<22} | {'Depth-Var':<18} | {'Entropy':<18} | {'Lipschitz'}")
print("-" * 110)

for lam in lambdas:
    data = results_dict[lam]
    
    mean_class = np.mean(data["class_accs"])
    std_class = np.std(data["class_accs"])
    
    mean_route = np.mean(data["route_accs"])
    std_route = np.std(data["route_accs"])
    
    mean_var = np.mean(data["depth_vars"])
    std_var = np.std(data["depth_vars"])
    
    mean_ent = np.mean(data["entropies"])
    std_ent = np.std(data["entropies"])
    
    mean_lip = np.mean(data["lipschitz_bounds"])
    std_lip = np.std(data["lipschitz_bounds"])
    
    print(f"{lam:<8.3f} | {mean_class:6.2f}% ± {std_class:5.2f}% | {mean_route:6.2f}% ± {std_route:5.2f}% | {mean_var:7.4f} ± {std_var:6.4f} | {mean_ent:7.4f} ± {std_ent:6.4f} | {mean_lip:7.2f} ± {std_lip:5.2f}")
