import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Load real-world pre-extracted features
checkpoint = torch.load("data/real_world_features.pt")
features = checkpoint["features"] # (2000, 192)
tasks = checkpoint["tasks"]       # (2000,)
classes = checkpoint["classes"]   # (2000,)

K = 4  # Number of tasks
D = 192  # Intermediate representation dimension
L = 14  # Number of sequential layers
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

def propagate_layers(inputs, router_func, prototypes, targets_class=None, use_oracle_class=False, use_soft_coordinates=True, tau_c=0.05):
    h = inputs.clone()
    alpha_history = []
    gamma_l = [0.1 + 0.9 * (l / L) for l in range(1, L + 1)]
    
    for l in range(L):
        alpha_l = router_func(h, l)
        alpha_history.append(alpha_l)
        
        if use_oracle_class and targets_class is not None:
            expert_updates = torch.stack([
                gamma_l[l] * (prototypes[k, targets_class] - h) for k in range(K)
            ], dim=1)
        else:
            expert_updates_list = []
            for k in range(K):
                scores = torch.matmul(h, prototypes[k].t())
                if use_soft_coordinates:
                    S_kc = torch.softmax(scores / tau_c, dim=-1)
                    best_proto = torch.matmul(S_kc, prototypes[k])
                else:
                    best_c = torch.argmax(scores, dim=-1)
                    best_proto = prototypes[k, best_c]
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
    
    acc_class = (correct_class / B) * 100.0
    acc_routing = (correct_routing / B) * 100.0
    
    return acc_class, acc_routing

def compute_gating_metrics(alphas, targets_task):
    B, L_val, K_val = alphas.shape
    pred_gating = torch.argmax(alphas[:, 4:], dim=-1)
    correct_gating = (pred_gating == targets_task.unsqueeze(1)).float()
    mean_gating_acc = correct_gating.mean().item() * 100.0
    
    targets_task_expanded = targets_task.view(B, 1, 1).expand(B, L_val - 4, 1)
    gathered_probs = torch.gather(alphas[:, 4:], dim=-1, index=targets_task_expanded).squeeze(-1)
    gating_ce = -torch.log(gathered_probs + 1e-10).mean().item()
    
    return mean_gating_acc, gating_ce

def run_real_world_experiment():
    results_summary = {
        "Oracle": {"class_acc": [], "route_acc": [], "gating_acc": [], "gating_ce": []},
        "Uniform": {"class_acc": [], "route_acc": [], "gating_acc": [], "gating_ce": []},
        "SABLE": {"class_acc": [], "route_acc": [], "gating_acc": [], "gating_ce": []},
        "ChemMerge": {"class_acc": [], "route_acc": [], "gating_acc": [], "gating_ce": []},
        "Shared Router": {"class_acc": [], "route_acc": [], "gating_acc": [], "gating_ce": []},
        "L2-Fixed Router": {"class_acc": [], "route_acc": [], "gating_acc": [], "gating_ce": []},
        "Linear Router": {"class_acc": [], "route_acc": [], "gating_acc": [], "gating_ce": []},
        "CR-Router": {"class_acc": [], "route_acc": [], "gating_acc": [], "gating_ce": []}
    }
    
    # Precompute stable class prototypes from the entire pool of 50 samples
    prototypes = torch.zeros(K, 10, D)
    for k in range(K):
        for c in range(10):
            proto = features[(tasks == k) & (classes == c)].mean(dim=0)
            prototypes[k, c] = proto / proto.norm()
            
    for seed in SEEDS:
        print(f"--- seed {seed} (real-world) ---")
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Sample 16 calibration samples and 100 test samples per task from the pool
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
        
        # 1. Oracle
        def oracle_router(h, l):
            eye = torch.eye(K)
            return eye[test_targets_task]
        final_h_or, alphas_or = propagate_layers(test_inputs, oracle_router, prototypes, use_oracle_class=False)
        acc_class_or, acc_route_or = evaluate_model(final_h_or, test_targets_task, test_targets_class, prototypes)
        gate_acc_or, gate_ce_or = compute_gating_metrics(alphas_or, test_targets_task)
        results_summary["Oracle"]["class_acc"].append(acc_class_or)
        results_summary["Oracle"]["route_acc"].append(acc_route_or)
        results_summary["Oracle"]["gating_acc"].append(gate_acc_or)
        results_summary["Oracle"]["gating_ce"].append(gate_ce_or)
        
        # 2. Uniform
        def uniform_router(h, l):
            B_size = h.shape[0]
            return torch.ones(B_size, K) / K
        final_h_un, alphas_un = propagate_layers(test_inputs, uniform_router, prototypes, use_oracle_class=False)
        acc_class_un, acc_route_un = evaluate_model(final_h_un, test_targets_task, test_targets_class, prototypes)
        gate_acc_un, gate_ce_un = compute_gating_metrics(alphas_un, test_targets_task)
        results_summary["Uniform"]["class_acc"].append(acc_class_un)
        results_summary["Uniform"]["route_acc"].append(acc_route_un)
        results_summary["Uniform"]["gating_acc"].append(gate_acc_un)
        results_summary["Uniform"]["gating_ce"].append(gate_ce_un)
        
        # 3. SABLE
        def sable_router(h, l):
            B_size = h.shape[0]
            sims = torch.zeros(B_size, K)
            for k_idx in range(K):
                p_task = prototypes[k_idx]
                dists = torch.cdist(h, p_task)
                sims[:, k_idx] = -dists.min(dim=-1)[0]
            return torch.softmax(sims / 0.1, dim=-1)
        final_h_sable, alphas_sable = propagate_layers(test_inputs, sable_router, prototypes, use_oracle_class=False)
        acc_class_sable, acc_route_sable = evaluate_model(final_h_sable, test_targets_task, test_targets_class, prototypes)
        gate_acc_sable, gate_ce_sable = compute_gating_metrics(alphas_sable, test_targets_task)
        results_summary["SABLE"]["class_acc"].append(acc_class_sable)
        results_summary["SABLE"]["route_acc"].append(acc_route_sable)
        results_summary["SABLE"]["gating_acc"].append(gate_acc_sable)
        results_summary["SABLE"]["gating_ce"].append(gate_ce_sable)
        
        # 4. ChemMerge
        def propagate_chemerge(inputs, prototypes, use_oracle_class=False):
            h_val = inputs.clone()
            B_size = inputs.shape[0]
            alpha_history_val = []
            c_conc = torch.ones(B_size, K) / K
            gamma_l = [0.1 + 0.9 * (l / L) for l in range(1, L + 1)]
            
            for l in range(L):
                sims = torch.zeros(B_size, K)
                for k_idx in range(K):
                    p_task = prototypes[k_idx]
                    dists = torch.cdist(h_val, p_task)
                    sims[:, k_idx] = -dists.min(dim=-1)[0]
                r_scores = torch.softmax(sims / 0.1, dim=-1)
                
                dt = 0.5
                k_rate = 0.8
                dc_dt = k_rate * (r_scores - c_conc)
                c_conc = torch.clamp(c_conc + dc_dt * dt, 1e-5, 1.0)
                c_conc = c_conc / c_conc.sum(dim=-1, keepdim=True)
                alpha_history_val.append(c_conc)
                
                expert_updates_list = []
                for k_idx in range(K):
                    scores = torch.matmul(h_val, prototypes[k_idx].t())
                    S_kc = torch.softmax(scores / 0.05, dim=-1)
                    best_proto = torch.matmul(S_kc, prototypes[k_idx])
                    expert_updates_list.append(gamma_l[l] * (best_proto - h_val))
                expert_updates = torch.stack(expert_updates_list, dim=1)
                blended_update = torch.sum(c_conc.unsqueeze(-1) * expert_updates, dim=1)
                h_val = h_val + blended_update
            return h_val, torch.stack(alpha_history_val, dim=1)
        final_h_chem, alphas_chem = propagate_chemerge(test_inputs, prototypes)
        acc_class_chem, acc_route_chem = evaluate_model(final_h_chem, test_targets_task, test_targets_class, prototypes)
        gate_acc_chem, gate_ce_chem = compute_gating_metrics(alphas_chem, test_targets_task)
        results_summary["ChemMerge"]["class_acc"].append(acc_class_chem)
        results_summary["ChemMerge"]["route_acc"].append(acc_route_chem)
        results_summary["ChemMerge"]["gating_acc"].append(gate_acc_chem)
        results_summary["ChemMerge"]["gating_ce"].append(gate_ce_chem)
        
        # 5. Shared Router
        shared_router = ParametricRouter(use_contraction_regularizer=False)
        optimizer_sr = optim.Adam(shared_router.parameters(), lr=0.01)
        for epoch in range(50):
            optimizer_sr.zero_grad()
            h_prop = cal_inputs.clone()
            cal_loss = 0.0
            # Force identical routing coefficients at all depths
            alpha_0 = shared_router.get_coefficients(h_prop, 0)
            for l in range(L):
                gamma_l = [0.1 + 0.9 * (l / L) for l in range(1, L + 1)]
                if l < 4:
                    target_dist = torch.ones(alpha_0.shape[0], K) / K
                    cal_loss += -torch.sum(target_dist * torch.log(alpha_0 + 1e-10), dim=-1).mean()
                else:
                    cal_loss += nn.CrossEntropyLoss()(alpha_0, cal_targets_task)
                expert_updates = torch.stack([
                    gamma_l[l] * (prototypes[k_idx, cal_targets_class] - h_prop) for k_idx in range(K)
                ], dim=1)
                blended_update = torch.sum(alpha_0.unsqueeze(-1) * expert_updates, dim=1)
                h_prop = h_prop + blended_update
            cal_loss = cal_loss / L
            cal_loss.backward()
            optimizer_sr.step()
        def shared_router_func(h_val, l):
            with torch.no_grad():
                return shared_router.get_coefficients(h_val, 0)
        final_h_sr, alphas_sr = propagate_layers(test_inputs, shared_router_func, prototypes, use_oracle_class=False)
        acc_class_sr, acc_route_sr = evaluate_model(final_h_sr, test_targets_task, test_targets_class, prototypes)
        gate_acc_sr, gate_ce_sr = compute_gating_metrics(alphas_sr, test_targets_task)
        results_summary["Shared Router"]["class_acc"].append(acc_class_sr)
        results_summary["Shared Router"]["route_acc"].append(acc_route_sr)
        results_summary["Shared Router"]["gating_acc"].append(gate_acc_sr)
        results_summary["Shared Router"]["gating_ce"].append(gate_ce_sr)
        
        # 6. Linear Router (Unregularized)
        lr_router = ParametricRouter(use_contraction_regularizer=False)
        optimizer_lr = optim.Adam(lr_router.parameters(), lr=0.01)
        for epoch in range(50):
            optimizer_lr.zero_grad()
            h_prop = cal_inputs.clone()
            cal_loss = 0.0
            gamma_l = [0.1 + 0.9 * (l / L) for l in range(1, L + 1)]
            for l in range(L):
                alpha_l = lr_router.get_coefficients(h_prop, l)
                if l < 4:
                    target_dist = torch.ones(alpha_l.shape[0], K) / K
                    cal_loss += -torch.sum(target_dist * torch.log(alpha_l + 1e-10), dim=-1).mean()
                else:
                    cal_loss += nn.CrossEntropyLoss()(alpha_l, cal_targets_task)
                expert_updates = torch.stack([
                    gamma_l[l] * (prototypes[k_idx, cal_targets_class] - h_prop) for k_idx in range(K)
                ], dim=1)
                blended_update = torch.sum(alpha_l.unsqueeze(-1) * expert_updates, dim=1)
                h_prop = h_prop + blended_update
            cal_loss = cal_loss / L
            cal_loss.backward()
            optimizer_lr.step()
        def lr_router_func(h_val, l):
            with torch.no_grad():
                return lr_router.get_coefficients(h_val, l)
        final_h_lr, alphas_lr = propagate_layers(test_inputs, lr_router_func, prototypes, use_oracle_class=False)
        acc_class_lr, acc_route_lr = evaluate_model(final_h_lr, test_targets_task, test_targets_class, prototypes)
        gate_acc_lr, gate_ce_lr = compute_gating_metrics(alphas_lr, test_targets_task)
        results_summary["Linear Router"]["class_acc"].append(acc_class_lr)
        results_summary["Linear Router"]["route_acc"].append(acc_route_lr)
        results_summary["Linear Router"]["gating_acc"].append(gate_acc_lr)
        results_summary["Linear Router"]["gating_ce"].append(gate_ce_lr)
        
        # 7. L2-Fixed Router
        l2_fixed_router = ParametricRouter(use_contraction_regularizer=False)
        l2_fixed_router.log_tau = nn.ParameterList([
            nn.Parameter(torch.tensor(np.log(0.05)), requires_grad=False) for _ in range(L)
        ])
        optimizer_l2 = optim.Adam(l2_fixed_router.parameters(), lr=0.01, weight_decay=1e-3)
        for epoch in range(50):
            optimizer_l2.zero_grad()
            h_prop = cal_inputs.clone()
            cal_loss = 0.0
            gamma_l = [0.1 + 0.9 * (l / L) for l in range(1, L + 1)]
            for l in range(L):
                alpha_l = l2_fixed_router.get_coefficients(h_prop, l)
                if l < 4:
                    target_dist = torch.ones(alpha_l.shape[0], K) / K
                    cal_loss += -torch.sum(target_dist * torch.log(alpha_l + 1e-10), dim=-1).mean()
                else:
                    cal_loss += nn.CrossEntropyLoss()(alpha_l, cal_targets_task)
                expert_updates = torch.stack([
                    gamma_l[l] * (prototypes[k_idx, cal_targets_class] - h_prop) for k_idx in range(K)
                ], dim=1)
                blended_update = torch.sum(alpha_l.unsqueeze(-1) * expert_updates, dim=1)
                h_prop = h_prop + blended_update
            cal_loss = cal_loss / L
            cal_loss.backward()
            optimizer_l2.step()
        def l2_fixed_router_func(h_val, l):
            with torch.no_grad():
                return l2_fixed_router.get_coefficients(h_val, l)
        final_h_l2, alphas_l2 = propagate_layers(test_inputs, l2_fixed_router_func, prototypes, use_oracle_class=False)
        acc_class_l2, acc_route_l2 = evaluate_model(final_h_l2, test_targets_task, test_targets_class, prototypes)
        gate_acc_l2, gate_ce_l2 = compute_gating_metrics(alphas_l2, test_targets_task)
        results_summary["L2-Fixed Router"]["class_acc"].append(acc_class_l2)
        results_summary["L2-Fixed Router"]["route_acc"].append(acc_route_l2)
        results_summary["L2-Fixed Router"]["gating_acc"].append(gate_acc_l2)
        results_summary["L2-Fixed Router"]["gating_ce"].append(gate_ce_l2)
        
        # 8. CR-Router
        cr_router = ParametricRouter(use_contraction_regularizer=True)
        optimizer_cr = optim.Adam(cr_router.parameters(), lr=0.01)
        for epoch in range(50):
            optimizer_cr.zero_grad()
            h_prop = cal_inputs.clone()
            cal_loss = 0.0
            gamma_l = [0.1 + 0.9 * (l / L) for l in range(1, L + 1)]
            for l in range(L):
                alpha_l = cr_router.get_coefficients(h_prop, l)
                if l < 4:
                    target_dist = torch.ones(alpha_l.shape[0], K) / K
                    cal_loss += -torch.sum(target_dist * torch.log(alpha_l + 1e-10), dim=-1).mean()
                else:
                    cal_loss += nn.CrossEntropyLoss()(alpha_l, cal_targets_task)
                expert_updates = torch.stack([
                    gamma_l[l] * (prototypes[k_idx, cal_targets_class] - h_prop) for k_idx in range(K)
                ], dim=1)
                blended_update = torch.sum(alpha_l.unsqueeze(-1) * expert_updates, dim=1)
                h_prop = h_prop + blended_update
            cal_loss = cal_loss / L
            reg_spec = sum(torch.sum(W ** 2) for W in cr_router.W_route)
            reg_temp = sum(1.0 / (torch.exp(log_tau) ** 2) for log_tau in cr_router.log_tau)
            total_loss = cal_loss + 0.01 * reg_spec + 0.01 * reg_temp
            total_loss.backward()
            optimizer_cr.step()
        def cr_router_func(h_val, l):
            with torch.no_grad():
                return cr_router.get_coefficients(h_val, l)
        final_h_cr, alphas_cr = propagate_layers(test_inputs, cr_router_func, prototypes, use_oracle_class=False)
        acc_class_cr, acc_route_cr = evaluate_model(final_h_cr, test_targets_task, test_targets_class, prototypes)
        gate_acc_cr, gate_ce_cr = compute_gating_metrics(alphas_cr, test_targets_task)
        results_summary["CR-Router"]["class_acc"].append(acc_class_cr)
        results_summary["CR-Router"]["route_acc"].append(acc_route_cr)
        results_summary["CR-Router"]["gating_acc"].append(gate_acc_cr)
        results_summary["CR-Router"]["gating_ce"].append(gate_ce_cr)
        
        print(f"CR-Router class acc: {acc_class_cr:.2f}% | routing acc: {acc_route_cr:.2f}%")
        print(f"L2-Fixed class acc: {acc_class_l2:.2f}% | routing acc: {acc_route_l2:.2f}%")
        
    print("\n================== REAL-WORLD RESULTS SUMMARY ==================")
    for m in results_summary:
        class_mean = np.mean(results_summary[m]["class_acc"])
        class_sd = np.std(results_summary[m]["class_acc"])
        route_mean = np.mean(results_summary[m]["route_acc"])
        route_sd = np.std(results_summary[m]["route_acc"])
        gate_mean = np.mean(results_summary[m]["gating_acc"])
        gate_sd = np.std(results_summary[m]["gating_acc"])
        print(f"Method: {m}")
        print(f"  Classification Acc: {class_mean:.2f}% ± {class_sd:.2f}%")
        print(f"  Representation Routing Acc: {route_mean:.2f}% ± {route_sd:.2f}%")
        print(f"  Direct Gating Acc: {gate_mean:.2f}% ± {gate_sd:.2f}%")

if __name__ == "__main__":
    run_real_world_experiment()
