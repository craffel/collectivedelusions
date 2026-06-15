import os
import torch
import torch.nn as nn
import numpy as np
from run_experiments import *

# Override optimize_dirichlet_pem with PEM-Div
def optimize_dirichlet_pem_div(cal_energies_norm, P_all, lambda_div=1.0, epochs=200, lr=0.05, tau0=0.2, delta=0.05):
    N, K = cal_energies_norm.size()
    w = torch.full((K,), np.log(tau0), requires_grad=True, device=device)
    optimizer = torch.optim.Adam([w], lr=lr)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        tau = torch.clamp(torch.exp(w), min=0.01, max=10.0)
        
        exponent = torch.clamp(cal_energies_norm / tau, min=-50.0, max=50.0)
        a_b = torch.exp(exponent)
        
        exponent0 = torch.clamp(cal_energies_norm / tau0, min=-50.0, max=50.0)
        a0_b = torch.exp(exponent0)
        
        # Compute ensembling weights
        alpha = a_b / torch.sum(a_b, dim=-1, keepdim=True) # (N, K)
        
        # Compute ensembled predictions: hat{p}_c(x_b) = \sum_k \alpha_{k, b} \cdot p_{k, c}(x_b)
        ensembled_probs = torch.sum(alpha.unsqueeze(-1) * P_all, dim=1)
        
        # Add a tiny epsilon to prevent log(0)
        ensembled_probs_clamped = torch.clamp(ensembled_probs, min=1e-15)
        
        # Shannon entropy normalized to [0, 1] by dividing by ln(C) where C = K
        entropy = -torch.sum(ensembled_probs_clamped * torch.log(ensembled_probs_clamped), dim=-1) # (N,)
        individual_entropy = torch.mean(entropy) / np.log(K)
        
        # Batch-averaged ensembling weights: alpha shape (N, K)
        mean_alpha = torch.mean(alpha, dim=0) # (K,)
        mean_alpha_clamped = torch.clamp(mean_alpha, min=1e-15)
        diversity_entropy = -torch.sum(mean_alpha_clamped * torch.log(mean_alpha_clamped)) / np.log(K)
        
        # PEM-Div loss: individual_entropy - lambda_div * diversity_entropy
        cal_loss = individual_entropy - lambda_div * diversity_entropy
        
        # Analytical Dirichlet KL
        sum_a = torch.sum(a_b, dim=-1)
        sum_a0 = torch.sum(a0_b, dim=-1)
        
        kl_samples = (torch.lgamma(sum_a) - torch.lgamma(sum_a0)
                      - torch.sum(torch.lgamma(a_b), dim=-1)
                      + torch.sum(torch.lgamma(a0_b), dim=-1)
                      + torch.sum((a_b - a0_b) * (torch.digamma(a_b) - torch.digamma(sum_a).unsqueeze(-1)), dim=-1))
        
        mean_kl = torch.mean(kl_samples)
        
        # PAC bound using McAllester's theorem over bounded [0, 1] loss
        bound = cal_loss + torch.sqrt((mean_kl + np.log(2.0 * np.sqrt(N) / delta)) / (2.0 * N))
        bound.backward()
        optimizer.step()
        
    return torch.clamp(torch.exp(w), min=0.01, max=10.0).detach()

noise_scales = [0.05, 0.15, 0.40, 1.20]
lambdas = [0.0, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0]

for l in lambdas:
    print(f"\n--- SWEEPING lambda_div = {l} ---")
    orthogonal_accs = []
    overlapping_accs = []
    
    for seed in range(10):
        # Orthogonal (rho = 0.0)
        set_seed(seed)
        v_orth = generate_base_vectors(D=192, K=4, rho=0.0)
        n_prior = 64 // 2
        n_opt = 64 - n_prior
        prior_h0, prior_labels, _ = generate_samples(v_orth, n_prior, noise_scales, D=192, K=4)
        opt_h0, opt_labels, _ = generate_samples(v_orth, n_opt, noise_scales, D=192, K=4)
        test_h0, test_labels, _ = generate_samples(v_orth, 250, noise_scales, D=192, K=4)
        
        V_d = []
        for k in range(4):
            mask = (prior_labels == k)
            Z_k = prior_h0[mask]
            U, S, Vt = torch.linalg.svd(Z_k, full_matrices=False)
            V_d.append(Vt.T[:, :8])
            
        opt_energies = torch.zeros(opt_h0.size(0), 4, device=device)
        for k in range(4):
            opt_energies[:, k] = torch.norm(torch.matmul(opt_h0, V_d[k]), dim=-1)
        test_energies = torch.zeros(test_h0.size(0), 4, device=device)
        for k in range(4):
            test_energies[:, k] = torch.norm(torch.matmul(test_h0, V_d[k]), dim=-1)
            
        opt_energies_norm = opt_energies / torch.sum(opt_energies, dim=-1, keepdim=True)
        test_energies_norm = test_energies / torch.sum(test_energies, dim=-1, keepdim=True)
        P_all = precompute_expert_all_class_probs(opt_h0, v_orth, K=4, eta=0.05)
        tau_dirichlet_pem = optimize_dirichlet_pem_div(opt_energies_norm, P_all, lambda_div=l, tau0=0.2)
        
        dirichlet_pem_exponent = torch.clamp(test_energies_norm / tau_dirichlet_pem, min=-50.0, max=50.0)
        dirichlet_pem_alpha = torch.exp(dirichlet_pem_exponent) / torch.sum(torch.exp(dirichlet_pem_exponent), dim=-1, keepdim=True)
        dirichlet_pem_hL = propagate_network(test_h0, v_orth, dirichlet_pem_alpha, eta=0.05)
        acc, _ = compute_predictions_and_accuracy(dirichlet_pem_hL, v_orth, test_labels)
        orthogonal_accs.append(acc)
        
        # Overlapping (rho = 0.33)
        set_seed(seed)
        v_over = generate_base_vectors(D=192, K=4, rho=0.33)
        prior_h0, prior_labels, _ = generate_samples(v_over, n_prior, noise_scales, D=192, K=4)
        opt_h0, opt_labels, _ = generate_samples(v_over, n_opt, noise_scales, D=192, K=4)
        test_h0, test_labels, _ = generate_samples(v_over, 250, noise_scales, D=192, K=4)
        
        V_d = []
        for k in range(4):
            mask = (prior_labels == k)
            Z_k = prior_h0[mask]
            U, S, Vt = torch.linalg.svd(Z_k, full_matrices=False)
            V_d.append(Vt.T[:, :8])
            
        opt_energies = torch.zeros(opt_h0.size(0), 4, device=device)
        for k in range(4):
            opt_energies[:, k] = torch.norm(torch.matmul(opt_h0, V_d[k]), dim=-1)
        test_energies = torch.zeros(test_h0.size(0), 4, device=device)
        for k in range(4):
            test_energies[:, k] = torch.norm(torch.matmul(test_h0, V_d[k]), dim=-1)
            
        opt_energies_norm = opt_energies / torch.sum(opt_energies, dim=-1, keepdim=True)
        test_energies_norm = test_energies / torch.sum(test_energies, dim=-1, keepdim=True)
        P_all = precompute_expert_all_class_probs(opt_h0, v_over, K=4, eta=0.05)
        tau_dirichlet_pem = optimize_dirichlet_pem_div(opt_energies_norm, P_all, lambda_div=l, tau0=0.2)
        
        dirichlet_pem_exponent = torch.clamp(test_energies_norm / tau_dirichlet_pem, min=-50.0, max=50.0)
        dirichlet_pem_alpha = torch.exp(dirichlet_pem_exponent) / torch.sum(torch.exp(dirichlet_pem_exponent), dim=-1, keepdim=True)
        dirichlet_pem_hL = propagate_network(test_h0, v_over, dirichlet_pem_alpha, eta=0.05)
        acc, _ = compute_predictions_and_accuracy(dirichlet_pem_hL, v_over, test_labels)
        overlapping_accs.append(acc)
        
    print(f"  Orthogonal accuracy (rho=0.0):  {np.mean(orthogonal_accs)*100:.2f}% +/- {np.std(orthogonal_accs)*100:.2f}%")
    print(f"  Overlapping accuracy (rho=0.33): {np.mean(overlapping_accs)*100:.2f}% +/- {np.std(overlapping_accs)*100:.2f}%")
