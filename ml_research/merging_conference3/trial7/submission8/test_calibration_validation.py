import torch
import torch.nn as nn
import numpy as np
import os
from simulate import generate_expert_heads, generate_data, set_seed, train_router, evaluate_model, compute_pfsr_coefficients, compute_confidence

D = 192
K = 4
d = D // K
C = 10

def get_loo_cv_threshold(W, train_z, train_tasks, train_classes, candidates=[0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]):
    N = train_z.shape[0]
    loo_confidences = []
    loo_parametric_alphas = []
    loo_pfsr_alphas = []
    
    # Run LOO-CV training
    for i in range(N):
        # Left-out split
        indices = [idx for idx in range(N) if idx != i]
        sub_z = train_z[indices]
        sub_tasks = train_tasks[indices]
        
        # Train router on N-1 samples
        sub_router = train_router(sub_z, sub_tasks, wd=0.1, epochs=150, lr=0.01)
        
        # Evaluate on the single left-out sample
        left_out_z = train_z[i:i+1]
        
        with torch.no_grad():
            logits = sub_router(left_out_z)
            alpha_param = torch.softmax(logits, dim=1)
            conf = compute_confidence(alpha_param, metric="max")
            
        alpha_pfsr, _ = compute_pfsr_coefficients(left_out_z, W)
        
        loo_confidences.append(conf.item())
        loo_parametric_alphas.append(alpha_param[0])
        loo_pfsr_alphas.append(alpha_pfsr[0])
        
    # Find the threshold that maximizes joint accuracy over left-out samples
    best_th = 0.85
    best_acc = -1.0
    
    for th in candidates:
        correct = 0
        for i in range(N):
            # Compute hybrid alpha
            if loo_confidences[i] >= th:
                alpha_hybrid = loo_parametric_alphas[i]
            else:
                alpha_hybrid = loo_pfsr_alphas[i]
                
            # Compute classification logit
            logits_c = torch.zeros(C)
            for k in range(K):
                z_kb = train_z[i, k*d : (k+1)*d]
                z_kb_norm = z_kb / torch.norm(z_kb)
                expert_logits = torch.matmul(W[k], z_kb_norm)
                logits_c += alpha_hybrid[k] * expert_logits
                
            pred = torch.argmax(logits_c).item()
            if pred == train_classes[i].item():
                correct += 1
                
        acc = (correct / N) * 100.0
        if acc > best_acc:
            best_acc = acc
            best_th = th
        elif acc == best_acc:
            # Tie breaker: choose the one closer to 0.85
            if abs(th - 0.85) < abs(best_th - 0.85):
                best_th = th
                
    return best_th

def get_random_projection_threshold(router, num_samples=1000, alpha_sig=0.05):
    # Generate random noise vectors
    z_noise = torch.randn(num_samples, D)
    # Normalize noise vectors
    z_noise = z_noise / torch.norm(z_noise, dim=1, keepdim=True)
    
    with torch.no_grad():
        logits = router(z_noise)
        alpha_param = torch.softmax(logits, dim=1)
        conf = compute_confidence(alpha_param, metric="max")
        
    # Select the (1 - alpha_sig) percentile as threshold
    sorted_conf, _ = torch.sort(conf)
    idx = int((1.0 - alpha_sig) * num_samples)
    idx = min(max(idx, 0), num_samples - 1)
    threshold = sorted_conf[idx].item()
    return threshold

def run_calibration_scarcity_verification():
    print("Initializing Calibration Verification Suite...")
    set_seed(42)
    W = generate_expert_heads()
    z_test, test_tasks, test_classes = generate_data(W, 500, [42])
    
    N_list = [16, 32]
    seeds = [10, 20, 30, 40, 50]
    
    results = {
        N: {
            "Static 0.85": [],
            "LOO-CV": [],
            "RP-Prior": []
        } for N in N_list
    }
    
    for N in N_list:
        print(f"\n--- Sweeping N = {N} ---")
        num_per_task = N // 4
        
        for seed in seeds:
            set_seed(seed)
            z_cal_seed, tasks_cal_seed, classes_cal_seed = generate_data(W, num_per_task, [seed])
            
            # Train router
            router = train_router(z_cal_seed, tasks_cal_seed, wd=0.1, epochs=150, lr=0.01)
            
            # 1. Static 0.85 Threshold
            acc_static = evaluate_model(z_test, test_tasks, test_classes, W, router=router, router_type="cghr",
                                        conf_metric="max", conf_threshold=0.85, batch_size=256, use_mbh=False, stream_type="homogeneous")
            results[N]["Static 0.85"].append(acc_static)
            
            # 2. LOO-CV Adaptive Threshold
            loo_th = get_loo_cv_threshold(W, z_cal_seed, tasks_cal_seed, classes_cal_seed)
            acc_loo = evaluate_model(z_test, test_tasks, test_classes, W, router=router, router_type="cghr",
                                      conf_metric="max", conf_threshold=loo_th, batch_size=256, use_mbh=False, stream_type="homogeneous")
            results[N]["LOO-CV"].append(acc_loo)
            
            # 3. Random Projection Prior Adaptive Threshold
            rp_th = get_random_projection_threshold(router)
            acc_rp = evaluate_model(z_test, test_tasks, test_classes, W, router=router, router_type="cghr",
                                    conf_metric="max", conf_threshold=rp_th, batch_size=256, use_mbh=False, stream_type="homogeneous")
            results[N]["RP-Prior"].append(acc_rp)
            
            print(f"  Seed {seed}: Static 0.85 Acc = {acc_static:.2f}% | LOO-CV (th={loo_th:.2f}) Acc = {acc_loo:.2f}% | RP-Prior (th={rp_th:.2f}) Acc = {acc_rp:.2f}%")
            
    print("\n=== FINAL REFINEMENT COMPARISON ===")
    for N in N_list:
        print(f"\nSample Complexity N = {N}:")
        for method in ["Static 0.85", "LOO-CV", "RP-Prior"]:
            mean_acc = np.mean(results[N][method])
            std_acc = np.std(results[N][method])
            print(f"  {method:12s}: {mean_acc:.2f}% +- {std_acc:.2f}%")

if __name__ == "__main__":
    run_calibration_scarcity_verification()
