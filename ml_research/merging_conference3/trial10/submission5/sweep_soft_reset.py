import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from compare_coupled_baselines import get_20newsgroups_data, train_experts, ExpertMLP, set_seed

def run_evaluation_soft_reset(lambda_reset):
    train_texts, train_targets, test_texts, test_targets = get_20newsgroups_data()
    train_indices = [i for i, text in enumerate(train_texts) if len(text.strip()) > 10]
    test_indices = [i for i, text in enumerate(test_texts) if len(text.strip()) > 10]
    train_texts = [train_texts[i] for i in train_indices]
    train_targets = train_targets[train_indices]
    test_texts = [test_texts[i] for i in test_indices]
    test_targets = test_targets[test_indices]
    
    vectorizer = TfidfVectorizer(max_features=1024, stop_words='english')
    X_train = vectorizer.fit_transform(train_texts).toarray()
    X_test = vectorizer.transform(test_texts).toarray()
    
    K = 4
    seeds = [42, 101, 2023, 777, 999]
    
    results = {"acc": [], "jitter": []}
    
    for seed in seeds:
        set_seed(seed)
        experts = train_experts(X_train, train_targets, K=K, input_dim=1024)
        
        # Calibrate centroids
        centroids_l3 = []
        for k in range(K):
            idx = np.where(train_targets == k)[0][:64]
            centroids_l3.append(np.mean(X_train[idx], axis=0))
        centroids_l3 = np.array(centroids_l3)
        
        centroids_l4 = []
        for k in range(K):
            idx = np.where(train_targets == k)[0][:64]
            with torch.no_grad():
                h_list = []
                for expert in experts:
                    h, _ = expert(torch.FloatTensor(X_train[idx]))
                    h_list.append(h.numpy())
                avg_h = np.mean(np.array(h_list), axis=0)
                centroids_l4.append(np.mean(avg_h, axis=0))
        centroids_l4 = np.array(centroids_l4)
        
        # Test stream
        block_len = 50
        stream_targets = []
        stream_features = []
        test_idx_by_domain = {k: np.where(test_targets == k)[0] for k in range(K)}
        
        num_blocks = 16
        for b in range(num_blocks):
            domain = b % K
            available_idx = test_idx_by_domain[domain]
            selected_idx = np.random.choice(available_idx, block_len, replace=True)
            for idx in selected_idx:
                stream_targets.append(domain)
                stream_features.append(X_test[idx])
                
        stream_targets = np.array(stream_targets)
        stream_features = np.array(stream_features)
        num_samples = len(stream_targets)
        
        accuracies = []
        jitters = []
        temp = 0.05
        eta = 0.10
        
        prev_s = np.ones(K) / np.sqrt(K)
        all_layer_alphas = []
        
        for i in range(num_samples):
            feat = stream_features[i]
            target_domain = stream_targets[i]
            
            s_state = np.copy(prev_s)
            alpha_l3 = s_state ** 2
            sample_alphas = [alpha_l3]
            
            # Layer 4 ensembling
            norm_feat = np.linalg.norm(feat) + 1e-8
            norm_c3 = np.linalg.norm(centroids_l3, axis=1) + 1e-8
            sim3 = np.dot(centroids_l3, feat) / (norm_feat * norm_c3)
            
            shifted_sim3 = (sim3 - np.max(sim3)) / temp
            w3 = np.exp(shifted_sim3)
            w3 = w3 / np.sum(w3)
            
            w_sphere3 = w3 / (np.linalg.norm(w3) + 1e-6)
            
            # Apply soft continuous reset damping
            c_val3 = np.dot(s_state, w_sphere3)
            phi = np.arccos(np.clip(c_val3, -1.0, 1.0))
            s_uniform = np.ones(K) / np.sqrt(K)
            
            s_state_blended = np.cos(lambda_reset * phi) * s_state + np.sin(lambda_reset * phi) * s_uniform
            s_state = s_state_blended / (np.linalg.norm(s_state_blended) + 1e-8)
            
            c_val3_new = np.dot(s_state, w_sphere3)
            if np.abs(c_val3_new) < 1.0 - 1e-6:
                v_orth = w_sphere3 - c_val3_new * s_state
                u_vec = v_orth / (np.linalg.norm(v_orth) + 1e-8)
                theta = eta * np.arccos(np.clip(c_val3_new, -1.0, 1.0))
                s_state = np.cos(theta) * s_state + np.sin(theta) * u_vec
            alpha_l4 = s_state ** 2
            sample_alphas.append(alpha_l4)
            
            # Layer 5 ensembling
            with torch.no_grad():
                h_vals = []
                out_vals = []
                for expert in experts:
                    h, out = expert(torch.FloatTensor(feat))
                    h_vals.append(h.numpy())
                    out_vals.append(out.numpy())
                h_vals = np.array(h_vals)
                out_vals = np.array(out_vals)
                
            blended_h = np.dot(alpha_l4, h_vals)
            norm_bh = np.linalg.norm(blended_h) + 1e-8
            norm_c4 = np.linalg.norm(centroids_l4, axis=1) + 1e-8
            sim4 = np.dot(centroids_l4, blended_h) / (norm_bh * norm_c4)
            
            shifted_sim4 = (sim4 - np.max(sim4)) / temp
            w4 = np.exp(shifted_sim4)
            w4 = w4 / np.sum(w4)
            
            w_sphere4 = w4 / (np.linalg.norm(w4) + 1e-6)
            c_val4 = np.dot(s_state, w_sphere4)
            if np.abs(c_val4) < 1.0 - 1e-6:
                v_orth = w_sphere4 - c_val4 * s_state
                u_vec = v_orth / (np.linalg.norm(v_orth) + 1e-8)
                theta = eta * np.arccos(np.clip(c_val4, -1.0, 1.0))
                s_state = np.cos(theta) * s_state + np.sin(theta) * u_vec
            alpha_l5 = s_state ** 2
            sample_alphas.append(alpha_l5)
            
            all_layer_alphas.append(sample_alphas)
            
            blended_out = np.dot(alpha_l5, out_vals)
            pred_domain = np.argmax(blended_out)
            accuracies.append(pred_domain == target_domain)
            
            # State save for next query
            prev_s = np.copy(s_state)
                
        mean_acc = np.mean(accuracies)
        for i in range(num_samples):
            alphas = all_layer_alphas[i]
            j34 = np.mean((alphas[1] - alphas[0]) ** 2)
            j45 = np.mean((alphas[2] - alphas[1]) ** 2)
            jitters.append((j34 + j45) / 2.0)
        mean_jitter = np.mean(jitters)
        
        results["acc"].append(mean_acc)
        results["jitter"].append(mean_jitter)
        
    accs_std = np.array(results["acc"]) * 100
    jits_std = np.array(results["jitter"])
    print(f"Lambda: {lambda_reset:.2f} | Acc: {np.mean(accs_std):.2f}% ± {np.std(accs_std):.2f}% | Jitter: {np.mean(jits_std)*10000:.4f} ± {np.std(jits_std)*10000:.4f}")

if __name__ == "__main__":
    lambdas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    print("Sweeping Soft Continuous Reset Lambdas for NLP task...")
    for l in lambdas:
        run_evaluation_soft_reset(l)
