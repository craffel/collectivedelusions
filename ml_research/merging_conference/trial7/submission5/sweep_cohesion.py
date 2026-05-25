import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from evaluate_ttmm import ExpertCNN, load_experts, get_datasets, precompute_unified_prototypes, compute_batch_cohesion

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experts = load_experts()
    datasets_dict = get_datasets()
    proto_data = precompute_unified_prototypes(experts, datasets_dict)
    
    mnist_loader = DataLoader(datasets_dict["mnist_test"], batch_size=64, shuffle=True)
    kmnist_loader = DataLoader(datasets_dict["kmnist_test"], batch_size=64, shuffle=True)
    fashion_loader = DataLoader(datasets_dict["fashion_test"], batch_size=64, shuffle=True)
    
    stream_batches = []
    mnist_iter = iter(mnist_loader)
    for _ in range(10):
        imgs, lbls = next(mnist_iter)
        stream_batches.append((imgs, lbls, "MNIST"))
        
    kmnist_iter = iter(kmnist_loader)
    for _ in range(10):
        imgs, lbls = next(kmnist_iter)
        stream_batches.append((imgs, lbls, "KMNIST"))
        
    fashion_iter = iter(fashion_loader)
    for _ in range(10):
        imgs, lbls = next(fashion_iter)
        stream_batches.append((imgs, lbls, "FashionMNIST"))
        
    print("\n--- Batch Cohesion Scores across Stream ---")
    mnist_scores = []
    kmnist_scores = []
    fashion_scores = []
    
    for idx, (images, labels, domain) in enumerate(stream_batches):
        images = images.to(device)
        with torch.no_grad():
            feats = proto_data["static_model"].extract_features(images)
        c_mnist = compute_batch_cohesion(feats, proto_data["prototypes_mnist"], proto_data["mu_static"])
        c_kmnist = compute_batch_cohesion(feats, proto_data["prototypes_kmnist"], proto_data["mu_static"])
        max_cohesion = max(c_mnist, c_kmnist)
        
        print(f"Batch {idx+1:02d} | Domain: {domain:<12} | C_MNIST: {c_mnist:.4f} | C_KMNIST: {c_kmnist:.4f} | Max Cohesion: {max_cohesion:.4f}")
        
        if domain == "MNIST":
            mnist_scores.append(max_cohesion)
        elif domain == "KMNIST":
            kmnist_scores.append(max_cohesion)
        elif domain == "FashionMNIST":
            fashion_scores.append(max_cohesion)
            
    print("\n--- Statistics ---")
    print(f"MNIST Max Cohesion      | Min: {min(mnist_scores):.4f} | Max: {max(mnist_scores):.4f} | Mean: {np.mean(mnist_scores):.4f}")
    print(f"KMNIST Max Cohesion     | Min: {min(kmnist_scores):.4f} | Max: {max(kmnist_scores):.4f} | Mean: {np.mean(kmnist_scores):.4f}")
    print(f"FashionMNIST Cohesion   | Min: {min(fashion_scores):.4f} | Max: {max(fashion_scores):.4f} | Mean: {np.mean(fashion_scores):.4f}")

    # Sweep thresholds and print NDR / FPR
    print("\n--- Threshold Sweep (\tau_N) ---")
    print(f"{'Threshold':<10} | {'NDR (%)':<10} | {'FPR (%)':<10}")
    print("-" * 35)
    for tau in np.arange(0.1, 0.7, 0.05):
        # Known domains are batches 0-19 (MNIST & KMNIST), novel is batches 20-29 (FashionMNIST)
        ndr_count = 0  # True positives: novel batch flagged as novel (Fashion, max_cohesion < tau)
        fpr_count = 0  # False positives: known batch flagged as novel (MNIST/KMNIST, max_cohesion < tau)
        
        for idx, max_cohesion in enumerate(mnist_scores + kmnist_scores):
            if max_cohesion < tau:
                fpr_count += 1
                
        for idx, max_cohesion in enumerate(fashion_scores):
            if max_cohesion < tau:
                ndr_count += 1
                
        ndr = 100.0 * ndr_count / len(fashion_scores)
        fpr = 100.0 * fpr_count / (len(mnist_scores) + len(kmnist_scores))
        print(f"{tau:.2f}      | {ndr:.1f}%     | {fpr:.1f}%")

if __name__ == "__main__":
    main()
