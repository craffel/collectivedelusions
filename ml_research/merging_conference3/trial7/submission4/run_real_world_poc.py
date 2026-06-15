import torch
import torchvision.models as models
import numpy as np

def run_poc():
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    print("Loading pre-trained ResNet18...")
    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    resnet.eval()

    # Penultimate feature dim is 512, classes are 1000
    fc_weights = resnet.fc.weight.data # [1000, 512]
    
    # Define 3 tasks from ImageNet-1K classes
    # Dogs: 151-160 (10 classes)
    dog_classes = list(range(151, 161))
    # Cats: 281-285 (5 classes)
    cat_classes = list(range(281, 286))
    # Vehicles: 817, 829, 511, 436, 437, 438, 555, 656, 751, 847 (10 classes)
    vehicle_classes = [817, 829, 511, 436, 437, 438, 555, 656, 751, 847]

    task_classes = [dog_classes, cat_classes, vehicle_classes]
    num_tasks = len(task_classes)
    
    # 1. Extract expert weights and compute SVD centroids
    centroids = []
    for k, classes in enumerate(task_classes):
        W_k = fc_weights[classes] # [num_classes, 512]
        
        # Compute SVD centroid
        U, S, Vh = torch.linalg.svd(W_k, full_matrices=False)
        # Vh shape is [num_classes, 512]
        v_k = Vh[0] # First principal component
        
        # Normalize centroid
        v_k_norm = v_k / (torch.norm(v_k) + 1e-8)
        centroids.append(v_k_norm)
        
    centroids = torch.stack(centroids, dim=0) # [3, 512]
    
    # 2. Compute OTSP orthonormal basis using Löwdin Orthogonalization
    Gram = torch.matmul(centroids, centroids.t())
    print("\nPairwise Cosine Similarities (Gram Overlap Matrix):")
    print(f"Dogs vs Cats: {Gram[0, 1].item():.4f}")
    print(f"Dogs vs Vehicles: {Gram[0, 2].item():.4f}")
    print(f"Cats vs Vehicles: {Gram[1, 2].item():.4f}")
    
    evals, evecs = torch.linalg.eigh(Gram)
    inv_sqrt_evals = 1.0 / torch.sqrt(evals + 1e-6)
    Gram_inv_sqrt = torch.matmul(evecs, torch.matmul(torch.diag(inv_sqrt_evals), evecs.t()))
    Q = torch.matmul(Gram_inv_sqrt, centroids) # [3, 512]
    
    # 3. Generate realistic penultimate representations by adding noise to class prototypes
    samples_per_class = 50
    X = []
    task_labels = []
    class_labels = []
    
    noise_std = 0.15 # realistic noise standard deviation
    
    for k, classes in enumerate(task_classes):
        for c in classes:
            prototype = fc_weights[c] # [512]
            # Generate noisy representations
            noise = torch.randn(samples_per_class, 512) * noise_std
            z = prototype.unsqueeze(0) + noise
            X.append(z)
            task_labels.extend([k] * samples_per_class)
            class_labels.extend([c] * samples_per_class)
            
    X = torch.cat(X, dim=0) # [num_samples, 512]
    task_labels = torch.tensor(task_labels, dtype=torch.long)
    class_labels = torch.tensor(class_labels, dtype=torch.long)
    
    # 4. Evaluate PFSR and OTSP Gating
    def evaluate_router(features, Q_basis, orig_centroids, labels, tau=0.3):
        # Normalize features
        features_norm = features / (torch.norm(features, dim=1, keepdim=True) + 1e-8)
        
        # PFSR: absolute projection onto original centroids
        u_pfsr = torch.abs(torch.matmul(features_norm, orig_centroids.t()))
        alpha_pfsr = torch.softmax(u_pfsr / tau, dim=-1)
        preds_pfsr = torch.argmax(alpha_pfsr, dim=-1)
        acc_pfsr = (preds_pfsr == labels).float().mean().item()
        
        # OTSP: absolute projection onto orthonormal basis
        u_otsp = torch.abs(torch.matmul(features_norm, Q_basis.t()))
        alpha_otsp = torch.softmax(u_otsp / tau, dim=-1)
        preds_otsp = torch.argmax(alpha_otsp, dim=-1)
        acc_otsp = (preds_otsp == labels).float().mean().item()
        
        return acc_pfsr, acc_otsp
    
    # Run evaluation over a range of gating temperatures
    print("\nEvaluating Parameter-Free Routing Accuracy on ResNet18 Manifold:")
    for t in [0.01, 0.1, 0.3, 0.5, 1.0]:
        acc_pfsr, acc_otsp = evaluate_router(X, Q, centroids, task_labels, tau=t)
        print(f"Temperature tau = {t:4.2f} | PFSR Accuracy: {acc_pfsr*100:6.2f}% | OTSP Accuracy: {acc_otsp*100:6.2f}%")

if __name__ == '__main__':
    run_poc()
