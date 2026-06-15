import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dset
import timm
import matplotlib.pyplot as plt

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Define a simple Linear Router model in PyTorch
class LinearRouterPyTorch(nn.Module):
    def __init__(self, input_dim=192, num_tasks=4):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_tasks, bias=True)
        nn.init.normal_(self.linear.weight, std=0.01)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        logits = self.linear(x)
        return torch.softmax(logits, dim=-1)

def train_linear_router(X_cal, y_cal, epochs=150, lr=0.01, wd=1e-3):
    D = X_cal.shape[1]
    K = len(np.unique(y_cal))

    X_flat = torch.tensor(X_cal, dtype=torch.float32)
    y_flat = torch.tensor(y_cal, dtype=torch.long)

    router = LinearRouterPyTorch(D, K)
    optimizer = optim.Adam(router.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()

    router.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = router.linear(X_flat)
        loss = criterion(logits, y_flat)
        loss.backward()
        optimizer.step()

    router.eval()
    return router

# Define a Task-Specific Classifier for downstream 10-class evaluation
class TaskClassifier(nn.Module):
    def __init__(self, input_dim=192, num_classes=10):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        nn.init.normal_(self.linear.weight, std=0.01)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)

def train_task_classifier(X_cal, y_cal, epochs=150, lr=0.01, wd=1e-3):
    D = X_cal.shape[1]
    classifier = TaskClassifier(D, 10)
    optimizer = optim.Adam(classifier.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()

    X_tensor = torch.tensor(X_cal, dtype=torch.float32)
    y_tensor = torch.tensor(y_cal, dtype=torch.long)

    classifier.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = classifier(X_tensor)
        loss = criterion(logits, y_tensor)
        loss.backward()
        optimizer.step()

    classifier.eval()
    return classifier

def main():
    print("Initializing Real-World ViT-Tiny Experiment (Enhanced)...")
    
    # 1. Load pre-trained ViT-Tiny model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = timm.create_model('vit_tiny_patch16_224', pretrained=True)
    model.to(device)
    model.eval()
    
    # 2. Define transforms for 224x224 ImageNet-normalized inputs
    mnist_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)), # 1 -> 3 channels
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    cifar_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 3. Load Datasets
    print("Loading datasets...")
    mnist_set = dset.MNIST(root='/tmp/mnist', train=False, download=True, transform=mnist_transform)
    fmnist_set = dset.FashionMNIST(root='/tmp/fmnist', train=False, download=True, transform=mnist_transform)
    cifar_set = dset.CIFAR10(root='/tmp/cifar10', train=False, download=True, transform=cifar_transform)
    svhn_set = dset.SVHN(root='/tmp/svhn', split='test', download=True, transform=cifar_transform)
    
    datasets = [mnist_set, fmnist_set, cifar_set, svhn_set]
    task_names = ["MNIST (Task 0)", "Fashion-MNIST (Task 1)", "CIFAR-10 (Task 2)", "SVHN (Task 3)"]
    K = 4
    D = 192
    
    # 4. Helper function to extract Layer 12 activations
    def extract_layer12_activations(imgs):
        with torch.no_grad():
            x = model.patch_embed(imgs)
            x = model.pos_drop(x)
            if hasattr(model, 'patch_drop'):
                x = model.patch_drop(x)
            x = model.norm_pre(x)
            for i in range(12):
                x = model.blocks[i](x)
            cls_tokens = x[:, 0]
        return cls_tokens.cpu().numpy()
        
    # 5. Extract calibration samples (64 per task), compute centroids, and train task classifiers
    print("Extracting calibration activations and labels...")
    centroids = []
    gmc_centroids = [] # For Gaussian Mixture Centroids (M=3)
    cal_X = []
    cal_y = []
    task_classifiers = []
    
    from sklearn.cluster import KMeans
    
    for k in range(K):
        loader = torch.utils.data.DataLoader(datasets[k], batch_size=64, shuffle=False)
        imgs, labels = next(iter(loader))
        imgs = imgs.to(device)
        activations = extract_layer12_activations(imgs) # [64, 192]
        cal_X.append(activations)
        cal_y.append(np.full(64, k))
        
        # Train a 10-class linear probe classifier on this task's calibration features
        print(f"Training task classifier probe for {task_names[k]}...")
        task_classifier = train_task_classifier(activations, labels.numpy(), epochs=150)
        task_classifiers.append(task_classifier)
        
        centroid = np.mean(activations, axis=0)
        centroid /= np.linalg.norm(centroid) # L2 normalize
        centroids.append(centroid)
        
        # Compute GMC centroids using KMeans (M=3)
        print(f"Computing Gaussian Mixture Centroids (M=3) for {task_names[k]}...")
        kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto').fit(activations)
        task_gmc = kmeans.cluster_centers_ # [3, 192]
        task_gmc = task_gmc / np.linalg.norm(task_gmc, axis=1, keepdims=True) # L2 normalize each centroid
        gmc_centroids.append(task_gmc)
        
    centroids = np.array(centroids)
    print("Real-world centroids computed.")
    
    # Compute Cosine Similarity Matrix between physical centroids
    rho_real = np.zeros((K, K))
    for k in range(K):
        for j in range(K):
            rho_real[k, j] = np.dot(centroids[k], centroids[j])
            
    print("\n--- Physical Cosine Similarity Matrix (SIT rho) ---")
    print(rho_real)
    print("--------------------------------------------------\n")
    
    # Save physical similarity matrix plot
    os.makedirs("results", exist_ok=True)
    plt.figure(figsize=(6, 5))
    plt.imshow(rho_real, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    plt.colorbar()
    plt.title("Physical Cosine Similarity of Task Centroids (ViT-Tiny Layer 12)")
    plt.xticks(range(K), [t.split()[0] for t in task_names])
    plt.yticks(range(K), [t.split()[0] for t in task_names])
    for k in range(K):
        for j in range(K):
            plt.text(j, k, f"{rho_real[k, j]:.4f}", ha="center", va="center", color="black" if abs(rho_real[k,j]) < 0.6 else "white")
    plt.tight_layout()
    plt.savefig("results/physical_similarity_matrix.png", dpi=300)
    plt.close()
    
    # Train parametric baselines
    print("Training parametric Linear Routers (Few-Shot and Fully Optimized)...")
    # Few-Shot: 16 samples per task (64 total)
    cal_X_few = np.concatenate([cal_X[k][:16] for k in range(K)], axis=0)
    cal_y_few = np.concatenate([cal_y[k][:16] for k in range(K)], axis=0)
    few_shot_router = train_linear_router(cal_X_few, cal_y_few, epochs=150)
    
    # Fully Optimized: 64 samples per task (256 total)
    cal_X_full = np.concatenate([cal_X[k] for k in range(K)], axis=0)
    cal_y_full = np.concatenate([cal_y[k] for k in range(K)], axis=0)
    fully_optimized_router = train_linear_router(cal_X_full, cal_y_full, epochs=150)
    print("Parametric routers trained successfully.")
    
    # 6. Extract test activations (100 per task) and classification labels
    print("Extracting test activations...")
    test_X = []
    test_y = []
    test_y_class = []
    
    for k in range(K):
        loader = torch.utils.data.DataLoader(datasets[k], batch_size=100, shuffle=True)
        imgs, labels = next(iter(loader))
        imgs = imgs.to(device)
        activations = extract_layer12_activations(imgs) # [100, 192]
        test_X.append(activations)
        test_y.append(np.full(100, k))
        test_y_class.append(labels.numpy())
        
    test_X = np.array(test_X) # [4, 100, 192]
    test_y = np.array(test_y) # [4, 100]
    test_y_class = np.array(test_y_class) # [4, 100]
    
    # Flatten test data
    test_X_flat = test_X.reshape(-1, D)
    test_y_flat = test_y.reshape(-1)
    test_y_class_flat = test_y_class.reshape(-1)
    num_total = len(test_X_flat)

    # Diagnostic check for individual classifier accuracy on clean in-distribution test sets
    print("\n================ INDIVIDUAL IN-DOMAIN CLASSIFIER ACCURACY (%) ================")
    for k in range(K):
        h_torch = torch.tensor(test_X[k], dtype=torch.float32)
        with torch.no_grad():
            logits = task_classifiers[k](h_torch)
            preds = torch.argmax(logits, dim=-1).numpy()
        indomain_acc = np.mean(preds == test_y_class[k]) * 100.0
        print(f"{task_names[k]:<35} : {indomain_acc:.2f}%")
    print("==============================================================================\n")
    
    # 7. Evaluate Routers under Noise Sweep
    noise_scales = [0.0, 0.5, 1.0, 1.5, 2.0]
    methods = [
        "Linear Router (Few-Shot)", 
        "Linear Router (Fully-Optimized)", 
        "SABLE", 
        "SPS-ZCA", 
        "ESM-LVC (Ours)",
        "ESM-LVC (DM-BSC)",
        "ESM-LVC (GMC-BSC)"
    ]
    
    results_acc = {m: [] for m in methods}
    results_class_acc = {m: [] for m in methods}
    results_entropy = {m: [] for m in methods}
    
    centroids_torch = torch.tensor(centroids, dtype=torch.float32)
    
    for sigma in noise_scales:
        print(f"Evaluating with physical representation noise scale: {sigma}...")
        # Add isotropic Gaussian noise to test activations
        np.random.seed(42) # set local seed for noise generation consistency
        noise = np.random.normal(0, sigma, test_X_flat.shape)
        test_X_noisy = test_X_flat + noise
        
        for method in methods:
            correct_routes = 0
            alphas = []
            
            if method == "Linear Router (Few-Shot)":
                with torch.no_grad():
                    X_torch = torch.tensor(test_X_noisy, dtype=torch.float32)
                    alphas = few_shot_router(X_torch).numpy()
                    
            elif method == "Linear Router (Fully-Optimized)":
                with torch.no_grad():
                    X_torch = torch.tensor(test_X_noisy, dtype=torch.float32)
                    alphas = fully_optimized_router(X_torch).numpy()
                    
            elif method == "SABLE":
                for i in range(num_total):
                    h_b = torch.tensor(test_X_noisy[i], dtype=torch.float32)
                    u = torch.zeros(K)
                    for k in range(K):
                        u[k] = torch.sum(h_b * centroids_torch[k]) / (torch.norm(h_b) * torch.norm(centroids_torch[k]))
                    alpha = torch.softmax(u / 0.05, dim=-1).numpy()
                    alphas.append(alpha)
                alphas = np.array(alphas)
                    
            elif method == "SPS-ZCA":
                for i in range(num_total):
                    h_b = torch.tensor(test_X_noisy[i], dtype=torch.float32)
                    u = torch.zeros(K)
                    for k in range(K):
                        u[k] = torch.sum(h_b * centroids_torch[k]) / (torch.norm(h_b) * torch.norm(centroids_torch[k]))
                    alpha = torch.softmax(u / 0.001, dim=-1).numpy()
                    alphas.append(alpha)
                alphas = np.array(alphas)
                    
            elif method == "ESM-LVC (Ours)":
                # Auto threshold based on the real similarities
                off_diag = [rho_real[k, j] for k in range(K) for j in range(K) if k != j]
                avg_off_diag = np.mean(off_diag) if len(off_diag) > 0 else 0.0
                theta = avg_off_diag + 0.5 * (1.0 - avg_off_diag)
                
                # Symbiotic interaction tensor
                lam = 10.0
                Gamma = np.tanh(lam * (rho_real - theta))
                
                # Compute Adaptive Step-Size for DESS to guarantee stability
                G = np.sum(np.maximum(0.0, Gamma) * (1.0 - np.eye(K)), axis=1)
                max_G = np.max(G)
                u_max = 1.0
                eta_stable = 0.9
                if max_G < 1.0:
                    alpha_max = max(1.0, u_max / (1.0 - max_G))
                    delta_tau_adaptive = min(0.2, eta_stable / alpha_max)
                else:
                    N_steps = 5
                    alpha_max_t = 1.0
                    for _ in range(N_steps):
                        alpha_max_t = (1.0 + max_G) * alpha_max_t + u_max
                    delta_tau_adaptive = min(0.2, eta_stable / alpha_max_t)
                
                for i in range(num_total):
                    h_b = torch.tensor(test_X_noisy[i], dtype=torch.float32)
                    # Environmental attraction
                    u = np.zeros(K)
                    for k in range(K):
                        u[k] = torch.sum(h_b * centroids_torch[k]).item() / (torch.norm(h_b).item() * torch.norm(centroids_torch[k]).item())
                    
                    # Initial state
                    alpha_t = torch.softmax(torch.tensor(u / 0.03), dim=-1).numpy()
                    
                    # DESS with Adaptive Step Size
                    N_steps = 5
                    beta = 1.0
                    for step in range(N_steps):
                        d_alpha = alpha_t * (u + np.dot(Gamma, alpha_t) - beta * alpha_t)
                        alpha_t = alpha_t + delta_tau_adaptive * d_alpha
                        alpha_t = np.clip(alpha_t, 0.0, None)
                        
                    # Normalize and apply Adaptive Entropy-Driven Sharpening (AEDS)
                    # to prevent downstream classification dilution on disjoint tasks while
                    # dynamically regularizing under highly noisy/uncertain regimes.
                    eps_stabilizer = 1e-9
                    sum_alpha_t = np.sum(alpha_t)
                    if sum_alpha_t > 0:
                        alpha_norm = alpha_t / sum_alpha_t
                    else:
                        alpha_norm = np.full(K, 1.0 / K)
                    entropy_t = -np.sum(alpha_norm * np.log(alpha_norm + eps_stabilizer))
                    
                    # Exponential Information-Theoretic Adaptive Sharpening (E-ITAS) with confidence decay
                    norm_entropy = entropy_t / np.log(K)
                    eta_decay = 12.0
                    gamma_max = 6.0
                    gamma_dais = 1.0 + (gamma_max - 1.0) * np.exp(-eta_decay * norm_entropy)
                    
                    alpha_sharpened = np.power(alpha_t, gamma_dais)
                    sum_alpha = np.sum(alpha_sharpened)
                    if sum_alpha > 0:
                        alpha_final = alpha_sharpened / sum_alpha
                    else:
                        alpha_final = np.full(K, 1.0 / K)
                        
                    alphas.append(alpha_final)
                alphas = np.array(alphas)

            elif method == "ESM-LVC (DM-BSC)":
                # Auto threshold based on the real similarities
                off_diag = [rho_real[k, j] for k in range(K) for j in range(K) if k != j]
                avg_off_diag = np.mean(off_diag) if len(off_diag) > 0 else 0.0
                theta = avg_off_diag + 0.5 * (1.0 - avg_off_diag)
                
                # Symbiotic interaction tensor
                lam = 10.0
                Gamma = np.tanh(lam * (rho_real - theta))
                
                # Compute Adaptive Step-Size for DESS to guarantee stability
                G = np.sum(np.maximum(0.0, Gamma) * (1.0 - np.eye(K)), axis=1)
                max_G = np.max(G)
                u_max = 1.0
                eta_stable = 0.9
                if max_G < 1.0:
                    alpha_max = max(1.0, u_max / (1.0 - max_G))
                    delta_tau_adaptive = min(0.2, eta_stable / alpha_max)
                else:
                    N_steps = 5
                    alpha_max_t = 1.0
                    for _ in range(N_steps):
                        alpha_max_t = (1.0 + max_G) * alpha_max_t + u_max
                    delta_tau_adaptive = min(0.2, eta_stable / alpha_max_t)
                
                for i in range(num_total):
                    h_b = torch.tensor(test_X_noisy[i], dtype=torch.float32)
                    # Environmental attraction
                    u = np.zeros(K)
                    for k in range(K):
                        u[k] = torch.sum(h_b * centroids_torch[k]).item() / (torch.norm(h_b).item() * torch.norm(centroids_torch[k]).item())
                    
                    # Initial state
                    alpha_t = torch.softmax(torch.tensor(u / 0.03), dim=-1).numpy()
                    
                    # DESS with Adaptive Step Size
                    N_steps = 5
                    beta = 1.0
                    for step in range(N_steps):
                        d_alpha = alpha_t * (u + np.dot(Gamma, alpha_t) - beta * alpha_t)
                        alpha_t = alpha_t + delta_tau_adaptive * d_alpha
                        alpha_t = np.clip(alpha_t, 0.0, None)
                        
                    # Dirichlet-Multinomial Bayesian Self-Calibration (DM-BSC)
                    u_pos = np.maximum(0.0, u)
                    sum_u = np.sum(u_pos)
                    gamma_0 = 1.0
                    kappa = 12.0
                    gamma_max = 6.0
                    
                    S_b = K * gamma_0 + kappa * sum_u
                    C_Bayes = (kappa * sum_u) / S_b if S_b > 0 else 0.0
                    gamma_dais = 1.0 + (gamma_max - 1.0) * C_Bayes
                    
                    alpha_sharpened = np.power(alpha_t, gamma_dais)
                    sum_alpha = np.sum(alpha_sharpened)
                    if sum_alpha > 0:
                        alpha_final = alpha_sharpened / sum_alpha
                    else:
                        alpha_final = np.full(K, 1.0 / K)
                        
                    alphas.append(alpha_final)
                alphas = np.array(alphas)

            elif method == "ESM-LVC (GMC-BSC)":
                # Auto threshold based on the real similarities
                off_diag = [rho_real[k, j] for k in range(K) for j in range(K) if k != j]
                avg_off_diag = np.mean(off_diag) if len(off_diag) > 0 else 0.0
                theta = avg_off_diag + 0.5 * (1.0 - avg_off_diag)
                
                # Symbiotic interaction tensor
                lam = 10.0
                Gamma = np.tanh(lam * (rho_real - theta))
                
                # Compute Adaptive Step-Size for DESS to guarantee stability
                G = np.sum(np.maximum(0.0, Gamma) * (1.0 - np.eye(K)), axis=1)
                max_G = np.max(G)
                u_max = 1.0
                eta_stable = 0.9
                if max_G < 1.0:
                    alpha_max = max(1.0, u_max / (1.0 - max_G))
                    delta_tau_adaptive = min(0.2, eta_stable / alpha_max)
                else:
                    N_steps = 5
                    alpha_max_t = 1.0
                    for _ in range(N_steps):
                        alpha_max_t = (1.0 + max_G) * alpha_max_t + u_max
                    delta_tau_adaptive = min(0.2, eta_stable / alpha_max_t)
                
                for i in range(num_total):
                    # Environmental attraction using GMC
                    u = np.zeros(K)
                    h_b_np = test_X_noisy[i]
                    h_b_norm = h_b_np / np.linalg.norm(h_b_np)
                    for k in range(K):
                        u[k] = np.max(np.dot(gmc_centroids[k], h_b_norm))
                    
                    # Initial state
                    alpha_t = torch.softmax(torch.tensor(u / 0.03), dim=-1).numpy()
                    
                    # DESS with Adaptive Step Size
                    N_steps = 5
                    beta = 1.0
                    for step in range(N_steps):
                        d_alpha = alpha_t * (u + np.dot(Gamma, alpha_t) - beta * alpha_t)
                        alpha_t = alpha_t + delta_tau_adaptive * d_alpha
                        alpha_t = np.clip(alpha_t, 0.0, None)
                        
                    # Dirichlet-Multinomial Bayesian Self-Calibration (DM-BSC)
                    u_pos = np.maximum(0.0, u)
                    sum_u = np.sum(u_pos)
                    gamma_0 = 1.0
                    kappa = 12.0
                    gamma_max = 6.0
                    
                    S_b = K * gamma_0 + kappa * sum_u
                    C_Bayes = (kappa * sum_u) / S_b if S_b > 0 else 0.0
                    gamma_dais = 1.0 + (gamma_max - 1.0) * C_Bayes
                    
                    alpha_sharpened = np.power(alpha_t, gamma_dais)
                    sum_alpha = np.sum(alpha_sharpened)
                    if sum_alpha > 0:
                        alpha_final = alpha_sharpened / sum_alpha
                    else:
                        alpha_final = np.full(K, 1.0 / K)
                        
                    alphas.append(alpha_final)
                alphas = np.array(alphas)
                
            # Calculate routing accuracy, downstream classification accuracy, and entropy
            correct_routes = 0
            correct_classes = 0
            for i in range(num_total):
                predicted_task = np.argmax(alphas[i])
                true_task = test_y_flat[i]
                if predicted_task == true_task:
                    correct_routes += 1
                
                # Get downstream class prediction using task-specific classifier heads
                h_torch = torch.tensor(test_X_noisy[i], dtype=torch.float32)
                P_tasks = []
                for k_head in range(K):
                    with torch.no_grad():
                        logits = task_classifiers[k_head](h_torch)
                        probs = torch.softmax(logits, dim=-1).numpy()
                        P_tasks.append(probs)
                P_tasks = np.array(P_tasks) # [K, 10]
                
                # Soft-mix task-specific predictions into a joint 40-class distribution
                P_joint = (alphas[i][:, None] * P_tasks).flatten() # shape [40]
                predicted_class = np.argmax(P_joint)
                true_class = true_task * 10 + test_y_class_flat[i]
                if predicted_class == true_class:
                    correct_classes += 1
            
            acc = (correct_routes / num_total) * 100.0
            class_acc = (correct_classes / num_total) * 100.0
            
            # Calculate mean entropy (avoid log(0))
            eps = 1e-9
            entropies = -np.sum(alphas * np.log(alphas + eps), axis=1)
            mean_ent = np.mean(entropies)
            
            results_acc[method].append(acc)
            results_class_acc[method].append(class_acc)
            results_entropy[method].append(mean_ent)
            
    # Print summary tables
    print("\n================== PHYSICAL ROUTING ACCURACY (%) ==================")
    header = f"{'Method':<35}" + "".join([f"Noisy σ={s:<8.1f}" for s in noise_scales])
    print(header)
    print("-" * len(header))
    for m in methods:
        row = f"{m:<35}" + "".join([f"{results_acc[m][idx]:<14.2f}" for idx in range(len(noise_scales))])
        print(row)
    print("===================================================================\n")

    print("\n============== PHYSICAL DOWNSTREAM CLASSIFICATION ACCURACY (%) ==============")
    header_class = f"{'Method':<35}" + "".join([f"Noisy σ={s:<8.1f}" for s in noise_scales])
    print(header_class)
    print("-" * len(header_class))
    for m in methods:
        row = f"{m:<35}" + "".join([f"{results_class_acc[m][idx]:<14.2f}" for idx in range(len(noise_scales))])
        print(row)
    print("===================================================================\n")

    print("===================== MEAN ROUTING ENTROPY =======================")
    header_ent = f"{'Method':<35}" + "".join([f"Noisy σ={s:<8.1f}" for s in noise_scales])
    print(header_ent)
    print("-" * len(header_ent))
    for m in methods:
        row = f"{m:<35}" + "".join([f"{results_entropy[m][idx]:<14.4f}" for idx in range(len(noise_scales))])
        print(row)
    print("===================================================================\n")
    
    # Save physical routing noise robustness plot
    plt.figure(figsize=(8, 5))
    styles = ['^-', 'v-', 'o-', 's-', 'D-', 'x-', '*-']
    colors = ['tab:red', 'tab:orange', 'tab:green', 'tab:blue', 'tab:purple', 'tab:brown', 'tab:pink']
    for idx, m in enumerate(methods):
        plt.plot(noise_scales, results_acc[m], styles[idx], label=m, color=colors[idx], linewidth=2, markersize=8)
    plt.xlabel("Representation Space Noise Scale ($\sigma$)", fontsize=12)
    plt.ylabel("Routing Accuracy (%)", fontsize=12)
    plt.title("Noise Robustness on Physical ViT-Tiny Activation Space", fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig("results/physical_routing_noise_robustness.png", dpi=300)
    plt.close()
    
    # Save results to a text file for easy LaTeX inclusion
    with open("results/real_vit_results.txt", "w") as f:
        f.write("Physical Cosine Similarity Matrix (SIT rho):\n")
        f.write(np.array2string(rho_real, precision=4, separator=', ') + "\n\n")
        
        f.write("Physical Routing Accuracy (%):\n")
        f.write(header + "\n")
        for m in methods:
            row = f"{m:<35}" + "".join([f"{results_acc[m][idx]:<14.2f}" for idx in range(len(noise_scales))])
            f.write(row + "\n")
            
        f.write("\nPhysical Downstream Classification Accuracy (%):\n")
        f.write(header_class + "\n")
        for m in methods:
            row = f"{m:<35}" + "".join([f"{results_class_acc[m][idx]:<14.2f}" for idx in range(len(noise_scales))])
            f.write(row + "\n")
            
        f.write("\nMean Routing Entropy:\n")
        f.write(header_ent + "\n")
        for m in methods:
            row = f"{m:<35}" + "".join([f"{results_entropy[m][idx]:<14.4f}" for idx in range(len(noise_scales))])
            f.write(row + "\n")
            
    print("Enhanced Real-World ViT-Tiny physical experiment completed and results/plots saved successfully.")

if __name__ == "__main__":
    main()
