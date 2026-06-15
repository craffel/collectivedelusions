import os
import torch
import timm
import numpy as np
import time
from torch.utils.data import Subset, DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Set seeds
torch.manual_seed(42)
np.random.seed(42)

class GrayscaleToRGB(object):
    def __call__(self, img):
        return img.convert('RGB')

def get_datasets(data_dir='./data'):
    transform_rgb = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_gray = transforms.Compose([
        GrayscaleToRGB(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    mnist_train = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform_gray)
    fmnist_train = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform_gray)
    cifar_train = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_rgb)
    svhn_train = datasets.SVHN(root=data_dir, split='train', download=True, transform=transform_rgb)
    
    mnist_test = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform_gray)
    fmnist_test = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform_gray)
    cifar_test = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_rgb)
    svhn_test = datasets.SVHN(root=data_dir, split='test', download=True, transform=transform_rgb)
    
    return {
        'MNIST': (mnist_train, mnist_test),
        'FashionMNIST': (fmnist_train, fmnist_test),
        'CIFAR10': (cifar_train, cifar_test),
        'SVHN': (svhn_train, svhn_test)
    }

def get_layer_group_idx(key):
    if 'patch_embed' in key or 'cls_token' in key or 'pos_embed' in key:
        return 0
    elif 'blocks.' in key:
        parts = key.split('.')
        return int(parts[1]) + 1
    elif 'norm.' in key:
        return 13
    else:
        return None

def merge_weights(base_state_dict, task_vectors, alpha_bar, device='cpu'):
    merged_state_dict = {}
    for key in base_state_dict:
        group_idx = get_layer_group_idx(key)
        if group_idx is not None:
            merged_val = base_state_dict[key].clone().to(device)
            for k in range(len(task_vectors)):
                coeff = alpha_bar[group_idx, k].to(device)
                merged_val = merged_val + coeff * task_vectors[k][key].to(device)
            merged_state_dict[key] = merged_val
        else:
            merged_state_dict[key] = base_state_dict[key].clone().to(device)
    return merged_state_dict

def evaluate_merged_model_with_coeffs(base_model, task_vectors, alpha_bar, task_heads, test_loader, task_idx, device='cpu'):
    pretrained_state = torch.load('checkpoints/base_model.pt', weights_only=True)
    merged_state_dict = merge_weights(pretrained_state, task_vectors, alpha_bar, device=device)
    base_model.load_state_dict(merged_state_dict, strict=False)
    
    base_model.head.weight.data.copy_(task_heads[task_idx]['weight'].to(device))
    base_model.head.bias.data.copy_(task_heads[task_idx]['bias'].to(device))
    
    base_model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = base_model(x)
            correct += (logits.argmax(dim=-1) == y).sum().item()
            total += y.size(0)
    return correct / total

class GatedChaosMergeModel(torch.nn.Module):
    def __init__(self, d=4, K=4, L=14, init_val=-2.0):
        super().__init__()
        self.d = d
        self.K = K
        self.L = L
        self.W_init = torch.nn.Parameter(torch.randn(K, d) * 0.02)
        self.b_init = torch.nn.Parameter(torch.zeros(K))
        self.gamma_raw = torch.nn.Parameter(torch.zeros(L))
        self.R_raw = torch.nn.Parameter(torch.ones(L, K) * 0.3)
        self.Phi = torch.nn.Parameter(torch.randn(L, K, d) * 0.02)
        self.phi = torch.nn.Parameter(torch.zeros(L, K))
        self.gate_raw = torch.nn.Parameter(torch.ones(L) * init_val)
        self.register_buffer('P', torch.randn(192, d))
        self.P.data = torch.nn.functional.normalize(self.P.data, dim=0)

    def forward_coefficients(self, x_features, average_batch=True):
        B = x_features.shape[0]
        psi_tilde = torch.matmul(x_features, self.P)
        psi = torch.nn.functional.normalize(psi_tilde, p=2, dim=-1, eps=1e-8)
        s = torch.sigmoid(torch.matmul(psi, self.W_init.t()) + self.b_init)
        all_states = []
        for l in range(self.L):
            gamma_l = torch.sigmoid(self.gamma_raw[l])
            def f(u):
                return 4.0 * u * (1.0 - u)
            f_s = f(s)
            sum_f_s = torch.sum(f_s, dim=-1, keepdim=True)
            s_bar = (1.0 - gamma_l) * f_s + (gamma_l / self.K) * sum_f_s
            delta = 1e-5
            s_bar_clipped = torch.clamp(s_bar, delta, 1.0 - delta)
            logits = torch.log(s_bar_clipped / (1.0 - s_bar_clipped))
            perturbation = torch.matmul(psi, self.Phi[l].t()) + self.phi[l]
            s_candidate = torch.sigmoid(logits + perturbation)
            gate_l = torch.sigmoid(self.gate_raw[l])
            s = (1.0 - gate_l) * s + gate_l * s_candidate
            all_states.append(s)
        all_states = torch.stack(all_states, dim=0)
        R = self.R_raw.unsqueeze(1)
        alpha = R * all_states
        if average_batch:
            return torch.mean(alpha, dim=1)
        return alpha

def kmeans_pytorch(X, K, num_iters=100):
    # PyTorch implementation of K-Means clustering on the unit sphere (Spherical K-Means)
    B, D = X.shape
    # Normalize X to sphere
    X = torch.nn.functional.normalize(X, p=2, dim=-1)
    
    # Initialize centroids randomly from data points
    centroids = X[torch.randperm(B)[:K]].clone()
    
    labels = torch.zeros(B, dtype=torch.long, device=X.device)
    for _ in range(num_iters):
        # Cosine similarity as distance metric on sphere (maximize dot product)
        similarities = torch.matmul(X, centroids.t()) # B x K
        labels = torch.argmax(similarities, dim=1)
        
        # Update centroids
        new_centroids = []
        for k in range(K):
            mask = (labels == k)
            if mask.sum() > 0:
                new_cent = X[mask].mean(dim=0)
                new_cent = torch.nn.functional.normalize(new_cent, p=2, dim=-1)
                new_centroids.append(new_cent)
            else:
                new_centroids.append(centroids[k])
        new_centroids = torch.stack(new_centroids)
        if torch.allclose(centroids, new_centroids, atol=1e-6):
            break
        centroids = new_centroids
    return labels, centroids

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    task_names = ['MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN']
    data_dict = get_datasets()
    
    train_size = 2000
    calib_size = 64
    test_size = 128  # Reduce test size slightly to ensure fast execution
    
    calib_loaders = {}
    test_loaders = {}
    
    calib_features_dict = {}
    test_features_dict = {}
    
    base_model = timm.create_model('vit_tiny_patch16_224', pretrained=False)
    base_model.head = torch.nn.Linear(192, 10)
    base_model.load_state_dict(torch.load('checkpoints/base_model.pt', weights_only=True))
    base_model.to(device)
    
    pretrained_state = torch.load('checkpoints/base_model.pt', weights_only=True)
    
    expert_models = {}
    task_heads = []
    
    for task_idx, name in enumerate(task_names):
        train_ds, test_ds = data_dict[name]
        calib_idx = list(range(min(train_size, train_size + calib_size)))
        test_idx = list(range(min(test_size, len(test_ds))))
        
        calib_sub = Subset(train_ds, calib_idx)
        test_sub = Subset(test_ds, test_idx)
        
        calib_loaders[name] = DataLoader(calib_sub, batch_size=16, shuffle=False)
        test_loaders[name] = DataLoader(test_sub, batch_size=16, shuffle=False)
        
        expert_path = f'checkpoints/expert_{name}.pt'
        expert = timm.create_model('vit_tiny_patch16_224', pretrained=False)
        expert.head = torch.nn.Linear(192, 10)
        expert.load_state_dict(torch.load(expert_path, weights_only=True))
        expert_models[name] = expert
        
        task_heads.append({
            'weight': expert.head.weight.data.clone(),
            'bias': expert.head.bias.data.clone()
        })
        
    base_model.eval()
    with torch.no_grad():
        for name in task_names:
            feats_list = []
            for x, _ in calib_loaders[name]:
                x = x.to(device)
                z = base_model.patch_embed(x).mean(dim=1)
                feats_list.append(z.cpu())
            calib_features_dict[name] = torch.cat(feats_list, dim=0).to(device)
            
            feats_list = []
            for x, _ in test_loaders[name]:
                x = x.to(device)
                z = base_model.patch_embed(x).mean(dim=1)
                feats_list.append(z.cpu())
            test_features_dict[name] = torch.cat(feats_list, dim=0).to(device)

    task_vectors = []
    for name in task_names:
        expert_state = expert_models[name].state_dict()
        vector = {}
        for key in pretrained_state:
            if pretrained_state[key].dtype in [torch.int64, torch.uint8]:
                continue
            if get_layer_group_idx(key) is None:
                continue
            vector[key] = expert_state[key] - pretrained_state[key]
        task_vectors.append(vector)

    calib_mean_feats = torch.stack([calib_features_dict[n].mean(dim=0) for n in task_names]) # K x D

    # Train ChaosMerge
    print("Training Gated ChaosMerge (G-CML)...")
    chaos_model = GatedChaosMergeModel().to(device)
    optimizer = torch.optim.Adam(chaos_model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    for step in range(50):
        optimizer.zero_grad()
        total_loss = 0.
        alpha_spec = chaos_model.forward_coefficients(calib_mean_feats, average_batch=False)
        for idx, name in enumerate(task_names):
            x_cal, y_cal = next(iter(calib_loaders[name]))
            x_cal, y_cal = x_cal.to(device), y_cal.to(device)
            task_alpha = alpha_spec[:, idx, :]
            merged_state = merge_weights(pretrained_state, task_vectors, task_alpha, device=device)
            task_merged_state = {k: v for k, v in merged_state.items()}
            task_merged_state['head.weight'] = task_heads[idx]['weight'].to(device)
            task_merged_state['head.bias'] = task_heads[idx]['bias'].to(device)
            logits = torch.func.functional_call(base_model, task_merged_state, (x_cal,))
            total_loss += criterion(logits, y_cal)
        total_loss.backward()
        optimizer.step()

    # Create heterogeneous mixed-task test batch
    print("\nCreating Heterogeneous Test Batch...")
    all_x = []
    all_y = []
    all_task_ids = []
    
    for idx, name in enumerate(task_names):
        for x, y in test_loaders[name]:
            all_x.append(x)
            all_y.append(y)
            all_task_ids.append(torch.ones_like(y) * idx)
            
    all_x = torch.cat(all_x, dim=0).to(device)
    all_y = torch.cat(all_y, dim=0).to(device)
    all_task_ids = torch.cat(all_task_ids, dim=0).to(device)
    
    # 1. Oracle (Perfect Task ID availability) Downstream Evaluation
    print("Evaluating Oracle ChaosMerge (using perfect Task IDs)...")
    t0 = time.time()
    oracle_correct = 0
    with torch.no_grad():
        for idx, name in enumerate(task_names):
            # Extract features for this task's test set
            test_feats = test_features_dict[name]
            # Compute centroid of test set
            centroid = test_feats.mean(dim=0, keepdim=True)
            # Compute coefficients
            alpha_test = chaos_model.forward_coefficients(centroid, average_batch=False)[:, 0, :]
            # Evaluate
            acc = evaluate_merged_model_with_coeffs(base_model, task_vectors, alpha_test, task_heads, test_loaders[name], idx, device=device)
            oracle_correct += int(acc * test_size)
    oracle_acc = oracle_correct / (test_size * 4)
    oracle_time = time.time() - t0
    print(f"Oracle Accuracy: {oracle_acc*100:.2f}% | Latency: {oracle_time:.4f}s")

    # 2. Extract features and project to 4D sphere
    print("\nExtracting test features for the entire heterogeneous batch...")
    with torch.no_grad():
        feats_list = []
        for i in range(0, len(all_x), 32):
            batch_x = all_x[i:i+32]
            z = base_model.patch_embed(batch_x).mean(dim=1)
            feats_list.append(z)
        all_feats = torch.cat(feats_list, dim=0)
        
        # Project and normalize to 4D sphere
        psi_tilde = torch.matmul(all_feats, chaos_model.P)
        psi = torch.nn.functional.normalize(psi_tilde, p=2, dim=-1, eps=1e-8)

    # 3. Unsupervised Spherical K-Means Clustering
    print("Running Unsupervised Spherical K-Means (K=4 clusters) in 4D space...")
    cluster_labels, cluster_centroids = kmeans_pytorch(psi, K=4)
    
    # Determine the mapping from cluster labels to true task labels (using majority voting)
    cluster_to_task = {}
    correct_clusters = 0
    for k in range(4):
        mask = (cluster_labels == k)
        if mask.sum() > 0:
            true_tasks = all_task_ids[mask]
            majority_task = torch.mode(true_tasks).values.item()
            cluster_to_task[k] = majority_task
            # Count correct cluster assignments
            correct_clusters += (true_tasks == majority_task).sum().item()
        else:
            cluster_to_task[k] = k # fallback
            
    clustering_accuracy = correct_clusters / len(all_task_ids)
    print(f"Projected 4D Clustering Purity/Accuracy: {clustering_accuracy*100:.2f}%")

    # 4. Unsupervised Downstream Evaluation
    print("\nEvaluating Unsupervised ChaosMerge (K-Means Centroid Weights)...")
    t0 = time.time()
    unsupervised_correct = 0
    with torch.no_grad():
        for k in range(4):
            mask = (cluster_labels == k)
            if mask.sum() == 0:
                continue
            
            # Sub-batch for this cluster
            sub_x = all_x[mask]
            sub_y = all_y[mask]
            sub_task_ids = all_task_ids[mask]
            
            # Compute cluster centroid feature
            centroid = all_feats[mask].mean(dim=0, keepdim=True)
            
            # Map this cluster to its majority task's classification head
            assigned_task = cluster_to_task[k]
            
            # Compute coefficients
            alpha_test = chaos_model.forward_coefficients(centroid, average_batch=False)[:, 0, :]
            
            # Assemble weights
            merged_state_dict = merge_weights(pretrained_state, task_vectors, alpha_test, device=device)
            base_model.load_state_dict(merged_state_dict, strict=False)
            
            # Load the head of the mapped majority task (to predict)
            base_model.head.weight.data.copy_(task_heads[assigned_task]['weight'].to(device))
            base_model.head.bias.data.copy_(task_heads[assigned_task]['bias'].to(device))
            
            # Predict
            logits = base_model(sub_x)
            preds = logits.argmax(dim=-1)
            unsupervised_correct += (preds == sub_y).sum().item()
            
    unsupervised_acc = unsupervised_correct / len(all_y)
    unsupervised_time = time.time() - t0
    print(f"Unsupervised Accuracy: {unsupervised_acc*100:.2f}% | Latency: {unsupervised_time:.4f}s")
    
    print("\n--- Summary of Clustering Loophole Experiment ---")
    print(f"1. Clustering Accuracy in 4D space: {clustering_accuracy*100:.2f}%")
    print(f"2. Downstream Classification Accuracy:")
    print(f"   - Oracle (Perfect Task ID): {oracle_acc*100:.2f}%")
    print(f"   - Unsupervised (K-Means):   {unsupervised_acc*100:.2f}%")
    print(f"   - Downstream Penalty:       {(oracle_acc - unsupervised_acc)*100:.2f}% absolute drop")
    print(f"3. Latency & Throughput Trade-off:")
    print(f"   - Oracle Latency:           {oracle_time:.4f}s")
    print(f"   - Unsupervised Latency:     {unsupervised_time:.4f}s")
    print(f"   - Latency Multiplier:       {unsupervised_time / (oracle_time + 1e-8):.2f}x")

if __name__ == '__main__':
    main()
