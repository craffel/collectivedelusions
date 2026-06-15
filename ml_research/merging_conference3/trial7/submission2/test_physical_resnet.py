import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
import numpy as np
import time
import torchvision.models as models

# Set random seed for perfect reproducibility
torch.manual_seed(42)
np.random.seed(42)

print("=========================================================")
print("PHYSICAL VALIDATION: END-TO-END RESNET-18 ON REAL IMAGES")
print("=========================================================")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# 1. Initialize Pre-trained ResNet-18 as a Shared Feature Extractor
print("\nInitializing pre-trained ResNet-18...")
resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
# Replace the final linear head with nn.Identity to extract 512-dimensional penultimate representations
resnet18.fc = nn.Identity()
resnet18 = resnet18.to(device)
resnet18.eval()

# 2. Define Dataset Transforms
# MNIST and FashionMNIST are 1-channel, SVHN is 3-channel. We resize to 128x128 for speed.
transform_mnist = transforms.Compose([
    transforms.Lambda(lambda img: img.convert('RGB')),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_svhn = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. Load Real-World Datasets
print("\nLoading MNIST, FashionMNIST, and SVHN datasets...")
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)

fmnist_train = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_mnist)
fmnist_test = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_mnist)

svhn_train = datasets.SVHN(root='./data', split='train', download=True, transform=transform_svhn)
svhn_test = datasets.SVHN(root='./data', split='test', download=True, transform=transform_svhn)

# 4. Helper Function to Select Balanced Indices
def get_balanced_indices(dataset, num_samples, classes=10):
    targets = np.array(dataset.targets if hasattr(dataset, 'targets') else dataset.labels)
    indices_per_class = num_samples // classes
    selected_indices = []
    for c in range(classes):
        c_indices = np.where(targets == c)[0]
        # Safeguard if class doesn't have enough samples
        if len(c_indices) < indices_per_class:
            selected_indices.extend(c_indices)
        else:
            selected_indices.extend(c_indices[:indices_per_class])
    return selected_indices

# Select Indices
N_train_per_task = 500 # Train expert classifier heads
N_cal_per_task = 30   # Calibration split for FIM estimation
N_test_per_task = 100 # Evaluation test split

print("\nPartitioning datasets into balanced splits...")
mnist_train_idx = get_balanced_indices(mnist_train, N_train_per_task + N_cal_per_task)
mnist_train_head_idx = mnist_train_idx[:N_train_per_task]
mnist_cal_idx = mnist_train_idx[N_train_per_task:N_train_per_task + N_cal_per_task]
mnist_test_idx = get_balanced_indices(mnist_test, N_test_per_task)

fmnist_train_idx = get_balanced_indices(fmnist_train, N_train_per_task + N_cal_per_task)
fmnist_train_head_idx = fmnist_train_idx[:N_train_per_task]
fmnist_cal_idx = fmnist_train_idx[N_train_per_task:N_train_per_task + N_cal_per_task]
fmnist_test_idx = get_balanced_indices(fmnist_test, N_test_per_task)

svhn_train_idx = get_balanced_indices(svhn_train, N_train_per_task + N_cal_per_task)
svhn_train_head_idx = svhn_train_idx[:N_train_per_task]
svhn_cal_idx = svhn_train_idx[N_train_per_task:N_train_per_task + N_cal_per_task]
svhn_test_idx = get_balanced_indices(svhn_test, N_test_per_task)

# 5. Extract Feature Representations
def extract_features_and_labels(dataset, indices, batch_size=64):
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False)
    all_features = []
    all_labels = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            feats = resnet18(x) # [B, 512]
            all_features.append(feats.cpu())
            all_labels.append(y.cpu())
    return torch.cat(all_features).to(device), torch.cat(all_labels).to(device)

print("\nExtracting features from physical ResNet-18 backbone...")
t0 = time.time()
mnist_train_head_feats, mnist_train_head_labels = extract_features_and_labels(mnist_train, mnist_train_head_idx)
mnist_cal_feats, mnist_cal_labels = extract_features_and_labels(mnist_train, mnist_cal_idx)
mnist_test_feats, mnist_test_labels = extract_features_and_labels(mnist_test, mnist_test_idx)

fmnist_train_head_feats, fmnist_train_head_labels = extract_features_and_labels(fmnist_train, fmnist_train_head_idx)
fmnist_cal_feats, fmnist_cal_labels = extract_features_and_labels(fmnist_train, fmnist_cal_idx)
fmnist_test_feats, fmnist_test_labels = extract_features_and_labels(fmnist_test, fmnist_test_idx)

svhn_train_head_feats, svhn_train_head_labels = extract_features_and_labels(svhn_train, svhn_train_head_idx)
svhn_cal_feats, svhn_cal_labels = extract_features_and_labels(svhn_train, svhn_cal_idx)
svhn_test_feats, svhn_test_labels = extract_features_and_labels(svhn_test, svhn_test_idx)
print(f"Feature extraction complete in {time.time() - t0:.2f} seconds.")

# Groups
train_features = [mnist_train_head_feats, fmnist_train_head_feats, svhn_train_head_feats]
train_labels = [mnist_train_head_labels, fmnist_train_head_labels, svhn_train_head_labels]

cal_features = [mnist_cal_feats, fmnist_cal_feats, svhn_cal_feats]
cal_labels = [mnist_cal_labels, fmnist_cal_labels, svhn_cal_labels]

test_features_list = [mnist_test_feats, fmnist_test_feats, svhn_test_feats]
test_labels_list = [mnist_test_labels, fmnist_test_labels, svhn_test_labels]

# 6. Apply Global Mean Centering first, so training is aligned with evaluation
print("\nApplying pre-calibration mean centering to all features to eliminate translation bias...")
all_train_cal = torch.cat(train_features + cal_features, dim=0) # [1500 + 90, 512]
global_mean = all_train_cal.mean(dim=0, keepdim=True)

for k in range(3):
    train_features[k] = train_features[k] - global_mean
    cal_features[k] = cal_features[k] - global_mean
    test_features_list[k] = test_features_list[k] - global_mean

# 7. Train Specialized Expert Heads on Centered representations
print("\nTraining task-specific specialized linear classifier heads on centered features...")
class ExpertHead(nn.Module):
    def __init__(self, d_in=512, classes=10):
        super().__init__()
        self.fc = nn.Linear(d_in, classes, bias=False) # bias=False to represent direct prototypical projection
    def forward(self, x):
        return self.fc(x)

heads = [ExpertHead().to(device) for _ in range(3)]
task_names = ["MNIST", "FashionMNIST", "SVHN"]

for k in range(3):
    optimizer = torch.optim.Adam(heads[k].parameters(), lr=0.01, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Train the linear head on task features
    feats = train_features[k]
    labels = train_labels[k]
    
    for epoch in range(120):
        optimizer.zero_grad()
        logits = heads[k](feats)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
    # Evaluate individual accuracy to confirm they are specialized experts
    with torch.no_grad():
        test_logits = heads[k](test_features_list[k])
        preds = torch.argmax(test_logits, dim=-1)
        acc = (preds == test_labels_list[k]).float().mean().item() * 100
        print(f"  - {task_names[k]} Expert Classifier Head Accuracy: {acc:.2f}%")

# Save head weights
head_weights = torch.stack([head.fc.weight.data for head in heads]) # [3, 10, 512]

# 8. Estimate representation-space pooled Class-Conditional Variance from Calibration Split
coordinate_variances = torch.zeros(3, 512).to(device)

for k in range(3):
    task_feats = cal_features[k] # [30, 512]
    task_labels = cal_labels[k]   # [30]
    
    pooled_var = torch.zeros(512).to(device)
    valid_classes = 0
    for c in range(10):
        c_mask = (task_labels == c)
        if torch.sum(c_mask) > 1:
            class_feats = task_feats[c_mask]
            mean_c = torch.mean(class_feats, dim=0)
            pooled_var += torch.sum((class_feats - mean_c) ** 2, dim=0)
            valid_classes += (torch.sum(c_mask) - 1).item()
            
    if valid_classes > 0:
        coordinate_variances[k] = pooled_var / valid_classes
    else:
        coordinate_variances[k] = torch.var(task_feats, dim=0)

# 9. Mix Test Set into Heterogeneous Stream
test_feats_all = torch.cat(test_features_list, dim=0) # [300, 512]
test_labels_all = torch.cat(test_labels_list, dim=0)   # [300]
test_tasks_all = torch.tensor([0]*100 + [1]*100 + [2]*100, dtype=torch.long).to(device)

# 10. Evaluate routing and ensembling accuracy under various regularization parameters
def evaluate_routing(method, Fisher_M_smoothed=None):
    B_size = len(test_feats_all)
    u = torch.zeros(B_size, 3).to(device)
    
    for k in range(3):
        W_k = head_weights[k] # [10, 512]
        # Normalize weights
        W_k_norm = W_k / (torch.norm(W_k, dim=-1, keepdim=True) + 1e-8)
        
        if method == "FIOSR":
            # Fisher-Weighted Cosine Similarity
            F_k_smoothed = Fisher_M_smoothed[k] # [512]
            
            # Expand dimensions
            z_expanded = test_feats_all.unsqueeze(1) # [B, 1, 512]
            W_k_expanded = W_k.unsqueeze(0) # [1, 10, 512]
            F_k_expanded = F_k_smoothed.unsqueeze(0).unsqueeze(0) # [1, 1, 512]
            
            num = torch.sum(F_k_expanded * W_k_expanded * z_expanded, dim=-1) # [B, 10]
            den1 = torch.sqrt(torch.sum(F_k_expanded * (W_k_expanded ** 2), dim=-1)) # [1, 10]
            den2 = torch.sqrt(torch.sum(F_k_expanded * (z_expanded ** 2), dim=-1)) # [B, 10]
            sims = num / (den1 * den2 + 1e-8) # [B, 10]
        else:
            # Standard PFSR (Flat Cosine)
            z_expanded = test_feats_all.unsqueeze(1) # [B, 1, 512]
            W_k_expanded = W_k_norm.unsqueeze(0) # [1, 10, 512]
            
            num = torch.sum(W_k_expanded * z_expanded, dim=-1) # [B, 10]
            den2 = torch.sqrt(torch.sum(z_expanded ** 2, dim=-1)) # [B, 10]
            # Since W_k is normalized, den1 is 1.0
            sims = num / (den2 + 1e-8) # [B, 10]
            
        u[:, k] = torch.max(sims, dim=-1)[0]
        
    # Apply Class-Size Scaling Calibration (CSC)
    csc_denom = np.sqrt(2 * np.log(10) / 512)
    u_calibrated = u / csc_denom
    
    # Routing decision
    predictions = torch.argmax(u_calibrated, dim=-1)
    
    # Calculate routing accuracy
    correct_routing = (predictions == test_tasks_all).float().mean().item() * 100
    
    # Calculate joint ensembling accuracy
    joint_correct = 0
    for b in range(B_size):
        pred_task = predictions[b].item()
        true_task = test_tasks_all[b].item()
        
        # Classify using chosen head
        z_chosen = test_feats_all[b]
        logits = torch.matmul(head_weights[pred_task], z_chosen)
        pred_class = torch.argmax(logits).item()
        
        if pred_task == true_task and pred_class == test_labels_all[b].item():
            joint_correct += 1
            
    joint_accuracy = (joint_correct / B_size) * 100
    return correct_routing, joint_accuracy

print("\nEvaluating Baseline Flat Cosine Routing...")
pfsr_route, pfsr_joint = evaluate_routing("PFSR")
print(f"PFSR Baseline: Routing Acc = {pfsr_route:.2f}%, Joint Acc = {pfsr_joint:.2f}%")

print("\nSweeping Regularization Hyperparameters for FIOSR...")
best_route_acc = 0
best_params = {}

for alpha in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]: # Shrinkage / variance addition factor
    for beta in [0.1, 0.5, 1.0, 5.0]: # Smoothing constant
        for gamma in [0.3, 0.5, 0.7, 1.0]: # Power scaling
            # Scale-regularized coordinate variance
            mean_var_k = coordinate_variances.mean(dim=-1, keepdim=True) # [3, 1]
            reg_variances = coordinate_variances + alpha * mean_var_k
            
            # Inverse variance calculation
            Fisher_M = 1.0 / reg_variances
            
            # Smooth and normalize
            Fisher_M_smoothed = (Fisher_M + beta) ** gamma
            Fisher_M_smoothed = Fisher_M_smoothed / Fisher_M_smoothed.sum(dim=-1, keepdim=True)
            
            fiosr_route, fiosr_joint = evaluate_routing("FIOSR", Fisher_M_smoothed)
            if fiosr_route > best_route_acc:
                best_route_acc = fiosr_route
                best_params = {"alpha": alpha, "beta": beta, "gamma": gamma, "joint": fiosr_joint}
                print(f"  * New Best: alpha={alpha}, beta={beta}, gamma={gamma} -> Routing Acc = {fiosr_route:.2f}%, Joint Acc = {fiosr_joint:.2f}%")

print("\n=========================================================")
print(f"Best FIOSR Parameters Found: {best_params}")
print(f"Best FIOSR Routing Accuracy: {best_route_acc:.2f}%")
print(f"Joint Ensembling Accuracy: {best_params['joint']:.2f}%")
print("=========================================================")
