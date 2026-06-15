import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
import numpy as np

# Set random seeds for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

class SimpleMLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def get_datasets():
    try:
        import torchvision
        import torchvision.transforms as transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        return train_dataset, test_dataset
    except Exception as e:
        print(f"Failed to load MNIST: {e}. Generating synthetic dataset.")
        np.random.seed(42)
        X_train = np.random.randn(6000, 784).astype(np.float32)
        y_train = np.random.randint(0, 10, size=(6000,)).astype(np.int64)
        X_test = np.random.randn(1000, 784).astype(np.float32)
        y_test = np.random.randint(0, 10, size=(1000,)).astype(np.int64)
        for i in range(10):
            mask_tr = (y_train == i)
            X_train[mask_tr, i*20:(i+1)*20] += 5.0
            mask_te = (y_test == i)
            X_test[mask_te, i*20:(i+1)*20] += 5.0
        return TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)), TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

def filter_split_dataset(dataset, classes):
    indices = []
    if hasattr(dataset, 'targets'):
        targets = dataset.targets
    elif hasattr(dataset, 'tensors'):
        targets = dataset.tensors[1]
    else:
        targets = [dataset[i][1] for i in range(len(dataset))]
    for idx, target in enumerate(targets):
        if int(target) in classes:
            indices.append(idx)
    return Subset(dataset, indices)

def eval_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    return correct / total if total > 0 else 0.0

def project_weights_to_ortho(model):
    """
    Project fc1 and fc2 weights onto the orthogonal group O(d) using SVD.
    W_proj = U @ V^T (padded/sliced for non-square)
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if len(param.shape) == 2 and "weight" in name and ("fc1" in name or "fc2" in name):
                out_d, in_d = param.shape
                U, S, Vh = torch.linalg.svd(param.data, full_matrices=False)
                # SVD projection: Set singular values to 1
                param.data.copy_(torch.matmul(U, Vh))
                print(f"Projected {name} (shape {param.shape}) to orthogonal manifold.")

def train_model_ortho(model, train_loader, epochs=3, lr=0.01, ortho_lambda=2.0):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Orthogonal Regularization
            reg_loss = 0.0
            for name, param in model.named_parameters():
                if len(param.shape) == 2 and "weight" in name and ("fc1" in name or "fc2" in name):
                    out_d, in_d = param.shape
                    if out_d >= in_d:
                        diff = torch.matmul(param.t(), param) - torch.eye(in_d)
                    else:
                        diff = torch.matmul(param, param.t()) - torch.eye(out_d)
                    reg_loss += torch.norm(diff, p='fro')**2
            
            total_loss = loss + ortho_lambda * reg_loss
            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
    return model

def get_rotation_procrustes(W_k, W_0):
    A = torch.matmul(W_0.t(), W_k)
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    R = torch.matmul(U, Vh)
    return R

if __name__ == '__main__':
    set_seed(42)
    print("=== Empirical Validation of Post-Hoc Base Model Projection ===")
    
    train_dataset, test_dataset = get_datasets()
    
    # Task dataloaders
    train_loader_t1 = DataLoader(filter_split_dataset(train_dataset, range(5)), batch_size=64, shuffle=True)
    train_loader_t2 = DataLoader(filter_split_dataset(train_dataset, range(5, 10)), batch_size=64, shuffle=True)
    
    test_loader_t1 = DataLoader(filter_split_dataset(test_dataset, range(5)), batch_size=128, shuffle=False)
    test_loader_t2 = DataLoader(filter_split_dataset(test_dataset, range(5, 10)), batch_size=128, shuffle=False)
    test_loader_all = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # 1. Load the original unconstrained base model W_0
    base_model = SimpleMLP()
    base_model.load_state_dict(torch.load('results/base_model.pt', map_location='cpu'))
    
    acc_t1_unconstrained = eval_model(base_model, test_loader_t1)
    acc_t2_unconstrained = eval_model(base_model, test_loader_t2)
    acc_all_unconstrained = eval_model(base_model, test_loader_all)
    
    print(f"\n[Unconstrained Base Model W_0]")
    print(f"Task 1 (0-4) Acc: {acc_t1_unconstrained:.4f}")
    print(f"Task 2 (5-9) Acc: {acc_t2_unconstrained:.4f}")
    print(f"Overall Acc: {acc_all_unconstrained:.4f}")
    
    # 2. Project W_0 to get W_{0, proj}
    base_model_proj = SimpleMLP()
    base_model_proj.load_state_dict(torch.load('results/base_model.pt', map_location='cpu'))
    project_weights_to_ortho(base_model_proj)
    
    acc_t1_proj = eval_model(base_model_proj, test_loader_t1)
    acc_t2_proj = eval_model(base_model_proj, test_loader_t2)
    acc_all_proj = eval_model(base_model_proj, test_loader_all)
    
    print(f"\n[Projected Base Model W_0_proj]")
    print(f"Task 1 (0-4) Acc: {acc_t1_proj:.4f} (diff: {acc_t1_proj - acc_t1_unconstrained:+.4f})")
    print(f"Task 2 (5-9) Acc: {acc_t2_proj:.4f} (diff: {acc_t2_proj - acc_t2_unconstrained:+.4f})")
    print(f"Overall Acc: {acc_all_proj:.4f} (diff: {acc_all_proj - acc_all_unconstrained:+.4f})")
    
    # 3. Fine-tune Experts with Orthogonal Regularization starting from W_0_proj
    expert1 = SimpleMLP()
    expert1.load_state_dict(base_model_proj.state_dict())
    print("\nFine-tuning Expert 1 from projected base model...")
    expert1 = train_model_ortho(expert1, train_loader_t1, epochs=3, lr=0.01, ortho_lambda=2.0)
    
    expert2 = SimpleMLP()
    expert2.load_state_dict(base_model_proj.state_dict())
    print("Fine-tuning Expert 2 from projected base model...")
    expert2 = train_model_ortho(expert2, train_loader_t2, epochs=3, lr=0.01, ortho_lambda=2.0)
    
    # 4. Perform Riemannian Merging (OrthoMerge / RIMO t=1.0)
    merged_model = SimpleMLP()
    merged_model.load_state_dict(base_model_proj.state_dict())
    
    # We will merge fc1 and fc2
    with torch.no_grad():
        for layer_name in ["fc1", "fc2"]:
            W_0 = dict(base_model_proj.named_parameters())[f"{layer_name}.weight"]
            W_1 = dict(expert1.named_parameters())[f"{layer_name}.weight"]
            W_2 = dict(expert2.named_parameters())[f"{layer_name}.weight"]
            
            # SVD on each
            out_d, in_d = W_0.shape
            
            # 4.1 Procrustes rotation
            R1 = get_rotation_procrustes(W_1, W_0)
            R2 = get_rotation_procrustes(W_2, W_0)
            
            # Residuals
            rho1 = W_1 - torch.matmul(W_0, R1)
            rho2 = W_2 - torch.matmul(W_0, R2)
            
            residual_norm = 0.5 * (torch.norm(rho1, p='fro') + torch.norm(rho2, p='fro')).item()
            print(f"[{layer_name}] Average Procrustes residual norm: {residual_norm:.4f}")
            
            # 4.2 Matrix logarithm / inverse Cayley to map to Lie algebra
            # For orthogonal matrices, rotation matrices are R1 and R2.
            # To map to Lie algebra tangent space of O(d), we use inverse Cayley:
            # Q = (R - I)(R + I)^-1
            I_rot = torch.eye(in_d)
            Q1 = torch.matmul(R1 - I_rot, torch.linalg.inv(R1 + I_rot))
            Q2 = torch.matmul(R2 - I_rot, torch.linalg.inv(R2 + I_rot))
            
            # Average generators in tangent space (t=1.0)
            Q_merged = 0.5 * (Q1 + Q2)
            
            # Map back via forward Cayley
            R_merged = torch.matmul(I_rot + Q_merged, torch.linalg.inv(I_rot - Q_merged))
            
            # Average residuals linearly
            rho_merged = 0.5 * (rho1 + rho2)
            
            # Reconstruct merged weights
            W_merged = torch.matmul(W_0, R_merged) + rho_merged
            dict(merged_model.named_parameters())[f"{layer_name}.weight"].copy_(W_merged)
            
        # For fc3 and biases, use standard task arithmetic
        for name, param in merged_model.named_parameters():
            if "weight" in name and "fc3" in name:
                p0 = dict(base_model_proj.named_parameters())[name]
                p1 = dict(expert1.named_parameters())[name]
                p2 = dict(expert2.named_parameters())[name]
                param.copy_(p0 + 0.5 * (p1 - p0) + 0.5 * (p2 - p0))
            elif "bias" in name:
                p0 = dict(base_model_proj.named_parameters())[name]
                p1 = dict(expert1.named_parameters())[name]
                p2 = dict(expert2.named_parameters())[name]
                param.copy_(p0 + 0.5 * (p1 - p0) + 0.5 * (p2 - p0))
                
    acc_t1_merged = eval_model(merged_model, test_loader_t1)
    acc_t2_merged = eval_model(merged_model, test_loader_t2)
    acc_all_merged = eval_model(merged_model, test_loader_all)
    
    print(f"\n[Riemannian Merged Model from Projected Base]")
    print(f"Task 1 (0-4) Acc: {acc_t1_merged:.4f}")
    print(f"Task 2 (5-9) Acc: {acc_t2_merged:.4f}")
    print(f"Overall Acc: {acc_all_merged:.4f}")
