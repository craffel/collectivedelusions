import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

D = 192
K = 4
C = 10
L = 14

# Generate orthonormal basis for each task block (48 dimensions each)
block_dim = D // K  # 48
prototypes = []
for k in range(K):
    # Random orthogonal matrix of size 48x48
    q, r = torch.linalg.qr(torch.randn(block_dim, block_dim))
    # Take first 10 columns as class prototypes
    prototypes.append(q[:, :C])  # Shape: [48, 10]

def generate_data(num_samples, noise_internal, noise_external):
    features = []
    labels = []
    tasks = []
    for k in range(K):
        task_proto = prototypes[k]  # [48, 10]
        for _ in range(num_samples):
            c = np.random.randint(C)
            proto = task_proto[:, c]  # [48]
            
            # Construct 192-dim feature
            feat = torch.zeros(D)
            for j in range(K):
                if j == k:
                    feat[j*block_dim:(j+1)*block_dim] = proto + torch.randn(block_dim) * noise_internal[k]
                else:
                    feat[j*block_dim:(j+1)*block_dim] = torch.randn(block_dim) * noise_external[k]
            
            features.append(feat)
            labels.append(c)
            tasks.append(k)
            
    return torch.stack(features), torch.tensor(labels), torch.tensor(tasks)

# Calibrated noise parameters
noise_internal = [0.001, 0.16, 0.32, 0.90]
noise_external = [0.28, 0.42, 0.49, 0.58]

print("Generating train, calibration, and test sets...")
train_feats, train_labels, train_tasks = generate_data(1000, noise_internal, noise_external)
calib_feats, calib_labels, calib_tasks = generate_data(16, noise_internal, noise_external)
test_feats, test_labels, test_tasks = generate_data(250, noise_internal, noise_external)

# Train specialized experts
experts_local = []
experts_global_W = []
experts_global_B = []

for k in range(K):
    print(f"Training expert for task {k}...")
    mask = (train_tasks == k)
    x_train = train_feats[mask][:, k*block_dim:(k+1)*block_dim]
    y_train = train_labels[mask]
    
    model = nn.Linear(block_dim, C)
    optimizer = optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(50):
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
    experts_local.append(model)
    
    W_glob = torch.zeros(C, D)
    W_glob[:, k*block_dim:(k+1)*block_dim] = model.weight.data
    B_glob = model.bias.data
    
    experts_global_W.append(W_glob)
    experts_global_B.append(B_glob)

# Evaluate Standalone Experts
standalone_accs = []
for k in range(K):
    mask = (test_tasks == k)
    x_test = test_feats[mask]
    y_test = test_labels[mask]
    
    with torch.no_grad():
        outputs = x_test @ experts_global_W[k].t() + experts_global_B[k]
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == y_test).float().mean().item()
        standalone_accs.append(acc)

print("\nStandalone Expert Accuracies:")
for k, acc in enumerate(standalone_accs):
    print(f"Task {k}: {acc*100:.2f}%")

# Evaluate Uniform Merging
uniform_accs = []
for k in range(K):
    mask = (test_tasks == k)
    x_test = test_feats[mask]
    y_test = test_labels[mask]
    
    merged_weight = torch.zeros(C, D)
    merged_bias = torch.zeros(C)
    for j in range(K):
        merged_weight += 0.25 * experts_global_W[j]
        merged_bias += 0.25 * experts_global_B[j]
        
    with torch.no_grad():
        outputs = x_test @ merged_weight.t() + merged_bias
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == y_test).float().mean().item()
        uniform_accs.append(acc)

print("\nUniform Merging Accuracies:")
for k, acc in enumerate(uniform_accs):
    print(f"Task {k}: {acc*100:.2f}%")
print(f"Mean: {np.mean(uniform_accs)*100:.2f}%")
