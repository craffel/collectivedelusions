import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as dsets
import torchvision.transforms as transforms

# Set random seeds for reproducibility
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seeds(42)

# --- REAL-WORLD DATA SETUP ---
print("Loading datasets (MNIST and FashionMNIST)...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

mnist_train = dsets.MNIST(root='./data', train=True, download=True, transform=transform)
fashion_train = dsets.FashionMNIST(root='./data', train=True, download=True, transform=transform)

mnist_test = dsets.MNIST(root='./data', train=False, download=True, transform=transform)
fashion_test = dsets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# Extract subsets for fast CPU training
def get_subset(dataset, num_samples):
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    subset_indices = indices[:num_samples]
    loader = torch.utils.data.DataLoader(dataset, batch_size=num_samples, sampler=torch.utils.data.SubsetRandomSampler(subset_indices))
    images, labels = next(iter(loader))
    return images, labels

mnist_train_x, mnist_train_y = get_subset(mnist_train, 1000)
fashion_train_x, fashion_train_y = get_subset(fashion_train, 1000)

mnist_test_x, mnist_test_y = get_subset(mnist_test, 500)
fashion_test_x, fashion_test_y = get_subset(fashion_test, 500)

# Reshape images to flat vectors (784 dimensions)
mnist_train_flat = mnist_train_x.view(mnist_train_x.size(0), -1)
fashion_train_flat = fashion_train_x.view(fashion_train_x.size(0), -1)
mnist_test_flat = mnist_test_x.view(mnist_test_x.size(0), -1)
fashion_test_flat = fashion_test_x.view(fashion_test_x.size(0), -1)

# Combined test set
test_x = torch.cat([mnist_test_flat, fashion_test_flat], dim=0)
test_y = torch.cat([mnist_test_y, fashion_test_y], dim=0)
test_task = torch.cat([torch.zeros(500, dtype=torch.long), torch.ones(500, dtype=torch.long)], dim=0)

# --- 4-LAYER MLP ARCHITECTURE ---
class DeepMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 10)
        
    def forward(self, x, return_intermediates=False):
        # Layer 1
        h1 = self.fc1(x)
        x1 = F.relu(h1)
        # Layer 2
        h2 = self.fc2(x1)
        x2 = F.relu(h2)
        # Layer 3
        h3 = self.fc3(x2)
        x3 = F.relu(h3)
        # Layer 4
        logits = self.fc4(x3)
        
        if return_intermediates:
            return logits, [x, x1, x2, x3, logits]
        return logits, x3

# Initialize base model
base_model = DeepMLP()

# Train base model jointly first on a mix of MNIST and FashionMNIST (representing pre-training / multitask base)
print("\nJointly Pre-training Base Model on Multitask Mixture...")
joint_train_x = torch.cat([mnist_train_flat[:500], fashion_train_flat[:500]], dim=0)
joint_train_y = torch.cat([mnist_train_y[:500], fashion_train_y[:500]], dim=0)
optimizer_base = optim.AdamW(base_model.parameters(), lr=1.5e-3, weight_decay=1e-3)
base_model.train()
for epoch in range(6):
    optimizer_base.zero_grad()
    logits, _ = base_model(joint_train_x)
    loss = F.cross_entropy(logits, joint_train_y)
    loss.backward()
    optimizer_base.step()
base_model.eval()

# Train specialized experts starting from pre-trained base model
K = 2  # MNIST, FashionMNIST
expert_models = []

print("\nTraining MNIST Deep MLP Expert...")
mnist_expert = DeepMLP()
mnist_expert.load_state_dict(base_model.state_dict())
optimizer0 = optim.AdamW(mnist_expert.parameters(), lr=1.5e-3, weight_decay=1e-3)
mnist_expert.train()
for epoch in range(12):
    optimizer0.zero_grad()
    logits, _ = mnist_expert(mnist_train_flat)
    loss = F.cross_entropy(logits, mnist_train_y)
    loss.backward()
    optimizer0.step()
mnist_expert.eval()
expert_models.append(mnist_expert)

print("Training FashionMNIST Deep MLP Expert...")
fashion_expert = DeepMLP()
fashion_expert.load_state_dict(base_model.state_dict())
optimizer1 = optim.AdamW(fashion_expert.parameters(), lr=1.5e-3, weight_decay=1e-3)
fashion_expert.train()
for epoch in range(12):
    optimizer1.zero_grad()
    logits, _ = fashion_expert(fashion_train_flat)
    loss = F.cross_entropy(logits, fashion_train_y)
    loss.backward()
    optimizer1.step()
fashion_expert.eval()
expert_models.append(fashion_expert)

# Standalone expert accuracy
with torch.no_grad():
    mnist_logits, _ = mnist_expert(mnist_test_flat)
    mnist_acc = (mnist_logits.argmax(dim=-1) == mnist_test_y).float().mean().item() * 100.0
    
    fashion_logits, _ = fashion_expert(fashion_test_flat)
    fashion_acc = (fashion_logits.argmax(dim=-1) == fashion_test_y).float().mean().item() * 100.0

expert_ceilings = [mnist_acc, fashion_acc]
print(f"\nStandalone Expert Accuracies:")
print(f"  MNIST Expert on MNIST: {mnist_acc:.2f}%")
print(f"  FashionMNIST Expert on FashionMNIST: {fashion_acc:.2f}%")
print(f"  Expert Ceiling Joint Mean: {np.mean(expert_ceilings):.2f}%")

# Helper accuracy metrics
def get_acc(logits, targets):
    return (logits.argmax(dim=-1) == targets).float().mean().item() * 100.0

# --- TASK CENTROIDS SETUP ---
# Compute centroids from Layer 1 activations of support samples passed through base model
with torch.no_grad():
    _, intermediates_mnist = base_model(mnist_train_flat[:16], return_intermediates=True)
    _, intermediates_fashion = base_model(fashion_train_flat[:16], return_intermediates=True)
    # Extract the representation from the penultimate layer (layer 3)
    z_mnist = intermediates_mnist[3]  # Shape (16, 32)
    z_fashion = intermediates_fashion[3]  # Shape (16, 32)

centroids = torch.stack([z_mnist.mean(dim=0), z_fashion.mean(dim=0)], dim=0)  # Shape (K, 32)
centroids_norm = centroids / (centroids.norm(dim=-1, keepdim=True) + 1e-12)

def compute_routing_coefficients(z_features, tau=0.05):
    z_norm = z_features / (z_features.norm(dim=-1, keepdim=True) + 1e-12)
    sims = torch.matmul(z_norm, centroids_norm.t())  # Shape (B, K)
    coeffs = torch.softmax(sims / tau, dim=-1)
    return coeffs

# Compute input-space centroids (Layer 0, Shape 784)
mnist_inputs = mnist_train_flat[:16]
fashion_inputs = fashion_train_flat[:16]
input_centroids = torch.stack([mnist_inputs.mean(dim=0), fashion_inputs.mean(dim=0)], dim=0)  # Shape (K, 784)
input_centroids_norm = input_centroids / (input_centroids.norm(dim=-1, keepdim=True) + 1e-12)

def compute_input_routing_coefficients(X_inputs, tau=0.05):
    X_norm = X_inputs / (X_inputs.norm(dim=-1, keepdim=True) + 1e-12)
    sims = torch.matmul(X_norm, input_centroids_norm.t())  # Shape (B, K)
    coeffs = torch.softmax(sims / tau, dim=-1)
    return coeffs

# --- DECOMPOSE ALL 4 LAYERS INTO LORA ADAPTERS ---
L_layers = 4
layers_keys = ['fc1', 'fc2', 'fc3', 'fc4']
r_rank = 8

A_adapters = []  # shape: (K, L_layers)
B_adapters = []
delta_biases = []

W_bases = [base_model.fc1.weight.data, base_model.fc2.weight.data, base_model.fc3.weight.data, base_model.fc4.weight.data]
b_bases = [base_model.fc1.bias.data, base_model.fc2.bias.data, base_model.fc3.bias.data, base_model.fc4.bias.data]

for k in range(K):
    expert = expert_models[k]
    W_experts = [expert.fc1.weight.data, expert.fc2.weight.data, expert.fc3.weight.data, expert.fc4.weight.data]
    b_experts = [expert.fc1.bias.data, expert.fc2.bias.data, expert.fc3.bias.data, expert.fc4.bias.data]
    
    expert_A = []
    expert_B = []
    expert_bias = []
    
    for l in range(L_layers):
        W_b = W_bases[l]
        b_b = b_bases[l]
        W_e = W_experts[l]
        b_e = b_experts[l]
        
        V_kl = W_e - W_b
        delta_b = b_e - b_b
        
        # SVD decomposition
        U, S, Vh = torch.linalg.svd(V_kl, full_matrices=False)
        rank_to_use = min(r_rank, V_kl.shape[0], V_kl.shape[1])
        Ur = U[:, :rank_to_use]
        Sr = torch.diag(torch.sqrt(S[:rank_to_use]))
        Vhr = Vh[:rank_to_use, :]
        
        A_kl = torch.matmul(Ur, Sr)  # Shape (out_dim, r)
        B_kl = torch.matmul(Sr, Vhr)  # Shape (r, in_dim)
        
        expert_A.append(A_kl)
        expert_B.append(B_kl)
        expert_bias.append(delta_b)
        
    A_adapters.append(expert_A)
    B_adapters.append(expert_B)
    delta_biases.append(expert_bias)

# --- SABLE MULTI-LAYER FORWARD EVALUATOR ---
def evaluate_sable_multilayer(X, M=2, L_route=0, true_single_pass=False):
    with torch.no_grad():
        if true_single_pass:
            if L_route == 3:
                # True single-pass Late Adaptation starting at Layer 4 (index 3).
                # Run unadapted base layers 1, 2, and 3 sequentially in a single pass.
                feat = X
                # Layer 1
                feat = F.relu(F.linear(feat, W_bases[0], b_bases[0]))
                # Layer 2
                feat = F.relu(F.linear(feat, W_bases[1], b_bases[1]))
                # Layer 3
                feat = F.relu(F.linear(feat, W_bases[2], b_bases[2]))
                
                # At this point, feat is the intermediate penultimate activation z (shape 32)!
                # We compute routing coefficients on-the-fly from this z:
                coeffs = compute_routing_coefficients(feat, tau=0.05)
                if M == 1:
                    top_vals, top_idx = torch.topk(coeffs, 1, dim=-1)
                    mask = torch.zeros_like(coeffs)
                    mask.scatter_(dim=-1, index=top_idx, src=torch.ones_like(top_vals))
                    coeffs = mask
                    
                # Layer 4 (adapted with SABLE)
                H_base = F.linear(feat, W_bases[3], b_bases[3])
                H_experts = torch.zeros(K, X.shape[0], W_bases[3].shape[0])
                for k in range(K):
                    A_kl = A_adapters[k][3]
                    B_kl = B_adapters[k][3]
                    delta_b_kl = delta_biases[k][3]
                    
                    proj = torch.matmul(feat, B_kl.t())
                    out = torch.matmul(proj, A_kl.t()) + delta_b_kl
                    H_experts[k] = out
                    
                H_blended = torch.sum(coeffs.t().unsqueeze(-1) * H_experts, dim=0)
                h_final = H_base + H_blended
                return h_final, [X, None, None, feat, h_final]
                
            elif L_route == 0:
                # True single-pass early routing starting at Layer 1.
                # Compute coefficients immediately using input-space centroids.
                coeffs = compute_input_routing_coefficients(X, tau=0.05)
                if M == 1:
                    top_vals, top_idx = torch.topk(coeffs, 1, dim=-1)
                    mask = torch.zeros_like(coeffs)
                    mask.scatter_(dim=-1, index=top_idx, src=torch.ones_like(top_vals))
                    coeffs = mask
                    
                feat = X
                sable_activations = [feat]
                for l in range(L_layers):
                    W_b = W_bases[l]
                    b_b = b_bases[l]
                    H_base = F.linear(feat, W_b, b_b)
                    
                    H_experts = torch.zeros(K, X.shape[0], W_b.shape[0])
                    for k in range(K):
                        A_kl = A_adapters[k][l]
                        B_kl = B_adapters[k][l]
                        delta_b_kl = delta_biases[k][l]
                        
                        proj = torch.matmul(feat, B_kl.t())
                        out = torch.matmul(proj, A_kl.t()) + delta_b_kl
                        H_experts[k] = out
                        
                    H_blended = torch.sum(coeffs.t().unsqueeze(-1) * H_experts, dim=0)
                    h_final = H_base + H_blended
                    
                    if l < L_layers - 1:
                        feat = F.relu(h_final)
                        sable_activations.append(feat)
                    else:
                        feat = h_final
                        sable_activations.append(feat)
                return feat, sable_activations
            else:
                raise NotImplementedError(f"True single-pass is only implemented for L_route=0 (early) and L_route=3 (late) in this physical script.")
                
        # Original 2-pass code (for comparing/validating)
        feat = X
        sable_activations = [feat]
        
        # We need the routing coefficients. SABLE computes them at the penultimate layer of the base network,
        # or at Layer 0. Let's compute them by passing a forward pass through the base network first to get routing features z.
        _, z = base_model(X)
        coeffs = compute_routing_coefficients(z)
        
        if M == 1:
            # Hard routing
            top_vals, top_idx = torch.topk(coeffs, 1, dim=-1)
            mask = torch.zeros_like(coeffs)
            mask.scatter_(dim=-1, index=top_idx, src=torch.ones_like(top_vals))
            coeffs = mask
            
        # Forward pass with SABLE ensembling
        feat = X
        sable_activations = [feat]
        
        for l in range(L_layers):
            W_b = W_bases[l]
            b_b = b_bases[l]
            
            H_base = F.linear(feat, W_b, b_b)
            
            # If layer is adapted
            if l >= L_route:
                H_experts = torch.zeros(K, X.shape[0], W_b.shape[0])
                for k in range(K):
                    A_kl = A_adapters[k][l]
                    B_kl = B_adapters[k][l]
                    delta_b_kl = delta_biases[k][l]
                    
                    proj = torch.matmul(feat, B_kl.t())
                    out = torch.matmul(proj, A_kl.t()) + delta_b_kl
                    H_experts[k] = out
                    
                H_blended = torch.sum(coeffs.t().unsqueeze(-1) * H_experts, dim=0)
                h_final = H_base + H_blended
            else:
                h_final = H_base
                
            if l < L_layers - 1:
                feat = F.relu(h_final)
                sable_activations.append(feat)
            else:
                feat = h_final
                sable_activations.append(feat)
                
        return feat, sable_activations

# --- UNIFORM MERGING ---
W_uniform = []
b_uniform = []
for l in range(L_layers):
    W_l = 0.5 * expert_models[0].fc1.weight.data + 0.5 * expert_models[1].fc1.weight.data if l==0 else \
          0.5 * expert_models[0].fc2.weight.data + 0.5 * expert_models[1].fc2.weight.data if l==1 else \
          0.5 * expert_models[0].fc3.weight.data + 0.5 * expert_models[1].fc3.weight.data if l==2 else \
          0.5 * expert_models[0].fc4.weight.data + 0.5 * expert_models[1].fc4.weight.data
    b_l = 0.5 * expert_models[0].fc1.bias.data + 0.5 * expert_models[1].fc1.bias.data if l==0 else \
          0.5 * expert_models[0].fc2.bias.data + 0.5 * expert_models[1].fc2.bias.data if l==1 else \
          0.5 * expert_models[0].fc3.bias.data + 0.5 * expert_models[1].fc3.bias.data if l==2 else \
          0.5 * expert_models[0].fc4.bias.data + 0.5 * expert_models[1].fc4.bias.data
    W_uniform.append(W_l)
    b_uniform.append(b_l)

def evaluate_uniform_multilayer(X):
    feat = X
    for l in range(L_layers):
        feat = F.linear(feat, W_uniform[l], b_uniform[l])
        if l < L_layers - 1:
            feat = F.relu(feat)
    return feat

# --- MEASURE EMPIRICAL REPRESENTATIONAL DRIFT (CKA/Cosine distance) ---
# For SABLE ensembling, we track the representational cosine similarity of SABLE's activations
# compared to the activations of the true specialized expert at each hidden layer l!
with torch.no_grad():
    # Pass MNIST test samples through MNIST Expert
    mnist_logits_e, mnist_acts_e = mnist_expert(mnist_test_flat, return_intermediates=True)
    # Pass Fashion test samples through Fashion Expert
    fashion_logits_e, fashion_acts_e = fashion_expert(fashion_test_flat, return_intermediates=True)
    
    # Run multi-layer SABLE on the test set
    _, test_sable_acts = evaluate_sable_multilayer(test_x, M=2, L_route=0)
    
    # Extract the portion corresponding to MNIST and FashionMNIST
    sable_acts_mnist = [act[:500] for act in test_sable_acts]
    sable_acts_fashion = [act[500:] for act in test_sable_acts]
    
    # Calculate representational cosine similarity layer-by-layer
    # Cosine Similarity = \frac{A \cdot B}{\|A\| \|B\|} averaged over the batch
    def get_layer_cosine_sim(acts_a, acts_b):
        sims = []
        for l in range(len(acts_a)):
            a = acts_a[l]
            b = acts_b[l]
            a_norm = a / (a.norm(dim=-1, keepdim=True) + 1e-12)
            b_norm = b / (b.norm(dim=-1, keepdim=True) + 1e-12)
            cos = torch.sum(a_norm * b_norm, dim=-1).mean().item()
            sims.append(cos)
        return sims

    drift_mnist = get_layer_cosine_sim(sable_acts_mnist, mnist_acts_e)
    drift_fashion = get_layer_cosine_sim(sable_acts_fashion, fashion_acts_e)

print("\n" + "="*50)
print("QUANTITATIVE REPRESENTATIONAL DRIFT ANALYSIS (MULTILAYER SABLE)")
print("="*50)
print(f"{'Layer':<10} | {'MNIST Cosine Sim':<20} | {'FashionMNIST Cosine Sim':<25}")
print("-" * 65)
for l in range(len(drift_mnist)):
    layer_name = f"Input" if l == 0 else f"Hidden {l}" if l < len(drift_mnist)-1 else "Logits"
    print(f"{layer_name:<10} | {drift_mnist[l]:>18.4f} | {drift_fashion[l]:>23.4f}")
print("="*50)

# Evaluate joint accuracies under streaming heterogeneity
sable_soft_logits, _ = evaluate_sable_multilayer(test_x, M=2, L_route=0)
sable_hard_logits, _ = evaluate_sable_multilayer(test_x, M=1, L_route=0)
uniform_logits = evaluate_uniform_multilayer(test_x)

sable_soft_acc = get_acc(sable_soft_logits, test_y)
sable_hard_acc = get_acc(sable_hard_logits, test_y)
uniform_acc = get_acc(uniform_logits, test_y)

# True Single-Pass Evaluations
sable_sp_early_logits, _ = evaluate_sable_multilayer(test_x, M=2, L_route=0, true_single_pass=True)
sable_sp_early_acc = get_acc(sable_sp_early_logits, test_y)

sable_late_2pass_logits, _ = evaluate_sable_multilayer(test_x, M=2, L_route=3)
sable_late_2pass_acc = get_acc(sable_late_2pass_logits, test_y)

sable_late_sp_logits, _ = evaluate_sable_multilayer(test_x, M=2, L_route=3, true_single_pass=True)
sable_late_sp_acc = get_acc(sable_late_sp_logits, test_y)

print("\n" + "="*65)
print("REAL-WORLD PHYSICAL MULTI-LAYER SABLE ACCURACY RESULTS")
print("="*65)
print(f"{'Method':<52} | {'Accuracy':<10}")
print("-" * 65)
print(f"{'Expert Ceiling Mean':<52} | {np.mean(expert_ceilings):>9.2f}%")
print(f"{'Uniform Merging':<52} | {uniform_acc:>9.2f}%")
print(f"{'SABLE Hard Multi-Layer (M=1, L_route=0)':<52} | {sable_hard_acc:>9.2f}%")
print(f"{'SABLE Soft Multi-Layer (M=2, L_route=0) [2-Pass]':<52} | {sable_soft_acc:>9.2f}%")
print(f"{'SABLE Soft Early-Route (M=2, L_route=0) [Single-Pass]':<52} | {sable_sp_early_acc:>9.2f}%")
print(f"{'SABLE Soft Late-Adapt (M=2, L_route=3) [2-Pass]':<52} | {sable_late_2pass_acc:>9.2f}%")
print(f"{'SABLE Soft Late-Adapt (M=2, L_route=3) [Single-Pass]':<52} | {sable_late_sp_acc:>9.2f}%")
print("="*65)
