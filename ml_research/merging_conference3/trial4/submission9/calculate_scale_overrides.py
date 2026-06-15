import torch
import timm
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

tasks = ['MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN']

# Load base model
print("Loading base pre-trained model...")
base_checkpoint = torch.load('checkpoints/base_model.pt', map_location=device)
base_state_dict = {k: v.to(device) for k, v in base_checkpoint['state_dict'].items()}

# Load expert models
expert_state_dicts = []
for task in tasks:
    print(f"Loading expert model for {task}...")
    expert_checkpoint = torch.load(f'checkpoints/{task}_expert.pt', map_location=device)
    sd = {k: v.to(device) for k, v in expert_checkpoint['state_dict'].items()}
    expert_state_dicts.append(sd)

# Extract and flatten task vectors
all_keys = [k for k in base_state_dict.keys() if 'head' not in k and base_state_dict[k].is_floating_point()]

K = len(tasks)
task_vectors = []
global_stds = []

for k in range(K):
    all_vals = []
    for key in all_keys:
        all_vals.append((expert_state_dicts[k][key] - base_state_dict[key]).view(-1))
    all_vals_flat = torch.cat(all_vals)
    task_vectors.append(all_vals_flat)
    std = torch.std(all_vals_flat).item()
    global_stds.append(std)

print("Global STDs:", global_stds)

# Let's stack task vectors to get a tensor of shape [K, D]
TV = torch.stack(task_vectors) # [K, D]
stds = torch.tensor(global_stds, device=device).unsqueeze(1) # [K, 1]

# Now let's define Lambda cases:
# Case 1: lambda = 1.0
lambdas_1 = torch.tensor([1.0, 1.0, 1.0, 1.0], device=device).unsqueeze(1)
# Case 2: TLC-Tune Lambda (Dense)
lambdas_tlc = torch.tensor([1.396, 1.198, 0.745, 1.026], device=device).unsqueeze(1)

def analyze_lambdas(lambdas, name):
    # Standardized magnitudes: M = |lambda * tau / std|
    M = torch.abs(lambdas * TV / stds) # [K, D]
    
    # Dominant expert indices:
    k_star = torch.argmax(M, dim=0) # [D]
    
    # Physical unstandardized absolute scaled updates: P = |lambda * tau|
    P = torch.abs(lambdas * TV) # [K, D]
    
    # Let's find the unstandardized magnitude of the selected expert at each coordinate
    D_size = P.shape[1]
    col_indices = torch.arange(D_size, device=device)
    P_selected = P[k_star, col_indices] # [D]
    
    # The maximum unstandardized magnitude across all experts at each coordinate
    P_max, k_max = torch.max(P, dim=0) # [D]
    
    # A scale override occurs when the unstandardized magnitude of another expert is strictly larger than P_selected
    overrides = P_max > P_selected
    num_overrides = torch.sum(overrides).item()
    override_fraction = num_overrides / D_size
    
    print(f"\n--- Analysis for {name} ---")
    print(f"Total parameters (D): {D_size}")
    print(f"Number of scale overrides: {num_overrides} ({override_fraction*100:.2f}%)")
    
    # Let's see which task's coordinate routing was overridden and by whom!
    # Specifically, we can compute a matrix of (selected_task, overriding_task) counts
    if num_overrides > 0:
        overridden_k_star = k_star[overrides] # [num_overrides]
        overriding_k_max = k_max[overrides] # [num_overrides]
        
        matrix = torch.zeros((K, K), dtype=torch.long, device=device)
        for i in range(K):
            for j in range(K):
                matrix[i, j] = torch.sum((overridden_k_star == i) & (overriding_k_max == j))
                
        print("Scale override source-destination matrix:")
        print("Columns: Overriding task (MNIST, FashionMNIST, CIFAR10, SVHN)")
        print("Rows: Elected task (MNIST, FashionMNIST, CIFAR10, SVHN)")
        for i in range(K):
            row_str = " ".join(f"{matrix[i, j].item():8d}" for j in range(K))
            print(f"{tasks[i]:13s}: {row_str}")

analyze_lambdas(lambdas_1, "Untuned EPM (Lambdas=1.0)")
analyze_lambdas(lambdas_tlc, "TLC-Tuned EPM (Dense)")
