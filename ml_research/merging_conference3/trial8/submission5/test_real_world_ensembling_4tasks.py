import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import timm
import time
import numpy as np
import math

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

print("="*80)
print("REAL-WORLD 4-TASK LORA ADAPTER ENSEMBLING WITH PEAR (ALL 12 BLOCKS)")
print("="*80)

# 1. Load Pre-trained ViT-Tiny and Freeze Backbone
device = torch.device("cpu")
vit = timm.create_model('vit_tiny_patch16_224', pretrained=True)
for param in vit.parameters():
    param.requires_grad = False

# Helper function to extract features for routing
def extract_vit_features(model, x, layer=2):
    with torch.no_grad():
        # Layer 0: Patch Embedding
        x_embed = model.patch_embed(x)  # (B, N, D)
        if layer == 0:
            return x_embed.mean(dim=1)
        
        # Positional Embedding and prefix tokens
        x_pos = model._pos_embed(x_embed)
        x_drop = model.patch_drop(x_pos)
        x_norm = model.norm_pre(x_drop)
        
        # Layer 1: Block 0
        x1 = model.blocks[0](x_norm)
        if layer == 1:
            return x1[:, 1:, :].mean(dim=1)
            
        # Layer 2: Block 1
        x2 = model.blocks[1](x1)
        if layer == 2:
            return x2[:, 1:, :].mean(dim=1)
    return None

# 2. Setup Data Loading for all 4 Tasks
transform_gray = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

transform_rgb = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

datasets_dict = {
    "MNIST": datasets.MNIST(root='./data', train=False, download=True, transform=transform_gray),
    "Fashion-MNIST": datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_gray),
    "CIFAR-10": datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_rgb),
    "SVHN": datasets.SVHN(root='./data', split='test', download=True, transform=transform_rgb)
}

# Sample 64 calibration and 64 test samples per task
N_cal = 64
N_test = 64
task_names = ["MNIST", "Fashion-MNIST", "CIFAR-10", "SVHN"]
data_splits = {}

for name in task_names:
    ds = datasets_dict[name]
    indices = np.random.choice(len(ds), N_cal + N_test, replace=False)
    cal_idx = indices[:N_cal]
    test_idx = indices[N_cal:]
    
    data_splits[name] = {
        "cal": Subset(ds, cal_idx),
        "test": Subset(ds, test_idx)
    }

# 3. Formulate manual LoRA patch for ALL 12 Blocks' Attn QKV
class LoRAWrappedQKVAllTasks(nn.Module):
    def __init__(self, original_qkv, r=8):
        super().__init__()
        self.original_qkv = original_qkv
        self.r = r
        
        # Adapters for all 4 tasks
        self.lora_A = nn.ParameterDict({
            'MNIST': nn.Parameter(torch.zeros(r, 192)),
            'Fashion-MNIST': nn.Parameter(torch.zeros(r, 192)),
            'CIFAR-10': nn.Parameter(torch.zeros(r, 192)),
            'SVHN': nn.Parameter(torch.zeros(r, 192))
        })
        self.lora_B = nn.ParameterDict({
            'MNIST': nn.Parameter(torch.zeros(576, r)),
            'Fashion-MNIST': nn.Parameter(torch.zeros(576, r)),
            'CIFAR-10': nn.Parameter(torch.zeros(576, r)),
            'SVHN': nn.Parameter(torch.zeros(576, r))
        })
        
        # Initialize (A: Kaiming, B: Zero)
        for k in self.lora_A.keys():
            nn.init.kaiming_uniform_(self.lora_A[k], a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[k])
            
        # Blending coefficients (dynamic per-sample)
        self.alphas = {
            'MNIST': None,
            'Fashion-MNIST': None,
            'CIFAR-10': None,
            'SVHN': None
        }
        
    def forward(self, x):
        # Base forward pass
        out = self.original_qkv(x)
        B, N, _ = x.shape
        
        # Compute LoRA updates for each task
        for task in ['MNIST', 'Fashion-MNIST', 'CIFAR-10', 'SVHN']:
            alpha = self.alphas[task]
            if alpha is not None:
                a = alpha.view(B, 1, 1)
                up = (x @ self.lora_A[task].t()) @ self.lora_B[task].t()
                out = out + a * up
        return out

# Replace the original QKV in ALL 12 Blocks
wrapped_qkvs = []
for l in range(12):
    orig_qkv = vit.blocks[l].attn.qkv
    wrapped = LoRAWrappedQKVAllTasks(orig_qkv, r=8)
    vit.blocks[l].attn.qkv = wrapped
    wrapped_qkvs.append(wrapped)

def set_alphas(tasks_alphas):
    for wrapped in wrapped_qkvs:
        for task in ['MNIST', 'Fashion-MNIST', 'CIFAR-10', 'SVHN']:
            wrapped.alphas[task] = tasks_alphas.get(task, None)

def reset_alphas():
    for wrapped in wrapped_qkvs:
        for task in ['MNIST', 'Fashion-MNIST', 'CIFAR-10', 'SVHN']:
            wrapped.alphas[task] = None

# Create task classification heads
heads = nn.ModuleDict({
    'MNIST': nn.Linear(192, 10),
    'Fashion-MNIST': nn.Linear(192, 10),
    'CIFAR-10': nn.Linear(192, 10),
    'SVHN': nn.Linear(192, 10)
})

# 4. Calibrate PEAR Centroids and Dispersion on Layer 2 (Early-Layer Routing Compromise)
print("\nExtracting calibration features for PEAR on Layer 2...")
centroids = {}
dispersion = {}

# Reset alphas to None to ensure we extract base (unadapted) features
reset_alphas()

for name in task_names:
    subset = data_splits[name]["cal"]
    dl = DataLoader(subset, batch_size=N_cal, shuffle=False)
    x_batch, _ = next(iter(dl))
    
    feats = extract_vit_features(vit, x_batch, layer=2)
    centroid = feats.mean(dim=0)
    centroids[name] = centroid
    
    # Cosine similarity to centroid
    denom_feat = feats.norm(p=2, dim=1)
    denom_cent = centroid.norm(p=2)
    sims = (feats @ centroid) / (denom_feat * denom_cent + 1e-8)
    disp = sims.mean().item()
    dispersion[name] = disp
    print(f"{name:15} Centroid Norm: {centroid.norm().item():.4f} | Dispersion: {disp:.4f}")

# Helper to run PEAR routing on Layer 2 features
def compute_pear_weights(feats, tau=0.05):
    norm_feats = feats.norm(p=2, dim=1)
    scores = []
    for name in task_names:
        centroid = centroids[name]
        disp = dispersion[name]
        sim = (feats @ centroid) / (norm_feats * centroid.norm() + 1e-8)
        sim_cal = sim / disp
        scores.append(sim_cal)
    
    scores = torch.stack(scores, dim=1)  # (B, 4)
    weights = torch.softmax(scores / tau, dim=1)
    
    # Return as dict of task_name -> weights tensor
    return {name: weights[:, k] for k, name in enumerate(task_names)}

# 5. Train Task-Specific LoRA Adapters and Heads
criterion = nn.CrossEntropyLoss()

for name in task_names:
    print(f"\nTraining {name} Task-Specific Adapter (all 12 layers) and Head...")
    
    # Gather parameters to optimize (all LoRA adapters for this task + head)
    params = []
    for wrapped in wrapped_qkvs:
        params.append(wrapped.lora_A[name])
        params.append(wrapped.lora_B[name])
    params += list(heads[name].parameters())
    
    optimizer = optim.Adam(params, lr=0.005)
    dl_train = DataLoader(data_splits[name]["cal"], batch_size=16, shuffle=True)
    
    for epoch in range(15):
        task_loss = 0.0
        correct = 0
        total = 0
        for x_b, y_b in dl_train:
            # Set routing to pure current task
            tasks_alphas = {t: torch.zeros(x_b.shape[0]) for t in task_names}
            tasks_alphas[name] = torch.ones(x_b.shape[0])
            set_alphas(tasks_alphas)
            
            optimizer.zero_grad()
            features = vit.forward_features(x_b)
            z = vit.forward_head(features, pre_logits=True)
            logits = heads[name](z)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
            
            task_loss += loss.item() * x_b.shape[0]
            correct += (logits.argmax(dim=1) == y_b).sum().item()
            total += x_b.shape[0]
            
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1:2d}/15 | Loss: {task_loss/total:.4f} | Accuracy: {correct/total*100:.2f}%")

# 6. Evaluation on Heterogeneous Test Set
print("\nPreparing 4-Task Heterogeneous Test Set (64 MNIST + 64 F-MNIST + 64 CIFAR-10 + 64 SVHN)...")
test_batches = {}
for name in task_names:
    dl = DataLoader(data_splits[name]["test"], batch_size=N_test, shuffle=False)
    test_batches[name] = next(iter(dl))

x_test_list = [test_batches[name][0] for name in task_names]
y_test_list = [test_batches[name][1] for name in task_names]

x_test = torch.cat(x_test_list, dim=0)
y_test = torch.cat(y_test_list, dim=0)

# Reset alphas to None to ensure we extract base features
reset_alphas()

# Extract Layer 2 features for PEAR routing on the full test set
feats_test_l2 = extract_vit_features(vit, x_test, layer=2)
pear_alphas = compute_pear_weights(feats_test_l2, tau=0.05)

results = {}
K = len(task_names)

# Helper to calculate individual task accuracy and joint mean
def evaluate_method_accuracy(method_name, get_alphas_fn, dynamic_heads=False):
    with torch.no_grad():
        # Set alphas
        tasks_alphas = get_alphas_fn()
        set_alphas(tasks_alphas)
        
        # Forward pass
        features = vit.forward_features(x_test)
        z = vit.forward_head(features, pre_logits=True)
        
        # Evaluate task-by-task
        task_correct = {}
        for k, name in enumerate(task_names):
            start_idx = k * N_test
            end_idx = (k + 1) * N_test
            
            z_task = z[start_idx:end_idx]
            y_task = y_test[start_idx:end_idx]
            
            if dynamic_heads:
                # Weighted ensembling of all heads
                combined_logits = torch.zeros(N_test, 10)
                for t_name in task_names:
                    w = tasks_alphas[t_name][start_idx:end_idx].view(-1, 1)
                    combined_logits += w * heads[t_name](z_task)
                preds = combined_logits.argmax(dim=1)
            else:
                # Direct task head
                logits = heads[name](z_task)
                preds = logits.argmax(dim=1)
                
            correct = (preds == y_task).sum().item()
            task_correct[name] = correct
            
        # Store results
        results[method_name] = {}
        total_correct = 0
        for name in task_names:
            acc = task_correct[name] / N_test * 100
            results[method_name][name] = acc
            total_correct += task_correct[name]
        results[method_name]["Joint Mean"] = total_correct / (K * N_test) * 100
    # Clean up alphas to None
    reset_alphas()

# --- Method A: Static Uniform Merging ---
def get_uniform_alphas():
    return {name: torch.full((x_test.shape[0],), 1.0 / K) for name in task_names}
evaluate_method_accuracy("Static Uniform", get_uniform_alphas, dynamic_heads=True)

# --- Method B: SABLE SOTA (Late Adaptation) ---
# SABLE leaves all blocks unadapted, ensembling via heads using PEAR routing weights
with torch.no_grad():
    results["SABLE SOTA (Late Adapt)"] = {}
    total_correct = 0
    # Shut off LoRAs in backbone
    sable_alphas = {name: torch.zeros(x_test.shape[0]) for name in task_names}
    set_alphas(sable_alphas)
    
    features = vit.forward_features(x_test)
    z = vit.forward_head(features, pre_logits=True)
    
    for k, name in enumerate(task_names):
        start_idx = k * N_test
        end_idx = (k + 1) * N_test
        z_task = z[start_idx:end_idx]
        y_task = y_test[start_idx:end_idx]
        
        combined_logits = torch.zeros(N_test, 10)
        for t_name in task_names:
            w = pear_alphas[t_name][start_idx:end_idx].view(-1, 1)
            combined_logits += w * heads[t_name](z_task)
        preds = combined_logits.argmax(dim=1)
        correct = (preds == y_task).sum().item()
        
        acc = correct / N_test * 100
        results["SABLE SOTA (Late Adapt)"][name] = acc
        total_correct += correct
    results["SABLE SOTA (Late Adapt)"]["Joint Mean"] = total_correct / (K * N_test) * 100
    reset_alphas()

# --- Method C: PEAR (Ours) ---
def get_pear_alphas():
    return pear_alphas
evaluate_method_accuracy("PEAR (Ours)", get_pear_alphas, dynamic_heads=True)

# --- Method D: Expert Ceiling (Perfect Routing) ---
def get_ceiling_alphas():
    alphas = {}
    for name in task_names:
        tasks_alphas = torch.zeros(x_test.shape[0])
        k = task_names.index(name)
        tasks_alphas[k*N_test:(k+1)*N_test] = 1.0
        alphas[name] = tasks_alphas
    return alphas
evaluate_method_accuracy("Expert Ceiling", get_ceiling_alphas, dynamic_heads=False)

# Format and display table
print("\n" + "="*90)
print(f"{'REAL-WORLD 4-TASK LORA CLASSIFICATION RESULTS (ALL 12 LAYERS LOPE)':^90}")
print("="*90)
print(f"{'Method':25} | " + " | ".join([f"{n:12}" for n in task_names]) + " | Joint Mean")
print("-"*90)
for method in ["Static Uniform", "SABLE SOTA (Late Adapt)", "PEAR (Ours)", "Expert Ceiling"]:
    res = results[method]
    row = f"{method:25} | " + " | ".join([f"{res[n]:11.2f}%" for n in task_names]) + f" | {res['Joint Mean']:10.2f}%"
    print(row)
print("="*90)

# Save results for use in the paper's LaTeX
with open("real_world_lora_results_4tasks.txt", "w") as f:
    for m in ["Static Uniform", "SABLE SOTA (Late Adapt)", "PEAR (Ours)", "Expert Ceiling"]:
        f.write(f"--- {m} ---\n")
        f.write(f"Joint Mean: {results[m]['Joint Mean']:.2f}%\n")
        for n in task_names:
            f.write(f"{n}: {results[m][n]:.2f}%\n")

print("\nSuccessfully saved 4-task real-world LoRA results to 'real_world_lora_results_4tasks.txt'")
