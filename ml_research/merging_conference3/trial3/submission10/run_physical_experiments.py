import os
import json
import io
import requests
import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPVisionModel, CLIPProcessor
from torch.func import functional_call

# ---------------------------------------------------------
# 1. IMAGES DOWNLOAD & PREPARATION
# ---------------------------------------------------------

def download_image(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return Image.open(io.BytesIO(response.content))
    except Exception as e:
        print(f"Failed to download {url}: {e}")
    # Fallback: create a random synthetic image
    return Image.fromarray(torch.randint(0, 255, (224, 224, 3), dtype=torch.uint8).numpy())

# Task 1 URLs (Animals: Cat & Dog)
task1_urls = [
    'https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg',
    'https://upload.wikimedia.org/wikipedia/commons/d/d9/Collage_of_Nine_Dogs.jpg'
]

# Task 2 URLs (Vehicles: Car & Train)
task2_urls = [
    'https://upload.wikimedia.org/wikipedia/commons/4/44/BMW_M3_E92_by_G-Power.jpg',
    'https://upload.wikimedia.org/wikipedia/commons/9/9b/TGV_Sud-Est_at_Paris-Gare_de_Lyon.jpg'
]

print("Downloading and preparing physical images...")
task1_images = [download_image(url) for url in task1_urls]
task2_images = [download_image(url) for url in task2_urls]

# Initialize processor
processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

inputs1 = processor(images=task1_images, return_tensors='pt')
inputs2 = processor(images=task2_images, return_tensors='pt')

pixel_values1 = inputs1.pixel_values
pixel_values2 = inputs2.pixel_values

# ---------------------------------------------------------
# 2. DESIGN MATRIX GENERATORS
# ---------------------------------------------------------

def get_monomial_design_matrix(L, degree):
    l_indices = torch.linspace(0.0, 1.0, L)
    V = []
    for j in range(degree + 1):
        V.append(l_indices ** j)
    return torch.stack(V, dim=1) # Shape: (L, degree + 1)

def get_chebyshev_design_matrix(L, degree):
    l_indices = torch.linspace(0, L - 1, L)
    x = 2.0 * l_indices / (L - 1) - 1.0
    C = []
    C.append(torch.ones_like(x)) # T_0(x) = 1
    if degree >= 1:
        C.append(x)              # T_1(x) = x
    for j in range(2, degree + 1):
        C.append(2.0 * x * C[-1] - C[-2]) # T_j(x) = 2x * T_{j-1}(x) - T_{j-2}(x)
    return torch.stack(C, dim=1) # Shape: (L, degree + 1)

# ---------------------------------------------------------
# 3. SETUP MODEL & CLASSIFIER HEADS
# ---------------------------------------------------------

L = 12
K = 2
degree = 2
lr = 1e-2
num_steps = 20
gamma_csd = 0.2

print("Loading CLIP Vision Model...")
model = CLIPVisionModel.from_pretrained('openai/clip-vit-base-patch32')
base_params = {n: p for n, p in model.named_parameters()}

# Select the target parameters to merge: Attention projection weights in all layers
target_param_names = []
for name, param in model.named_parameters():
    if "vision_model.encoder.layers." in name and any(proj in name for proj in ["q_proj", "k_proj", "v_proj", "out_proj"]) and "weight" in name:
        target_param_names.append(name)

print(f"Selected {len(target_param_names)} parameters across {L} layers for weight merging.")

# Generate task-specific perturbations (task vectors)
torch.manual_seed(42)
deltas = {0: {}, 1: {}}
for name in target_param_names:
    param = base_params[name]
    # Set task vector scale to 0.02
    deltas[0][name] = torch.randn_like(param) * 0.02
    deltas[1][name] = torch.randn_like(param) * 0.02

# Initialize classification heads (mapping 768 -> 10 classes)
torch.manual_seed(42)
W1 = torch.nn.Parameter(torch.randn(10, 768) * 0.1)
W2 = torch.nn.Parameter(torch.randn(10, 768) * 0.1)

# ---------------------------------------------------------
# 4. ENTROPY EVALUATOR
# ---------------------------------------------------------

def entropy(logits):
    probs = torch.softmax(logits, dim=-1)
    return -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()

# ---------------------------------------------------------
# 5. CORE OPTIMIZATION PIPELINE
# ---------------------------------------------------------

def run_physical_merging(method_name):
    print(f"\nRunning Physical Merging: {method_name}")
    
    # Precompute design matrices
    V = get_monomial_design_matrix(L, degree)
    C = get_chebyshev_design_matrix(L, degree)
    
    # Initialize parameters and optimizer
    if method_name == 'Task Arithmetic':
        # Static coefficients at 0.5
        lambdas = torch.full((K, L), 0.5)
        optimizer = None
    elif method_name == 'AdaMerging':
        # Unconstrained spatial coefficients
        lambdas = torch.full((K, L), 0.5, requires_grad=True)
        optimizer = torch.optim.Adam([lambdas], lr=lr)
    elif method_name == 'PolyMerge':
        # Monomial coefficients gammas
        gammas = [torch.full((K,), 0.5 if j==0 else 0.0, requires_grad=True) for j in range(degree + 1)]
        optimizer = torch.optim.Adam(gammas, lr=lr)
    elif method_name == 'ChebyMerge':
        # Chebyshev coefficients alphas
        alphas = [torch.full((K,), 0.5 if j==0 else 0.0, requires_grad=True) for j in range(degree + 1)]
        optimizer = torch.optim.Adam(alphas, lr=lr)
    elif method_name == 'ChebyMerge + CSD':
        # Chebyshev coefficients alphas with Controllable Spectral Decay
        alphas = [torch.full((K,), 0.5 if j==0 else 0.0, requires_grad=True) for j in range(degree + 1)]
        param_groups = []
        for j in range(degree + 1):
            param_groups.append({
                'params': [alphas[j]],
                'lr': lr * (gamma_csd ** j)
            })
        optimizer = torch.optim.Adam(param_groups)
        
    losses = []
    
    # Run optimization
    for step in range(num_steps):
        # Step 1: Compute spatial coefficients
        if method_name == 'Task Arithmetic':
            lambdas_step = lambdas
        elif method_name == 'AdaMerging':
            lambdas_step = lambdas
        elif method_name == 'PolyMerge':
            # Reconstruction via monomial basis
            lambdas_step = sum(gammas[j].unsqueeze(1) * V[:, j].unsqueeze(0) for j in range(degree + 1))
        elif method_name in ['ChebyMerge', 'ChebyMerge + CSD']:
            # Reconstruction via Chebyshev basis
            lambdas_step = sum(alphas[j].unsqueeze(1) * C[:, j].unsqueeze(0) for j in range(degree + 1))
            
        # Step 2: Build merged state dict for functional call
        merged_params = {}
        for name, param in base_params.items():
            if name in target_param_names:
                # Extract layer index from name: e.g., 'vision_model.encoder.layers.0.self_attn.q_proj.weight'
                parts = name.split(".")
                l_idx = int(parts[parts.index("layers") + 1])
                # Compute merged parameter: param + \sum_k \lambda_k_l * delta_k
                merged_params[name] = param + lambdas_step[0, l_idx] * deltas[0][name] + lambdas_step[1, l_idx] * deltas[1][name]
            else:
                merged_params[name] = param
                
        # Step 3: Differentiable Forward Passes
        outputs1 = functional_call(model, merged_params, args=(), kwargs={'pixel_values': pixel_values1})
        outputs2 = functional_call(model, merged_params, args=(), kwargs={'pixel_values': pixel_values2})
        
        pooler_output1 = outputs1.pooler_output
        pooler_output2 = outputs2.pooler_output
        
        # Logits
        logits1 = pooler_output1 @ W1.t()
        logits2 = pooler_output2 @ W2.t()
        
        # Loss (entropy minimization)
        loss = entropy(logits1) + entropy(logits2)
        losses.append(loss.item())
        
        print(f"Step {step:02d} | TTA Entropy Loss: {loss.item():.6f}")
        
        # Step 4: Backpropagation and Optimizer Update
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    # Calculate condition number of the design matrix / Hessian approximation
    if method_name in ['Task Arithmetic', 'AdaMerging']:
        cond_num = 1.0 # Unconstrained base
    elif method_name == 'PolyMerge':
        cond_num = torch.linalg.cond(V.t() @ V).item()
    elif method_name in ['ChebyMerge', 'ChebyMerge + CSD']:
        cond_num = torch.linalg.cond(C.t() @ C).item()
        
    return {
        'losses': losses,
        'final_loss': losses[-1],
        'condition_number': cond_num
    }

# Run all methods
results = {}
methods = ['Task Arithmetic', 'AdaMerging', 'PolyMerge', 'ChebyMerge', 'ChebyMerge + CSD']
for m in methods:
    results[m] = run_physical_merging(m)

# Print final comparison table
print("\n" + "="*60)
print(f"{'Method Name':<22} | {'Initial Loss':<12} | {'Final Loss':<10} | {'Condition #':<12}")
print("="*60)
for m in methods:
    init_l = results[m]['losses'][0]
    final_l = results[m]['losses'][-1]
    cond = results[m]['condition_number']
    print(f"{m:<22} | {init_l:<12.6f} | {final_l:<10.6f} | {cond:<12.4f}")
print("="*60)

# Save results to JSON
os.makedirs('results', exist_ok=True)
with open('results/physical_metrics.json', 'w') as f:
    json.dump(results, f, indent=2)
print("Physical metrics saved successfully to results/physical_metrics.json!")
