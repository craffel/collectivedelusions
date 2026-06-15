import sys
# Ensure we use our locally installed hf-hub to avoid ImportErrors
sys.path.insert(0, './local_packages')

import os
import json
import torch
import torch.nn as nn
from PIL import Image
import torchvision.datasets as dset
from transformers import CLIPModel, CLIPProcessor, CLIPVisionModel
from torch.func import functional_call
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ---------------------------------------------------------
# 1. LOAD MODEL PARTS & COMPUTE TASK VECTORS
# ---------------------------------------------------------

t0 = time.time()
print("Loading pre-trained base CLIPModel...")
clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(device)
processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
print(f"Loaded base model in {time.time() - t0:.2f}s")

t0 = time.time()
print("Loading fine-tuned MNIST Vision Model...")
model_mnist = CLIPVisionModel.from_pretrained('tanganke/clip-vit-base-patch32_mnist').to(device)
print(f"Loaded MNIST model in {time.time() - t0:.2f}s")

t0 = time.time()
print("Loading fine-tuned SVHN Vision Model...")
model_svhn = CLIPVisionModel.from_pretrained('tanganke/clip-vit-base-patch32_svhn').to(device)
print(f"Loaded SVHN model in {time.time() - t0:.2f}s")

base_params = {n: p for n, p in clip_model.vision_model.named_parameters()}
mnist_params = {n: p for n, p in model_mnist.vision_model.named_parameters()}
svhn_params = {n: p for n, p in model_svhn.vision_model.named_parameters()}

# Select parameters to merge: We merge ALL parameters in all 12 transformer encoder layers
# This includes attention query, key, value, projections, MLP weights, and layer norms.
target_param_names = []
for name in base_params.keys():
    if "encoder.layers." in name:
        target_param_names.append(name)

print(f"Selected {len(target_param_names)} parameters across 12 vision layers for physical weight merging.")

# Compute actual fine-tuned task vectors
t0 = time.time()
task_vectors = {
    0: {}, # MNIST task vector
    1: {}  # SVHN task vector
}
with torch.no_grad():
    for name in target_param_names:
        task_vectors[0][name] = mnist_params[name] - base_params[name]
        task_vectors[1][name] = svhn_params[name] - base_params[name]
print(f"Computed task vectors in {time.time() - t0:.2f}s")

# ---------------------------------------------------------
# 2. PRE-COMPUTE TEXT EMBEDDINGS (CLASSIFIERS)
# ---------------------------------------------------------

t0 = time.time()
print("Pre-computing text classifier embeddings...")
# MNIST and SVHN classes represent digits 0-9
mnist_classes = [f"a photo of the number {i}" for i in range(10)]
svhn_classes = [f"a photo of the number {i}" for i in range(10)]

# Pre-compute text embeds for both tasks
def precompute_text_embeds(classes):
    inputs = processor(text=classes, padding=True, return_tensors='pt').to(device)
    with torch.no_grad():
        text_outputs = clip_model.text_model(**inputs)
        text_embeds = clip_model.text_projection(text_outputs.pooler_output)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    return text_embeds

text_embeds_mnist = precompute_text_embeds(mnist_classes)
text_embeds_svhn = precompute_text_embeds(svhn_classes)
print(f"Pre-computed text embeds in {time.time() - t0:.2f}s")

# ---------------------------------------------------------
# 3. PREPARE REAL ADAPTATION & TEST SETS
# ---------------------------------------------------------

t0 = time.time()
print("Loading actual MNIST and SVHN datasets from torchvision...")
mnist_dataset = dset.MNIST(root='./data', train=False, download=True)
svhn_dataset = dset.SVHN(root='./data', split='test', download=True)
print(f"Loaded datasets in {time.time() - t0:.2f}s")

# Generate adaptation stream (50 images each, total 100 images)
torch.manual_seed(42)
mnist_adapt_indices = torch.randperm(len(mnist_dataset))[:50].tolist()
svhn_adapt_indices = torch.randperm(len(svhn_dataset))[:50].tolist()

mnist_adapt_imgs = [mnist_dataset[i][0] for i in mnist_adapt_indices]
svhn_adapt_imgs = [svhn_dataset[i][0] for i in svhn_adapt_indices]

# Generate test sets (100 images each, total 200 images for final evaluation)
mnist_test_indices = torch.randperm(len(mnist_dataset))[50:150].tolist()
svhn_test_indices = torch.randperm(len(svhn_dataset))[50:150].tolist()

mnist_test_imgs = [mnist_dataset[i][0] for i in mnist_test_indices]
mnist_test_labels = torch.tensor([mnist_dataset[i][1] for i in mnist_test_indices]).to(device)

svhn_test_imgs = [svhn_dataset[i][0] for i in svhn_test_indices]
svhn_test_labels = torch.tensor([svhn_dataset[i][1] for i in svhn_test_indices]).to(device)

t0 = time.time()
print("Processing images through CLIPProcessor...")
# Process images through CLIPProcessor
inputs_mnist_adapt = processor(images=mnist_adapt_imgs, return_tensors='pt').pixel_values.to(device)
inputs_svhn_adapt = processor(images=svhn_adapt_imgs, return_tensors='pt').pixel_values.to(device)

inputs_mnist_test = processor(images=mnist_test_imgs, return_tensors='pt').pixel_values.to(device)
inputs_svhn_test = processor(images=svhn_test_imgs, return_tensors='pt').pixel_values.to(device)
print(f"Processed images in {time.time() - t0:.2f}s")

# ---------------------------------------------------------
# 4. SPECTRAL DESIGN MATRIX BUILDERS
# ---------------------------------------------------------

L = 12
K = 2
degree = 2
lr = 1e-2
num_steps = 20
gamma_csd = 0.2

def get_monomial_design_matrix(L, degree):
    l_indices = torch.linspace(0.0, 1.0, L).to(device)
    V = []
    for j in range(degree + 1):
        V.append(l_indices ** j)
    return torch.stack(V, dim=1) # Shape: (L, degree + 1)

def get_chebyshev_design_matrix(L, degree):
    l_indices = torch.linspace(0, L - 1, L).to(device)
    x = 2.0 * l_indices / (L - 1) - 1.0
    C = []
    C.append(torch.ones_like(x)) # T_0(x) = 1
    if degree >= 1:
        C.append(x)              # T_1(x) = x
    for j in range(2, degree + 1):
        C.append(2.0 * x * C[-1] - C[-2]) # T_j(x) = 2x * T_{j-1}(x) - T_{j-2}(x)
    return torch.stack(C, dim=1) # Shape: (L, degree + 1)

# ---------------------------------------------------------
# 5. CORE OPTIMIZATION & EVALUATION
# ---------------------------------------------------------

def entropy(logits):
    probs = torch.softmax(logits, dim=-1)
    return -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()

def evaluate_accuracy(merged_vision_params, test_pixel_values, text_embeds, labels):
    with torch.no_grad():
        vision_outputs = functional_call(clip_model.vision_model, merged_vision_params, args=(), kwargs={'pixel_values': test_pixel_values})
        image_embeds = clip_model.visual_projection(vision_outputs.pooler_output)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        logits = image_embeds @ text_embeds.t()
        preds = logits.argmax(dim=-1)
        acc = (preds == labels).float().mean().item()
    return acc * 100.0

def run_real_merging(method_name):
    print(f"\nRunning Real Model Merging: {method_name}")
    
    # Precompute design matrices
    V = get_monomial_design_matrix(L, degree)
    C = get_chebyshev_design_matrix(L, degree)
    
    # Initialize parameters and optimizer
    if method_name == 'Task Arithmetic':
        lambdas = torch.full((K, L), 0.5, device=device)
        optimizer = None
    elif method_name == 'AdaMerging':
        lambdas = torch.full((K, L), 0.5, device=device, requires_grad=True)
        optimizer = torch.optim.Adam([lambdas], lr=lr)
    elif method_name == 'PolyMerge':
        gammas = [torch.full((K,), 0.5 if j==0 else 0.0, device=device, requires_grad=True) for j in range(degree + 1)]
        optimizer = torch.optim.Adam(gammas, lr=lr)
    elif method_name == 'ChebyMerge':
        alphas = [torch.full((K,), 0.5 if j==0 else 0.0, device=device, requires_grad=True) for j in range(degree + 1)]
        optimizer = torch.optim.Adam(alphas, lr=lr)
    elif method_name == 'ChebyMerge + CSD':
        alphas = [torch.full((K,), 0.5 if j==0 else 0.0, device=device, requires_grad=True) for j in range(degree + 1)]
        param_groups = []
        for j in range(degree + 1):
            param_groups.append({
                'params': [alphas[j]],
                'lr': lr * (gamma_csd ** j)
            })
        optimizer = torch.optim.Adam(param_groups)
        
    losses = []
    
    # 1. Initial Evaluation (Step 0)
    # Define initial spatial coefficients
    if method_name == 'Task Arithmetic':
        lambdas_step = lambdas
    elif method_name == 'AdaMerging':
        lambdas_step = lambdas
    elif method_name == 'PolyMerge':
        lambdas_step = sum(gammas[j].unsqueeze(1) * V[:, j].unsqueeze(0) for j in range(degree + 1))
    elif method_name in ['ChebyMerge', 'ChebyMerge + CSD']:
        lambdas_step = sum(alphas[j].unsqueeze(1) * C[:, j].unsqueeze(0) for j in range(degree + 1))
        
    # Build initial merged state dict
    merged_params = {}
    for name, param in base_params.items():
        if name in target_param_names:
            parts = name.split(".")
            l_idx = int(parts[parts.index("layers") + 1])
            merged_params[name] = param + lambdas_step[0, l_idx] * task_vectors[0][name] + lambdas_step[1, l_idx] * task_vectors[1][name]
        else:
            merged_params[name] = param
            
    init_mnist_acc = evaluate_accuracy(merged_params, inputs_mnist_test, text_embeds_mnist, mnist_test_labels)
    init_svhn_acc = evaluate_accuracy(merged_params, inputs_svhn_test, text_embeds_svhn, svhn_test_labels)
    init_avg_acc = (init_mnist_acc + init_svhn_acc) / 2.0
    print(f"Step 00 | Initial Accuracies: MNIST={init_mnist_acc:.2f}%, SVHN={init_svhn_acc:.2f}%, Avg={init_avg_acc:.2f}%")
    
    # 2. Run TTA Optimization
    for step in range(num_steps):
        # Step 2.1: Reconstruct spatial coefficients
        if method_name == 'Task Arithmetic':
            lambdas_step = lambdas
        elif method_name == 'AdaMerging':
            lambdas_step = lambdas
        elif method_name == 'PolyMerge':
            lambdas_step = sum(gammas[j].unsqueeze(1) * V[:, j].unsqueeze(0) for j in range(degree + 1))
        elif method_name in ['ChebyMerge', 'ChebyMerge + CSD']:
            lambdas_step = sum(alphas[j].unsqueeze(1) * C[:, j].unsqueeze(0) for j in range(degree + 1))
            
        # Step 2.2: Build merged state dict
        merged_params = {}
        for name, param in base_params.items():
            if name in target_param_names:
                parts = name.split(".")
                l_idx = int(parts[parts.index("layers") + 1])
                merged_params[name] = param + lambdas_step[0, l_idx] * task_vectors[0][name] + lambdas_step[1, l_idx] * task_vectors[1][name]
            else:
                merged_params[name] = param
                
        # Step 2.3: Differentiable Forward Passes on Adaptation Stream
        outputs_mnist = functional_call(clip_model.vision_model, merged_params, args=(), kwargs={'pixel_values': inputs_mnist_adapt})
        outputs_svhn = functional_call(clip_model.vision_model, merged_params, args=(), kwargs={'pixel_values': inputs_svhn_adapt})
        
        # Project features
        embeds_mnist = clip_model.visual_projection(outputs_mnist.pooler_output)
        embeds_mnist = embeds_mnist / embeds_mnist.norm(dim=-1, keepdim=True)
        
        embeds_svhn = clip_model.visual_projection(outputs_svhn.pooler_output)
        embeds_svhn = embeds_svhn / embeds_svhn.norm(dim=-1, keepdim=True)
        
        # Logits
        logits_mnist = embeds_mnist @ text_embeds_mnist.t()
        logits_svhn = embeds_svhn @ text_embeds_svhn.t()
        
        # Unsupervised joint entropy loss
        loss = entropy(logits_mnist) + entropy(logits_svhn)
        losses.append(loss.item())
        
        # Step 2.4: Backpropagation and Optimizer Update
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    # 3. Final Evaluation
    # Reconstruct final coefficients
    with torch.no_grad():
        if method_name == 'Task Arithmetic':
            lambdas_step = lambdas
        elif method_name == 'AdaMerging':
            lambdas_step = lambdas
        elif method_name == 'PolyMerge':
            lambdas_step = sum(gammas[j].unsqueeze(1) * V[:, j].unsqueeze(0) for j in range(degree + 1))
        elif method_name in ['ChebyMerge', 'ChebyMerge + CSD']:
            lambdas_step = sum(alphas[j].unsqueeze(1) * C[:, j].unsqueeze(0) for j in range(degree + 1))
            
        final_merged_params = {}
        for name, param in base_params.items():
            if name in target_param_names:
                parts = name.split(".")
                l_idx = int(parts[parts.index("layers") + 1])
                final_merged_params[name] = param + lambdas_step[0, l_idx] * task_vectors[0][name] + lambdas_step[1, l_idx] * task_vectors[1][name]
            else:
                final_merged_params[name] = param
                
    final_mnist_acc = evaluate_accuracy(final_merged_params, inputs_mnist_test, text_embeds_mnist, mnist_test_labels)
    final_svhn_acc = evaluate_accuracy(final_merged_params, inputs_svhn_test, text_embeds_svhn, svhn_test_labels)
    final_avg_acc = (final_mnist_acc + final_svhn_acc) / 2.0
    print(f"Final Accuracies  : MNIST={final_mnist_acc:.2f}%, SVHN={final_svhn_acc:.2f}%, Avg={final_avg_acc:.2f}%")
    
    # Calculate condition number of the design matrix
    if method_name in ['Task Arithmetic', 'AdaMerging']:
        cond_num = 1.0
    elif method_name == 'PolyMerge':
        cond_num = torch.linalg.cond(V.t() @ V).item()
    elif method_name in ['ChebyMerge', 'ChebyMerge + CSD']:
        cond_num = torch.linalg.cond(C.t() @ C).item()
        
    return {
        'losses': losses,
        'initial_loss': losses[0],
        'final_loss': losses[-1],
        'initial_accuracies': {
            'MNIST': init_mnist_acc,
            'SVHN': init_svhn_acc,
            'Average': init_avg_acc
        },
        'final_accuracies': {
            'MNIST': final_mnist_acc,
            'SVHN': final_svhn_acc,
            'Average': final_avg_acc
        },
        'condition_number': cond_num
    }

# Run all methods
results = {}
methods = ['Task Arithmetic', 'AdaMerging', 'PolyMerge', 'ChebyMerge', 'ChebyMerge + CSD']
for m in methods:
    results[m] = run_real_merging(m)

# Print final comparison table of physical accuracies
print("\n" + "="*80)
print(f"{'Method Name':<22} | {'Init Avg Acc':<12} | {'Final Avg Acc':<13} | {'Init Ent':<10} | {'Final Ent':<10} | {'Cond Num':<10}")
print("="*80)
for m in methods:
    init_acc = results[m]['initial_accuracies']['Average']
    final_acc = results[m]['final_accuracies']['Average']
    init_ent = results[m]['losses'][0]
    final_ent = results[m]['losses'][-1]
    cond = results[m]['condition_number']
    print(f"{m:<22} | {init_acc:<12.2f}% | {final_acc:<13.2f}% | {init_ent:<10.4f} | {final_ent:<10.4f} | {cond:<10.4f}")
print("="*80)

# Save results to JSON
os.makedirs('results', exist_ok=True)
with open('results/real_physical_metrics.json', 'w') as f:
    json.dump(results, f, indent=2)
print("Real physical metrics saved successfully to results/real_physical_metrics.json!")
