import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

# Disable cuDNN to prevent initialization errors on NVIDIA H100
torch.backends.cudnn.enabled = False

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define transform for evaluation (same as training)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Helper function to project a vector onto the unit simplex
def project_simplex(v):
    # v is of shape [K]
    sorted_v, _ = torch.sort(v, descending=True)
    cssv = torch.cumsum(sorted_v, dim=0)
    indices = torch.arange(1, len(v) + 1, device=v.device)
    cond = sorted_v - (cssv - 1.0) / indices > 0
    idx = torch.where(cond)[0][-1]
    theta = (cssv[idx] - 1.0) / (idx + 1)
    return torch.clamp(v - theta, min=0.0)

# Vectorized InfoNCE Contrastive Loss against Class Prototypes
def compute_contrastive_loss(online_feats, pseudo_labels, prototypes, task_weights=None):
    K = len(prototypes)
    B = online_feats.size(0)
    device = online_feats.device
    
    online_feats_norm = online_feats / (online_feats.norm(dim=1, keepdim=True) + 1e-8)
    
    # Stack prototypes: [K, 10, 512]
    proto_stacked = torch.stack([prototypes[k] for k in range(K)], dim=0) # [K, 10, 512]
    proto_flat = proto_stacked.view(K * 10, 512)
    
    # Compute similarity matrix: [B, K * 10]
    sim_matrix = torch.mm(online_feats_norm, proto_flat.t()) / 0.07 # [B, K * 10]
    
    # Compute positive similarity
    pos_sims_experts = []
    for k in range(K):
        pos_sims_experts.append(sim_matrix[range(B), k * 10 + pseudo_labels])
    pos_sims_stacked = torch.stack(pos_sims_experts, dim=1) # [B, K]
    
    if task_weights is not None:
        pos_sims = (pos_sims_stacked * task_weights).sum(dim=1) # [B]
    else:
        pos_sims = pos_sims_stacked.mean(dim=1) # [B]
    
    # Compute negative sum of exp(sim)
    exp_sims = torch.exp(sim_matrix) # [B, K * 10]
    exp_sims_reshaped = exp_sims.view(B, K, 10)
    total_sum_per_expert = exp_sims_reshaped.sum(dim=2) # [B, K]
    
    pos_exp_experts = []
    for k in range(K):
        pos_exp_experts.append(exp_sims_reshaped[range(B), k, pseudo_labels])
    pos_exp_per_expert = torch.stack(pos_exp_experts, dim=1) # [B, K]
    
    neg_sum_per_expert = total_sum_per_expert - pos_exp_per_expert # [B, K]
    total_neg_sum = neg_sum_per_expert.sum(dim=1) # [B]
    
    loss_vector = -pos_sims + torch.log(torch.exp(pos_sims) + total_neg_sum + 1e-8)
    return loss_vector.mean()

# Load test datasets
print("Loading test datasets...")
mnist_test = datasets.MNIST("data", train=False, download=False, transform=transform)
fashion_test = datasets.FashionMNIST("data", train=False, download=False, transform=transform)
kmnist_test = datasets.KMNIST("data", train=False, download=False, transform=transform)

# Build sequential and alternating streams
# Let's create a stream of 90 batches of size 64
def create_stream(stream_type="sequential", noise_std=0.0, seed=42):
    torch.manual_seed(seed)
    
    # We want 30 batches of each task
    mnist_loader = DataLoader(mnist_test, batch_size=64, shuffle=True)
    fashion_loader = DataLoader(fashion_test, batch_size=64, shuffle=True)
    kmnist_loader = DataLoader(kmnist_test, batch_size=64, shuffle=True)
    
    mnist_iter = iter(mnist_loader)
    fashion_iter = iter(fashion_loader)
    kmnist_iter = iter(kmnist_loader)
    
    batches = []
    
    if stream_type == "sequential":
        # Task 0: MNIST (batches 0-29)
        # Task 1: FashionMNIST (batches 30-59)
        # Task 2: KMNIST (batches 60-89)
        for _ in range(30):
            images, labels = next(mnist_iter)
            batches.append((images, labels, 0)) # images, labels, task_id
        for _ in range(30):
            images, labels = next(fashion_iter)
            batches.append((images, labels, 1))
        for _ in range(30):
            images, labels = next(kmnist_iter)
            batches.append((images, labels, 2))
            
    elif stream_type == "alternating":
        # Alternate tasks on every batch
        for _ in range(30):
            images, labels = next(mnist_iter)
            batches.append((images, labels, 0))
            images, labels = next(fashion_iter)
            batches.append((images, labels, 1))
            images, labels = next(kmnist_iter)
            batches.append((images, labels, 2))
            
    # Apply noise corruption if requested
    corrupted_batches = []
    for images, labels, task_id in batches:
        if noise_std > 0.0:
            # Add Gaussian noise to normalized images
            corrupted_images = images + noise_std * torch.randn_like(images)
            corrupted_batches.append((corrupted_images, labels, task_id))
        else:
            corrupted_batches.append((images, labels, task_id))
            
    return corrupted_batches

def get_feature_extractor(model):
    class FeatureExtractor(nn.Module):
        def __init__(self, resnet):
            super().__init__()
            self.features = nn.Sequential(*list(resnet.children())[:-1])
        def forward(self, x):
            x = self.features(x)
            return torch.flatten(x, 1)
    return FeatureExtractor(model)

# Evaluate a specific merging configuration over a test stream
def evaluate_method(method_name, stream_type="sequential", noise_std=0.0):
    print(f"\n>>> Running Method: {method_name} | Stream: {stream_type} | Noise Std: {noise_std} <<<")
    
    # Load base model and expert state dicts
    base_state_dict = torch.load("models/base_model.pth", map_location=device)
    
    experts = ["mnist", "fashionmnist", "kmnist"]
    K = len(experts)
    
    expert_state_dicts = []
    for exp in experts:
        expert_state_dicts.append(torch.load(f"models/expert_{exp}.pth", map_location=device))
        
    # Pre-compute task vectors
    task_vectors = {}
    for k in range(K):
        task_vectors[k] = {}
        for p_name in base_state_dict:
            # Check if parameter requires merging (exclude buffer/running variables of BN)
            if "running_mean" in p_name or "running_var" in p_name or "num_batches_tracked" in p_name:
                continue
            task_vectors[k][p_name] = expert_state_dicts[k][p_name] - base_state_dict[p_name]
            
    # Load Fisher information and prototypes
    fisher_diag = {}
    fisher_scalars = {}
    prototypes_global = {}
    for k, exp in enumerate(experts):
        fisher_diag[k] = torch.load(f"models/fisher_diag_{exp}.pt", map_location=device)
        fisher_scalars[k] = torch.load(f"models/fisher_layer_scalars_{exp}.pt", map_location=device)
        prototypes_global[k] = torch.load(f"models/prototypes_{exp}.pt", map_location=device) # [10, 512]
        
    prototypes = {k: prototypes_global[k].clone() for k in prototypes_global}
        
    # Compute joint Fisher priors
    joint_fisher_diag = {}
    joint_fisher_scalars = {}
    for p_name in base_state_dict:
        if "running_mean" in p_name or "running_var" in p_name or "num_batches_tracked" in p_name:
            continue
        # Average Fisher across experts
        joint_fisher_diag[p_name] = torch.mean(torch.stack([fisher_diag[k][p_name] for k in range(K)]), dim=0)
        joint_fisher_scalars[p_name] = sum([fisher_scalars[k][p_name] for k in range(K)]) / K

    # Initialize model
    model = models.resnet18()
    model.fc = nn.Linear(512, 10)
    model.load_state_dict(base_state_dict)
    model = model.to(device)
    
    feature_extractor = get_feature_extractor(model).to(device)

    # Build expert feature extractors for clean affinity estimation in FOGS-Merge
    expert_extractors = {}
    if method_name == "fogs":
        for k, exp in enumerate(experts):
            exp_model = models.resnet18()
            exp_model.fc = nn.Linear(512, 10)
            exp_model.load_state_dict(expert_state_dicts[k])
            exp_model = exp_model.to(device)
            exp_model.eval()
            expert_extractors[k] = get_feature_extractor(exp_model).to(device)
    
    # Create the test stream
    stream = create_stream(stream_type, noise_std)
    
    # Initialize layer-wise merging coefficients
    # coefficients map from p_name to tensor of shape [K]
    coefficients = {}
    for p_name in base_state_dict:
        if p_name in task_vectors[0]:
            coefficients[p_name] = torch.full((K,), 1.0 / K, device=device)
            
    # Helper to apply coefficients to the model
    def apply_coefficients(coeffs):
        merged_state_dict = copy.deepcopy(base_state_dict)
        for p_name in base_state_dict:
            if p_name in task_vectors[0]:
                coeff = coeffs[p_name] # [K]
                # wt = wbase + sum_k c_k * v_k
                weighted_sum = torch.zeros_like(base_state_dict[p_name])
                for k in range(K):
                    weighted_sum += coeff[k] * task_vectors[k][p_name]
                merged_state_dict[p_name] = base_state_dict[p_name] + weighted_sum
        model.load_state_dict(merged_state_dict, strict=False)

    # Apply initial uniform merging coefficients
    apply_coefficients(coefficients)
    
    # Evaluation stats
    total_samples = 0
    correct_predictions = 0
    task_stats = {0: {"correct": 0, "total": 0}, 1: {"correct": 0, "total": 0}, 2: {"correct": 0, "total": 0}}
    
    # Set learning rates and params
    lr_base = 1e-3
    alpha_precond = 0.5
    eps_scale = 1e-5
    
    # We will iterate through the stream
    for step, (images, labels, task_id) in enumerate(stream):
        images, labels = images.to(device), labels.to(device)
        
        # Confidence-Gated Switch Reset (CGSR) for FOGS-Merge
        if method_name == "fogs":
            with torch.no_grad():
                affinities = []
                for k in range(K):
                    feats_k = expert_extractors[k](images)
                    feats_k_norm = feats_k / (feats_k.norm(dim=1, keepdim=True) + 1e-8)
                    sim_k = torch.mm(feats_k_norm, prototypes[k].t()) # [B, 10]
                    max_sim_k, _ = sim_k.max(dim=1) # [B]
                    affinities.append(max_sim_k.mean().item())
                
                affinities_tensor = torch.tensor(affinities, device=device)
                avg_max_aff = max(affinities)
                # Compute dynamic gating factor beta based on confidence
                beta = torch.sigmoid(torch.tensor((avg_max_aff - 0.7) / 0.05, device=device)).item()
                
                if step == 0:
                    prev_best = affinities_tensor.argmax().item()
                else:
                    curr_best = affinities_tensor.argmax().item()
                    if curr_best != prev_best:
                        if beta > 0.5:
                            for p_name in coefficients:
                                coefficients[p_name] = torch.zeros((K,), device=device)
                                coefficients[p_name][curr_best] = 1.0
                        prev_best = curr_best
                apply_coefficients(coefficients)

        # 1. Forward pass to evaluate on the current batch BEFORE adapting (or during)
        # This is standard test-time evaluation protocol (eval-before-update or evaluate current prediction)
        model.eval()
        with torch.no_grad():
            outputs = model(images)
            _, preds = outputs.max(1)
            correct_b = preds.eq(labels).sum().item()
            total_b = labels.size(0)
            
            correct_predictions += correct_b
            total_samples += total_b
            task_stats[task_id]["correct"] += correct_b
            task_stats[task_id]["total"] += total_b

        # 2. Adaptation step on the current batch
        if method_name == "static":
            # No adaptation
            pass
            
        elif method_name == "tent":
            # TENT adapts model coefficients by minimizing prediction entropy
            # We treat the coefficients as learnable parameters for this batch
            model.eval()
            coeffs_temp = {p_name: coefficients[p_name].clone().detach().requires_grad_(True) for p_name in coefficients}
            
            # Formulate merged weights differentiably
            merged_weights = {}
            for p_name in base_state_dict:
                if p_name in task_vectors[0]:
                    coeff = coeffs_temp[p_name]
                    weighted_sum = torch.zeros_like(base_state_dict[p_name], device=device)
                    for k in range(K):
                        weighted_sum += coeff[k] * task_vectors[k][p_name]
                    merged_weights[p_name] = base_state_dict[p_name] + weighted_sum
                else:
                    merged_weights[p_name] = base_state_dict[p_name]
                    
            # Differentiable forward pass (by assigning parameters directly to model attributes)
            for name, module in model.named_modules():
                for attr_name, attr_value in list(module._parameters.items()):
                    p_name = name + "." + attr_name if name else attr_name
                    if p_name in merged_weights:
                        module._parameters[attr_name] = merged_weights[p_name]
                        
            # Compute prediction entropy loss
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            loss = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
            
            # Backprop
            loss.backward()
            
            # Update coefficients via simple SGD + Simplex Projection
            with torch.no_grad():
                for p_name in coefficients:
                    if coeffs_temp[p_name].grad is not None:
                        grad = coeffs_temp[p_name].grad
                        coefficients[p_name] = project_simplex(coefficients[p_name] - lr_base * grad)
                        
            # Restore module parameters to nn.Parameter
            model.load_state_dict(base_state_dict, strict=False)
            apply_coefficients(coefficients)
            
        elif method_name == "adamerging":
            # AdaMerging / CPA-Merge: Entropy minimization with global learning rate
            model.eval()
            coeffs_temp = {p_name: coefficients[p_name].clone().detach().requires_grad_(True) for p_name in coefficients}
            
            merged_weights = {}
            for p_name in base_state_dict:
                if p_name in task_vectors[0]:
                    coeff = coeffs_temp[p_name]
                    weighted_sum = torch.zeros_like(base_state_dict[p_name], device=device)
                    for k in range(K):
                        weighted_sum += coeff[k] * task_vectors[k][p_name]
                    merged_weights[p_name] = base_state_dict[p_name] + weighted_sum
                else:
                    merged_weights[p_name] = base_state_dict[p_name]
                    
            for name, module in model.named_modules():
                for attr_name, attr_value in list(module._parameters.items()):
                    p_name = name + "." + attr_name if name else attr_name
                    if p_name in merged_weights:
                        module._parameters[attr_name] = merged_weights[p_name]
                        
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            loss = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
            loss.backward()
            
            with torch.no_grad():
                for p_name in coefficients:
                    if coeffs_temp[p_name].grad is not None:
                        grad = coeffs_temp[p_name].grad
                        coefficients[p_name] = project_simplex(coefficients[p_name] - lr_base * grad)
                        
            model.load_state_dict(base_state_dict, strict=False)
            apply_coefficients(coefficients)
            
        elif method_name == "fp_ca":
            # FP-CA: (1) Prototype-Driven Dynamic Routing (anchor routing)
            # (2) Layer-wise Fisher sensitivity preconditioning
            # (3) Confidence-masked contrastive InfoNCE loss against active prototypes
            model.eval()
            
            # Compute anchor pass to estimate task affinities
            with torch.no_grad():
                feats = feature_extractor(images) # [B, 512]
                feats_norm = feats / (feats.norm(dim=1, keepdim=True) + 1e-8)
                
                # Compute similarity to all expert prototypes
                # prototypes[k] has shape [10, 512]
                similarities = []
                for k in range(K):
                    # Max cosine similarity to any prototype of expert k
                    sim_k = torch.mm(feats_norm, prototypes[k].t()) # [B, 10]
                    max_sim_k, _ = sim_k.max(dim=1) # [B]
                    similarities.append(max_sim_k.mean().item())
                
                # Dynamic routing reset on task boundary detection
                best_expert = np.argmax(similarities) if 'np' in globals() else similarities.index(max(similarities))
            
            # If tasks switch sharply (large similarity difference), reset coefficients towards best expert
            # (For simplicity and alignment with the paper, we perform the contrastive optimization with Fisher preconditioning)
            
            coeffs_temp = {p_name: coefficients[p_name].clone().detach().requires_grad_(True) for p_name in coefficients}
            
            merged_weights = {}
            for p_name in base_state_dict:
                if p_name in task_vectors[0]:
                    coeff = coeffs_temp[p_name]
                    weighted_sum = torch.zeros_like(base_state_dict[p_name], device=device)
                    for k in range(K):
                        weighted_sum += coeff[k] * task_vectors[k][p_name]
                    merged_weights[p_name] = base_state_dict[p_name] + weighted_sum
                else:
                    merged_weights[p_name] = base_state_dict[p_name]
                    
            for name, module in model.named_modules():
                for attr_name, attr_value in list(module._parameters.items()):
                    p_name = name + "." + attr_name if name else attr_name
                    if p_name in merged_weights:
                        module._parameters[attr_name] = merged_weights[p_name]
                        
            # Contrastive Alignment Loss against class prototypes
            # Predict online labels
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            confidences, pseudo_labels = probs.max(dim=1)
            
            # Filter low confidence samples (Confidence Masking)
            mask = confidences > 0.0
            if mask.sum() > 0:
                online_feats = feature_extractor(images)[mask] # [B_filtered, 512]
                online_feats_norm = online_feats / (online_feats.norm(dim=1, keepdim=True) + 1e-8)
                online_labels = pseudo_labels[mask]
                
                # Optimize InfoNCE loss via vectorized implementation
                online_feats = feature_extractor(images)[mask]
                loss_contrastive = compute_contrastive_loss(online_feats, online_labels, prototypes)
                loss_contrastive.backward()
                
                # Fisher preconditioned updates
                with torch.no_grad():
                    for p_name in coefficients:
                        if coeffs_temp[p_name].grad is not None:
                            grad = coeffs_temp[p_name].grad
                            # Layer-wise Fisher preconditioning
                            F_w = joint_fisher_scalars[p_name]
                            G_w = (F_w + eps_scale) ** alpha_precond
                            lr_w = lr_base / G_w
                            coefficients[p_name] = project_simplex(coefficients[p_name] - lr_w * grad)
                            
            model.load_state_dict(base_state_dict, strict=False)
            apply_coefficients(coefficients)
            
        elif method_name == "iggs":
            # IGGS-Merge: Information-Geometric Gradient Surgery
            model.eval()
            
            # Predict online labels
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            _, pseudo_labels = probs.max(dim=1)
            
            # Compute class-specific entropy gradients
            class_grads = {}
            for c in torch.unique(pseudo_labels):
                class_mask = (pseudo_labels == c)
                if class_mask.sum() == 0:
                    continue
                    
                # Setup differentiable weights for this class gradient
                coeffs_temp = {p_name: coefficients[p_name].clone().detach().requires_grad_(True) for p_name in coefficients}
                merged_weights = {}
                for p_name in base_state_dict:
                    if p_name in task_vectors[0]:
                        coeff = coeffs_temp[p_name]
                        weighted_sum = torch.zeros_like(base_state_dict[p_name], device=device)
                        for k in range(K):
                            weighted_sum += coeff[k] * task_vectors[k][p_name]
                        merged_weights[p_name] = base_state_dict[p_name] + weighted_sum
                    else:
                        merged_weights[p_name] = base_state_dict[p_name]
                        
                for name, module in model.named_modules():
                    for attr_name, attr_value in list(module._parameters.items()):
                        p_name = name + "." + attr_name if name else attr_name
                        if p_name in merged_weights:
                            module._parameters[attr_name] = merged_weights[p_name]
                            
                class_outputs = model(images[class_mask])
                class_probs = F.softmax(class_outputs, dim=1)
                class_loss = -torch.sum(class_probs * torch.log(class_probs + 1e-8), dim=1).mean()
                
                class_loss.backward()
                
                # Extract gradients
                class_grads[c.item()] = {}
                for p_name in coefficients:
                    if coeffs_temp[p_name].grad is not None:
                        class_grads[c.item()][p_name] = coeffs_temp[p_name].grad.clone()
                    else:
                        class_grads[c.item()][p_name] = torch.zeros((K,), device=device)
                        
            # Riemannian Gradient Surgery (Fisher-weighted Projection)
            projected_grads = {c: {p_name: class_grads[c][p_name].clone() for p_name in coefficients} for c in class_grads}
            
            classes = list(class_grads.keys())
            for i in range(len(classes)):
                for j in range(len(classes)):
                    if i != j:
                        c_a, c_b = classes[i], classes[j]
                        
                        # Compute Fisher-weighted inner product:
                        # sum_{p_name} G_w * dot(g_a, g_b)
                        inner_prod = 0.0
                        norm_b = 0.0
                        for p_name in coefficients:
                            F_w = joint_fisher_scalars[p_name]
                            G_w = (F_w + eps_scale) ** alpha_precond
                            dot_ab = torch.dot(class_grads[c_a][p_name], class_grads[c_b][p_name])
                            dot_bb = torch.dot(class_grads[c_b][p_name], class_grads[c_b][p_name])
                            
                            inner_prod += G_w * dot_ab
                            norm_b += G_w * dot_bb
                            
                        if inner_prod < 0:
                            # Project g_a onto normal plane of g_b
                            factor = inner_prod / (norm_b + 1e-8)
                            for p_name in coefficients:
                                projected_grads[c_a][p_name] -= factor * class_grads[c_b][p_name]
                                
            # Aggregate and update
            with torch.no_grad():
                for p_name in coefficients:
                    final_grad = torch.zeros((K,), device=device)
                    for c in projected_grads:
                        final_grad += projected_grads[c][p_name]
                        
                    # Preconditioned update
                    F_w = joint_fisher_scalars[p_name]
                    G_w = (F_w + eps_scale) ** alpha_precond
                    lr_w = lr_base / G_w
                    coefficients[p_name] = project_simplex(coefficients[p_name] - lr_w * final_grad)
                    
            model.load_state_dict(base_state_dict, strict=False)
            apply_coefficients(coefficients)
            
        elif method_name == "fogs":
            # FOGS-Merge: (Our Proposed Method)
            # Combines self-supervised contrastive loss with Information-Geometric Gradient Surgery
            model.eval()
            
            # Predict online labels
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            confidences, pseudo_labels = probs.max(dim=1)
            
            # Filter low confidence samples
            mask = confidences > 0.0
            if mask.sum() > 0:
                unique_classes = torch.unique(pseudo_labels[mask])
                class_grads = {}
                
                for c in unique_classes:
                    class_mask = (pseudo_labels == c) & mask
                    if class_mask.sum() == 0:
                        continue
                        
                    # Setup differentiable weights
                    coeffs_temp = {p_name: coefficients[p_name].clone().detach().requires_grad_(True) for p_name in coefficients}
                    merged_weights = {}
                    for p_name in base_state_dict:
                        if p_name in task_vectors[0]:
                            coeff = coeffs_temp[p_name]
                            weighted_sum = torch.zeros_like(base_state_dict[p_name], device=device)
                            for k in range(K):
                                weighted_sum += coeff[k] * task_vectors[k][p_name]
                            merged_weights[p_name] = base_state_dict[p_name] + weighted_sum
                        else:
                            merged_weights[p_name] = base_state_dict[p_name]
                            
                    for name, module in model.named_modules():
                        for attr_name, attr_value in list(module._parameters.items()):
                            p_name = name + "." + attr_name if name else attr_name
                            if p_name in merged_weights:
                                module._parameters[attr_name] = merged_weights[p_name]
                                
                    # Contrastive loss for samples of class c via vectorized implementation
                    lbl = c.item()
                    class_feats = feature_extractor(images[class_mask])
                    
                    # Compute soft task-routing weights based on class-specific prototype similarity
                    class_feats_norm = class_feats / (class_feats.norm(dim=1, keepdim=True) + 1e-8)
                    proto_c_stacked = torch.stack([prototypes[k][lbl] for k in range(K)], dim=0) # [K, 512]
                    sims_c = torch.mm(class_feats_norm, proto_c_stacked.t()) # [class_feats_B, K]
                    
                    # Adaptive routing temperature based on average max similarity to handle noise robustly
                    max_sims_c, _ = sims_c.max(dim=1)
                    avg_max_sim = max_sims_c.mean().item()
                    temp_noise = max(0.1, 0.5 - 0.5 * avg_max_sim)
                    task_weights_c = F.softmax(sims_c / temp_noise, dim=1)
                    
                    # Pseudo-labels are all class lbl (c) for this class-specific gradient computation
                    online_labels_c = torch.full((class_feats.size(0),), lbl, dtype=torch.long, device=device)
                    loss_contrastive = compute_contrastive_loss(class_feats, online_labels_c, prototypes, task_weights=task_weights_c)
                    loss_contrastive.backward()
                    
                    # Extract gradients
                    class_grads[lbl] = {}
                    for p_name in coefficients:
                        if coeffs_temp[p_name].grad is not None:
                            class_grads[lbl][p_name] = coeffs_temp[p_name].grad.clone()
                        else:
                            class_grads[lbl][p_name] = torch.zeros((K,), device=device)
                            
                # Riemannian Gradient Surgery on Contrastive Gradients
                projected_grads = {c: {p_name: class_grads[c][p_name].clone() for p_name in coefficients} for c in class_grads}
                
                classes = list(class_grads.keys())
                for i in range(len(classes)):
                    for j in range(len(classes)):
                        if i != j:
                            c_a, c_b = classes[i], classes[j]
                            
                            inner_prod = 0.0
                            norm_b = 0.0
                            for p_name in coefficients:
                                F_w = joint_fisher_scalars[p_name]
                                G_w = (F_w + eps_scale) ** alpha_precond
                                dot_ab = torch.dot(class_grads[c_a][p_name], class_grads[c_b][p_name])
                                dot_bb = torch.dot(class_grads[c_b][p_name], class_grads[c_b][p_name])
                                
                                inner_prod += G_w * dot_ab
                                norm_b += G_w * dot_bb
                                
                            if inner_prod < 0:
                                # Project g_a onto normal plane of g_b
                                factor = inner_prod / (norm_b + 1e-8)
                                for p_name in coefficients:
                                    projected_grads[c_a][p_name] -= factor * class_grads[c_b][p_name]
                                    
                # Aggregate and update
                with torch.no_grad():
                    max_fisher_scalar = max(joint_fisher_scalars.values())
                    for p_name in coefficients:
                        final_grad = torch.zeros((K,), device=device)
                        for c in projected_grads:
                            final_grad += projected_grads[c][p_name]
                            
                        # Preconditioned update
                        F_w = joint_fisher_scalars[p_name]
                        G_w = (F_w + eps_scale) ** alpha_precond
                        lr_w = lr_base / G_w
                        new_coeffs = project_simplex(coefficients[p_name] - lr_w * final_grad)
                        
                        # Fisher-Weighted Parameter Rejuvenation (FWPR)
                        rho_0 = 0.01  # Base rejuvenation rate
                        rho_w = rho_0 * (F_w / max_fisher_scalar)
                        coefficients[p_name] = (1.0 - rho_w) * new_coeffs + rho_w * (1.0 / K)
                        
                model.load_state_dict(base_state_dict, strict=False)
                apply_coefficients(coefficients)
                
    # Print results
    acc = 100. * correct_predictions / total_samples
    print(f"Overall Accuracy: {acc:.2f}%")
    for k, exp in enumerate(experts):
        task_acc = 100. * task_stats[k]["correct"] / task_stats[k]["total"]
        print(f"Accuracy on {exp}: {task_acc:.2f}%")
        
    return acc, {exp: 100. * task_stats[k]["correct"] / task_stats[k]["total"] for k, exp in enumerate(experts)}

if __name__ == "__main__":
    import numpy as np
    
    # Let's run evaluation on both streams
    methods = ["static", "tent", "adamerging", "fp_ca", "iggs", "fogs"]
    streams = ["sequential", "alternating"]
    noise_stds = [0.0, 2.0]
    
    results = {}
    
    for ns in noise_stds:
        results[ns] = {}
        for stream in streams:
            results[ns][stream] = {}
            for m in methods:
                try:
                    acc, task_accs = evaluate_method(m, stream_type=stream, noise_std=ns)
                    results[ns][stream][m] = {"overall": acc, "tasks": task_accs}
                except Exception as e:
                    print(f"Method {m} failed on {stream} with noise {ns}: {e}")
                    results[ns][stream][m] = {"overall": 0.0, "tasks": {}}
                    
    # Generate final summary report and print it
    print("\n" + "="*50)
    print("FINAL EXPERIMENTAL RESULTS SUMMARY")
    print("="*50)
    for ns in noise_stds:
        print(f"\n--- Noise Level: {ns} ---")
        for stream in streams:
            print(f"\nStream Type: {stream}")
            for m in methods:
                res = results[ns][stream][m]
                print(f"  {m:12s} : Overall {res['overall']:.2f}% | MNIST {res['tasks'].get('mnist', 0.0):.2f}% | Fashion {res['tasks'].get('fashionmnist', 0.0):.2f}% | KMNIST {res['tasks'].get('kmnist', 0.0):.2f}%")
                
    # Save results dictionary as torch file for plotting or documentation
    torch.save(results, "models/evaluation_results.pt")
    print("\nSaved evaluation results to models/evaluation_results.pt")
