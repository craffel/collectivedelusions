import os
import copy
import random
import numpy as np
import torch
torch.backends.cudnn.enabled = False
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

# Set seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Evaluation script running on device: {device}")

# Define transforms
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load datasets
def get_dataset(name, train=False):
    if name == 'MNIST':
        return torchvision.datasets.MNIST(root='./data', train=train, download=False, transform=transform)
    elif name == 'FashionMNIST':
        return torchvision.datasets.FashionMNIST(root='./data', train=train, download=False, transform=transform)
    elif name == 'KMNIST':
        return torchvision.datasets.KMNIST(root='./data', train=train, download=False, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {name}")

# Construct non-stationary test streams
def construct_stream(stream_type='sequential', num_batches=60, batch_size=64):
    """
    Constructs a stream of batches.
    stream_type: 'sequential' (large blocks of MNIST, then FashionMNIST, then KMNIST)
                 'alternating' (MNIST, FashionMNIST, KMNIST, MNIST, FashionMNIST, ...)
    """
    datasets = {
        0: get_dataset('MNIST', train=False),
        1: get_dataset('FashionMNIST', train=False),
        2: get_dataset('KMNIST', train=False)
    }
    
    # Create loaders for each dataset to pull batches from
    loaders = {
        i: DataLoader(datasets[i], batch_size=batch_size, shuffle=True, drop_last=True)
        for i in range(3)
    }
    
    iters = {i: iter(loaders[i]) for i in range(3)}
    
    stream = []
    
    if stream_type == 'alternating':
        for b in range(num_batches):
            task_id = b % 3
            try:
                images, labels = next(iters[task_id])
            except StopIteration:
                iters[task_id] = iter(loaders[task_id])
                images, labels = next(iters[task_id])
            stream.append((images, labels, task_id))
            
    elif stream_type == 'sequential':
        block_size = num_batches // 3
        for task_id in range(3):
            for _ in range(block_size):
                try:
                    images, labels = next(iters[task_id])
                except StopIteration:
                    iters[task_id] = iter(loaders[task_id])
                    images, labels = next(iters[task_id])
                stream.append((images, labels, task_id))
                
    return stream

# Apply environmental corruptions to a batch of images
def apply_corruption(images, corruption_type='none', severity=1):
    if corruption_type == 'none':
        return images
    
    images = images.clone()
    if corruption_type == 'gaussian_noise':
        # Add random normal noise (severity maps to std of noise)
        noise_std = 0.15 * severity
        noise = torch.randn_like(images) * noise_std
        images = images + noise
        images = torch.clamp(images, -1.0, 1.0)
        
    elif corruption_type == 'gaussian_blur':
        # Apply Gaussian Blur (severity maps to kernel size & sigma)
        kernel_size = 2 * severity + 1
        sigma = 0.5 * severity + 0.5
        blur = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
        images = blur(images)
        
    elif corruption_type == 'contrast':
        # Apply contrast reduction/enhancement
        factor = 1.0 - 0.25 * severity if severity > 0 else 1.0
        images = transforms.functional.adjust_contrast(images, factor)
        
    return images

# Helper: Compute Entropy
def compute_entropy(probs, eps=1e-8):
    return -torch.sum(probs * torch.log(probs + eps), dim=-1)

# Helper: Projection onto the simplex (for CPA-Merge and custom simplex updates)
def project_simplex(v):
    """v is a 1D tensor of raw weights"""
    n_features = v.size(0)
    u, _ = torch.sort(v, descending=True)
    cssv = torch.cumsum(u, dim=0)
    ind = torch.arange(n_features, device=v.device) + 1
    cond = u - (cssv - 1.0) / ind > 0
    # find maximum index satisfying the condition
    # cond is a boolean tensor; argmax on its flipped version is not safe
    # we can find the last index that is True
    nonzero = torch.nonzero(cond)
    if len(nonzero) == 0:
        rho = 0
    else:
        rho = nonzero[-1].item()
    theta = (cssv[rho] - 1.0) / (rho + 1)
    w = torch.clamp(v - theta, min=0)
    return w

# Base / Expert weights loading
def load_models():
    # Base model
    base_model = models.resnet18()
    base_model.fc = nn.Linear(512, 10)
    base_model.load_state_dict(torch.load("base_model.pt", map_location=device))
    base_model = base_model.to(device)
    base_model.eval()
    
    # Expert models
    experts = []
    task_names = ['mnist', 'fashionmnist', 'kmnist']
    for name in task_names:
        expert = models.resnet18()
        expert.fc = nn.Linear(512, 10)
        expert.load_state_dict(torch.load(f"expert_{name}.pt", map_location=device))
        expert = expert.to(device)
        expert.eval()
        experts.append(expert)
        
    return base_model, experts

# Step 2: Compute Diagonal Empirical Fisher Information
def compute_fisher_sensitivity(base_model, experts):
    print("Computing diagonal Fisher Information sensitivity for each expert...")
    joint_fisher = {}
    
    task_classes = [
        torchvision.datasets.MNIST,
        torchvision.datasets.FashionMNIST,
        torchvision.datasets.KMNIST
    ]
    
    for k, expert in enumerate(experts):
        dataset_class = task_classes[k]
        cal_dataset = dataset_class(root='./data', train=True, download=False, transform=transform)
        # Take a subset of 256 calibration samples
        torch.manual_seed(42)
        indices = torch.randperm(len(cal_dataset))[:256].tolist()
        cal_subset = Subset(cal_dataset, indices)
        cal_loader = DataLoader(cal_subset, batch_size=32, shuffle=False)
        
        # Set to eval but keep tracking gradients
        expert.eval()
        
        # Initialize Fisher accumulator for parameters
        fisher_accum = {name: torch.zeros_like(p) for name, p in expert.named_parameters() if p.requires_grad and not name.startswith("fc.")}
        
        criterion = nn.CrossEntropyLoss()
        count = 0
        
        for images, labels in cal_loader:
            images, labels = images.to(device), labels.to(device)
            expert.zero_grad()
            outputs = expert(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            with torch.no_grad():
                for name, p in expert.named_parameters():
                    if name in fisher_accum:
                        if p.grad is not None:
                            fisher_accum[name] += (p.grad ** 2) * images.size(0)
            count += images.size(0)
            
        # Average and compute average layer-wise scalar sensitivity
        for name in fisher_accum:
            fisher_accum[name] /= count
            # Average over all individual parameters in the tensor
            tensor_avg = fisher_accum[name].mean().item()
            if name not in joint_fisher:
                joint_fisher[name] = []
            joint_fisher[name].append(tensor_avg)
            
    # Compute the joint layer-wise Fisher sensitivity (average across experts)
    final_joint_fisher = {}
    for name in joint_fisher:
        final_joint_fisher[name] = sum(joint_fisher[name]) / len(joint_fisher[name])
        
    return final_joint_fisher

# Step 3: Compute Class Prototypes for CPA-Merge
def extract_class_prototypes(experts):
    print("Extracting class prototypes for CPA-Merge routing...")
    prototypes = {k: {} for k in range(3)} # task -> class -> prototype_vector
    
    task_classes = [
        torchvision.datasets.MNIST,
        torchvision.datasets.FashionMNIST,
        torchvision.datasets.KMNIST
    ]
    
    for k, expert in enumerate(experts):
        # We need to extract the backbone embeddings (inputs to the fc layer)
        # We can register a hook or bypass the fc forward
        # Let's write a simple helper to extract embeddings
        dataset_class = task_classes[k]
        cal_dataset = dataset_class(root='./data', train=True, download=False, transform=transform)
        
        torch.manual_seed(42)
        indices = torch.randperm(len(cal_dataset))[:512].tolist()
        cal_subset = Subset(cal_dataset, indices)
        cal_loader = DataLoader(cal_subset, batch_size=64, shuffle=False)
        
        expert.eval()
        
        # Store class embeddings
        class_embeddings = {c: [] for c in range(10)}
        
        # Feature extraction function
        def get_features(images):
            # Pass through ResNet-18 layers excluding fc
            with torch.no_grad():
                x = expert.conv1(images)
                x = expert.bn1(x)
                x = expert.relu(x)
                x = expert.maxpool(x)
                x = expert.layer1(x)
                x = expert.layer2(x)
                x = expert.layer3(x)
                x = expert.layer4(x)
                x = expert.avgpool(x)
                x = torch.flatten(x, 1)
                return x
                
        for images, labels in cal_loader:
            images = images.to(device)
            feats = get_features(images)
            for feat, label in zip(feats, labels):
                c = label.item()
                class_embeddings[c].append(feat.cpu())
                
        # Compute mean and normalize
        for c in range(10):
            if len(class_embeddings[c]) > 0:
                mean_vec = torch.stack(class_embeddings[c]).mean(dim=0)
                mean_vec = mean_vec / (mean_vec.norm(p=2) + 1e-8)
                prototypes[k][c] = mean_vec.to(device)
            else:
                # Fallback to zero vector if class is missing in subset
                prototypes[k][c] = torch.zeros(512, device=device)
                
    return prototypes

# Evaluation function
def run_evaluation(base_model, experts, stream, fisher_sens=None, prototypes=None, method='uniform', lr=0.01, severity=1, corruption_type='none'):
    from torch.func import functional_call
    # Clone models to prevent side-effects
    base_model = copy.deepcopy(base_model)
    experts = [copy.deepcopy(e) for e in experts]
    
    # Mock fc layer of base_model with Identity for functional_call feature extraction
    base_model.fc = nn.Identity()
    
    # Identify mergible tensors
    mergible_keys = [name for name, p in base_model.named_parameters() if p.requires_grad and not name.startswith("fc.")]
    # Also merge BN running mean/var and track parameters
    # Let's get all state_dict keys that are parameters or buffers of the backbone
    state_dict_keys = [k for k in base_model.state_dict().keys() if not k.startswith("fc.") and torch.is_floating_point(base_model.state_dict()[k])]
    
    # Initialize merging coefficients for each mergible tensor
    # K=3 experts. We initialize to uniform routing
    raw_weights = {}
    for k in state_dict_keys:
        raw_weights[k] = torch.zeros(3, device=device, requires_grad=True)
        
    # Optimizer for raw_weights (if adapting)
    optimizer = optim.SGD(list(raw_weights.values()), lr=lr)
    
    # Keep track of metrics
    correct_samples = 0
    total_samples = 0
    loss_history = []
    
    # EMA loss track (for PC-Merge OPR)
    ema_loss = None
    beta_ema = 0.9
    
    # For CPA-Merge confidence threshold
    cpa_conf_threshold = 0.85
    
    # Setup baseline metrics
    for step, (images, labels, task_id) in enumerate(stream):
        images, labels = images.to(device), labels.to(device)
        
        # Apply environmental corruptions
        images = apply_corruption(images, corruption_type, severity)
        
        # 1. PRE-STEP ADAPTATION/RESET LOGIC (PC-Merge / CPA-Merge)
        
        # CPA-Merge: PD-Routing
        if method == 'cpa_merge' and prototypes is not None:
            # Anchor pass through uniform merged model
            # Construct a uniform merged state dict for anchor pass
            anchor_state_dict = copy.deepcopy(base_model.state_dict())
            uniform_w = torch.tensor([1/3, 1/3, 1/3], device=device)
            for k in state_dict_keys:
                merged_tensor = anchor_state_dict[k].clone()
                merged_tensor.zero_()
                for exp_idx, exp in enumerate(experts):
                    diff = exp.state_dict()[k].to(device) - base_model.state_dict()[k].to(device)
                    merged_tensor += uniform_w[exp_idx] * diff
                anchor_state_dict[k] = base_model.state_dict()[k].to(device) + merged_tensor
                
            # Create anchor model and extract features
            anchor_model = copy.deepcopy(base_model)
            anchor_model.load_state_dict(anchor_state_dict)
            anchor_model.eval()
            
            with torch.no_grad():
                # Extract features
                x = anchor_model.conv1(images)
                x = anchor_model.bn1(x)
                x = anchor_model.relu(x)
                x = anchor_model.maxpool(x)
                x = anchor_model.layer1(x)
                x = anchor_model.layer2(x)
                x = anchor_model.layer3(x)
                x = anchor_model.layer4(x)
                x = anchor_model.avgpool(x)
                z_anchor = torch.flatten(x, 1)
                z_anchor = z_anchor / (z_anchor.norm(p=2, dim=1, keepdim=True) + 1e-8)
                
            # Compute similarity with prototypes for each task
            task_scores = torch.zeros(3, device=device)
            for exp_idx in range(3):
                # max cosine similarity across 10 classes, averaged over batch
                max_sims = []
                for i in range(images.size(0)):
                    sample_vec = z_anchor[i]
                    max_sim = -1.0
                    for c in range(10):
                        proto_vec = prototypes[exp_idx][c]
                        sim = torch.dot(sample_vec, proto_vec).item()
                        if sim > max_sim:
                            max_sim = sim
                    max_sims.append(max_sim)
                task_scores[exp_idx] = sum(max_sims) / len(max_sims)
                
            # Software routing via low temperature softmax
            routing_prior = torch.softmax(task_scores / 0.02, dim=0)
            
            # Reset active raw coefficients to the log of routing prior (to match softmax output)
            with torch.no_grad():
                for k in state_dict_keys:
                    # Clear gradient history
                    if raw_weights[k].grad is not None:
                        raw_weights[k].grad.zero_()
                    # Initialize raw weights directly to map to routing_prior
                    # Since we use softmax, raw_weights = log(routing_prior) + C
                    raw_weights[k].copy_(torch.log(routing_prior + 1e-8))
                    
        # 2. CONSTRUCT MERGED MODEL
        # Softmax Model Merging using functional_call for differentiability
        merged_params_and_buffers = {}
        
        # Merge parameters (differentiable)
        for k, v in base_model.named_parameters():
            if k in state_dict_keys:
                norm_w = torch.softmax(raw_weights[k], dim=0)
                merged_tensor = torch.zeros_like(v)
                for exp_idx, exp in enumerate(experts):
                    diff = exp.state_dict()[k].to(device) - base_model.state_dict()[k].to(device)
                    merged_tensor = merged_tensor + norm_w[exp_idx] * diff
                merged_params_and_buffers[k] = base_model.state_dict()[k].to(device) + merged_tensor
            else:
                merged_params_and_buffers[k] = v.to(device)
                
        # Merge buffers (non-differentiable)
        with torch.no_grad():
            for k, v in base_model.named_buffers():
                if k in state_dict_keys:
                    norm_w = torch.softmax(raw_weights[k], dim=0)
                    merged_tensor = torch.zeros_like(v, dtype=torch.float32)
                    for exp_idx, exp in enumerate(experts):
                        diff = exp.state_dict()[k].to(device) - base_model.state_dict()[k].to(device)
                        merged_tensor = merged_tensor + norm_w[exp_idx] * diff
                    merged_params_and_buffers[k] = base_model.state_dict()[k].to(device) + merged_tensor
                else:
                    merged_params_and_buffers[k] = v.to(device)
                    
        # 3. COMPUTE ADAPTATION LOSS
        # Head selection: use true task head for evaluation (as standard)
        # In this multi-task setup, we evaluate utilizing the head of the current task
        active_head = experts[task_id].fc
        
        # Extract features differentiably using functional_call
        features = functional_call(base_model, merged_params_and_buffers, images)
        
        outputs = active_head(features)
        probs = torch.softmax(outputs, dim=-1)
        
        # Prediction entropy loss
        batch_entropy = compute_entropy(probs).mean()
        
        # Loss for optimization
        loss = batch_entropy
        
        # CPA-Merge: Confidence-Masked Contrastive Alignment
        if method == 'cpa_merge' and prototypes is not None:
            max_probs, preds = probs.max(dim=1)
            high_conf_mask = max_probs > cpa_conf_threshold
            masked_indices = torch.nonzero(high_conf_mask).squeeze(1)
            
            if len(masked_indices) > 0:
                # Calculate InfoNCE contrastive loss with active task prototypes
                z_norm = features / (features.norm(p=2, dim=1, keepdim=True) + 1e-8)
                contra_loss = 0.0
                for idx in masked_indices:
                    sample_feat = z_norm[idx]
                    pred_class = preds[idx].item()
                    # Cosine similarities to all classes of active task prototypes
                    sim_c = torch.zeros(10, device=device)
                    for c in range(10):
                        sim_c[c] = torch.dot(sample_feat, prototypes[task_id][c])
                    # InfoNCE loss
                    pos_score = sim_c[pred_class] / 0.1
                    sum_exp_scores = torch.logsumexp(sim_c / 0.1, dim=0)
                    contra_loss += -(pos_score - sum_exp_scores)
                contra_loss /= len(masked_indices)
                loss = batch_entropy + 0.1 * contra_loss
                
        # 4. PERFORM GRADIENT STEP
        if method != 'uniform':
            optimizer.zero_grad()
            if method != 'pc_merge':
                loss.backward()
            
            # Optimization adjustments depending on the method
            
            # LFWA (Layer-wise Fisher-Weighted Adaptation)
            if method == 'lfwa' and fisher_sens is not None:
                with torch.no_grad():
                    for k in state_dict_keys:
                        if raw_weights[k].grad is not None:
                            # Apply dynamic damping via layer-wise Fisher weighting
                            # η_w = η * (F_w + ε)^(-α). Let's use α = 0.5, ε = 1e-5
                            F_w = fisher_sens.get(k, 0.0)
                            damping_factor = (F_w + 1e-5) ** (-0.5)
                            raw_weights[k].grad *= damping_factor
                            
            # PC-Merge (Class-Specific Gradient Surgery + OPR)
            elif method == 'pc_merge':
                # Clear standard gradient and do class-specific PCGrad
                # Note: For strict implementation, we would group samples by class and do pairwise projections.
                # Since we want to resolve gradient conflicts, let's implement PCGrad on the raw coefficients!
                # Group samples by predicted classes
                with torch.no_grad():
                    max_probs, preds = probs.max(dim=1)
                
                class_grads = {}
                for c in range(10):
                    c_mask = (preds == c)
                    if c_mask.sum() == 0:
                        continue
                    # Compute class-specific entropy loss
                    c_loss = compute_entropy(probs[c_mask]).mean()
                    
                    # Zero grads
                    optimizer.zero_grad()
                    c_loss.backward(retain_graph=True)
                    
                    # Store gradients of raw_weights
                    grads_c = {k: raw_weights[k].grad.clone() for k in state_dict_keys if raw_weights[k].grad is not None}
                    class_grads[c] = grads_c
                
                # Pairwise projection (Surgery)
                active_classes = list(class_grads.keys())
                projected_grads = {k: torch.zeros_like(raw_weights[k]) for k in state_dict_keys}
                
                if len(active_classes) > 0:
                    for k in state_dict_keys:
                        if k not in class_grads[active_classes[0]]:
                            continue
                        # Gather gradients for this tensor across classes
                        g_list = [class_grads[c][k].clone() for c in active_classes]
                        # PCGrad pairwise surgery
                        for i in range(len(g_list)):
                            random.shuffle(g_list) # Randomize order for symmetry
                        for i in range(len(g_list)):
                            for j in range(len(g_list)):
                                if i != j:
                                    dot_prod = torch.dot(g_list[i], g_list[j])
                                    if dot_prod < 0:
                                        # Project g_i onto normal plane of g_j
                                        g_list[i] -= (dot_prod / (torch.norm(g_list[j])**2 + 1e-8)) * g_list[j]
                        # Final gradient is the sum
                        for g in g_list:
                            projected_grads[k] += g
                            
                # Apply the projected gradients back to raw_weights
                with torch.no_grad():
                    optimizer.zero_grad()
                    for k in state_dict_keys:
                        raw_weights[k].grad = projected_grads[k].clone()
                            
            # EWFR-Merge (Our Proposed Method)
            elif method == 'ewfr_merge' and fisher_sens is not None:
                # L_total = L_ent + Σ γ_w * (λ_w - λ_init)^2
                # where γ_w = γ_0 * F_w * H(Xt)
                norm_entropy = batch_entropy.item() / np.log(10.0)
                gamma_0 = 10.0
                
                with torch.no_grad():
                    for k in state_dict_keys:
                        if raw_weights[k].grad is not None:
                            # F_w is the joint Fisher sensitivity of tensor k
                            F_w = fisher_sens.get(k, 0.0)
                            # Strength of regularization pulling back to uniform (0.0 for raw weights)
                            reg_strength = gamma_0 * F_w * norm_entropy
                            # Add regularization gradient: 2 * reg_strength * (λ - λ_init)
                            raw_weights[k].grad += 2.0 * reg_strength * raw_weights[k]
                            
                            # Apply layer-wise scaling (LFWA style) to stabilize high lr
                            damping_factor = (F_w + 1e-5) ** (-0.5)
                            raw_weights[k].grad *= damping_factor
                            
            # Update coefficients
            optimizer.step()
            
            # Post-step OPR boundary check (PC-Merge)
            if method == 'pc_merge':
                # Loss spike detection for task boundary
                current_loss_val = batch_entropy.item()
                if ema_loss is None:
                    ema_loss = current_loss_val
                else:
                    # Check for loss spike
                    threshold = 2.5 # corrupted threshold is lower
                    if current_loss_val > threshold * ema_loss:
                        # OPR trigger: reset coefficients to uniform (0) and clear momentum
                        with torch.no_grad():
                            for k in state_dict_keys:
                                raw_weights[k].zero_()
                                if raw_weights[k].grad is not None:
                                    raw_weights[k].grad.zero_()
                        ema_loss = current_loss_val
                    else:
                        ema_loss = beta_ema * ema_loss + (1.0 - beta_ema) * current_loss_val
                        
        # 5. EVALUATE CORRECTNESS ON ACTUAL LABEL
        # Predict on active head
        with torch.no_grad():
            # Reconstruct model with newly updated weights for clean forward using functional_call
            eval_params_and_buffers = {}
            for k, v in base_model.named_parameters():
                if k in state_dict_keys:
                    norm_w = torch.softmax(raw_weights[k], dim=0)
                    merged_tensor = torch.zeros_like(v)
                    for exp_idx, exp in enumerate(experts):
                        diff = exp.state_dict()[k].to(device) - base_model.state_dict()[k].to(device)
                        merged_tensor = merged_tensor + norm_w[exp_idx] * diff
                    eval_params_and_buffers[k] = base_model.state_dict()[k].to(device) + merged_tensor
                else:
                    eval_params_and_buffers[k] = v.to(device)
            for k, v in base_model.named_buffers():
                if k in state_dict_keys:
                    norm_w = torch.softmax(raw_weights[k], dim=0)
                    merged_tensor = torch.zeros_like(v, dtype=torch.float32)
                    for exp_idx, exp in enumerate(experts):
                        diff = exp.state_dict()[k].to(device) - base_model.state_dict()[k].to(device)
                        merged_tensor = merged_tensor + norm_w[exp_idx] * diff
                    eval_params_and_buffers[k] = base_model.state_dict()[k].to(device) + merged_tensor
                else:
                    eval_params_and_buffers[k] = v.to(device)
            
            eval_features = functional_call(base_model, eval_params_and_buffers, images)
            eval_outputs = experts[task_id].fc(eval_features)
            _, preds = eval_outputs.max(1)
            correct_samples += preds.eq(labels).sum().item()
            total_samples += labels.size(0)
            loss_history.append(batch_entropy.item())
            
    avg_accuracy = 100.0 * correct_samples / total_samples
    avg_loss = sum(loss_history) / len(loss_history)
    return avg_accuracy, avg_loss

def main():
    print("Loading models...")
    base_model, experts = load_models()
    
    # Compute Fisher sensitivity for LFWA and EWFR
    fisher_sens = compute_fisher_sensitivity(base_model, experts)
    
    # Extract prototypes for CPA-Merge
    prototypes = extract_class_prototypes(experts)
    
    # Define experiment protocols
    # Stream types: alternating, sequential
    # Corruptions: none, gaussian_noise, gaussian_blur, contrast
    # Severities: 1, 2, 3
    
    stream_types = ['alternating', 'sequential']
    corruptions = [
        ('none', 0),
        ('gaussian_noise', 2),
        ('gaussian_blur', 2),
        ('contrast', 2)
    ]
    
    methods = ['uniform', 'adamerging', 'lfwa', 'pc_merge', 'cpa_merge', 'ewfr_merge']
    
    results = {}
    
    # Let's run a comprehensive evaluation sweep
    for stream_type in stream_types:
        results[stream_type] = {}
        print(f"\n========================================\nStream Type: {stream_type.upper()}\n========================================")
        
        # We run 30 batches for alternating, 30 batches for sequential (10 of each dataset)
        num_batches = 30 if stream_type == 'alternating' else 30
        batch_size = 64
        
        for corr_type, severity in corruptions:
            results[stream_type][corr_type] = {}
            print(f"\nCorruption: {corr_type.upper()} (Severity: {severity})")
            
            # Construct test stream (keep it same for all methods in the comparison)
            set_seed(42) # Ensure identical batch choices and shuffling
            stream = construct_stream(stream_type=stream_type, num_batches=num_batches, batch_size=batch_size)
            
            for method in methods:
                # Set identical seed for TTA updates reproducibility
                set_seed(42)
                
                # Use optimal learning rate for each method
                if method == 'lfwa':
                    lr = 0.50
                elif method == 'pc_merge':
                    lr = 1.00
                elif method == 'ewfr_merge':
                    lr = 1.00
                else:
                    lr = 0.01
                    
                acc, loss = run_evaluation(
                    base_model=base_model,
                    experts=experts,
                    stream=stream,
                    fisher_sens=fisher_sens,
                    prototypes=prototypes,
                    method=method,
                    lr=lr,
                    severity=severity,
                    corruption_type=corr_type
                )
                
                results[stream_type][corr_type][method] = acc
                print(f"[{method.upper():12s}] Acc: {acc:.2f}% | Entropy Loss: {loss:.4f}")
                
    # Print final summary table
    print("\n\n" + "#"*50 + "\n  FINAL EVALUATION RESULTS COMPARISON\n" + "#"*50)
    for stream_type in stream_types:
        print(f"\nStream: {stream_type.upper()}")
        header = f"{'Method':15s} | {'Clean':8s} | {'G-Noise':8s} | {'G-Blur':8s} | {'Contrast':8s}"
        print(header)
        print("-" * len(header))
        for method in methods:
            clean_acc = results[stream_type]['none'][method]
            noise_acc = results[stream_type]['gaussian_noise'][method]
            blur_acc = results[stream_type]['gaussian_blur'][method]
            contrast_acc = results[stream_type]['contrast'][method]
            print(f"{method.upper():15s} | {clean_acc:7.2f}% | {noise_acc:7.2f}% | {blur_acc:7.2f}% | {contrast_acc:7.2f}%")
            
    # Save results to a file for paper generation and plotting
    import json
    with open("experiment_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nResults saved to experiment_results.json")

if __name__ == "__main__":
    main()
