import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
from models import SimpleCNN

def get_entropy(logits):
    probs = F.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
    return entropy.mean()

def apply_soft_bn_fusion(experts, weights, merged_model):
    """
    Applies Soft BN Buffer Fusion to combine running_mean and running_var of BatchNorm layers.
    """
    weights = weights.detach()
    for name, module in merged_model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            mean_key = f"{name}.running_mean"
            var_key = f"{name}.running_var"
            
            # Fused running mean: mu_fused = sum_k w_k * mu_k
            mu_fused = torch.zeros_like(module.running_mean)
            for k, expert_state in enumerate(experts):
                mu_fused += weights[k] * expert_state[mean_key]
                
            # Fused running variance: var_fused = sum_k w_k * (var_k + (mu_k - mu_fused)^2)
            var_fused = torch.zeros_like(module.running_var)
            for k, expert_state in enumerate(experts):
                diff = expert_state[mean_key] - mu_fused
                var_fused += weights[k] * (expert_state[var_key] + diff ** 2)
                
            module.running_mean.copy_(mu_fused)
            module.running_var.copy_(var_fused)

def merge_weights_layerwise(experts, weights_dict, merged_model):
    """
    Merges expert parameters layer-by-layer using layer-specific weights_dict while preserving autograd.
    weights_dict maps layer names/prefixes to tensor of coefficients of shape (K,).
    """
    for name, module in merged_model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
            for param_name in ["weight", "bias"]:
                if hasattr(module, param_name) and getattr(module, param_name) is not None:
                    key = f"{name}.{param_name}" if name else param_name
                    
                    # Find the appropriate merging coefficients for this key
                    # We can match layer-specific coefficients by parsing the key prefix (e.g. 'conv1', 'fc1')
                    coefs = None
                    for prefix, w in weights_dict.items():
                        if key.startswith(prefix):
                            coefs = w
                            break
                    if coefs is None:
                        coefs = weights_dict.get("global")
                        
                    param_sum = 0
                    for k, expert_state in enumerate(experts):
                        param_sum = param_sum + coefs[k] * expert_state[key]
                        
                    # Delete the parameter from module's parameters dict to avoid PyTorch errors and set as attribute
                    if param_name in module._parameters:
                        del module._parameters[param_name]
                    setattr(module, param_name, param_sum)

# ----------------- 1. STATIC MERGING -----------------
def run_static_merging(experts, stream, device="cpu"):
    """
    Static merging: uses constant uniform [0.5, 0.5] weights and average BN statistics.
    """
    merged_model = SimpleCNN().to(device)
    K = len(experts)
    uniform_weights = torch.ones(K, device=device) / K
    
    # Merge weights
    weights_dict = {"global": uniform_weights}
    merge_weights_layerwise(experts, weights_dict, merged_model)
    apply_soft_bn_fusion(experts, uniform_weights, merged_model)
    
    merged_model.eval()
    correct = 0
    total = 0
    accuracies = []
    
    for x, y, domain in stream:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = merged_model(x)
            _, predicted = logits.max(1)
            batch_correct = predicted.eq(y).sum().item()
            correct += batch_correct
            total += y.size(0)
            accuracies.append(batch_correct / y.size(0))
            
    return correct / total, accuracies

# ----------------- 2. ADAMERGING -----------------
def run_adamerging(experts, stream, device="cpu", lr=1e-2, inner_steps=3):
    """
    AdaMerging: unconstrained layer-wise entropy minimization on merging coefficients.
    """
    merged_model = SimpleCNN().to(device)
    K = len(experts)
    
    # Identify layers to merge layer-specifically
    # We define merging logit parameters for main layers: conv1, conv2, fc1, fc2, and a global one
    layer_prefixes = ["conv1", "conv2", "fc1", "fc2", "global"]
    logits_dict = {prefix: nn.Parameter(torch.zeros(K, device=device)) for prefix in layer_prefixes}
    
    optimizer = optim.Adam(logits_dict.values(), lr=lr)
    
    correct = 0
    total = 0
    accuracies = []
    
    for x, y, domain in stream:
        x, y = x.to(device), y.to(device)
        
        # Adaptation loop on the current batch
        for step in range(inner_steps):
            optimizer.zero_grad()
            
            # Map logits to simplex using softmax
            weights_dict = {prefix: F.softmax(logits, dim=-1) for prefix, logits in logits_dict.items()}
            
            # Merge weights and fuse BN
            merge_weights_layerwise(experts, weights_dict, merged_model)
            apply_soft_bn_fusion(experts, weights_dict["global"], merged_model)
            
            merged_model.eval() # Enable gradient calculation
            logits = merged_model(x)
            loss = get_entropy(logits)
            loss.backward()
            optimizer.step()
            
        # Evaluation
        merged_model.eval()
        with torch.no_grad():
            weights_dict = {prefix: F.softmax(logits, dim=-1) for prefix, logits in logits_dict.items()}
            merge_weights_layerwise(experts, weights_dict, merged_model)
            apply_soft_bn_fusion(experts, weights_dict["global"], merged_model)
            
            logits = merged_model(x)
            _, predicted = logits.max(1)
            batch_correct = predicted.eq(y).sum().item()
            correct += batch_correct
            total += y.size(0)
            accuracies.append(batch_correct / y.size(0))
            
    return correct / total, accuracies

# ----------------- 3. DF-BAYES-TTMM -----------------
def run_df_bayes_ttmm(experts, stream, device="cpu", lr=1e-2, inner_steps=3, gamma=15.0, beta=0.5, tau_n=1.2):
    """
    DF-Bayes-TTMM: Bayesian Mixture-of-Experts routing, Soft BN Buffer Fusion, and Uncertainty Novelty Detection.
    """
    merged_model = SimpleCNN().to(device)
    K = len(experts)
    
    # Pre-instantiate models for computing individual expert entropy
    expert_models = []
    for state_dict in experts:
        m = SimpleCNN().to(device)
        m.load_state_dict(state_dict)
        m.eval()
        expert_models.append(m)
        
    layer_prefixes = ["conv1", "conv2", "fc1", "fc2", "global"]
    logits_dict = {prefix: nn.Parameter(torch.zeros(K, device=device)) for prefix in layer_prefixes}
    optimizer = optim.Adam(logits_dict.values(), lr=lr)
    
    correct = 0
    total = 0
    accuracies = []
    
    # Store previous logits for MAP regularizer
    logits_prev = {prefix: logits.clone().detach() for prefix, logits in logits_dict.items()}
    
    for x, y, domain in stream:
        x, y = x.to(device), y.to(device)
        
        # 1. Compute expert entropies and Bayesian posterior soft-routing weights
        expert_entropies = []
        with torch.no_grad():
            for m in expert_models:
                expert_entropies.append(get_entropy(m(x)).item())
                
        avg_entropy = sum(expert_entropies) / K
        
        # Bayesian posteriors: w_k = exp(-gamma * H_k) / sum_j exp(-gamma * H_j)
        likelihoods = torch.tensor([torch.exp(-torch.tensor(gamma * h)) for h in expert_entropies], device=device)
        posterior_weights = likelihoods / (torch.sum(likelihoods) + 1e-9)
        
        # 2. Novelty detection
        is_novel = avg_entropy > tau_n
        
        if not is_novel:
            # Route directly to the best expert (soft EMA transition)
            best_idx = torch.argmax(posterior_weights).item()
            # Set logits towards the routed expert
            for prefix in layer_prefixes:
                with torch.no_grad():
                    logits_dict[prefix].copy_(torch.zeros(K, device=device))
                    logits_dict[prefix][best_idx] = 3.0 # Encourage routing
        else:
            # Novel task: Adapt using entropy minimization with MAP prior
            for step in range(inner_steps):
                optimizer.zero_grad()
                weights_dict = {prefix: F.softmax(logits, dim=-1) for prefix, logits in logits_dict.items()}
                
                merge_weights_layerwise(experts, weights_dict, merged_model)
                apply_soft_bn_fusion(experts, weights_dict["global"], merged_model)
                
                merged_model.eval()
                logits = merged_model(x)
                
                entropy_loss = get_entropy(logits)
                
                # MAP regularizer: sum_l ||logits_l - logits_prev_l||^2
                map_loss = 0
                for prefix in layer_prefixes:
                    map_loss += torch.sum((logits_dict[prefix] - logits_prev[prefix]) ** 2)
                    
                loss = entropy_loss + (beta / 2.0) * map_loss
                loss.backward()
                optimizer.step()
                
            # Update prev logits
            for prefix in layer_prefixes:
                logits_prev[prefix] = logits_dict[prefix].clone().detach()
                
        # Evaluation
        merged_model.eval()
        with torch.no_grad():
            weights_dict = {prefix: F.softmax(logits, dim=-1) for prefix, logits in logits_dict.items()}
            merge_weights_layerwise(experts, weights_dict, merged_model)
            apply_soft_bn_fusion(experts, weights_dict["global"], merged_model)
            
            logits = merged_model(x)
            _, predicted = logits.max(1)
            batch_correct = predicted.eq(y).sum().item()
            correct += batch_correct
            total += y.size(0)
            accuracies.append(batch_correct / y.size(0))
            
    return correct / total, accuracies

def copy_model_weights_functional(src_model, dst_model, alpha=None):
    """
    Copies (or updates via EMA) weights and BN buffers from src_model to dst_model,
    handling attributes that are regular tensors as well as buffers.
    """
    for (name_src, mod_src), (name_dst, mod_dst) in zip(src_model.named_modules(), dst_model.named_modules()):
        if isinstance(mod_src, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
            for param_name in ["weight", "bias"]:
                if hasattr(mod_src, param_name) and getattr(mod_src, param_name) is not None:
                    val_src = getattr(mod_src, param_name)
                    if param_name in mod_dst._parameters:
                        del mod_dst._parameters[param_name]
                    if alpha is None:
                        setattr(mod_dst, param_name, val_src.clone().detach())
                    else:
                        val_dst = getattr(mod_dst, param_name)
                        new_val = alpha * val_dst + (1.0 - alpha) * val_src
                        setattr(mod_dst, param_name, new_val.clone().detach())
        if isinstance(mod_src, nn.BatchNorm2d):
            if alpha is None:
                mod_dst.running_mean.copy_(mod_src.running_mean)
                mod_dst.running_var.copy_(mod_src.running_var)
            else:
                mod_dst.running_mean.copy_(alpha * mod_dst.running_mean + (1.0 - alpha) * mod_src.running_mean)
                mod_dst.running_var.copy_(alpha * mod_dst.running_var + (1.0 - alpha) * mod_src.running_var)

def get_mutual_information(logits):
    probs = F.softmax(logits, dim=-1)
    sample_entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1).mean()
    mean_probs = probs.mean(dim=0)
    marginal_entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-9))
    return (marginal_entropy - sample_entropy).item()

# ----------------- 4. FRTR-TTMM (OURS) -----------------
def run_frtr_ttmm(experts, stream, device="cpu", lr=1e-2, inner_steps=3, gamma=15.0, beta=1.5, alpha_ema=0.9):
    """
    FRTR-TTMM (Ours): Feedback-Resistant Teacher-Regularized TTMM.
    Uses an EMA teacher model to regularize student adaptation, preventing the feedback trap.
    """
    K = len(experts)
    student_model = SimpleCNN().to(device)
    teacher_model = SimpleCNN().to(device)
    
    # Initialize teacher and student with uniform weights
    layer_prefixes = ["conv1", "conv2", "fc1", "fc2", "global"]
    logits_dict = {prefix: nn.Parameter(torch.zeros(K, device=device)) for prefix in layer_prefixes}
    optimizer = optim.Adam(logits_dict.values(), lr=lr)
    
    # Setup initial student weights and copy to teacher model
    weights_dict = {prefix: F.softmax(logits, dim=-1) for prefix, logits in logits_dict.items()}
    merge_weights_layerwise(experts, weights_dict, student_model)
    apply_soft_bn_fusion(experts, weights_dict["global"], student_model)
    
    copy_model_weights_functional(student_model, teacher_model)
    
    expert_models = []
    for state_dict in experts:
        m = SimpleCNN().to(device)
        m.load_state_dict(state_dict)
        m.eval()
        expert_models.append(m)
        
    correct = 0
    total = 0
    accuracies = []
    
    previous_best_idx = None
    
    for b_idx, (x, y, domain) in enumerate(stream):
        x, y = x.to(device), y.to(device)
        
        # 1. Mutual Information-Guided Routing (MI-Routing)
        expert_mis = []
        with torch.no_grad():
            for m in expert_models:
                expert_mis.append(get_mutual_information(m(x)))
                
        # Use softmax over Gamma * Mutual Information
        mi_tensor = torch.tensor(expert_mis, device=device)
        routing_prior = F.softmax(gamma * mi_tensor, dim=-1)
        
        current_best_idx = torch.argmax(routing_prior).item()
        
        # Initialize student logits with the routing prior to accelerate convergence
        prior_logits = torch.log(routing_prior + 1e-9)
        for prefix in layer_prefixes:
            with torch.no_grad():
                logits_dict[prefix].copy_(prior_logits)
                
        # Merge student with the new initialized prior weights before setting up teacher
        weights_dict = {prefix: F.softmax(logits, dim=-1) for prefix, logits in logits_dict.items()}
        merge_weights_layerwise(experts, weights_dict, student_model)
        apply_soft_bn_fusion(experts, weights_dict["global"], student_model)
        
        # Reset teacher if task shifted
        if previous_best_idx is None:
            previous_best_idx = current_best_idx
            copy_model_weights_functional(student_model, teacher_model)
        elif current_best_idx != previous_best_idx:
            copy_model_weights_functional(student_model, teacher_model)
            previous_best_idx = current_best_idx
            
        # 2. Adaptation with EMA Teacher Regularization
        # Compute teacher predictions on current batch (frozen/stable target)
        teacher_model.eval()
        with torch.no_grad():
            teacher_logits = teacher_model(x)
            teacher_probs = F.softmax(teacher_logits, dim=-1)
            
        for step in range(inner_steps):
            optimizer.zero_grad()
            weights_dict = {prefix: F.softmax(logits, dim=-1) for prefix, logits in logits_dict.items()}
            
            # Merge student weights and BN running statistics
            merge_weights_layerwise(experts, weights_dict, student_model)
            apply_soft_bn_fusion(experts, weights_dict["global"], student_model)
            
            student_model.eval()
            student_logits = student_model(x)
            
            # Unsupervised student entropy loss
            entropy_loss = get_entropy(student_logits)
            
            # Teacher-student consistency loss (KL divergence)
            student_log_probs = F.log_softmax(student_logits, dim=-1)
            kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
            
            # Total Loss: Student Entropy + beta * KL(Teacher || Student)
            loss = entropy_loss + beta * kl_loss
            loss.backward()
            optimizer.step()
            
        # 3. Post-Adaptation Evaluation
        student_model.eval()
        with torch.no_grad():
            weights_dict = {prefix: F.softmax(logits, dim=-1) for prefix, logits in logits_dict.items()}
            merge_weights_layerwise(experts, weights_dict, student_model)
            apply_soft_bn_fusion(experts, weights_dict["global"], student_model)
            
            logits = student_model(x)
            _, predicted = logits.max(1)
            batch_correct = predicted.eq(y).sum().item()
            correct += batch_correct
            total += y.size(0)
            accuracies.append(batch_correct / y.size(0))
            
        # 4. Update the EMA Teacher model parameters
        with torch.no_grad():
            copy_model_weights_functional(student_model, teacher_model, alpha=alpha_ema)
            
    return correct / total, accuracies
