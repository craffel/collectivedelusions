import torch
import torch.nn as nn
import numpy as np

def get_merged_state_dict(expert_state_dicts, mode='wa', lam=0.3, base_state_dict=None):
    """
    Returns the merged state dict of the model backbones.
    Classification heads (named with 'fc') are not merged.
    """
    merged_state_dict = {}
    keys = list(expert_state_dicts[0].keys())
    
    for key in keys:
        if 'fc' in key:
            # Classification heads are task-specific and not merged
            merged_state_dict[key] = expert_state_dicts[0][key].clone()
            continue
            
        tensors = [sd[key].float() for sd in expert_state_dicts]
        
        if mode == 'wa':
            merged_state_dict[key] = torch.stack(tensors).mean(dim=0)
        elif mode == 'ta':
            if base_state_dict is None:
                raise ValueError("base_state_dict is required for Task Arithmetic")
            base_tensor = base_state_dict[key].float().to(tensors[0].device)
            task_vectors = [t - base_tensor for t in tensors]
            sum_task_vectors = torch.stack(task_vectors).sum(dim=0)
            merged_state_dict[key] = base_tensor + lam * sum_task_vectors
        else:
            raise ValueError(f"Unknown merge mode: {mode}")
            
    return merged_state_dict

def find_conv_bn_pairs(model):
    """
    Traverses the ResNet model and returns list of (conv_name, conv_module, bn_name, bn_module)
    for all Conv2d layers that are followed by BatchNorm2d layers.
    """
    pairs = []
    modules = list(model.named_modules())
    for i in range(len(modules) - 1):
        name, module = modules[i]
        next_name, next_module = modules[i+1]
        # In ResNet, a conv is immediately followed by a BN
        if isinstance(module, nn.Conv2d) and isinstance(next_module, nn.BatchNorm2d):
            pairs.append((name, module, next_name, next_module))
    return pairs

@torch.no_grad()
def apply_sp_taac(merged_model, expert_models, calibration_batches, device, eps=1e-5):
    """
    Applies SP-TAAC (Sparsity-Preserving Task-Agnostic Activation Calibration)
    to early layers of the merged model (conv1, layer1, layer2).
    """
    merged_model.eval()
    for m in expert_models:
        m.eval()
        
    pairs = find_conv_bn_pairs(merged_model)
    early_pairs = [p for p in pairs if any(p[0].startswith(prefix) for prefix in ['conv1', 'layer1', 'layer2'])]
    
    # We calibrate layer-by-layer sequentially
    for conv_name, conv, bn_name, bn in early_pairs:
        # We need to collect activations at this BN output
        # Define hook to capture activations
        merged_act = []
        expert_acts = [[] for _ in expert_models]
        
        # Helper to find corresponding BN in expert models
        expert_bns = []
        for em in expert_models:
            ebn = dict(em.named_modules())[bn_name]
            expert_bns.append(ebn)
            
        def hook_merged(module, input, output):
            merged_act.append(output.detach())
        def make_hook_expert(idx):
            def hook_expert(module, input, output):
                expert_acts[idx].append(output.detach())
            return hook_expert
            
        # Register hooks
        h_m = bn.register_forward_hook(hook_merged)
        h_es = [ebn.register_forward_hook(make_hook_expert(idx)) for idx, ebn in enumerate(expert_bns)]
        
        # Pass calibration batches
        for batch in calibration_batches:
            batch = batch.to(device)
            _ = merged_model(batch)
            for idx, em in enumerate(expert_models):
                K = len(expert_models)
                N = batch.size(0) // K
                expert_batch = batch[idx*N : (idx+1)*N]
                _ = em(expert_batch)
                
        # Remove hooks
        h_m.remove()
        for h in h_es:
            h.remove()
            
        # Compute statistics
        merged_act_tensor = torch.cat(merged_act, dim=0) # (B_total, C, H, W)
        expert_acts_tensors = [torch.cat(acts, dim=0) for acts in expert_acts] # list of (B_total, C, H, W)
        
        # Standard deviation of merged model
        sigma_merged = merged_act_tensor.std()
        
        # Standard deviation of expert models concatenated
        all_experts_act = torch.cat(expert_acts_tensors, dim=0)
        sigma_target = all_experts_act.std()
        
        # Compute global scaling factor gamma
        gamma = sigma_target / (sigma_merged + eps)
        
        # Scale BN weight and bias in-place
        bn.weight.data.copy_(bn.weight.data * gamma)
        if bn.bias is not None:
            bn.bias.data.copy_(bn.bias.data * gamma)
            
        print(f"SP-TAAC applied to {bn_name}: gamma={gamma.item():.4f}")

@torch.no_grad()
def apply_slr_wbc(merged_model, expert_models, calibration_batches, rank=2, reg=0.5, device='cuda', eps=1e-5, layer_prefixes=['layer3', 'layer4']):
    """
    Applies SLR-WBC (SVD-based Low-Rank Weight and BatchNorm Calibration)
    to deep layers of the merged model (layer3, layer4).
    """
    merged_model.eval()
    for m in expert_models:
        m.eval()
        
    pairs = find_conv_bn_pairs(merged_model)
    deep_pairs = [p for p in pairs if any(p[0].startswith(prefix) for prefix in layer_prefixes)]
    
    for conv_name, conv, bn_name, bn in deep_pairs:
        # We need to collect:
        # 1. Input activations to the Conv layer (X) under the current merged model
        # 2. Outputs of the BN layer of the experts (H_target,k)
        conv_inputs = []
        expert_bn_outputs = [[] for _ in expert_models]
        
        # Find corresponding modules in experts
        expert_convs = []
        expert_bns = []
        for em in expert_models:
            expert_convs.append(dict(em.named_modules())[conv_name])
            expert_bns.append(dict(em.named_modules())[bn_name])
            
        def hook_conv_input(module, input, output):
            # input[0] is the input tensor to the Conv2d layer
            conv_inputs.append(input[0].detach())
            
        def make_hook_expert_bn(idx):
            def hook_expert_bn(module, input, output):
                expert_bn_outputs[idx].append(output.detach())
            return hook_expert_bn
            
        h_conv = conv.register_forward_hook(hook_conv_input)
        h_bns = [ebn.register_forward_hook(make_hook_expert_bn(idx)) for idx, ebn in enumerate(expert_bns)]
        
        # Pass calibration batches
        for batch in calibration_batches:
            batch = batch.to(device)
            _ = merged_model(batch)
            for idx, em in enumerate(expert_models):
                K = len(expert_models)
                N = batch.size(0) // K
                expert_batch = batch[idx*N : (idx+1)*N]
                _ = em(expert_batch)
                
        h_conv.remove()
        for h in h_bns:
            h.remove()
            
        X = torch.cat(conv_inputs, dim=0) # Shape: (K*B, C_in, H_in, W_in)
        H_target_list = [torch.cat(outputs, dim=0) for outputs in expert_bn_outputs] # List of (K*B, C_out, H_out, W_out)
        
        # 1. Unfold and flatten input activations to X_matrix (d_in, M)
        # ResNet-18 convs have kernel_size, padding, stride, dilation
        X_unfold = nn.functional.unfold(
            X, 
            kernel_size=conv.kernel_size, 
            dilation=conv.dilation, 
            padding=conv.padding, 
            stride=conv.stride
        ) # (K*B, d_in, L)
        
        d_in = X_unfold.size(1)
        M = X_unfold.size(0) * X_unfold.size(2)
        X_matrix = X_unfold.transpose(0, 1).flatten(1) # (d_in, M)
        
        # 2. Invert experts' BN operations to get target Conv outputs V_target
        V_target_k_list = []
        for k, H_k in enumerate(H_target_list):
            ebn = expert_bns[k]
            # Reshape ebn parameters for channel-wise broadcasting
            w = ebn.weight.view(1, -1, 1, 1)
            b = ebn.bias.view(1, -1, 1, 1)
            mu = ebn.running_mean.view(1, -1, 1, 1)
            sigma_sq = ebn.running_var.view(1, -1, 1, 1)
            
            # Invert: V = (H - b)/w * sqrt(sigma_sq + eps) + mu
            V_k = (H_k - b) / (w + eps) * torch.sqrt(sigma_sq + eps) + mu
            V_target_k_list.append(V_k)
            
        # Joint target V_target is concatenated along the batch dimension
        V_target = torch.cat(V_target_k_list, dim=0) # (K*K*B, C_out, H_out, W_out)
        C_out = V_target.size(1)
        V_target_matrix = V_target.transpose(0, 1).flatten(1) # (C_out, M)
        
        # 3. Solve ridge regression for optimal full-rank correction delta_W_star
        W_curr = conv.weight.data.view(C_out, d_in)
        E = V_target_matrix - W_curr @ X_matrix # (C_out, M)
        
        lambda_reg = reg * M
        A = X_matrix @ X_matrix.T + lambda_reg * torch.eye(d_in, device=device)
        B_T = (E @ X_matrix.T).T
        delta_W_star_T = torch.linalg.solve(A, B_T)
        delta_W_star = delta_W_star_T.T # (C_out, d_in)
        
        # 4. Truncate SVD to rank r
        U, S, Vh = torch.linalg.svd(delta_W_star, full_matrices=False)
        r = min(rank, len(S))
        S_trunc = S[:r]
        delta_W_r = U[:, :r] @ torch.diag(S_trunc) @ Vh[:r, :]
        
        # 5. Update weights in-place
        conv.weight.data += delta_W_r.view(conv.weight.shape)
        
        # 6. Update BatchNorm stats and affine parameters
        # Fast projection: V_new = W_new @ X_matrix
        W_new = conv.weight.data.view(C_out, d_in)
        V_new_matrix = W_new @ X_matrix # (C_out, M)
        
        # Reshape back to (K*K*B, C_out, H_out, W_out)
        V_new = V_new_matrix.view(C_out, V_target.size(0), V_target.size(2), V_target.size(3)).transpose(0, 1)
        
        # Empirical mean and variance of V_new
        mean_new = V_new.mean(dim=(0, 2, 3))
        var_new = V_new.var(dim=(0, 2, 3), unbiased=False)
        
        bn.running_mean.copy_(mean_new)
        bn.running_var.copy_(var_new)
        
        # Normalize activations
        mu_broadcast = mean_new.view(1, -1, 1, 1)
        sigma_sq_broadcast = var_new.view(1, -1, 1, 1)
        V_norm = (V_new - mu_broadcast) / torch.sqrt(sigma_sq_broadcast + eps)
        
        # Solve channel-wise BN affine least squares: wc * V_norm + bc ~ H_target
        # wc = mean(V_norm_c * H_target_c)
        # bc = mean(H_target_c)
        # Since H_target is concatenated from all experts, we concatenate them
        H_all = torch.cat(H_target_list, dim=0) # (K*K*B, C_out, H_out, W_out)
        
        w_c = (V_norm * H_all).mean(dim=(0, 2, 3))
        b_c = H_all.mean(dim=(0, 2, 3))
        
        # Clip scaling factor wc to [0.1, 10.0] for stability
        w_c = torch.clamp(w_c, 0.1, 10.0)
        
        bn.weight.data.copy_(w_c)
        bn.bias.data.copy_(b_c)
        
        print(f"SLR-WBC applied to {conv_name}/{bn_name}: SVD rank={r}, reg={reg}")

class WRSAHook:
    def __init__(self, c=0.30, eps=1e-5):
        self.c = c
        self.eps = eps
        self.M_T = None
        self.M_M = None
        self.gamma = None
        self.hook_handle = None
        
    def collect_expert_stats(self, expert_bns, calibration_batches, device):
        # Collect spectra for expert models on calibration batches
        spectra = []
        for ebn in expert_bns:
            ebn_acts = []
            def e_hook(module, input, output):
                ebn_acts.append(output.detach())
            h = ebn.register_forward_hook(e_hook)
            # Pass batches through expert model
            # Note: caller must run forward pass of expert models
            # Here we assume caller runs the passes and ebn_acts gets filled
            self.current_ebn_acts = ebn_acts
            self.current_h = h
            
    def register_wrsa_hook(self, bn, M_T, M_M):
        self.M_T = M_T.to(bn.weight.device)
        self.M_M = M_M.to(bn.weight.device)
        
        # Compute scale-invariant scaling factor gamma
        # Gamma = (M_T * M_M) / (M_M^2 + c^2 * M_T^2 + eps)
        # Clip max value by 1/(2c) as mathematically proven in WRSA theorem
        self.gamma = (self.M_T * self.M_M) / (self.M_M**2 + (self.c**2) * (self.M_T**2) + self.eps)
        max_bound = 1.0 / (2.0 * self.c)
        self.gamma = torch.clamp(self.gamma, max=max_bound)
        
        def hook_fn(module, input, output):
            # Apply WRSA scaling to output Fourier coefficients
            X = output # Shape: (B, C, H, W)
            X_fft = torch.fft.fft2(X, dim=(-2, -1))
            
            # Broadcast gamma: (H, W) to (1, 1, H, W)
            g = self.gamma.unsqueeze(0).unsqueeze(1)
            X_fft_cal = X_fft * g
            X_cal = torch.fft.ifft2(X_fft_cal, dim=(-2, -1)).real
            return X_cal
            
        self.hook_handle = bn.register_forward_hook(hook_fn)
        
    def remove(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()

@torch.no_grad()
def apply_wrsa(merged_model, expert_models, calibration_batches, c=0.30, device='cuda', eps=1e-5):
    """
    Applies WRSA (Wiener-Regularized Spectral Alignment) to deep layers of the merged model.
    Registers forward hooks that scale Fourier coefficients during the forward pass.
    """
    merged_model.eval()
    for m in expert_models:
        m.eval()
        
    pairs = find_conv_bn_pairs(merged_model)
    deep_pairs = [p for p in pairs if any(p[0].startswith(prefix) for prefix in ['layer3', 'layer4'])]
    
    wrsa_hooks = []
    
    for conv_name, conv, bn_name, bn in deep_pairs:
        # Collect merged activations to get M_M
        merged_acts = []
        def merged_hook(module, input, output):
            merged_acts.append(output.detach())
        h_m = bn.register_forward_hook(merged_hook)
        
        # Collect expert activations to get M_T
        expert_acts_list = [[] for _ in expert_models]
        expert_bns = [dict(em.named_modules())[bn_name] for em in expert_models]
        
        def make_expert_hook(idx):
            def expert_hook(module, input, output):
                expert_acts_list[idx].append(output.detach())
            return expert_hook
            
        h_es = [ebn.register_forward_hook(make_expert_hook(idx)) for idx, ebn in enumerate(expert_bns)]
        
        # Run calibration forward passes
        for batch in calibration_batches:
            batch = batch.to(device)
            _ = merged_model(batch)
            for idx, em in enumerate(expert_models):
                K = len(expert_models)
                N = batch.size(0) // K
                expert_batch = batch[idx*N : (idx+1)*N]
                _ = em(expert_batch)
                
        h_m.remove()
        for h in h_es:
            h.remove()
            
        X_merged = torch.cat(merged_acts, dim=0) # (K*B, C, H, W)
        # Compute M_M spectral profile
        M_M = torch.fft.fft2(X_merged, dim=(-2, -1)).abs().mean(dim=(0, 1))
        
        # Compute M_T spectral profile (averaged across experts)
        M_T_list = []
        for idx in range(len(expert_models)):
            X_exp = torch.cat(expert_acts_list[idx], dim=0)
            M_T_k = torch.fft.fft2(X_exp, dim=(-2, -1)).abs().mean(dim=(0, 1))
            M_T_list.append(M_T_k)
        M_T = torch.stack(M_T_list).mean(dim=0)
        
        # Create and register WRSA hook
        hook = WRSAHook(c=c, eps=eps)
        hook.register_wrsa_hook(bn, M_T, M_M)
        wrsa_hooks.append(hook)
        
        print(f"WRSA hook registered for {bn_name} with c={c}")
        
    return wrsa_hooks

@torch.no_grad()
def get_task_prototypes(merged_model, calibration_loaders, device='cuda'):
    """
    Extracts Layer 2 task prototypes for MSPR.
    calibration_loaders is a dict: {'mnist': loader, 'fmnist': loader, 'cifar10': loader}
    """
    merged_model.eval()
    prototypes = {}
    
    # We want to collect activations at Layer 2 (e.g. output of layer2)
    layer2_module = merged_model.layer2
    
    for task_name, loader in calibration_loaders.items():
        layer2_acts = []
        def hook_fn(module, input, output):
            # Output shape: (B, C, H, W)
            # Global Average Pooling:
            pooled = output.mean(dim=(2, 3)) # (B, C)
            layer2_acts.append(pooled.detach().cpu())
            
        handle = layer2_module.register_forward_hook(hook_fn)
        
        # Run through calibration set
        for inputs, _ in loader:
            inputs = inputs.to(device)
            _ = merged_model(inputs)
            
        handle.remove()
        
        # Average over all samples and L2-normalize
        pooled_all = torch.cat(layer2_acts, dim=0) # (N_cal, C)
        avg_proto = pooled_all.mean(dim=0) # (C,)
        normalized_proto = avg_proto / (avg_proto.norm(p=2) + 1e-8)
        prototypes[task_name] = normalized_proto
        
    return prototypes

def mspr_route_sample(merged_model, x, prototypes, device='cuda'):
    """
    Computes Layer 2 activation and routes to the best task head.
    x is a single input sample or batch of shape (B, C, H, W)
    """
    merged_model.eval()
    layer2_module = merged_model.layer2
    
    captured_pooled = []
    def hook_fn(module, input, output):
        pooled = output.mean(dim=(2, 3)) # (B, C)
        captured_pooled.append(pooled.detach())
        
    handle = layer2_module.register_forward_hook(hook_fn)
    
    # Run up to Layer 2 (we can run the whole forward pass since hooks run inline)
    _ = merged_model(x)
    handle.remove()
    
    v = captured_pooled[0] # (B, C)
    # L2-normalize each sample in the batch
    v_norm = v / (v.norm(p=2, dim=1, keepdim=True) + 1e-8)
    
    # Compute cosine similarities against task prototypes
    task_names = list(prototypes.keys())
    proto_matrix = torch.stack([prototypes[name].to(device) for name in task_names]) # (K, C)
    
    # Cosine similarities: (B, K) = (B, C) @ (C, K)
    similarities = v_norm @ proto_matrix.T
    best_task_indices = similarities.argmax(dim=1).cpu().numpy()
    
    routed_tasks = [task_names[idx] for idx in best_task_indices]
    return routed_tasks
