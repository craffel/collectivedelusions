import torch
import torch.nn as nn
import torch.nn.functional as F
from models import RoutingContext, RoutedConv2d, RoutedBatchNorm2d, RoutedResNet18

@torch.no_grad()
def calibrate_model(routed_model, expert_models, calib_loaders, rank=4, reg=0.5):
    """
    Calibrates the routed_model sequentially layer-by-layer using SLR-WBC.
    Also extracts prototypes at Layer 2.
    """
    device = next(routed_model.parameters()).device
    num_tasks = len(expert_models)
    
    # 1. Compute Layer 2 Prototypes
    print("Step 1: Computing Layer 2 Prototypes...")
    prototypes = torch.zeros(num_tasks, 128, device=device)
    
    # Turn off routing during prototype extraction
    RoutingContext.is_active = False
    
    for k in range(num_tasks):
        loader = calib_loaders[k]
        feat_list = []
        for x, _ in loader:
            x = x.to(device)
            # Run stem and early layers of merged model
            with torch.no_grad():
                out = routed_model.conv1(x)
                out = routed_model.bn1(out)
                out = routed_model.relu(out)
                out = routed_model.maxpool(out)
                out = routed_model.layer1(out)
                a2_feat = routed_model.layer2(out) # shape: [B, 128, H, W]
                
                # Global average pool to get 128-d vectors
                pooled = F.adaptive_avg_pool2d(a2_feat, (1, 1)).view(x.size(0), -1)
                feat_list.append(pooled)
                
        feat_all = torch.cat(feat_list, dim=0)
        prototypes[k] = feat_all.mean(dim=0)
        
    # Set prototypes in model
    routed_model.prototypes.data.copy_(prototypes)
    print("Prototypes extracted successfully.")
    
    # 2. Sequential Layer-by-Layer Calibration
    print("Step 2: Starting sequential SLR-WBC calibration of Layer 3 and Layer 4...")
    
    # List of deep blocks
    blocks = []
    for basic_block in routed_model.layer3:
        blocks.append(basic_block)
    for basic_block in routed_model.layer4:
        blocks.append(basic_block)
        
    for b_idx, block in enumerate(blocks):
        print(f"Calibrating Block {b_idx+1}/{len(blocks)}...")
        
        # In each block, calibrate:
        # conv1 + bn1, then conv2 + bn2, then downsample if exists.
        sublayers = []
        if block.downsample is not None:
            # Calibrate downsample conv & bn first, or sequentially with others?
            # Let's calibrate downsample first, then conv1, then conv2.
            # Downsample is conv (0) and bn (1)
            sublayers.append((block.downsample[0], block.downsample[1], "downsample"))
            
        sublayers.append((block.conv1, block.bn1, "conv1"))
        sublayers.append((block.conv2, block.bn2, "conv2"))
        
        for conv, bn, name in sublayers:
            if not isinstance(conv, RoutedConv2d) or not isinstance(bn, RoutedBatchNorm2d):
                continue
                
            print(f"  Calibrating {name}...")
            
            # For each task k, compute task-specific delta_W and update task_bns
            for k in range(num_tasks):
                expert = expert_models[k].to(device)
                loader = calib_loaders[k]
                
                # Set routing context to force task k path
                RoutingContext.is_active = True
                
                # Find matching Conv and BN layers in the expert model
                # Since the architecture is identical, we find the matching layer using block index and sublayer name
                expert_block = None
                if b_idx < len(routed_model.layer3):
                    expert_block = expert.layer3[b_idx]
                else:
                    expert_block = expert.layer4[b_idx - len(routed_model.layer3)]
                    
                if name == "downsample":
                    expert_conv = expert_block.downsample[0]
                    expert_bn = expert_block.downsample[1]
                elif name == "conv1":
                    expert_conv = expert_block.conv1
                    expert_bn = expert_block.bn1
                else:
                    expert_conv = expert_block.conv2
                    expert_bn = expert_block.bn2
                
                # Hooks to capture:
                # - input to merged model's Conv layer: X
                # - output of expert model's Conv layer: V_expert
                # - output of expert model's BN layer: H_target
                merged_inputs = []
                expert_conv_outputs = []
                expert_bn_outputs = []
                
                def hook_merged_in(m, i):
                    merged_inputs.append(i[0].detach())
                def hook_expert_conv_out(m, i, o):
                    expert_conv_outputs.append(o.detach())
                def hook_expert_bn_out(m, i, o):
                    expert_bn_outputs.append(o.detach())
                    
                h1 = conv.register_forward_pre_hook(hook_merged_in)
                h2 = expert_conv.register_forward_hook(hook_expert_conv_out)
                h3 = expert_bn.register_forward_hook(hook_expert_bn_out)
                
                # Run the calibration loaders to collect inputs and targets
                for x, _ in loader:
                    x = x.to(device)
                    # Dynamically adjust RoutingContext weights shape
                    mock_weights = torch.zeros(x.size(0), num_tasks, device=device)
                    mock_weights[:, k] = 1.0
                    RoutingContext.weights = mock_weights
                    RoutingContext.ood_gate = torch.ones(x.size(0), 1, 1, 1, device=device)
                    
                    # Run full forward pass on both models to trigger the hooks
                    with torch.no_grad():
                        _ = routed_model(x)
                        _ = expert(x)
                        
                h1.remove()
                h2.remove()
                h3.remove()
                
                # Concatenate across batches
                X = torch.cat(merged_inputs, dim=0)          # shape: [B, Cin, Hin, Win]
                V_expert = torch.cat(expert_conv_outputs, dim=0) # shape: [B, Cout, Hout, Wout]
                H_target = torch.cat(expert_bn_outputs, dim=0)   # shape: [B, Cout, Hout, Wout]
                
                B, C_in, H_in, W_in = X.size()
                _, C_out, H_out, W_out = V_expert.size()
                
                # Unfold input activations
                X_unfold = F.unfold(X, conv.kernel_size, conv.dilation, conv.padding, conv.stride)
                # X_unfold shape: [B, Cin * Kh * Kw, Hout * Wout]
                # Reshape to [din, B * Hout * Wout]
                d_in = X_unfold.size(1)
                X_matrix = X_unfold.transpose(0, 1).reshape(d_in, -1)
                M = X_matrix.size(1)
                
                # Reshape V_expert to [Cout, B * Hout * Wout]
                V_matrix = V_expert.transpose(0, 1).reshape(C_out, -1)
                
                # Current Conv weight
                W_curr = conv.weight.data.view(C_out, d_in)
                
                # Compute error E
                E = V_matrix - torch.matmul(W_curr, X_matrix)
                
                # Ridge regression solver
                lambd = reg * M
                cov = torch.matmul(X_matrix, X_matrix.t()) + lambd * torch.eye(d_in, device=device)
                cov_inv = torch.linalg.inv(cov)
                delta_W_star = torch.matmul(torch.matmul(E, X_matrix.t()), cov_inv)
                
                # SVD truncation to rank r
                r = min(rank, C_out, d_in)
                U, S, V = torch.linalg.svd(delta_W_star, full_matrices=False)
                U_r = U[:, :r]
                S_r = S[:r]
                V_r = V[:r, :]
                delta_W_r = torch.matmul(U_r, torch.matmul(torch.diag(S_r), V_r))
                
                # Save task-specific correction
                conv.delta_W[k].data.copy_(delta_W_r.view_as(conv.weight))
                
                # Now update task-specific BN for task k
                # Collect Conv outputs with the newly added correction
                new_conv_outputs = []
                def hook_new_conv_out(m, i, o):
                    new_conv_outputs.append(o.detach())
                h_new = conv.register_forward_hook(hook_new_conv_out)
                
                for x, _ in loader:
                    x = x.to(device)
                    mock_weights = torch.zeros(x.size(0), num_tasks, device=device)
                    mock_weights[:, k] = 1.0
                    RoutingContext.weights = mock_weights
                    RoutingContext.ood_gate = torch.ones(x.size(0), 1, 1, 1, device=device)
                    with torch.no_grad():
                        _ = routed_model(x)
                h_new.remove()
                
                V_new = torch.cat(new_conv_outputs, dim=0) # shape: [B, Cout, Hout, Wout]
                
                # Compute running stats
                mean = V_new.mean(dim=(0, 2, 3), keepdim=True)
                var = V_new.var(dim=(0, 2, 3), keepdim=True, unbiased=False)
                
                # Normalize V_new
                V_norm = (V_new - mean) / torch.sqrt(var + bn.eps)
                
                # Assign running stats to task_bns[k]
                bn.task_bns[k].running_mean.copy_(mean.squeeze())
                bn.task_bns[k].running_var.copy_(var.squeeze())
                
                # Solve channel-wise least squares for scale and bias
                scale = (V_norm * H_target).mean(dim=(0, 2, 3))
                bias = H_target.mean(dim=(0, 2, 3))
                
                # Clip scale for stability
                scale = torch.clamp(scale, 0.1, 10.0)
                
                bn.task_bns[k].weight.copy_(scale)
                bn.task_bns[k].bias.copy_(bias)
                
            print(f"  Calibrated {name} for all tasks.")
            
    # Reset routing context to standard
    RoutingContext.is_active = True
    RoutingContext.weights = None
    RoutingContext.ood_gate = None
    print("Sequential SLR-WBC calibration completed.")
