import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import copy

class MultiTaskResNet18(nn.Module):
    """
    ResNet-18 with 3 task-specific classification heads.
    """
    def __init__(self, pretrained=True):
        super(MultiTaskResNet18, self).__init__()
        # Load backbone
        if pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
        else:
            weights = None
        self.backbone = models.resnet18(weights=weights)
        
        # We replace the fc layer with an identity or just a custom dict of heads.
        # To make it a standard nn.Module, we will register the heads as ModuleDict.
        self.num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity() # Remove the original fc
        
        self.heads = nn.ModuleDict({
            'mnist': nn.Linear(self.num_features, 10),
            'fmnist': nn.Linear(self.num_features, 10),
            'cifar10': nn.Linear(self.num_features, 10)
        })
        
        # Keep track of active task for forward pass if not explicitly specified
        self.task_mapping = {0: 'mnist', 1: 'fmnist', 2: 'cifar10'}
        
    def forward(self, x, task_id=None):
        features = self.backbone(x)
        if task_id is None:
            # If no task_id, we can't route. Raise error or use a default.
            raise ValueError("task_id must be specified (0: mnist, 1: fmnist, 2: cifar10)")
            
        task_name = self.task_mapping[task_id]
        logits = self.heads[task_name](features)
        return logits

def extract_expert(multitask_model, task_id):
    """
    Returns a copy of the multitask model with only the backbone and the task-specific head.
    """
    expert = copy.deepcopy(multitask_model)
    task_name = multitask_model.task_mapping[task_id]
    # To keep it clean, we can freeze or delete other heads
    for k in list(expert.heads.keys()):
        if k != task_name:
            del expert.heads[k]
    return expert

def merge_models_weight_averaging(experts, target_model):
    """
    Averages the backbone weights and BatchNorm parameters across a list of expert models
    and copies them into target_model.
    Also copies the original expert classification heads into target_model directly.
    """
    K = len(experts)
    averaged_state_dict = copy.deepcopy(experts[0].state_dict())
    
    # We want to average only the backbone parameters.
    # The heads should be preserved as they are from the original experts!
    # So for head parameters, we just copy them from the respective expert.
    
    # 1. Average backbone parameters
    for key in averaged_state_dict.keys():
        if "backbone" in key:
            # Accumulate across experts
            param_sum = experts[0].state_dict()[key].clone().float()
            for i in range(1, K):
                param_sum += experts[i].state_dict()[key].float()
            averaged_state_dict[key] = (param_sum / K).to(averaged_state_dict[key].dtype)
            
    # 2. Copy head parameters directly from their respective expert
    for i, task_name in target_model.task_mapping.items():
        # Find head keys for this task
        for key in averaged_state_dict.keys():
            if f"heads.{task_name}" in key:
                # Get this parameter directly from experts[i]
                averaged_state_dict[key] = experts[i].state_dict()[key].clone()
                
    target_model.load_state_dict(averaged_state_dict)
    return target_model

def merge_models_task_arithmetic(experts, base_model, target_model, lam=0.3):
    """
    Merges experts using Task Arithmetic:
    theta_merged = theta_base + lam * sum(theta_expert - theta_base)
    Also averages BatchNorm parameters (running stats and affines) and copies heads.
    """
    K = len(experts)
    base_state_dict = base_model.state_dict()
    merged_state_dict = copy.deepcopy(base_state_dict)
    
    # 1. Apply Task Arithmetic to learnable backbone weights/biases
    for key in base_state_dict.keys():
        if "backbone" in key:
            # Check if it is a learnable parameter (weight/bias) or a running buffer (mean/var)
            is_buffer = "running_mean" in key or "running_var" in key or "num_batches_tracked" in key
            
            if not is_buffer:
                # Learnable parameter: apply task arithmetic
                delta_sum = torch.zeros_like(base_state_dict[key]).float()
                for i in range(K):
                    delta_sum += (experts[i].state_dict()[key].float() - base_state_dict[key].float())
                merged_state_dict[key] = (base_state_dict[key].float() + lam * delta_sum).to(base_state_dict[key].dtype)
            else:
                # Buffer parameter (like BN running stats): average them
                buffer_sum = experts[0].state_dict()[key].clone().float()
                for i in range(1, K):
                    buffer_sum += experts[i].state_dict()[key].float()
                merged_state_dict[key] = (buffer_sum / K).to(base_state_dict[key].dtype)
                
    # 2. Copy heads directly
    for i, task_name in target_model.task_mapping.items():
        for key in merged_state_dict.keys():
            if f"heads.{task_name}" in key:
                merged_state_dict[key] = experts[i].state_dict()[key].clone()
                
    target_model.load_state_dict(merged_state_dict)
    return target_model

# N-TAAC Calibration Implementation
def calibrate_model_ntaac(model, joint_calibration_loader, device, momentum=1.0):
    """
    Native Task-Agnostic Activation Calibration (N-TAAC).
    Puts the model in train mode, freezes learnable parameters, sets BN momentum,
    and runs a single forward pass over the joint calibration dataset.
    """
    model.to(device)
    # 1. Put model in train mode
    model.train()
    
    # 2. Freeze all parameters (prevent updates to weights/biases)
    for param in model.parameters():
        param.requires_grad = False
        
    # 3. Modify BatchNorm layers to set momentum = 1.0 and keep track of original momentum
    orig_momentums = {}
    bn_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            orig_momentums[name] = module.momentum
            module.momentum = momentum
            bn_layers.append(module)
            
    # 4. Run a single pass over the loader
    # Since loader contains joint samples (img, label, task_id)
    with torch.no_grad():
        for imgs, labels, task_ids in joint_calibration_loader:
            imgs = imgs.to(device)
            # Run forward pass for each task_id or generic forward pass if we bypass head
            # N-TAAC only cares about backbone BN updates, so we can just run forward through backbone
            _ = model.backbone(imgs)
            
    # 5. Restore original BN momentums and put back in eval mode
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            module.momentum = orig_momentums[name]
            
    model.eval()
    for param in model.parameters():
        param.requires_grad = True
    return model

# LSC / TSC Calibration Implementation
class LSCActivationHook:
    def __init__(self):
        self.activations = []
        
    def hook_fn(self, module, input, output):
        # Store standard deviation of activations globally across entire batch, channel, and spatial dimensions
        # output shape: [B, C, H, W]
        # We can calculate standard deviation globally over all dimensions
        # or we can keep it as a positive scalar.
        self.activations.append(output.detach())

def get_layer_stds(model, calibration_loader, device, task_id):
    """
    Passes calibration loader through model and computes global standard deviation for each BatchNorm layer.
    """
    model.to(device)
    model.eval()
    
    handles = []
    activation_hooks = []
    bn_modules = []
    
    # Find all BatchNorm layers
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            bn_modules.append(module)
            hook = LSCActivationHook()
            activation_hooks.append(hook)
            handles.append(module.register_forward_hook(hook.hook_fn))
            
    # Run forward pass
    with torch.no_grad():
        for imgs, labels in calibration_loader:
            imgs = imgs.to(device)
            _ = model(imgs, task_id=task_id)
            
    # Remove hooks
    for handle in handles:
        handle.remove()
        
    # Compute global std across all samples, channels, and spatial dims
    stds = []
    for hook in activation_hooks:
        # Concatenate all recorded activations along batch dimension
        act = torch.cat(hook.activations, dim=0) # [NumSamples, C, H, W]
        # Compute global standard deviation across all dimensions
        # Var(X) = E[X^2] - E[X]^2
        mean = act.mean()
        var = act.var(unbiased=False)
        std = torch.sqrt(var + 1e-5)
        stds.append(std.item())
        
    return stds

class LSCScalingHook:
    def __init__(self, scale_factor):
        self.scale_factor = scale_factor
        
    def hook_fn(self, module, input, output):
        return output * self.scale_factor

def apply_lsc_calibration(merged_model, experts_stds, merged_stds, tau=1.0):
    """
    Calculates LSC (or TSC if tau > 1.0) scaling factors and registers forward hooks
    to scale BatchNorm outputs during evaluation.
    Returns:
        dict: {task_id: [scaling_factors]}
    """
    # K tasks
    K = len(experts_stds)
    scaling_factors = {}
    
    for task_id in range(K):
        factors = []
        for l in range(len(merged_stds[task_id])):
            orig_std = experts_stds[task_id][l]
            merged_std = merged_stds[task_id][l]
            gamma = orig_std / merged_std
            
            # Apply threshold for TSC
            if tau > 1.0:
                if gamma < tau:
                    gamma = 1.0
            factors.append(gamma)
        scaling_factors[task_id] = factors
        
    return scaling_factors

class CalibratedMultiTaskModel(nn.Module):
    """
    Wrapper for MultiTaskResNet18 that dynamically applies LSC scaling factors
    during inference based on the active task_id.
    """
    def __init__(self, base_model, scaling_factors=None):
        super(CalibratedMultiTaskModel, self).__init__()
        self.base_model = base_model
        # scaling_factors: {task_id: [list of floats for each BN layer]}
        self.scaling_factors = scaling_factors
        self.hooks = []
        
    def register_calibration_hooks(self, task_id):
        # Remove any existing hooks
        self.remove_calibration_hooks()
        
        if self.scaling_factors is None or task_id not in self.scaling_factors:
            return
            
        factors = self.scaling_factors[task_id]
        bn_idx = 0
        
        # We iterate and register scaling hooks
        for name, module in self.base_model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                factor = factors[bn_idx]
                if factor != 1.0:
                    # Register a hook to scale output
                    def make_hook(f):
                        return lambda m, inp, out: out * f
                    hook = module.register_forward_hook(make_hook(factor))
                    self.hooks.append(hook)
                bn_idx += 1
                
    def remove_calibration_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
    def forward(self, x, task_id=None):
        if task_id is not None:
            self.register_calibration_hooks(task_id)
        out = self.base_model(x, task_id=task_id)
        self.remove_calibration_hooks()
        return out
