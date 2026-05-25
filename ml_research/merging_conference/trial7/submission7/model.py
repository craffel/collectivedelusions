import torch
import torch.nn as nn
import torchvision.models as models

def get_resnet18_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Modify conv1: 1 input channel, kernel size 3x3, stride 1, padding 1, no bias
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # Modify fc: 10 output classes
    model.fc = nn.Linear(512, 10)
    return model

def merge_weights_and_buffers(target_model, experts, lambdas):
    """
    Fuses weights and Batch Normalization buffers of multiple experts.
    lambdas: Either a 1D tensor of shape (K,) for global merging,
             or a dictionary mapping parameter names to 1D tensors of shape (K,) for layer-wise merging.
    """
    # 1. Merge Weights (Parameters)
    for name, param in target_model.named_parameters():
        if isinstance(lambdas, dict):
            l = lambdas[name]
        else:
            l = lambdas
            
        weighted_param = torch.zeros_like(param)
        for k, expert in enumerate(experts):
            expert_param = dict(expert.named_parameters())[name]
            weighted_param += l[k] * expert_param
        param.data.copy_(weighted_param)
        
    # 2. Merge Batch Normalization Buffers
    # We compute the average coefficient for each expert across all layers, autograd-detached.
    if isinstance(lambdas, dict):
        # average coefficients across all named parameters
        stacked_lambdas = torch.stack([l.detach() for l in lambdas.values()], dim=0) # (num_params, K)
        avg_lambdas = stacked_lambdas.mean(dim=0) # (K,)
    else:
        avg_lambdas = lambdas.detach() if isinstance(lambdas, torch.Tensor) else torch.tensor(lambdas, device=experts[0].fc.weight.device)
        
    for name, buf in target_model.named_buffers():
        if "running_mean" in name or "running_var" in name:
            weighted_buf = torch.zeros_like(buf)
            for k, expert in enumerate(experts):
                expert_buf = dict(expert.named_buffers())[name]
                weighted_buf += avg_lambdas[k] * expert_buf
            buf.copy_(weighted_buf)
        elif "num_batches_tracked" in name:
            # Just copy from the first expert
            expert_buf = dict(experts[0].named_buffers())[name]
            buf.copy_(expert_buf)
