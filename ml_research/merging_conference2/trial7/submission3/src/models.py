import torch
import torch.nn as nn

class TestTimeBatchNorm2d(nn.Module):
    def __init__(self, original_bn):
        super().__init__()
        self.num_features = original_bn.num_features
        self.eps = original_bn.eps
        self.momentum = original_bn.momentum
        self.affine = original_bn.affine
        
        # Copy parameters and buffers
        if self.affine:
            self.weight = nn.Parameter(original_bn.weight.data.clone())
            self.bias = nn.Parameter(original_bn.bias.data.clone())
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
            
        self.register_buffer('running_mean', original_bn.running_mean.data.clone())
        self.register_buffer('running_var', original_bn.running_var.data.clone())
        
        # Dynamic blending variables
        self.alpha = 0.0  # blending factor: 0.0 = static uncalibrated, 1.0 = full test-time stats
        
    def forward(self, x):
        if self.alpha == 0.0:
            # Revert to standard evaluation mode using static running statistics
            return nn.functional.batch_norm(
                x, self.running_mean, self.running_var,
                self.weight, self.bias, False, 0.0, self.eps
            )
        
        # Compute active batch statistics across batch and spatial dimensions
        # x has shape [B, C, H, W]
        # We compute mean and variance over dim=(0, 2, 3)
        batch_mean = x.mean(dim=(0, 2, 3))
        # Use unbiased=False for variance as standard in PyTorch batchnorm
        batch_var = x.var(dim=(0, 2, 3), unbiased=False)
        
        # Blend running stats with active batch stats
        active_mean = (1 - self.alpha) * self.running_mean + self.alpha * batch_mean
        active_var = (1 - self.alpha) * self.running_var + self.alpha * batch_var
        
        return nn.functional.batch_norm(
            x, active_mean, active_var,
            self.weight, self.bias, False, 0.0, self.eps
        )

def patch_bn_to_test_time(model):
    """
    Recursively replaces all nn.BatchNorm2d layers in a model with TestTimeBatchNorm2d.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            setattr(model, name, TestTimeBatchNorm2d(module))
        else:
            patch_bn_to_test_time(module)
    return model
