import torch
import torch.nn as nn
from torchvision import models

def fold_batchnorm(conv, bn):
    with torch.no_grad():
        w_conv = conv.weight
        b_conv = conv.bias if conv.bias is not None else torch.zeros(conv.out_channels, device=w_conv.device)
        
        mean = bn.running_mean
        var = bn.running_var
        eps = bn.eps
        w_bn = bn.weight
        b_bn = bn.bias
        
        scale = w_bn / torch.sqrt(var + eps)
        w_folded = w_conv * scale.view(-1, 1, 1, 1)
        b_folded = (b_conv - mean) * scale + b_bn
        
        folded_conv = nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=True
        )
        folded_conv.weight.copy_(w_folded)
        folded_conv.bias.copy_(b_folded)
        return folded_conv

def fold_resnet18(model):
    # Clone model
    import copy
    folded_model = copy.deepcopy(model)
    folded_model.eval()
    
    # Fold root conv1 & bn1
    folded_model.conv1 = fold_batchnorm(folded_model.conv1, folded_model.bn1)
    folded_model.bn1 = nn.Identity()
    
    # Fold layers
    for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
        layer = getattr(folded_model, layer_name)
        for block in layer:
            block.conv1 = fold_batchnorm(block.conv1, block.bn1)
            block.bn1 = nn.Identity()
            
            block.conv2 = fold_batchnorm(block.conv2, block.bn2)
            block.bn2 = nn.Identity()
            
            if block.downsample is not None:
                # Downsample is nn.Sequential(Conv2d, BatchNorm2d)
                block.downsample[0] = fold_batchnorm(block.downsample[0], block.downsample[1])
                block.downsample[1] = nn.Identity()
                
    return folded_model

def test_folding():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.eval()
    
    x = torch.randn(2, 3, 32, 32)
    
    # Original forward
    with torch.no_grad():
        out_orig = model(x)
        
    # Folded forward
    folded_model = fold_resnet18(model)
    with torch.no_grad():
        out_folded = folded_model(x)
        
    diff = torch.max(torch.abs(out_orig - out_folded))
    print("Folding max difference:", diff.item())
    assert diff.item() < 1e-5, f"Folding failed! Diff: {diff.item()}"
    print("BatchNorm folding verified successfully!")

if __name__ == '__main__':
    test_folding()
