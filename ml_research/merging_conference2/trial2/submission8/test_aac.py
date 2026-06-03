import torch
import torch.nn as nn
import torchvision.models as models
import math

def relu_propagation(mu, var, eps=1e-8):
    sigma = torch.sqrt(torch.clamp(var, min=eps))
    alpha = -mu / sigma
    
    # Standard normal PDF
    phi = torch.exp(-0.5 * alpha**2) / math.sqrt(2 * math.pi)
    
    # Standard normal CDF
    cdf = 0.5 * (1.0 + torch.erf(alpha / math.sqrt(2.0)))
    
    mean_relu = mu * (1.0 - cdf) + sigma * phi
    m2_relu = (mu**2 + var) * (1.0 - cdf) + mu * sigma * phi
    var_relu = m2_relu - mean_relu**2
    
    # Handle small variance case
    mean_relu = torch.where(sigma < 1e-5, torch.clamp(mu, min=0.0), mean_relu)
    var_relu = torch.where(sigma < 1e-5, torch.where(mu >= 0.0, var, torch.zeros_like(var)), var_relu)
    
    return mean_relu, torch.clamp(var_relu, min=0.0)

def propagate_stats(model, m_in, v_in):
    stats = {}
    m = m_in
    v = v_in
    
    # 1. conv1
    w_sum = model.conv1.weight.sum(dim=(2, 3))
    w_sq_sum = (model.conv1.weight**2).sum(dim=(2, 3))
    m = w_sum @ m
    v = w_sq_sum @ v
    stats['bn1'] = (m.clone(), v.clone())
    
    # 2. bn1 output
    m = model.bn1.bias
    v = model.bn1.weight**2
    
    # 3. relu
    m, v = relu_propagation(m, v)
    
    # 4. maxpool (identity mapping for stats)
    
    # 5. layer1, layer2, layer3, layer4
    def propagate_basic_block(block, m_x, v_x, prefix):
        # conv1
        w1_sum = block.conv1.weight.sum(dim=(2, 3))
        w1_sq_sum = (block.conv1.weight**2).sum(dim=(2, 3))
        m_c1 = w1_sum @ m_x
        v_c1 = w1_sq_sum @ v_x
        stats[prefix + '.bn1'] = (m_c1.clone(), v_c1.clone())
        
        # bn1
        m_bn1 = block.bn1.bias
        v_bn1 = block.bn1.weight**2
        
        # relu
        m_r1, v_r1 = relu_propagation(m_bn1, v_bn1)
        
        # conv2
        w2_sum = block.conv2.weight.sum(dim=(2, 3))
        w2_sq_sum = (block.conv2.weight**2).sum(dim=(2, 3))
        m_c2 = w2_sum @ m_r1
        v_c2 = w2_sq_sum @ v_r1
        stats[prefix + '.bn2'] = (m_c2.clone(), v_c2.clone())
        
        # bn2
        m_bn2 = block.bn2.bias
        v_bn2 = block.bn2.weight**2
        
        # downsample
        if block.downsample is not None:
            conv_ds = block.downsample[0]
            bn_ds = block.downsample[1]
            w_ds_sum = conv_ds.weight.sum(dim=(2, 3))
            w_ds_sq_sum = (conv_ds.weight**2).sum(dim=(2, 3))
            m_ds_c = w_ds_sum @ m_x
            v_ds_c = w_ds_sq_sum @ v_x
            stats[prefix + '.downsample.1'] = (m_ds_c.clone(), v_ds_c.clone())
            
            m_ds_out = bn_ds.bias
            v_ds_out = bn_ds.weight**2
        else:
            m_ds_out = m_x
            v_ds_out = v_x
            
        m_sum = m_bn2 + m_ds_out
        v_sum = v_bn2 + v_ds_out
        
        m_out, v_out = relu_propagation(m_sum, v_sum)
        return m_out, v_out

    for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
        layer = getattr(model, layer_name)
        for i, block in enumerate(layer):
            m, v = propagate_basic_block(block, m, v, f"{layer_name}.{i}")
            
    return stats

if __name__ == "__main__":
    print("Testing Analytical Activation Calibration propagation...")
    model = models.resnet18()
    m_in = torch.zeros(3)
    v_in = torch.ones(3)
    
    stats = propagate_stats(model, m_in, v_in)
    print(f"Propagated statistics successfully! Found {len(stats)} BatchNorm layers.")
    for name, (m, v) in list(stats.items())[:5]:
        print(f"Layer {name}: Mean shape {m.shape}, Var shape {v.shape}")
        print(f"  First 3 means: {m[:3].tolist()}, First 3 vars: {v[:3].tolist()}")
