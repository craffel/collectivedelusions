import torch
import torch.nn as nn
import torch.nn.functional as F

def test_center():
    B, C_in, C_out, H, W = 2, 4, 8, 32, 32
    k1, k2 = 3, 3
    p1 = 1
    p2 = k2 // 2
    
    # Create input
    x = torch.randn(B, C_in, H, W)
    
    # Create Conv2d and Depthwise Conv2d with bias
    conv1 = nn.Conv2d(C_in, C_out, kernel_size=k1, padding=p1, stride=1, bias=True)
    conv_dw = nn.Conv2d(C_out, C_out, kernel_size=k2, padding=p2, stride=1, groups=C_out, bias=True)
    
    # Forward sequentially
    y1 = conv1(x)
    z1 = conv_dw(y1)
    
    # Fuse weights
    w1 = conv1.weight
    b1 = conv1.bias
    wdw = conv_dw.weight
    bdw = conv_dw.bias
    
    w_fused = torch.zeros(C_out, C_in, k1 + k2 - 1, k1 + k2 - 1)
    for c in range(C_out):
        for i in range(C_in):
            slice1 = w1[c, i].unsqueeze(0).unsqueeze(0)
            kernel = wdw[c, 0].unsqueeze(0).unsqueeze(0)
            kernel_flipped = torch.flip(kernel, dims=[2, 3])
            fused_slice = F.conv2d(slice1, kernel_flipped, padding=k2 - 1)
            w_fused[c, i] = fused_slice.squeeze()
            
    b_fused = torch.zeros(C_out)
    for c in range(C_out):
        b_fused[c] = b1[c] * wdw[c, 0].sum() + bdw[c]
        
    # Create fused Conv2d layer
    conv_fused = nn.Conv2d(C_in, C_out, kernel_size=k1 + k2 - 1, padding=p1 + p2, stride=1, bias=True)
    conv_fused.weight.data.copy_(w_fused)
    conv_fused.bias.data.copy_(b_fused)
    
    z2 = conv_fused(x)
    
    # Check difference at the center (ignoring 2 pixels border)
    border = 2
    z1_center = z1[:, :, border:-border, border:-border]
    z2_center = z2[:, :, border:-border, border:-border]
    
    diff_center = torch.max(torch.abs(z1_center - z2_center))
    diff_all = torch.max(torch.abs(z1 - z2))
    print("Max difference (all):", diff_all.item())
    print("Max difference (center):", diff_center.item())

if __name__ == '__main__':
    test_center()
