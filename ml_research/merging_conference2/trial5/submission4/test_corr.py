import torch
import torch.nn as nn
import torch.nn.functional as F

def test_corr_loop():
    B, C_in, C_out, H, W = 2, 4, 8, 16, 16
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
            slice1 = w1[c, i].unsqueeze(0).unsqueeze(0) # (1, 1, k1, k1)
            kernel = wdw[c, 0].unsqueeze(0).unsqueeze(0) # (1, 1, k2, k2)
            # Use Option 2: F.conv2d(w1, flip(wdw), padding=k2-1)
            kernel_flipped = torch.flip(kernel, dims=[2, 3])
            fused_slice = F.conv2d(slice1, kernel_flipped, padding=k2 - 1)
            w_fused[c, i] = fused_slice.squeeze()
            
    # Fuse bias: b_fused[c] = b1[c] * sum(wdw[c, 0]) + bdw[c]
    b_fused = torch.zeros(C_out)
    for c in range(C_out):
        b_fused[c] = b1[c] * wdw[c, 0].sum() + bdw[c]
        
    # Create fused Conv2d layer
    conv_fused = nn.Conv2d(C_in, C_out, kernel_size=k1 + k2 - 1, padding=p1 + p2, stride=1, bias=True)
    conv_fused.weight.data.copy_(w_fused)
    conv_fused.bias.data.copy_(b_fused)
    
    z2 = conv_fused(x)
    
    # Check max difference on the interior pixels (excluding the boundary of size p2)
    border = p2
    z1_interior = z1[:, :, border:-border, border:-border]
    z2_interior = z2[:, :, border:-border, border:-border]
    
    diff_all = torch.max(torch.abs(z1 - z2))
    diff_interior = torch.max(torch.abs(z1_interior - z2_interior))
    print("Max difference (all):", diff_all.item())
    print("Max difference (interior):", diff_interior.item())
    assert diff_interior.item() < 1e-5, f"Parity check failed: {diff_interior.item()}"
    print("Loop-based fusion matches perfectly!")

if __name__ == '__main__':
    test_corr_loop()
