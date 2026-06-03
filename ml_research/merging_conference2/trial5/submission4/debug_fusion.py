import torch
import torch.nn as nn
import torch.nn.functional as F

def debug_fusion():
    # Keep it simple: 1D or 2D with size 3, no padding, to check exact formulas
    x = torch.arange(1.0, 26.0).view(1, 1, 5, 5)
    
    # Conv1 with size 3, no padding
    conv1 = nn.Conv2d(1, 1, kernel_size=3, padding=0, bias=False)
    conv1.weight.data = torch.arange(1.0, 10.0).view(1, 1, 3, 3)
    
    # Conv_dw with size 3, no padding
    conv_dw = nn.Conv2d(1, 1, kernel_size=3, padding=0, bias=False)
    conv_dw.weight.data = torch.tensor([[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]]).view(1, 1, 3, 3) # Identity!
    
    y = conv1(x)
    z = conv_dw(y)
    
    print("x:\n", x[0, 0])
    print("y:\n", y[0, 0])
    print("z:\n", z[0, 0])
    
    # Now let's try to compute fused weight
    # If conv_dw is identity, the fused weight should be conv1 weight!
    # Let's see what F.conv2d(w1, wdw, padding=2) produces:
    w1 = conv1.weight
    wdw = conv_dw.weight
    
    # If we do full convolution of w1 with wdw:
    # Since wdw is identity, the central 3x3 of the full convolution should be w1!
    # Let's print the full convolution
    fused = F.conv2d(w1, wdw, padding=2)
    print("w1:\n", w1[0, 0])
    print("wdw:\n", wdw[0, 0])
    print("fused:\n", fused[0, 0])
    
    # Wait, the fused weight has shape (1, 1, 5, 5)
    # If conv_dw is identity, the fused weight should act as a 3x3 filter.
    # If we pad the fused weight of shape 5x5 to conv_fused, we get:
    conv_fused = nn.Conv2d(1, 1, kernel_size=5, padding=0, bias=False)
    conv_fused.weight.data.copy_(fused)
    z_fused = conv_fused(x)
    print("z_fused:\n", z_fused[0, 0])
    print("Diff with z:", torch.max(torch.abs(z - z_fused)).item())

if __name__ == '__main__':
    debug_fusion()
