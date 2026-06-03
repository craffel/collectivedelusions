import torch
import torch.nn as nn
import torch.nn.functional as F

def test_non_identity():
    # Input
    x = torch.randn(1, 1, 7, 7)
    
    # Conv1: 3x3, no padding, no bias
    conv1 = nn.Conv2d(1, 1, kernel_size=3, padding=0, bias=False)
    w1 = torch.randn(1, 1, 3, 3)
    conv1.weight.data.copy_(w1)
    
    # Conv_dw: 3x3, no padding, no bias
    conv_dw = nn.Conv2d(1, 1, kernel_size=3, padding=0, bias=False)
    wdw = torch.randn(1, 1, 3, 3)
    conv_dw.weight.data.copy_(wdw)
    
    # Sequential forward
    y = conv1(x)
    z = conv_dw(y)
    
    # Now let's try to compute fused weight
    # We want a fused weight of shape (1, 1, 5, 5)
    # PyTorch conv2d(A, B) does correlation:
    # y = conv1(x) -> y[h, w] = sum_{dy, dx} w1[dy, dx] * x[h+dy, w+dx]
    # z = conv_dw(y) -> z[h, w] = sum_{dy2, dx2} wdw[dy2, dx2] * y[h+dy2, w+dx2]
    # Substituting:
    # z[h, w] = sum_{dy2, dx2} wdw[dy2, dx2] sum_{dy, dx} w1[dy, dx] * x[h+dy2+dy, w+dx2+dx]
    # Let H = dy2 + dy, W = dx2 + dx.
    # The weight for x[h+H, w+W] is:
    # w_fused[H, W] = sum_{dy2, dx2} wdw[dy2, dx2] * w1[H - dy2, W - dx2].
    # This is a standard 2D convolution: w_fused = wdw * w1.
    # In PyTorch, since F.conv2d(A, B) does correlation, we can get convolution by:
    # convolving A with B is the same as correlation of A with B flipped.
    # Specifically, F.conv2d(input=wdw, weight=torch.flip(w1, dims=[2,3]), padding=2) or vice versa.
    # Let's test different combinations of F.conv2d with padding=2.
    
    # Option 1: F.conv2d(wdw, torch.flip(w1, [2, 3]), padding=2)
    fused_1 = F.conv2d(wdw, torch.flip(w1, [2, 3]), padding=2)
    
    # Option 2: F.conv2d(w1, torch.flip(wdw, [2, 3]), padding=2)
    fused_2 = F.conv2d(w1, torch.flip(wdw, [2, 3]), padding=2)
    
    # Option 3: F.conv2d(torch.flip(wdw, [2,3]), w1, padding=2)
    fused_3 = F.conv2d(torch.flip(wdw, [2, 3]), w1, padding=2)
    
    # Option 4: F.conv2d(torch.flip(w1, [2,3]), wdw, padding=2)
    fused_4 = F.conv2d(torch.flip(w1, [2, 3]), wdw, padding=2)

    # Option 5: Let's do it using manual convolution / correlation with flip
    # In PyTorch, to convolve w1 with wdw, we can use F.conv2d:
    # w_fused = F.conv2d(w1, torch.flip(wdw, [2, 3]), padding=2)
    # Wait, let's verify if any of 1-4 gives exact match when we evaluate:
    for i, fused in enumerate([fused_1, fused_2, fused_3, fused_4], 1):
        conv_fused = nn.Conv2d(1, 1, kernel_size=5, padding=0, bias=False)
        conv_fused.weight.data.copy_(fused)
        z_fused = conv_fused(x)
        diff = torch.max(torch.abs(z - z_fused)).item()
        print(f"Option {i} diff: {diff}")

if __name__ == '__main__':
    test_non_identity()
