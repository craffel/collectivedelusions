import torch
import sys
import os

print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH"))
print("CUDA_HOME:", os.environ.get("CUDA_HOME"))
print("CUDA_PATH:", os.environ.get("CUDA_PATH"))

print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device Count:", torch.cuda.device_count())
    
    try:
        x = torch.randn(10, 10, device='cuda')
        y = torch.matmul(x, x)
        print("Matmul works!", y.shape)
    except Exception as e:
        print("Matmul failed:", e)
        
    try:
        print("Enabling cuDNN = True (default)")
        torch.backends.cudnn.enabled = True
        conv = torch.nn.Conv2d(1, 10, 3).cuda()
        img = torch.randn(1, 1, 224, 224, device='cuda')
        out = conv(img)
        print("Conv2d with cuDNN works!", out.shape)
    except Exception as e:
        print("Conv2d with cuDNN failed:", e)

    try:
        print("Disabling cuDNN = False")
        torch.backends.cudnn.enabled = False
        conv = torch.nn.Conv2d(1, 10, 3).cuda()
        img = torch.randn(1, 1, 224, 224, device='cuda')
        out = conv(img)
        print("Conv2d without cuDNN works!", out.shape)
    except Exception as e:
        print("Conv2d without cuDNN failed:", e)
