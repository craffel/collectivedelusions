import torch
import torch.nn as nn
# Disable cuDNN to bypass initialization errors
torch.backends.cudnn.enabled = False

print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device Name:", torch.cuda.get_device_name(0))
    x = torch.randn(1, 3, 128, 128).cuda()
    conv = nn.Conv2d(3, 64, 3, padding=1).cuda()
    y = conv(x)
    print("Successfully ran Conv2D on GPU! Shape:", y.shape)
else:
    print("No GPU available.")
