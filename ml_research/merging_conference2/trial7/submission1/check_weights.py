import torch
from torchvision.models import resnet18, ResNet18_Weights

w1 = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).state_dict()
w2 = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).state_dict()

# Check if they are identical
diff = 0.0
for k in w1.keys():
    diff += (w1[k] - w2[k]).abs().sum().item()
print("Difference between two fresh ImageNet weights:", diff)

# Check norms
norm = 0.0
for k in w1.keys():
    if "fc" not in k:
        norm += (w1[k]**2).sum().item()
print("L2 norm of base weights (excluding fc):", norm**0.5)

# Load our saved expert models
try:
    exp = torch.load("models/SGD_LowReg/MNIST_seed42.pt", map_location="cpu")
    diff_exp = 0.0
    for k in w1.keys():
        if "fc" not in k and k in exp:
            diff_exp += ((exp[k] - w1[k])**2).sum().item()
    print("Drift of MNIST_seed42 from base weights:", diff_exp**0.5)
except Exception as e:
    print("Error loading expert:", e)
