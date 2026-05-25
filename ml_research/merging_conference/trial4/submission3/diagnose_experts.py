import os
import torch
import torch.nn as nn
from torch.func import functional_call
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

SAVE_DIR = "/fsx/craffel/collectivedelusions/ml_research/merging_conference/trial4/submission3"

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class ExpertModel(nn.Module):
    def __init__(self, dropout_prob=0.1):
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.head = nn.Linear(512, 10)
        
    def forward(self, x):
        features = self.backbone(x)
        features = self.dropout(features)
        logits = self.head(features)
        return logits

# Load MNIST test dataset
mnist_test = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
mnist_loader = DataLoader(Subset(mnist_test, list(range(1600))), batch_size=32, shuffle=False)

# Load expert
ckpt_path = os.path.join(SAVE_DIR, "mnist_expert.pt")
checkpoint = torch.load(ckpt_path, map_location=device)

expert = ExpertModel().to(device)
expert.backbone.load_state_dict(checkpoint['backbone_state_dict'])
expert.head.load_state_dict(checkpoint['head_state_dict'])
expert.eval()

# 1. Standalone evaluation
correct = 0
total = 0
with torch.no_grad():
    for images, labels in mnist_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = expert(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
standalone_acc = 100.0 * correct / total
print(f"Standalone Expert Accuracy: {standalone_acc:.2f}%")

# 2. functional_call reconstruction evaluation with state_dict + buffer detach
base_backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
base_backbone.fc = nn.Identity()
base_backbone.to(device).eval()

# Extract from state_dict to include buffers
base_state = {k: v for k, v in base_backbone.state_dict().items()}
expert_state = {k: v for k, v in expert.backbone.state_dict().items()}

# Set parameter names to identify buffers
parameter_names = set(dict(base_backbone.named_parameters()).keys())

# Compute task vectors for all floating-point tensors
task_vector = {}
for k in base_state.keys():
    if base_state[k].is_floating_point():
        task_vector[k] = expert_state[k] - base_state[k]
    else:
        task_vector[k] = expert_state[k]

# Reconstruct merged backbone with lambda = [1, 0, 0] (with requires_grad=True to simulate adaptation)
lambda_coeff = torch.tensor([1.0, 0.0, 0.0], device=device, requires_grad=True)
merged_params = {}
for k in base_state.keys():
    if base_state[k].is_floating_point():
        tensor = base_state[k] + lambda_coeff[0] * task_vector[k]
        if k not in parameter_names:
            tensor = tensor.detach() # Detach buffers from autograd!
        merged_params[k] = tensor
    else:
        merged_params[k] = base_state[k]

correct_recon = 0
# Run forward pass and compute backward pass to verify differentiability
images, labels = next(iter(mnist_loader))
images, labels = images.to(device), labels.to(device)
features = functional_call(base_backbone, merged_params, images)
logits = expert.head(features)
loss = nn.CrossEntropyLoss()(logits, labels)
loss.backward()

print("Autograd successful! Lambda gradient:", lambda_coeff.grad)

# Evaluate overall reconstruction accuracy
correct_recon = 0
with torch.no_grad():
    for images, labels in mnist_loader:
        images, labels = images.to(device), labels.to(device)
        # Note: we recreate merged_params without grad tracking for speed during evaluation
        merged_params_eval = {}
        for k in base_state.keys():
            if base_state[k].is_floating_point():
                merged_params_eval[k] = base_state[k] + lambda_coeff[0].detach() * task_vector[k]
            else:
                merged_params_eval[k] = base_state[k]
        features = functional_call(base_backbone, merged_params_eval, images)
        logits = expert.head(features)
        _, predicted = logits.max(1)
        correct_recon += predicted.eq(labels).sum().item()
        
recon_acc = 100.0 * correct_recon / total
print(f"Reconstructed state_dict (with buffer detach) Expert Accuracy: {recon_acc:.2f}%")
