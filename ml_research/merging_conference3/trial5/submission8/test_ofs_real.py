import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import copy
import os
from run_experiments_optimized import OFSTuneModel, ExpertHeadsWrapper

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# Load real experts
print("Loading real base model...")
base_model = timm.create_model('vit_tiny_patch16_224', pretrained=True).to(device)
base_model.reset_classifier(0)

datasets_list = ['MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN']
experts = []
expert_heads = []

for ds in datasets_list:
    model_path = f'checkpoints/{ds.lower()}_expert.pth'
    expert = timm.create_model('vit_tiny_patch16_224', pretrained=True)
    expert.head = nn.Linear(expert.head.in_features, 10)
    expert.load_state_dict(torch.load(model_path, map_location='cpu'))
    expert = expert.to(device)

    expert_heads.append(copy.deepcopy(expert.head))
    expert.reset_classifier(0)
    experts.append(expert)

print("All experts loaded.")

ofs_model = OFSTuneModel(base_model, experts, k_tasks=4).to(device)
wrapped_ofs = ExpertHeadsWrapper(ofs_model, expert_heads).to(device)

# Dummy inputs (or real ones from calibration dataset)
images = torch.randn(4, 3, 224, 224).to(device)
labels = torch.randint(0, 10, (4,)).to(device)
task_indices = torch.tensor([0, 1, 2, 3]).to(device)

optimizer = torch.optim.Adam(ofs_model.parameters(), lr=1e-3)
optimizer.zero_grad()

features = ofs_model(images)
outputs = []
for b in range(images.size(0)):
    task_idx = task_indices[b].item()
    out = wrapped_ofs.heads[task_idx](features[b:b+1])
    outputs.append(out)
outputs = torch.cat(outputs, dim=0)

criterion = nn.CrossEntropyLoss()
loss = criterion(outputs, labels)
loss.backward()

print("Alphas grad after step:", ofs_model.alphas.grad)
if ofs_model.alphas.grad is not None:
    print("Grad sum:", ofs_model.alphas.grad.abs().sum().item())
else:
    print("Grad is None!")
