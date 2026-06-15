import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import copy
from torch.utils.data import DataLoader
from run_experiments_optimized import OFSTuneModel, ExpertHeadsWrapper, get_calibration_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# Load base model
print("Loading base model...")
base_model = timm.create_model('vit_tiny_patch16_224', pretrained=True).to(device)
base_model.reset_classifier(0)

# Load experts
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

# Load calibration dataset
cal_dataset = get_calibration_dataset()
cal_loader = DataLoader(cal_dataset, batch_size=16, shuffle=True)

ofs_model = OFSTuneModel(base_model, experts, k_tasks=4).to(device)
wrapped_ofs = ExpertHeadsWrapper(ofs_model, expert_heads).to(device)

optimizer = torch.optim.Adam(ofs_model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

print("Alphas at start:", ofs_model.alphas.clone().detach())

ofs_model.train()
for step in range(100):
    total_loss = 0.0
    for images, labels, task_indices in cal_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        features = ofs_model(images)
        outputs = []
        for b in range(images.size(0)):
            task_idx = task_indices[b].item()
            out = wrapped_ofs.heads[task_idx](features[b:b+1])
            outputs.append(out)
        outputs = torch.cat(outputs, dim=0)
        loss = criterion(outputs, labels) * images.size(0)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if step % 10 == 0:
        print(f"Step {step}, Loss: {total_loss:.4f}")
        print("Alphas mean:", ofs_model.alphas.mean().item(), "std:", ofs_model.alphas.std().item())

print("Alphas after training:", ofs_model.alphas.clone().detach())
