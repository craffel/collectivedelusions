import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import copy
from torch.utils.data import DataLoader
from run_experiments_optimized import OFSTuneModel, ExpertHeadsWrapper, get_calibration_dataset, get_dataset, evaluate_stream, TensorDataset
import random

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

# Create test loader
test_images = []
test_labels = []
test_task_indices = []
for task_idx, ds in enumerate(datasets_list):
    test_ds = get_dataset(ds, split='test')
    indices = list(range(len(test_ds)))
    random.seed(42)
    random.shuffle(indices)
    indices = indices[:50]  # Just 50 samples per task for quick evaluation
    for idx in indices:
        img, label = test_ds[idx]
        test_images.append(img)
        test_labels.append(label)
        test_task_indices.append(task_idx)

shuffled_indices = list(range(len(test_images)))
random.shuffle(shuffled_indices)
test_loader = DataLoader(
    TensorDataset(torch.stack(test_images)[shuffled_indices], torch.tensor(test_labels)[shuffled_indices], torch.tensor(test_task_indices)[shuffled_indices]),
    batch_size=16, shuffle=False
)

# Custom OFSTuneModel with frozen base model
class OFSTuneModelFrozen(nn.Module):
    def __init__(self, base_model, experts, k_tasks):
        super().__init__()
        self.base_model = copy.deepcopy(base_model)
        self.k_tasks = k_tasks
        self.alphas = nn.Parameter(torch.full((k_tasks, 12), 0.3))
        
        # Replace linear layers
        self.replace_linear_layers(self.base_model, experts)
        
        # FREEZE base model parameters!
        for p in self.base_model.parameters():
            p.requires_grad = False
            
    def replace_linear_layers(self, model, experts):
        blocks_module = model.blocks
        for idx in range(len(blocks_module)):
            block = blocks_module[idx]
            exp_blocks = [exp.blocks[idx] for exp in experts]
            self._replace_block_linears(block, exp_blocks, idx)

    def _replace_block_linears(self, module, exp_modules, block_idx):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                from run_experiments_optimized import OFSTuneLinear
                expert_linears = [getattr(exp, name) for exp in exp_modules]
                new_linear = OFSTuneLinear(child, expert_linears, self.k_tasks, block_idx, self)
                setattr(module, name, new_linear)
            else:
                self._replace_block_linears(child, [getattr(exp, name) for exp in exp_modules], block_idx)

    def forward(self, x):
        return self.base_model(x)

ofs_model = OFSTuneModelFrozen(base_model, experts, k_tasks=4).to(device)
wrapped_ofs = ExpertHeadsWrapper(ofs_model, expert_heads).to(device)

print("Check trainable parameters:")
for name, p in ofs_model.named_parameters():
    if p.requires_grad:
        print(f"  {name}: {p.shape}")

optimizer = torch.optim.Adam(ofs_model.parameters(), lr=1e-2) # try slightly higher lr for alphas
criterion = nn.CrossEntropyLoss()

ofs_model.train()
for step in range(100):
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

# Evaluate
acc = evaluate_stream(wrapped_ofs, test_loader, device)
print(f"Frozen OFS-Tune accuracy: {acc:.2f}%")
print("Alphas mean after training:", ofs_model.alphas.mean().item(), "std:", ofs_model.alphas.std().item())
