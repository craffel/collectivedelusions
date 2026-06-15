import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import copy
from run_experiments_optimized import OFSTuneModel, ExpertHeadsWrapper, OFSTuneLinear

# Override OFSTuneLinear.forward to print
original_forward = OFSTuneLinear.forward
def noisy_forward(self, x):
    print(f"OFSTuneLinear.forward called for block {self.block_idx}!")
    return original_forward(self, x)
OFSTuneLinear.forward = noisy_forward

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# Create dummy model
base_model = timm.create_model('vit_tiny_patch16_224', pretrained=False)
base_model.reset_classifier(0)

# Create 4 dummy experts
experts = [copy.deepcopy(base_model) for _ in range(4)]
expert_heads = [nn.Linear(192, 10) for _ in range(4)]

ofs_model = OFSTuneModel(base_model, experts, k_tasks=4).to(device)
wrapped_ofs = ExpertHeadsWrapper(ofs_model, expert_heads).to(device)

# Dummy inputs
images = torch.randn(2, 3, 224, 224).to(device)
labels = torch.randint(0, 10, (2,)).to(device)
task_indices = torch.tensor([0, 1]).to(device)

features = ofs_model(images)
print("Features shape:", features.shape)
