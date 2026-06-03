import torch
import torch.nn as nn
import torchvision.models as models
import copy
import os

device = torch.device("cpu")

# Load progenitor and experts
experts_dir = "./experts"
print("Loading progenitor...")
progenitor = models.resnet18()
progenitor_state = copy.deepcopy(progenitor.state_dict())

def get_expert(task_name):
    expert_path = os.path.join(experts_dir, f"expert_{task_name}.pt")
    model = models.resnet18()
    model.fc = nn.Linear(512, 10)
    model.load_state_dict(torch.load(expert_path, map_location=device))
    return model

print("Loading experts...")
expert_mnist = get_expert("mnist")
expert_fmnist = get_expert("fmnist")
expert_cifar10 = get_expert("cifar10")

experts = {
    "mnist": expert_mnist,
    "fmnist": expert_fmnist,
    "cifar10": expert_cifar10
}

# Import our functions from run_experiments_v2
from run_experiments_v2 import apply_channel_update_level_ipr, apply_channel_orthogonality_aware_ipr, get_standard_merge

print("Merging WA...")
wa_base = get_standard_merge(experts, progenitor_state, merge_type="WA")

print("Testing CU-IPR...")
wa_cu_ipr = apply_channel_update_level_ipr(wa_base, experts, progenitor_state)
print("CU-IPR Successful!")

print("Testing CO-IPR...")
wa_co_ipr = apply_channel_orthogonality_aware_ipr(wa_base, experts, progenitor_state)
print("CO-IPR Successful!")

print("All channel-level verification tests passed!")
