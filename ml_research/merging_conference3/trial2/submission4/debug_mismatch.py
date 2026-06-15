import os
import sys
import torch

# Add SyMerge and SyMerge/src to path
sys.path.append('SyMerge')
sys.path.append('SyMerge/src')
sys.path.append('local_packages')

from task_vectors import TaskVector

pretrained_checkpoint = 'checkpoints_tint/ViT-B-32/zeroshot.pt'
base = torch.load(pretrained_checkpoint, weights_only=False)
expert = torch.load('checkpoints_tint/ViT-B-32/MNIST/finetuned.pt', weights_only=False)

tv = TaskVector(pretrained_checkpoint, 'checkpoints_tint/ViT-B-32/MNIST/finetuned.pt')

base_sd = base.state_dict()
expert_actual_sd = expert.state_dict()
tv_sd = tv.vector

# Reconstructed expert_sd
expert_reconstructed_sd = {}
for k in base_sd:
    if k in tv_sd:
        expert_reconstructed_sd[k] = base_sd[k] + tv_sd[k]
    else:
        expert_reconstructed_sd[k] = base_sd[k]

print('Comparing reconstructed expert_sd with actual expert_sd:')
any_diff = False
for k in base_sd:
    diff = torch.max(torch.abs(expert_reconstructed_sd[k] - expert_actual_sd[k]))
    if diff > 1e-5:
        print(f"Key '{k}' differs by {diff:.6f}!")
        print("  Base mean:", torch.mean(base_sd[k]).item())
        print("  Expert actual mean:", torch.mean(expert_actual_sd[k]).item())
        print("  Task vector mean:", torch.mean(tv_sd[k]).item() if k in tv_sd else "not in tv")
        any_diff = True
        break
print('Any diff?', any_diff)
