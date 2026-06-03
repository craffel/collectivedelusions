import torch
from merge_eval import ExpertModel, get_bn_name

progenitor = ExpertModel()
progenitor.load_state_dict(torch.load("checkpoints/progenitor.pt", map_location="cpu"))
progenitor_state = progenitor.state_dict()

expert_states = []
for task in ["mnist", "fmnist", "cifar10"]:
    expert = ExpertModel()
    expert.load_state_dict(torch.load(f"checkpoints/{task}_expert.pt", map_location="cpu"))
    expert_states.append(expert.state_dict())

key = "backbone.conv1.weight"
w_init = progenitor_state[key]
w_merged = torch.stack([expert_states[t][key] for t in range(3)]).mean(dim=0)
w_merged_update = w_merged - w_init
w_expert_updates = [expert_states[t][key] - w_init for t in range(3)]

c = 0
norm_m = torch.norm(w_merged_update[c], p=2)
norm_exp = sum(torch.norm(exp_up[c], p=2) for exp_up in w_expert_updates) / 3
s_c = norm_exp / (norm_m + 1e-8)
print(f"Channel {c} norm_m: {norm_m:.6f}, norm_exp: {norm_exp:.6f}, s_c: {s_c:.6f}")

# Check running mean and running var of bn1
bn_mean_key = "backbone.bn1.running_mean"
bn_var_key = "backbone.bn1.running_var"
mean_merged = torch.stack([expert_states[t][bn_mean_key] for t in range(3)]).mean(dim=0)
var_merged = torch.stack([expert_states[t][bn_var_key] for t in range(3)]).mean(dim=0)

print(f"bn1 merged running_mean (first 5 channels): {mean_merged[:5]}")
print(f"bn1 merged running_var (first 5 channels): {var_merged[:5]}")
