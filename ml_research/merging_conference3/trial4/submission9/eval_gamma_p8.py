import torch
from run_evaluation import get_val_and_test_loaders, merge_epm, evaluate_model, timm, nn
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base_checkpoint = torch.load('checkpoints/base_model.pt', map_location=device)
base_model = timm.create_model('vit_tiny_patch16_224', pretrained=False)
base_model.head = nn.Linear(base_model.num_features, 10)
base_model.load_state_dict(base_checkpoint['state_dict'])
base_model = base_model.to(device)
base_state_dict = {k: v.to(device) for k, v in base_checkpoint['state_dict'].items()}

tasks = ['MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN']
expert_models = []
expert_state_dicts = []
for task in tasks:
    expert_checkpoint = torch.load(f'checkpoints/{task}_expert.pt', map_location=device)
    model = timm.create_model('vit_tiny_patch16_224', pretrained=False)
    model.head = nn.Linear(model.num_features, 10)
    model.load_state_dict(expert_checkpoint['state_dict'])
    model = model.to(device)
    expert_models.append(model)
    expert_state_dicts.append({k: v.to(device) for k, v in expert_checkpoint['state_dict'].items()})

test_loaders = []
for task in tasks:
    _, test_loader = get_val_and_test_loaders(task, val_size=128)
    test_loaders.append(test_loader)

global_stds = []
for k in range(len(tasks)):
    all_vals = []
    for key in base_state_dict.keys():
        if 'head' in key or not base_state_dict[key].is_floating_point():
            continue
        all_vals.append((expert_state_dicts[k][key] - base_state_dict[key]).view(-1))
    all_vals_flat = torch.cat(all_vals)
    global_stds.append(torch.std(all_vals_flat).item())

merged_sd = merge_epm(base_state_dict, expert_state_dicts, [1.0]*4, sparsity=0.8, global_stds=global_stds, gamma=1.0)
accs = evaluate_model(merged_sd, base_model, expert_models, test_loaders, device)
print('Accuracies at p=0.8 (gamma=1.0):', accs, 'Mean:', np.mean(accs))
