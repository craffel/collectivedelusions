import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import random

# Reuse the model definition from experiment
class MultiTaskResNet18(nn.Module):
    def __init__(self, tasks=['mnist', 'fashion', 'cifar10']):
        super().__init__()
        self.backbone = torchvision.models.resnet18(weights=None)
        self.backbone.fc = nn.Identity()
        self.heads = nn.ModuleDict({
            task: nn.Linear(512, 10) for task in tasks
        })
    def forward(self, x, task):
        features = self.backbone(x)
        return self.heads[task](features)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_data(data_dir='./data'):
    transform_gray = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform_rgb = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_ds = {
        'mnist': torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=transform_gray),
        'fashion': torchvision.datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform_gray),
        'cifar10': torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_rgb)
    }
    test_ds = {
        'mnist': torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=transform_gray),
        'fashion': torchvision.datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform_gray),
        'cifar10': torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_rgb)
    }
    return train_ds, test_ds

def create_subsets(train_ds, test_ds, n_train=2000, n_test=500, n_calib=128, seed=42):
    set_seed(seed)
    sub_train, sub_test, sub_calib = {}, {}, {}
    for task in train_ds.keys():
        len_tr = len(train_ds[task])
        indices_tr = np.random.choice(len_tr, n_train, replace=False)
        sub_train[task] = Subset(train_ds[task], indices_tr)
        rem_indices = list(set(range(len_tr)) - set(indices_tr))
        indices_cal = np.random.choice(rem_indices, n_calib, replace=False)
        sub_calib[task] = Subset(train_ds[task], indices_cal)
        len_te = len(test_ds[task])
        indices_te = np.random.choice(len_te, n_test, replace=False)
        sub_test[task] = Subset(test_ds[task], indices_te)
    return sub_train, sub_test, sub_calib

def get_backbone_state(model):
    return {k: v.clone().cpu() for k, v in model.backbone.state_dict().items()}

def set_backbone_state(model, state_dict):
    model.backbone.load_state_dict({k: v.to(next(model.parameters()).device) for k, v in state_dict.items()})

def merge_models(base_state, expert_states, lambdas):
    merged_state = {}
    for key in base_state.keys():
        if base_state[key].is_floating_point():
            update = torch.zeros_like(base_state[key])
            for task, exp_state in expert_states.items():
                update += lambdas[task] * (exp_state[key] - base_state[key])
            merged_state[key] = base_state[key] + update
        else:
            merged_state[key] = base_state[key].clone()
    return merged_state

def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Running on device: {device}')

    train_ds, test_ds = load_data()
    tasks = ['mnist', 'fashion', 'cifar10']

    # Load Base and Expert Weights
    base_state = torch.load('base_backbone.pt', map_location='cpu')
    expert_states = {}
    expert_heads = {}
    for task in tasks:
        expert_states[task] = torch.load(f'expert_backbone_{task}.pt', map_location='cpu')
        expert_heads[task] = torch.load(f'expert_head_{task}.pt', map_location='cpu')

    model = MultiTaskResNet18().to(device)
    lambdas = {'mnist': 0.3, 'fashion': 0.3, 'cifar10': 0.3}
    ta_state = merge_models(base_state, expert_states, lambdas)

    # We evaluate for N = 128 and N = 512
    for N in [128, 512]:
        print(f'\n=== EVALUATING HEAD INITIALIZATION EFFECT FOR N = {N} ===')
        sub_train, sub_test, sub_calib = create_subsets(train_ds, test_ds, n_calib=N)
        calib_loaders = {t: DataLoader(sub_calib[t], batch_size=128, shuffle=False) for t in sub_calib.keys()}
        test_loaders = {t: DataLoader(sub_test[t], batch_size=128, shuffle=False) for t in sub_test.keys()}

        for init_mode in ['expert', 'random']:
            print(f'\n--- Mode: {init_mode.upper()} Initialization ---')
            set_backbone_state(model, ta_state)
            
            # Setup classification heads
            for task in tasks:
                if init_mode == 'expert':
                    model.heads[task].load_state_dict({k: v.to(device) for k, v in expert_heads[task].items()})
                else:
                    # Random Kaiming init
                    nn.init.kaiming_normal_(model.heads[task].weight, nonlinearity='linear')
                    nn.init.constant_(model.heads[task].bias, 0)

            # Freeze backbone, optimize only heads on calibration dataset
            sft_acc = {}
            for task in tasks:
                head_copy = model.heads[task]
                optimizer = torch.optim.AdamW(head_copy.parameters(), lr=1e-3, weight_decay=1e-4)
                model.eval()
                
                for epoch in range(15):
                    for x, y in calib_loaders[task]:
                        x, y = x.to(device), y.to(device)
                        optimizer.zero_grad()
                        with torch.no_grad():
                            features = model.backbone(x)
                        outputs = head_copy(features)
                        loss = F.cross_entropy(outputs, y)
                        loss.backward()
                        optimizer.step()

                # Evaluate
                correct, total = 0, 0
                with torch.no_grad():
                    for x, y in test_loaders[task]:
                        x, y = x.to(device), y.to(device)
                        features = model.backbone(x)
                        outputs = head_copy(features)
                        _, pred = outputs.max(1)
                        correct += pred.eq(y).sum().item()
                        total += y.size(0)
                sft_acc[task] = 100.0 * correct / total
                print(f'  {task.upper()}: {sft_acc[task]:.2f}%')
            print(f'  AVERAGE: {np.mean(list(sft_acc.values())):.2f}%')

if __name__ == '__main__':
    main()
