import os
import torch
import timm
import numpy as np
from torch.utils.data import Subset, DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Set seeds
torch.manual_seed(42)
np.random.seed(42)

class GrayscaleToRGB(object):
    def __call__(self, img):
        return img.convert('RGB')

def get_datasets(data_dir='./data'):
    transform_rgb = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_gray = transforms.Compose([
        GrayscaleToRGB(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    mnist_train = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform_gray)
    fmnist_train = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform_gray)
    cifar_train = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_rgb)
    svhn_train = datasets.SVHN(root=data_dir, split='train', download=True, transform=transform_rgb)
    
    mnist_test = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform_gray)
    fmnist_test = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform_gray)
    cifar_test = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_rgb)
    svhn_test = datasets.SVHN(root=data_dir, split='test', download=True, transform=transform_rgb)
    
    return {
        'MNIST': (mnist_train, mnist_test),
        'FashionMNIST': (fmnist_train, fmnist_test),
        'CIFAR10': (cifar_train, cifar_test),
        'SVHN': (svhn_train, svhn_test)
    }

def get_layer_group_idx(key):
    if 'patch_embed' in key or 'cls_token' in key or 'pos_embed' in key:
        return 0
    elif 'blocks.' in key:
        parts = key.split('.')
        return int(parts[1]) + 1
    elif 'norm.' in key:
        return 13
    else:
        return None

def merge_weights(base_state_dict, task_vectors, alpha_bar, device='cpu'):
    merged_state_dict = {}
    for key in base_state_dict:
        group_idx = get_layer_group_idx(key)
        if group_idx is not None:
            merged_val = base_state_dict[key].clone().to(device)
            for k in range(len(task_vectors)):
                coeff = alpha_bar[group_idx, k].to(device)
                merged_val = merged_val + coeff * task_vectors[k][key].to(device)
            merged_state_dict[key] = merged_val
        else:
            merged_state_dict[key] = base_state_dict[key].clone().to(device)
    return merged_state_dict

def evaluate_merged_model(base_model, task_vectors, alpha_bar, task_heads, test_loader, task_idx, device='cpu'):
    pretrained_state = torch.load('checkpoints/base_model.pt', weights_only=True)
    merged_state_dict = merge_weights(pretrained_state, task_vectors, alpha_bar, device=device)
    base_model.load_state_dict(merged_state_dict, strict=False)
    
    base_model.head.weight.data.copy_(task_heads[task_idx]['weight'].to(device))
    base_model.head.bias.data.copy_(task_heads[task_idx]['bias'].to(device))
    
    base_model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = base_model(x)
            correct += (logits.argmax(dim=-1) == y).sum().item()
            total += y.size(0)
    return correct / total

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    task_names = ['MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN']
    data_dict = get_datasets()
    
    train_size = 2000
    calib_size = 64
    test_size = 500
    
    train_loaders = {}
    calib_loaders = {}
    test_loaders = {}
    
    base_model = timm.create_model('vit_tiny_patch16_224', pretrained=False)
    base_model.head = torch.nn.Linear(192, 10)
    base_model.load_state_dict(torch.load('checkpoints/base_model.pt', weights_only=True))
    base_model.to(device)
    
    pretrained_state = torch.load('checkpoints/base_model.pt', weights_only=True)
    
    expert_models = {}
    task_heads = []
    
    for task_idx, name in enumerate(task_names):
        train_ds, test_ds = data_dict[name]
        
        train_idx = list(range(min(train_size, len(train_ds))))
        calib_idx = list(range(min(train_size, train_size + calib_size)))
        test_idx = list(range(min(test_size, len(test_ds))))
        
        train_sub = Subset(train_ds, train_idx)
        calib_sub = Subset(train_ds, calib_idx)
        test_sub = Subset(test_ds, test_idx)
        
        train_loaders[name] = DataLoader(train_sub, batch_size=64, shuffle=True)
        calib_loaders[name] = DataLoader(calib_sub, batch_size=16, shuffle=False)
        test_loaders[name] = DataLoader(test_sub, batch_size=64, shuffle=False)
        
        expert_path = f'checkpoints/expert_{name}.pt'
        expert = timm.create_model('vit_tiny_patch16_224', pretrained=False)
        expert.head = torch.nn.Linear(192, 10)
        expert.load_state_dict(torch.load(expert_path, weights_only=True))
        expert_models[name] = expert
        
        task_heads.append({
            'weight': expert.head.weight.data.clone(),
            'bias': expert.head.bias.data.clone()
        })

    task_vectors = []
    for name in task_names:
        expert_state = expert_models[name].state_dict()
        vector = {}
        for key in pretrained_state:
            if pretrained_state[key].dtype in [torch.int64, torch.uint8]:
                continue
            if get_layer_group_idx(key) is None:
                continue
            vector[key] = expert_state[key] - pretrained_state[key]
        task_vectors.append(vector)

    print("\n--- Optimizing Task-Specific OFS-Tune (Supervised Static, Task-Conditional) ---")
    task_specific_coeffs = torch.nn.Parameter(torch.ones(4, 14, 4) * 0.3)  # K x L x K
    optimizer = torch.optim.Adam([task_specific_coeffs], lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    
    for step in range(50):
        optimizer.zero_grad()
        total_loss = 0.
        for idx, name in enumerate(task_names):
            x_cal, y_cal = next(iter(calib_loaders[name]))
            x_cal, y_cal = x_cal.to(device), y_cal.to(device)
            # Use task-specific coefficients
            alpha_task = task_specific_coeffs[idx] # L x K
            merged_state = merge_weights(pretrained_state, task_vectors, alpha_task, device=device)
            task_merged_state = {k: v for k, v in merged_state.items()}
            task_merged_state['head.weight'] = task_heads[idx]['weight'].to(device)
            task_merged_state['head.bias'] = task_heads[idx]['bias'].to(device)
            logits = torch.func.functional_call(base_model, task_merged_state, (x_cal,))
            total_loss += criterion(logits, y_cal)
        total_loss.backward()
        optimizer.step()
        if (step+1) % 10 == 0:
            print(f"Step {step+1}/50 - Loss: {total_loss.item():.4f}")
            
    accs = []
    for idx, name in enumerate(task_names):
        alpha_task = task_specific_coeffs[idx].data
        acc = evaluate_merged_model(base_model, task_vectors, alpha_task, task_heads, test_loaders[name], idx, device=device)
        print(f"Task-Specific OFS-Tune {name} Accuracy: {acc*100:.2f}%")
        accs.append(acc)
    print(f"Average Accuracy: {np.mean(accs)*100:.2f}%")

if __name__ == '__main__':
    main()
