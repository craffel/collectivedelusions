import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Subset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_calibration_data(n_cal):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    cifar_train = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=False)
    cifar_subset = Subset(cifar_train, list(range(n_cal)))
    cifar_loader = DataLoader(cifar_subset, batch_size=32, shuffle=False)
    
    svhn_train = torchvision.datasets.SVHN(root='./data', split='train', transform=transform, download=False)
    svhn_subset = Subset(svhn_train, list(range(n_cal)))
    svhn_loader = DataLoader(svhn_subset, batch_size=32, shuffle=False)
    
    return cifar_loader, svhn_loader

def compute_fisher_sensitivity_parameterized(n_cal):
    cifar_loader, svhn_loader = load_calibration_data(n_cal)
    criterion = nn.CrossEntropyLoss()
    
    cifar_model = models.resnet18().to(device)
    cifar_model.fc = nn.Linear(512, 10).to(device)
    cifar_model.load_state_dict(torch.load("models/cifar10_expert.pt", map_location=device))
    cifar_model.eval()
    
    fisher_cifar = {}
    encoder_params = [name for name, _ in cifar_model.named_parameters() if not name.startswith('fc')]
    
    for name in encoder_params:
        p = dict(cifar_model.named_parameters())[name]
        fisher_cifar[name] = torch.zeros_like(p.data)
        
    for images, labels in cifar_loader:
        images, labels = images.to(device), labels.to(device)
        for i in range(len(images)):
            img = images[i:i+1]
            lbl = labels[i:i+1]
            cifar_model.zero_grad()
            outputs = cifar_model(img)
            loss = criterion(outputs, lbl)
            loss.backward()
            
            for name in encoder_params:
                p = dict(cifar_model.named_parameters())[name]
                if p.grad is not None:
                    fisher_cifar[name] += (p.grad.data ** 2)
                    
    for name in encoder_params:
        fisher_cifar[name] /= float(n_cal)
        
    svhn_model = models.resnet18().to(device)
    svhn_model.fc = nn.Linear(512, 10).to(device)
    svhn_model.load_state_dict(torch.load("models/svhn_expert.pt", map_location=device))
    svhn_model.eval()
    
    fisher_svhn = {}
    for name in encoder_params:
        p = dict(svhn_model.named_parameters())[name]
        fisher_svhn[name] = torch.zeros_like(p.data)
        
    for images, labels in svhn_loader:
        images, labels = images.to(device), labels.to(device)
        for i in range(len(images)):
            img = images[i:i+1]
            lbl = labels[i:i+1]
            svhn_model.zero_grad()
            outputs = svhn_model(img)
            loss = criterion(outputs, lbl)
            loss.backward()
            
            for name in encoder_params:
                p = dict(svhn_model.named_parameters())[name]
                if p.grad is not None:
                    fisher_svhn[name] += (p.grad.data ** 2)
                    
    for name in encoder_params:
        fisher_svhn[name] /= float(n_cal)
        
    joint_fisher = {}
    for name in encoder_params:
        mean_cifar = fisher_cifar[name].mean().item()
        mean_svhn = fisher_svhn[name].mean().item()
        joint_fisher[name] = 0.5 * (mean_cifar + mean_svhn)
        
    return joint_fisher

for n in [10, 50, 100, 200]:
    jf = compute_fisher_sensitivity_parameterized(n)
    vals = list(jf.values())
    min_val, max_val, mean_val = min(vals), max(vals), sum(vals)/len(vals)
    
    eta = 0.001
    epsilon_scale = 1e-8
    lrs = [eta / ((v + epsilon_scale) ** 1.0) for v in vals]
    min_lr, max_lr, mean_lr = min(lrs), max(lrs), sum(lrs)/len(lrs)
    
    # Let's count how many layers have LR > 10.0 or something huge
    huge_lrs = sum(1 for lr in lrs if lr > 1.0)
    
    print(f"N_cal = {n:3d}: Fisher Min = {min_val:.8f}, Max = {max_val:.8f}, Mean = {mean_val:.8f}")
    print(f"           LR Min = {min_lr:.4f}, Max = {max_lr:.4f}, Mean = {mean_lr:.4f}, Count LR > 1.0: {huge_lrs}")
