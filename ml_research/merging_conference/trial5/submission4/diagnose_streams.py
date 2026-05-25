import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Subset
import numpy as np
from torch.func import functional_call

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

class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=0.4):
        self.std = std
        self.mean = mean
    def __call__(self, tensor):
        noise = torch.randn(tensor.size(), device=tensor.device) * self.std + self.mean
        return torch.clamp(tensor + noise, 0., 1.)

class ContrastShift(object):
    def __init__(self, alpha=0.3):
        self.alpha = alpha
    def __call__(self, tensor):
        return torch.clamp(0.5 + self.alpha * (tensor - 0.5), 0., 1.)

def get_transforms(corruption_type):
    base_transform = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
    if corruption_type == "clean":
        pass
    elif corruption_type == "noise":
        base_transform.append(AddGaussianNoise(std=0.4))
    elif corruption_type == "blur":
        base_transform.append(transforms.GaussianBlur(kernel_size=5, sigma=1.5))
    elif corruption_type == "contrast":
        base_transform.append(ContrastShift(alpha=0.3))
    base_transform.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return transforms.Compose(base_transform)

def run_evaluation(preloaded_batches, base_weights, cifar_weights, svhn_weights, cifar_head_weights, svhn_head_weights, joint_fisher, device):
    encoder_param_names = list(joint_fisher.keys())
    lambdas_cifar = {name: torch.tensor(0.5, requires_grad=True, device=device) for name in encoder_param_names}
    lambdas_svhn = {name: torch.tensor(0.5, requires_grad=True, device=device) for name in encoder_param_names}
    
    sorted_layers = sorted(joint_fisher.items(), key=lambda x: x[1], reverse=True)
    num_to_freeze = int(len(sorted_layers) * (20 / 100.0))
    frozen_layers = {layer[0] for layer in sorted_layers[:num_to_freeze]}
    
    eta_w = {}
    epsilon_scale = 1e-8
    for name in encoder_param_names:
        eta_w[name] = 0.001 / ((joint_fisher[name] + epsilon_scale) ** 1.0)

    param_groups = []
    for name in encoder_param_names:
        if name in frozen_layers:
            lambdas_cifar[name].requires_grad = False
            lambdas_svhn[name].requires_grad = False
        else:
            param_groups.append({'params': [lambdas_cifar[name]], 'lr': eta_w[name]})
            param_groups.append({'params': [lambdas_svhn[name]], 'lr': eta_w[name]})
            
    optimizer = torch.optim.SGD(param_groups, momentum=0.9)
    model = models.resnet18().to(device)
    model.load_state_dict(torch.load("models/base_pretrained.pt", map_location=device))
    model.fc = nn.Linear(512, 10).to(device)
    model.eval()
    
    total_correct = 0
    total_samples = 0
    
    for images, labels, task_type in preloaded_batches:
        active_head = cifar_head_weights if task_type == "cifar" else svhn_head_weights
        merged_params = {}
        for name in base_weights:
            coeff_name = name
            is_buffer = False
            if name.endswith(".running_mean") or name.endswith(".running_var") or name.endswith(".num_batches_tracked"):
                prefix = name.rsplit(".", 1)[0]
                coeff_name = f"{prefix}.weight"
                is_buffer = True
                
            if coeff_name in lambdas_cifar:
                l_c = lambdas_cifar[coeff_name].detach() if is_buffer else lambdas_cifar[coeff_name]
                l_s = lambdas_svhn[coeff_name].detach() if is_buffer else lambdas_svhn[coeff_name]
                merged_params[name] = (
                    base_weights[name] 
                    + l_c * (cifar_weights[name] - base_weights[name]) 
                    + l_s * (svhn_weights[name] - base_weights[name])
                )
            else:
                merged_params[name] = base_weights[name]
                
        for name in active_head:
            merged_params[f"fc.{name}"] = active_head[name]
            
        outputs = functional_call(model, merged_params, images)
        probs = F.softmax(outputs, dim=1)
        entropy_loss = -torch.mean(torch.sum(probs * torch.log(probs + 1e-5), dim=1))
        
        optimizer.zero_grad()
        entropy_loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            for name in encoder_param_names:
                lambdas_cifar[name].clamp_(0.0, 1.0)
                lambdas_svhn[name].clamp_(0.0, 1.0)
                
        with torch.no_grad():
            merged_params = {}
            for name in base_weights:
                coeff_name = name
                is_buffer = False
                if name.endswith(".running_mean") or name.endswith(".running_var") or name.endswith(".num_batches_tracked"):
                    prefix = name.rsplit(".", 1)[0]
                    coeff_name = f"{prefix}.weight"
                    is_buffer = True
                    
                if coeff_name in lambdas_cifar:
                    l_c = lambdas_cifar[coeff_name].detach() if is_buffer else lambdas_cifar[coeff_name]
                    l_s = lambdas_svhn[coeff_name].detach() if is_buffer else lambdas_svhn[coeff_name]
                    merged_params[name] = (
                        base_weights[name] 
                        + l_c * (cifar_weights[name] - base_weights[name]) 
                        + l_s * (svhn_weights[name] - base_weights[name])
                    )
                else:
                    merged_params[name] = base_weights[name]
                    
            for name in active_head:
                merged_params[f"fc.{name}"] = active_head[name]
                
            outputs = functional_call(model, merged_params, images)
            _, predicted = outputs.max(1)
            correct = predicted.eq(labels).sum().item()
            total_correct += correct
            total_samples += labels.size(0)
            
    return 100.0 * total_correct / total_samples

base_model = models.resnet18().to(device)
base_model.load_state_dict(torch.load("models/base_pretrained.pt", map_location=device))
base_weights = {name: param.clone().detach().to(device) for name, param in base_model.state_dict().items() if not name.startswith('fc')}

cifar_model = models.resnet18().to(device)
cifar_model.fc = nn.Linear(512, 10).to(device)
cifar_model.load_state_dict(torch.load("models/cifar10_expert.pt", map_location=device))
cifar_weights = {name: param.clone().detach().to(device) for name, param in cifar_model.state_dict().items() if not name.startswith('fc')}
cifar_head_weights = {name.replace("fc.", ""): param.clone().detach().to(device) for name, param in cifar_model.named_parameters() if name.startswith('fc.')}

svhn_model = models.resnet18().to(device)
svhn_model.fc = nn.Linear(512, 10).to(device)
svhn_model.load_state_dict(torch.load("models/svhn_expert.pt", map_location=device))
svhn_weights = {name: param.clone().detach().to(device) for name, param in svhn_model.state_dict().items() if not name.startswith('fc')}
svhn_head_weights = {name.replace("fc.", ""): param.clone().detach().to(device) for name, param in svhn_model.named_parameters() if name.startswith('fc.')}

corruptions = ["clean", "contrast", "blur"]
stream_types = ["alternating", "sequential"]

# Preload streams
preloaded_streams = {}
for corruption in corruptions:
    transform = get_transforms(corruption)
    cifar_test = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=False)
    cifar_subset = Subset(cifar_test, list(range(1024)))
    cifar_loader_preload = DataLoader(cifar_subset, batch_size=64, shuffle=False)
    cifar_batches = []
    for imgs, lbls in cifar_loader_preload:
        cifar_batches.append((imgs.to(device), lbls.to(device)))
        
    svhn_test = torchvision.datasets.SVHN(root='./data', split='test', transform=transform, download=False)
    svhn_subset = Subset(svhn_test, list(range(1024)))
    svhn_loader_preload = DataLoader(svhn_subset, batch_size=64, shuffle=False)
    svhn_batches = []
    for imgs, lbls in svhn_loader_preload:
        svhn_batches.append((imgs.to(device), lbls.to(device)))
        
    for stream_type in stream_types:
        preloaded_batches = []
        if stream_type == "alternating":
            for i in range(16):
                preloaded_batches.append((cifar_batches[i][0], cifar_batches[i][1], "cifar"))
                preloaded_batches.append((svhn_batches[i][0], svhn_batches[i][1], "svhn"))
        elif stream_type == "sequential":
            for i in range(16):
                preloaded_batches.append((cifar_batches[i][0], cifar_batches[i][1], "cifar"))
            for i in range(16):
                preloaded_batches.append((svhn_batches[i][0], svhn_batches[i][1], "svhn"))
        preloaded_streams[f"{corruption}_{stream_type}"] = preloaded_batches

for n in [10, 50, 100, 200]:
    jf = compute_fisher_sensitivity_parameterized(n)
    print(f"\nEvaluating stream accuracies for N_cal = {n}")
    accs = []
    for key, stream in preloaded_streams.items():
        acc = run_evaluation(stream, base_weights, cifar_weights, svhn_weights, cifar_head_weights, svhn_head_weights, jf, device)
        print(f"  {key}: {acc:.2f}%")
        accs.append(acc)
    print(f"  MEAN ACCURACY: {np.mean(accs):.2f}%")
