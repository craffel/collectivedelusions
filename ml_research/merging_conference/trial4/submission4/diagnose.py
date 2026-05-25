import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torch.func import functional_call
from torchvision.models import resnet18, ResNet18_Weights

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device for diagnosis:", device)

class ResNetBackbone(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.avgpool = base_model.avgpool

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

def get_dataset(name):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    if name == 'mnist':
        return torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform)
    elif name == 'fashion':
        return torchvision.datasets.FashionMNIST(root='./data', train=False, download=False, transform=transform)
    else:
        return torchvision.datasets.KMNIST(root='./data', train=False, download=False, transform=transform)

base_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
base_backbone = ResNetBackbone(base_model).to(device)
base_backbone_params = {k: v.clone().detach() for k, v in base_backbone.named_parameters()}

expert_backbones = []
expert_heads = []
task_vectors = []
fisher_priors = []

for i in range(3):
    ckpt = torch.load(f'checkpoints/expert_{i}.pt', map_location=device)
    eb = ResNetBackbone(base_model).to(device)
    eb.load_state_dict(ckpt['backbone_state_dict'])
    expert_backbones.append(eb)
    eh = nn.Linear(512, 10).to(device)
    eh.load_state_dict(ckpt['head_state_dict'])
    expert_heads.append(eh)
    eb_params = {k: v.clone().detach() for k, v in eb.named_parameters()}
    tv = {k: eb_params[k] - base_backbone_params[k] for k in base_backbone_params.keys()}
    task_vectors.append(tv)
    fim = torch.load(f'checkpoints/fim_{i}.pt', map_location=device)
    fisher_priors.append(fim)

mnist_test = get_dataset('mnist')
mnist_loader = DataLoader(mnist_test, batch_size=32, shuffle=False)
batches = []
for i, batch in enumerate(mnist_loader):
    if i >= 5: break
    batches.append((batch, 0)) # MNIST is task 0

def diagnose_method(method_name, lr_lambda=0.5, gamma=100.0):
    print(f"\n===== Diagnosing {method_name} (lr_lambda={lr_lambda}, gamma={gamma}) =====")
    adapted_heads = [copy.deepcopy(h).to(device) for h in expert_heads]
    lambda_val = torch.tensor([1/3, 1/3, 1/3], device=device, requires_grad=True)
    
    for step, ((inputs, targets), task_idx) in enumerate(batches):
        inputs, targets = inputs.to(device), targets.to(device)
        active_head = adapted_heads[task_idx]
        for p in active_head.parameters():
            p.requires_grad = True
            
        merged_params = {}
        for k in base_backbone_params.keys():
            merged_params[k] = base_backbone_params[k] + sum(
                lambda_val[i] * task_vectors[i][k] for i in range(3)
            )
            
        features = functional_call(base_backbone, merged_params, inputs)
        logits = active_head(features)
        
        # Calculate loss
        if method_name == 'unconstrained':
            probs = F.softmax(logits, dim=-1)
            loss = -(probs * torch.log(probs + 1e-12)).sum(dim=-1).mean()
            print(f"Step {step} | Loss (Entropy): {loss.item():.4f}")
        elif method_name == 'ewc_tta':
            with torch.no_grad():
                exp_feat = expert_backbones[task_idx](inputs)
                exp_logits = expert_heads[task_idx](exp_feat)
                exp_probs = F.softmax(exp_logits, dim=-1)
            merged_probs = F.softmax(logits, dim=-1)
            loss_kl = (exp_probs * (torch.log(exp_probs + 1e-12) - torch.log(merged_probs + 1e-12))).sum(dim=-1).mean()
            
            loss_ewc = 0.0
            fim = fisher_priors[task_idx]
            init_head = expert_heads[task_idx]
            for p_name, p in active_head.named_parameters():
                init_p = getattr(init_head, p_name)
                f_weight = fim[p_name]
                loss_ewc += 0.5 * (f_weight * (p - init_p) ** 2).sum()
                
            loss = loss_kl + gamma * loss_ewc
            print(f"Step {step} | Loss (KL): {loss_kl.item():.4f} | Loss (EWC): {loss_ewc.item():.4f} | Total: {loss.item():.4f}")
            
        loss.backward()
        print(f"       | lambda_val: {lambda_val.tolist()}")
        print(f"       | lambda_val.grad: {lambda_val.grad.tolist() if lambda_val.grad is not None else 'None'}")
        
        # Update
        with torch.no_grad():
            lambda_val.data -= lr_lambda * lambda_val.grad
            # simplex projection
            from tta import project_to_simplex
            lambda_val.data = project_to_simplex(lambda_val.data)
            
            for p in active_head.parameters():
                if p.grad is not None:
                    p.data -= 1e-4 * p.grad
                    
        lambda_val.grad = None
        active_head.zero_grad()

diagnose_method('unconstrained')
diagnose_method('ewc_tta')
