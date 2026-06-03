import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import numpy as np

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Datasets and loaders (same transforms as training)
def get_loaders():
    mnist_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))
    ])

    fmnist_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize((0.2860, 0.2860, 0.2860), (0.3530, 0.3530, 0.3530))
    ])

    cifar10_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_mnist = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=mnist_transform)
    train_fmnist = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True, transform=fmnist_transform)
    train_cifar = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=cifar10_transform)

    cal_mnist = DataLoader(Subset(train_mnist, range(1024)), batch_size=128, shuffle=False)
    cal_fmnist = DataLoader(Subset(train_fmnist, range(1024)), batch_size=128, shuffle=False)
    cal_cifar = DataLoader(Subset(train_cifar, range(1024)), batch_size=128, shuffle=False)

    return {
        "mnist": cal_mnist,
        "fmnist": cal_fmnist,
        "cifar10": cal_cifar
    }

def load_expert(task):
    model = resnet18()
    model.fc = nn.Linear(512, 10)
    model.load_state_dict(torch.load(f"models/{task}_expert.pt", map_location="cpu"))
    return model

def load_progenitor():
    model = resnet18()
    model.fc = nn.Linear(512, 10)
    backbone_sd = torch.load("models/progenitor_backbone.pt", map_location="cpu")
    model_sd = model.state_dict()
    for k, v in backbone_sd.items():
        if k in model_sd:
            model_sd[k] = v
    model.load_state_dict(model_sd)
    return model

def compute_diagonal_fisher(model, dataloader, device):
    model.eval()
    model = model.to(device)
    fisher = {}
    for name, param in model.named_parameters():
        if param.requires_grad and not name.startswith("fc."):
            fisher[name] = torch.zeros_like(param.data)
            
    criterion = nn.CrossEntropyLoss()
    count = 0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        for name, param in model.named_parameters():
            if param.grad is not None and name in fisher:
                fisher[name] += (param.grad.data ** 2) * inputs.size(0)
        count += inputs.size(0)
        
    for name in fisher:
        fisher[name] = fisher[name] / count
        fisher[name] = torch.clamp(fisher[name], min=1e-8)
        
    return fisher

def main():
    loaders = get_loaders()
    print("Loading models...")
    experts = {
        "mnist": load_expert("mnist"),
        "fmnist": load_expert("fmnist"),
        "cifar10": load_expert("cifar10")
    }
    progenitor = load_progenitor()
    
    W0 = {k: v.clone() for k, v in progenitor.state_dict().items() if not k.startswith("fc.")}
    W_experts = {
        task: {k: v.clone() for k, v in experts[task].state_dict().items() if not k.startswith("fc.")}
        for task in ["mnist", "fmnist", "cifar10"]
    }
    T_experts = {
        task: {k: W_experts[task][k] - W0[k] for k in W0}
        for task in ["mnist", "fmnist", "cifar10"]
    }
    
    print("Computing diagonal Fishers...")
    fishers = {
        "mnist": compute_diagonal_fisher(experts["mnist"], loaders["mnist"], device),
        "fmnist": compute_diagonal_fisher(experts["fmnist"], loaders["fmnist"], device),
        "cifar10": compute_diagonal_fisher(experts["cifar10"], loaders["cifar10"], device)
    }
    
    # Selected layers representing early, mid, late layers
    selected_layers = [
        "conv1.weight",
        "layer1.0.conv1.weight",
        "layer2.0.conv1.weight",
        "layer3.0.conv1.weight",
        "layer4.0.conv1.weight"
    ]
    
    alpha = 20.0
    print(f"\n--- Analyzing Scale Factors s* at alpha={alpha} ---")
    
    for k in selected_layers:
        t_merge = (T_experts["mnist"][k] + T_experts["fmnist"][k] + T_experts["cifar10"][k]) / 3.0
        orig_shape = t_merge.shape
        
        d_out = orig_shape[0]
        d_in = int(np.prod(orig_shape[1:]))
        t_merge_2d = t_merge.reshape(d_out, d_in).to(device)
        
        U, Sigma, Vh = torch.linalg.svd(t_merge_2d, full_matrices=False)
        V = Vh.t()
        C = len(Sigma)
        D = d_out * d_in
        
        Phi = torch.zeros((C, D), device=device)
        for c in range(C):
            phi_c_2d = Sigma[c] * torch.outer(U[:, c], V[:, c])
            Phi[c] = phi_c_2d.flatten()
            
        A = torch.zeros((C, C), device=device)
        b = torch.zeros(C, device=device)
        
        for task in ["mnist", "fmnist", "cifar10"]:
            F_task = fishers[task][k].reshape(D).to(device)
            T_task = T_experts[task][k].reshape(D).to(device)
            A_task = torch.matmul(Phi * F_task.unsqueeze(0), Phi.t())
            A += A_task
            b_task = torch.matmul(Phi, F_task * T_task)
            b += b_task
            
        mean_diag = torch.clamp(torch.diag(A).mean(), min=1e-15)
        lambda_reg = alpha * mean_diag
        A_reg = A + max(lambda_reg, 1e-6 * mean_diag) * torch.eye(C, device=device)
        b_reg = b + lambda_reg * torch.ones(C, device=device)
        
        s = torch.linalg.solve(A_reg, b_reg)
        s = torch.clamp(s, min=0.0, max=10.0).cpu().numpy()
        
        print(f"\nLayer: {k}")
        print(f"  Shape: {orig_shape}")
        print(f"  Number of Singular Components (C): {C}")
        print(f"  Scale factors s* statistics:")
        print(f"    Mean:   {np.mean(s):.4f}")
        print(f"    Std:    {np.std(s):.4f}")
        print(f"    Min:    {np.min(s):.4f}")
        print(f"    Max:    {np.max(s):.4f}")
        print(f"    First 5 scale factors: {s[:5]}")
        print(f"    Last 5 scale factors:  {s[-5:]}")

if __name__ == "__main__":
    main()
