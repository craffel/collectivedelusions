import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class ResNet18Custom(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)
        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)
        x = self.base_model.avgpool(x)
        feat = torch.flatten(x, 1)
        logits = self.base_model.fc(feat)
        return feat, logits

def get_resnet18_1channel():
    model = resnet18()
    conv1_new = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.conv1 = conv1_new
    model.fc = nn.Linear(512, 10)
    return model

def project_simplex(v):
    v_sorted, _ = torch.sort(v, descending=True)
    j = torch.arange(1, len(v) + 1, device=v.device)
    cumsum = torch.cumsum(v_sorted, dim=0)
    rho_cond = v_sorted - (cumsum - 1.0) / j > 0
    rho = torch.max(torch.where(rho_cond)[0]) + 1
    theta = (cumsum[rho - 1] - 1.0) / rho
    return torch.clamp(v - theta, min=0.0)

def compute_batch_fisher_fast(expert_model, batch_X, device):
    fisher = {name: torch.zeros_like(p) for name, p in expert_model.named_parameters() if p.requires_grad}
    expert_model.eval()
    
    _, logits = expert_model(batch_X)
    pseudo_labels = torch.argmax(logits, dim=-1)
    
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, pseudo_labels)
    expert_model.zero_grad()
    loss.backward()
    
    for name, param in expert_model.named_parameters():
        if param.requires_grad and param.grad is not None:
            fisher[name] = param.grad.data ** 2
            
    return fisher

def get_joint_fisher_fast(expert_models, batch_X, device):
    joint_fisher = {}
    K = len(expert_models)
    for k in range(K):
        fisher_k = compute_batch_fisher_fast(expert_models[k], batch_X, device)
        for name, val in fisher_k.items():
            clean_name = name.replace('base_model.', '')
            tensor_avg = val.mean().item()
            if clean_name not in joint_fisher:
                joint_fisher[clean_name] = 0.0
            joint_fisher[clean_name] += tensor_avg / K
    return joint_fisher

def compute_preconditioned_lrs(joint_fisher, base_lr=1e-3, eps=1e-6, alpha=1.0):
    sensitivities = list(joint_fisher.values())
    mean_sens = sum(sensitivities) / len(sensitivities)
    
    preconditioned_lrs = {}
    for name, sens in joint_fisher.items():
        norm_sens = sens / (mean_sens + 1e-12)
        lr_mult = (norm_sens + eps) ** (-alpha)
        lr_mult = torch.clamp(torch.tensor(lr_mult), 0.01, 10.0).item()
        preconditioned_lrs[name] = base_lr * lr_mult
    return preconditioned_lrs

def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    # Load datasets
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    kmnist_test = torchvision.datasets.KMNIST(root='./data', train=False, download=True, transform=transform)
    fmnist_test = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    mnist_test_subset = Subset(mnist_test, list(range(1920)))
    kmnist_test_subset = Subset(kmnist_test, list(range(1920)))
    fmnist_test_subset = Subset(fmnist_test, list(range(1920)))

    mnist_loader = DataLoader(mnist_test_subset, batch_size=64, shuffle=False)
    kmnist_loader = DataLoader(kmnist_test_subset, batch_size=64, shuffle=False)
    fmnist_loader = DataLoader(fmnist_test_subset, batch_size=64, shuffle=False)

    mnist_batches = [b for b in mnist_loader]
    kmnist_batches = [b for b in kmnist_loader]
    fmnist_batches = [b for b in fmnist_loader]

    # Sequential Clean Stream
    seq_stream_clean = mnist_batches[:30] + kmnist_batches[:30] + fmnist_batches[:30]
    seq_domains = [0]*30 + [1]*30 + [2]*30

    # Load experts
    expert_paths = {
        'mnist': 'mnist_expert.pt',
        'kmnist': 'kmnist_expert.pt',
        'fashionmnist': 'fashionmnist_expert.pt'
    }
    experts = {}
    for name, path in expert_paths.items():
        model = get_resnet18_1channel()
        model.load_state_dict(torch.load(path, map_location=device))
        model = ResNet18Custom(model).to(device)
        model.eval()
        experts[name] = model

    expert_list = [experts['mnist'], experts['kmnist'], experts['fashionmnist']]
    K = len(expert_list)

    base_model = get_resnet18_1channel().to(device)
    parameter_names = [name for name, p in base_model.named_parameters() if p.requires_grad]
    buffer_names = [name for name, b in base_model.named_buffers()]

    expert_params = []
    expert_buffers = []
    for k in range(K):
        expert_params.append({name: p.clone().detach() for name, p in expert_list[k].base_model.named_parameters()})
        expert_buffers.append({name: b.clone().detach() for name, b in expert_list[k].base_model.named_buffers()})

    base_buffers_dict = {name: b for name, b in base_model.named_buffers()}

    # S-Fisher from fast batch fisher
    print("Computing fast Source Fisher (S-Fisher)...")
    # Take 500 samples total (250 MNIST and 250 KMNIST)
    sf_subset_mnist = Subset(mnist_test_subset, list(range(250)))
    sf_subset_kmnist = Subset(kmnist_test_subset, list(range(250)))
    sf_loader_m = DataLoader(sf_subset_mnist, batch_size=250, shuffle=False)
    sf_loader_k = DataLoader(sf_subset_kmnist, batch_size=250, shuffle=False)
    for x_m, _ in sf_loader_m:
        sf_x_mnist = x_m
    for x_k, _ in sf_loader_k:
        sf_x_kmnist = x_k
    sf_X = torch.cat([sf_x_mnist, sf_x_kmnist], dim=0).to(device)
    s_fisher = get_joint_fisher_fast(expert_list, sf_X, device)

    # Coefficients
    coefficients = {name: torch.tensor([1/3, 1/3, 1/3], device=device, requires_grad=True) for name in parameter_names}
    running_fisher = {name: torch.tensor(s_fisher[name], device=device) for name in parameter_names}

    correct_total = 0
    samples_total = 0
    gamma_fisher = 0.1
    alpha_damping = 1.0

    print("Starting stream evaluation WITH fast Fisher preconditioned update...")
    for t, (batch_X, batch_y) in enumerate(seq_stream_clean):
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        true_dom = seq_domains[t]

        # Compute predictive entropy of individual experts on batch
        entropies = []
        with torch.no_grad():
            for k in range(K):
                _, logits_k = expert_list[k](batch_X)
                probs_k = torch.softmax(logits_k, dim=-1)
                ent_k = -torch.mean(torch.sum(probs_k * torch.log(probs_k + 1e-12), dim=-1)).item()
                entropies.append(ent_k)

        k_star = np.argmin(entropies)

        # Reset coefficients to Lambda_prior
        Lambda_prior = torch.tensor([0.005, 0.005, 0.005], device=device)
        Lambda_prior[k_star] = 0.99

        with torch.no_grad():
            for name in parameter_names:
                coefficients[name].copy_(Lambda_prior)

        # Enable gradient tracking on coefficients
        for name in parameter_names:
            coefficients[name].requires_grad_(True)
            if coefficients[name].grad is not None:
                coefficients[name].grad.zero_()

        # Differentiably merge weights
        params_dict = {}
        for name in parameter_names:
            coeff = coefficients[name]
            params_dict[name] = (
                coeff[0] * expert_params[0][name] +
                coeff[1] * expert_params[1][name] +
                coeff[2] * expert_params[2][name]
            )

        # Unsupervised forward pass
        logits = torch.func.functional_call(base_model, params_dict, (batch_X,))
        probs = torch.softmax(logits, dim=-1)
        loss = -torch.mean(torch.sum(probs * torch.log(probs + 1e-12), dim=-1))

        # Backpropagate
        loss.backward()

        # Update running fisher and compute preconditioned LRs
        batch_fisher = get_joint_fisher_fast(expert_list, batch_X, device)
        for name in parameter_names:
            running_fisher[name] = (1 - gamma_fisher) * running_fisher[name] + gamma_fisher * batch_fisher[name]
        
        lrs = compute_preconditioned_lrs({name: val.item() for name, val in running_fisher.items()}, base_lr=1e-3, alpha=alpha_damping)

        # Update coefficients
        with torch.no_grad():
            for name in parameter_names:
                grad = coefficients[name].grad
                if grad is not None:
                    updated = coefficients[name] - lrs[name] * grad
                    coefficients[name].copy_(project_simplex(updated))

        # AFTER update, compute coeff_avg and sync BN buffers for evaluation
        with torch.no_grad():
            coeff_sum = torch.zeros(K, device=device)
            for name in parameter_names:
                coeff_sum += coefficients[name]
            coeff_avg = coeff_sum / len(parameter_names)
            
            for name in buffer_names:
                b_val = 0.0
                for k in range(K):
                    b_val += coeff_avg[k].item() * expert_buffers[k][name]
                base_buffers_dict[name].copy_(b_val)

        # Evaluate merged model on batch
        params_dict_eval = {}
        for name in parameter_names:
            coeff = coefficients[name]
            params_dict_eval[name] = coeff[0] * expert_params[0][name] + coeff[1] * expert_params[1][name] + coeff[2] * expert_params[2][name]

        with torch.no_grad():
            logits = torch.func.functional_call(base_model, params_dict_eval, (batch_X,))
            _, preds = logits.max(1)
            batch_correct = preds.eq(batch_y).sum().item()
            batch_total = batch_X.size(0)
            batch_acc = (batch_correct / batch_total) * 100.0

            correct_total += batch_correct
            samples_total += batch_total

        if t in [0, 5, 30, 35, 60, 65]:
            print(f"Batch {t:2d} | True Domain: {true_dom} | Entropies: {np.array(entropies)} | Routed: {k_star} | Batch Acc: {batch_acc:.2f}%")

    print(f"Overall Accuracy: {(correct_total / samples_total)*100:.2f}%")

if __name__ == '__main__':
    main()
