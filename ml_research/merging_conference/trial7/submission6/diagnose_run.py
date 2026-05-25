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

    # Sequential Clean Stream: 30 batches MNIST, 30 batches KMNIST, 30 batches FMNIST
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

    # We evaluate "D-TT-Fisher" with diagnostic prints
    base_model = get_resnet18_1channel().to(device)
    parameter_names = [name for name, p in base_model.named_parameters() if p.requires_grad]
    buffer_names = [name for name, b in base_model.named_buffers()]

    expert_params = []
    expert_buffers = []
    for k in range(K):
        expert_params.append({name: p.clone().detach() for name, p in expert_list[k].base_model.named_parameters()})
        expert_buffers.append({name: b.clone().detach() for name, b in expert_list[k].base_model.named_buffers()})

    base_buffers_dict = {name: b for name, b in base_model.named_buffers()}

    # Coefficients initialized
    coefficients = {name: torch.tensor([1/3, 1/3, 1/3], device=device, requires_grad=True) for name in parameter_names}

    correct_total = 0
    samples_total = 0

    print("Starting stream evaluation...")
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

        # Merge BN buffers with coeff_avg (using Lambda_prior for now to see how BN is initialized)
        with torch.no_grad():
            for name in buffer_names:
                b_val = 0.0
                for k in range(K):
                    b_val += Lambda_prior[k].item() * expert_buffers[k][name]
                base_buffers_dict[name].copy_(b_val)

        # Compute merged parameters in params_dict using Lambda_prior
        params_dict = {}
        for name in parameter_names:
            params_dict[name] = Lambda_prior[0] * expert_params[0][name] + Lambda_prior[1] * expert_params[1][name] + Lambda_prior[2] * expert_params[2][name]

        # Evaluate merged model on batch
        with torch.no_grad():
            logits = torch.func.functional_call(base_model, params_dict, (batch_X,))
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
