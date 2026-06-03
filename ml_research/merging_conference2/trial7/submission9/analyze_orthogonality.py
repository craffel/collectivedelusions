import torch
import torchvision.models as models
import os
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

experts_dir = "./experts"

# Load Progenitor
progenitor = models.resnet18()
progenitor.fc = torch.nn.Linear(512, 10)
# torchvision resnet18 pretrained weights
from torchvision.models import ResNet18_Weights
progenitor_weights = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
progenitor_state = progenitor_weights.state_dict()

# Load experts
def get_expert(task_name):
    expert_path = os.path.join(experts_dir, f"expert_{task_name}.pt")
    model = models.resnet18()
    model.fc = torch.nn.Linear(512, 10)
    model.load_state_dict(torch.load(expert_path, map_location="cpu"))
    return model

expert_mnist = get_expert("mnist")
expert_fmnist = get_expert("fmnist")
expert_cifar10 = get_expert("cifar10")

experts = {
    "mnist": expert_mnist,
    "fmnist": expert_fmnist,
    "cifar10": expert_cifar10
}

keys = [k for k in progenitor_state.keys() if "fc" not in k and ("weight" in k or "bias" in k)]

print(f"{'Layer Name':40s} | {'Cos(1,2)':8s} | {'Cos(2,3)':8s} | {'Cos(1,3)':8s} | {'Avg Cos':8s} | {'Expected S':10s} | {'Actual S':10s}")
print("-" * 105)

cos_similarities = []
expected_S_list = []
actual_S_list = []

for key in keys:
    if "weight" in key and progenitor_state[key].dim() >= 2:
        # Get progenitor and expert weights
        w_init = progenitor_state[key].float()
        w_mnist = expert_mnist.state_dict()[key].float()
        w_fmnist = expert_fmnist.state_dict()[key].float()
        w_cifar10 = expert_cifar10.state_dict()[key].float()
        
        # Compute task vectors
        t_mnist = w_mnist - w_init
        t_fmnist = w_fmnist - w_init
        t_cifar10 = w_cifar10 - w_init
        
        # Flatten
        t1 = t_mnist.view(-1)
        t2 = t_fmnist.view(-1)
        t3 = t_cifar10.view(-1)
        
        # Cosine similarities
        cos12 = torch.dot(t1, t2) / (torch.norm(t1) * torch.norm(t2) + 1e-8)
        cos23 = torch.dot(t2, t3) / (torch.norm(t2) * torch.norm(t3) + 1e-8)
        cos13 = torch.dot(t1, t3) / (torch.norm(t1) * torch.norm(t3) + 1e-8)
        
        avg_cos = (cos12 + cos23 + cos13) / 3.0
        cos_similarities.append(avg_cos.item())
        
        # Expected scaling factor under pure orthogonality: sqrt(K) = sqrt(3) = 1.732
        # Actual scaling factor computed by U-IPR
        t_merged = (t_mnist + t_fmnist + t_cifar10) / 3.0
        norm_t_merged = torch.norm(t_merged)
        avg_norm_experts = (torch.norm(t1) + torch.norm(t2) + torch.norm(t3)) / 3.0
        
        S_actual = avg_norm_experts / (norm_t_merged + 1e-8)
        
        # Under pure orthogonality, norm of sum of 3 vectors with same norm N is sqrt(3) * N.
        # Averaged vector norm is sqrt(3)/3 * N = 1/sqrt(3) * N.
        # So S_expected = N / (1/sqrt(3) * N) = sqrt(3) = 1.732.
        S_expected = np.sqrt(3.0)
        
        expected_S_list.append(S_expected)
        actual_S_list.append(S_actual.item())
        
        print(f"{key:40s} | {cos12.item():8.4f} | {cos23.item():8.4f} | {cos13.item():8.4f} | {avg_cos.item():8.4f} | {S_expected:10.4f} | {S_actual.item():10.4f}")

print("-" * 105)
print(f"Average pairwise cosine similarity across all layer weights: {np.mean(cos_similarities):.4f}")
print(f"Average actual U-IPR scaling factor: {np.mean(actual_S_list):.4f} (Theoretical expected under pure orthogonality: 1.7321)")
