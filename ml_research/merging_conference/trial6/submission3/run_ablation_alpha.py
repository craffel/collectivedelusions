import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.func as func
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Disabling cuDNN for deterministic behavior and to bypass initialization errors
torch.backends.cudnn.enabled = False

# Helper function to get the base model
def get_base_model():
    model = resnet18()
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(512, 10)
    return model

# ResNet-18 Feature and Logit Wrapper
class ResNet18FeatureWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool
        self.fc = model.fc
        
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
        feats = torch.flatten(x, 1)
        logits = self.fc(feats)
        return feats, logits

# Simplex projection helper for PyTorch tensors
def project_simplex(v):
    if len(v.shape) == 1:
        u = torch.sort(v, descending=True)[0]
        cssv = torch.cumsum(u, dim=0) - 1.0
        ind = torch.arange(1, len(v) + 1, device=v.device)
        cond = u - cssv / ind > 0
        rho = torch.nonzero(cond)[-1].item()
        theta = cssv[rho] / (rho + 1)
        return torch.clamp(v - theta, min=0.0)
    else:
        shape = v.shape
        v_flat = v.view(-1, shape[-1])
        u, _ = torch.sort(v_flat, descending=True, dim=-1)
        cssv = torch.cumsum(u, dim=-1) - 1.0
        ind = torch.arange(1, shape[-1] + 1, device=v.device).repeat(v_flat.shape[0], 1)
        cond = u - cssv / ind > 0
        rho = (cond.long() * torch.arange(1, shape[-1] + 1, device=v.device)).argmax(dim=-1)
        theta = cssv[torch.arange(v_flat.shape[0]), rho] / (rho + 1)
        projected = torch.clamp(v_flat - theta.unsqueeze(-1), min=0.0)
        return projected.view(shape)

# Load base model, experts, and task vectors
print("Loading model checkpoints...")
base_model = get_base_model()
base_model.load_state_dict(torch.load("./checkpoints/base_model.pth", map_location=device))
base_model = base_model.to(device)

expert_mnist = get_base_model()
expert_mnist.load_state_dict(torch.load("./checkpoints/expert_mnist.pth", map_location=device))
expert_mnist = expert_mnist.to(device)

expert_kmnist = get_base_model()
expert_kmnist.load_state_dict(torch.load("./checkpoints/expert_kmnist.pth", map_location=device))
expert_kmnist = expert_kmnist.to(device)

expert_fmnist = get_base_model()
expert_fmnist.load_state_dict(torch.load("./checkpoints/expert_fmnist.pth", map_location=device))
expert_fmnist = expert_fmnist.to(device)

# Instantiate the wrapped model for functional calls
wrapped_base = ResNet18FeatureWrapper(base_model).to(device)
wrapped_base.eval()

# Load Fisher matrices
fisher_mnist = torch.load("./checkpoints/fisher_mnist.pth", map_location=device)
fisher_kmnist = torch.load("./checkpoints/fisher_kmnist.pth", map_location=device)
fisher_fmnist = torch.load("./checkpoints/fisher_fmnist.pth", map_location=device)

# Build task vectors for the K=3 experts
base_params = {name: param.clone().detach() for name, param in base_model.named_parameters()}
base_buffers = {name: buf.clone().detach() for name, buf in base_model.named_buffers()}

task_vectors = {}
for name, param in base_model.named_parameters():
    v1 = expert_mnist.state_dict()[name] - base_params[name]
    v2 = expert_kmnist.state_dict()[name] - base_params[name]
    v3 = expert_fmnist.state_dict()[name] - base_params[name]
    task_vectors[name] = [v1.detach(), v2.detach(), v3.detach()]

task_vector_buffers = {}
for name, buf in base_model.named_buffers():
    v1 = expert_mnist.state_dict()[name] - base_buffers[name]
    v2 = expert_kmnist.state_dict()[name] - base_buffers[name]
    v3 = expert_fmnist.state_dict()[name] - base_buffers[name]
    task_vector_buffers[name] = [v1.detach(), v2.detach(), v3.detach()]

# Compute Joint Fisher
print("Computing joint Fisher...")
mean_fisher_w = {}
for name, param in base_model.named_parameters():
    if param.requires_grad:
        f_mnist = fisher_mnist[name]
        f_kmnist = fisher_kmnist[name]
        f_fmnist = fisher_fmnist[name]
        f_joint = (f_mnist + f_kmnist + f_fmnist) / 3.0
        mean_fisher_w[name] = f_joint.mean().item()

# Prepare Stream Datasets
print("Preparing stream data loaders...")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

mnist_dataset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
fmnist_dataset = torchvision.datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
kmnist_dataset = torchvision.datasets.KMNIST(root="./data", train=False, download=True, transform=transform)

# Select subsets to form the 90 batches of size 64
np.random.seed(42)
mnist_indices = np.random.choice(len(mnist_dataset), 1920, replace=False)
kmnist_indices = np.random.choice(len(kmnist_dataset), 1920, replace=False)
fmnist_indices = np.random.choice(len(fmnist_dataset), 1920, replace=False)

mnist_subset = Subset(mnist_dataset, mnist_indices)
kmnist_subset = Subset(kmnist_dataset, kmnist_indices)
fmnist_subset = Subset(fmnist_dataset, fmnist_indices)

mnist_loader = DataLoader(mnist_subset, batch_size=64, shuffle=False)
kmnist_loader = DataLoader(kmnist_subset, batch_size=64, shuffle=False)
fmnist_loader = DataLoader(fmnist_subset, batch_size=64, shuffle=False)

# Re-assemble the stream
stream_batches = []
for batch in mnist_loader:
    stream_batches.append((batch[0], batch[1], 0))
for batch in kmnist_loader:
    stream_batches.append((batch[0], batch[1], 1))
for batch in fmnist_loader:
    stream_batches.append((batch[0], batch[1], 2))

# Pre-build static uniformly merged parameters
static_params = {}
for name, param in base_model.named_parameters():
    static_params[name] = base_params[name] + (1.0/3.0) * task_vectors[name][0] + (1.0/3.0) * task_vectors[name][1] + (1.0/3.0) * task_vectors[name][2]
static_buffers = {}
for name, buf in base_model.named_buffers():
    static_buffers[name] = base_buffers[name] + (1.0/3.0) * task_vector_buffers[name][0] + (1.0/3.0) * task_vector_buffers[name][1] + (1.0/3.0) * task_vector_buffers[name][2]
static_merged_all = {**static_params, **static_buffers}

def merged_forward(merged_params, inputs):
    return func.functional_call(wrapped_base, merged_params, inputs)

# Pre-compute class prototypes and expert means offline
print("Computing static merged prototypes...")
def compute_expert_prototypes_static(dataset, num_samples=500):
    loader = DataLoader(Subset(dataset, list(range(num_samples))), batch_size=64, shuffle=False)
    features_list = []
    labels_list = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            feats, _ = merged_forward(static_merged_all, inputs)
            features_list.append(feats.cpu())
            labels_list.append(labels)
    features = torch.cat(features_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    dataset_mean = features.mean(dim=0).to(device)
    prototypes = {}
    for c in range(10):
        class_feats = features[labels == c]
        if len(class_feats) > 0:
            prototypes[c] = class_feats.mean(dim=0).to(device)
        else:
            prototypes[c] = torch.zeros(512, device=device)
    return dataset_mean, prototypes

mean_mnist_ind, protos_mnist = compute_expert_prototypes_static(mnist_dataset)
mean_kmnist_ind, protos_kmnist = compute_expert_prototypes_static(kmnist_dataset)
mean_fmnist_ind, protos_fmnist = compute_expert_prototypes_static(fmnist_dataset)

# Centering offline prototypes
for c in range(10):
    protos_mnist[c] = protos_mnist[c] - mean_mnist_ind
    protos_kmnist[c] = protos_kmnist[c] - mean_kmnist_ind

# Proposed method parameterized by alpha_metric
def run_iggs_ow(batches, alpha_metric):
    accuracies = []
    novelty_detections = []
    
    alpha_ema = 0.1
    stability_constant = 1e-5
    novelty_threshold = 0.35
    
    lambdas_layerwise = {}
    for name, param in base_model.named_parameters():
        if param.requires_grad:
            lambdas_layerwise[name] = torch.tensor([1.0/3.0, 1.0/3.0, 1.0/3.0], device=device)
            
    G_w_inv = {}
    for name, param in base_model.named_parameters():
        if param.requires_grad:
            # Metric preconditioning
            G_w = (mean_fisher_w[name] + stability_constant) ** alpha_metric
            G_w_inv[name] = 1.0 / G_w
            
    protos_novel = None
    
    for batch_idx, (inputs, labels, task_id) in enumerate(batches):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Step 1: Anchor Pass
        with torch.no_grad():
            feats_anchor, logits_anchor = merged_forward(static_merged_all, inputs)
            mu_static = (1.0/3.0) * mean_mnist_ind + (1.0/3.0) * mean_kmnist_ind + (1.0/3.0) * mean_fmnist_ind
            feats_centered_anchor = feats_anchor - mu_static
            
        # Step 2: Unbiased Routing via Prototype Cohesion
        cohesion_e1_list = []
        for feat in feats_centered_anchor:
            sims = [F.cosine_similarity(feat, protos_mnist[c], dim=0) for c in range(10)]
            cohesion_e1_list.append(max(sims).item())
        cohesion_e1 = np.mean(cohesion_e1_list)
        
        cohesion_e2_list = []
        for feat in feats_centered_anchor:
            sims = [F.cosine_similarity(feat, protos_kmnist[c], dim=0) for c in range(10)]
            cohesion_e2_list.append(max(sims).item())
        cohesion_e2 = np.mean(cohesion_e2_list)
        
        max_cohesion = max(cohesion_e1, cohesion_e2)
        is_novel = max_cohesion < novelty_threshold
        novelty_detections.append(is_novel)
        
        # Reset to uniform when novelty is first detected
        if is_novel and protos_novel is None:
            for name in lambdas_layerwise:
                lambdas_layerwise[name] = torch.tensor([1.0/3.0, 1.0/3.0, 1.0/3.0], device=device)
            protos_novel = True
        
        lambdas_sum = torch.zeros(3, device=device)
        layer_count = 0
        for name in lambdas_layerwise:
            lambdas_sum += lambdas_layerwise[name]
            layer_count += 1
        mean_lambda = lambdas_sum / layer_count
        
        # Build merged parameters
        merged_params = {}
        for name, param in base_model.named_parameters():
            if param.requires_grad:
                l = lambdas_layerwise[name]
                merged_params[name] = base_params[name] + l[0] * task_vectors[name][0] + l[1] * task_vectors[name][1] + l[2] * task_vectors[name][2]
            else:
                merged_params[name] = param
                
        for name, buf in base_model.named_buffers():
            merged_params[name] = base_buffers[name] + mean_lambda[0] * task_vector_buffers[name][0] + mean_lambda[1] * task_vector_buffers[name][1] + mean_lambda[2] * task_vector_buffers[name][2]
            
        if not is_novel:
            best_expert = 0 if cohesion_e1 > cohesion_e2 else 1
            target_onehot = torch.eye(3, device=device)[best_expert]
            for name in lambdas_layerwise:
                lambdas_layerwise[name] = (1.0 - alpha_ema) * lambdas_layerwise[name] + alpha_ema * target_onehot
        else:
            # Novel domain: route based on predictive entropy of experts
            with torch.no_grad():
                # Expert 1
                merged_params_e1 = {name: base_params[name] + task_vectors[name][0] for name in base_params}
                for name, buf in base_model.named_buffers():
                    merged_params_e1[name] = base_buffers[name] + task_vector_buffers[name][0]
                _, outputs_e1 = merged_forward(merged_params_e1, inputs)
                entropy_e1 = -torch.sum(F.softmax(outputs_e1, dim=-1) * F.log_softmax(outputs_e1, dim=-1), dim=-1).mean().item()
                
                # Expert 2
                merged_params_e2 = {name: base_params[name] + task_vectors[name][1] for name in base_params}
                for name, buf in base_model.named_buffers():
                    merged_params_e2[name] = base_buffers[name] + task_vector_buffers[name][1]
                _, outputs_e2 = merged_forward(merged_params_e2, inputs)
                entropy_e2 = -torch.sum(F.softmax(outputs_e2, dim=-1) * F.log_softmax(outputs_e2, dim=-1), dim=-1).mean().item()
                
                # Expert 3
                merged_params_e3 = {name: base_params[name] + task_vectors[name][2] for name in base_params}
                for name, buf in base_model.named_buffers():
                    merged_params_e3[name] = base_buffers[name] + task_vector_buffers[name][2]
                _, outputs_e3 = merged_forward(merged_params_e3, inputs)
                entropy_e3 = -torch.sum(F.softmax(outputs_e3, dim=-1) * F.log_softmax(outputs_e3, dim=-1), dim=-1).mean().item()
                
            entropies = [entropy_e1, entropy_e2, entropy_e3]
            best_expert = np.argmin(entropies)
            target_onehot = torch.eye(3, device=device)[best_expert]
            
            with torch.no_grad():
                for name in lambdas_layerwise:
                    # Metric-preconditioned update
                    step = 0.1 * G_w_inv[name] * (lambdas_layerwise[name] - target_onehot)
                    lambdas_layerwise[name] = lambdas_layerwise[name] - step
                    lambdas_layerwise[name] = project_simplex(lambdas_layerwise[name])
                            
        with torch.no_grad():
            lambdas_sum = torch.zeros(3, device=device)
            for name in lambdas_layerwise:
                lambdas_sum += lambdas_layerwise[name]
            mean_lambda = lambdas_sum / len(lambdas_layerwise)
            
            merged_params = {}
            for name, param in base_model.named_parameters():
                if param.requires_grad:
                    l = lambdas_layerwise[name]
                    merged_params[name] = base_params[name] + l[0] * task_vectors[name][0] + l[1] * task_vectors[name][1] + l[2] * task_vectors[name][2]
                else:
                    merged_params[name] = param
            for name, buf in base_model.named_buffers():
                merged_params[name] = base_buffers[name] + mean_lambda[0] * task_vector_buffers[name][0] + mean_lambda[1] * task_vector_buffers[name][1] + mean_lambda[2] * task_vector_buffers[name][2]
                
            _, outputs = merged_forward(merged_params, inputs)
            _, predicted = outputs.max(dim=1)
            acc = predicted.eq(labels).float().mean().item()
            
        accuracies.append(acc)
        
    acc_task_a = np.mean(accuracies[0:30]) * 100
    acc_task_b = np.mean(accuracies[30:60]) * 100
    acc_task_c = np.mean(accuracies[60:90]) * 100
    overall_acc = np.mean(accuracies) * 100
    
    known_novel_flags = novelty_detections[0:60]
    fpr = np.mean(known_novel_flags) * 100
    
    novel_novel_flags = novelty_detections[60:90]
    ndr = np.mean(novel_novel_flags) * 100
    
    return {
        "acc_task_a": acc_task_a,
        "acc_task_b": acc_task_b,
        "acc_task_c": acc_task_c,
        "overall_acc": overall_acc,
        "ndr": ndr,
        "fpr": fpr
    }

def main():
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    results = {}
    
    for alpha in alphas:
        print(f"\nRunning ablation for alpha = {alpha}...")
        res = run_iggs_ow(stream_batches, alpha)
        results[str(alpha)] = res
        print(f"  MNIST (Task A) Acc:   {res['acc_task_a']:.2f}%")
        print(f"  KMNIST (Task B) Acc:  {res['acc_task_b']:.2f}%")
        print(f"  FashionMNIST (Task C) Acc: {res['acc_task_c']:.2f}%")
        print(f"  Overall Acc:          {res['overall_acc']:.2f}%")
        
    # Save results
    os.makedirs("./results", exist_ok=True)
    with open("./results/ablation_alpha.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nAblation results saved to results/ablation_alpha.json")
    
    # Plot results
    plt.figure(figsize=(8, 5))
    accs = [results[str(alpha)]["acc_task_c"] for alpha in alphas]
    plt.plot(alphas, accs, marker='o', linewidth=2, color='darkblue', label='FashionMNIST (Novel)')
    plt.title("Ablation Study: Effect of Sensitivity Damping Factor (α) on Novel Domain")
    plt.xlabel("Sensitivity Damping Factor (α)")
    plt.ylabel("FashionMNIST Accuracy (%)")
    plt.grid(True, linestyle='--')
    plt.legend()
    plt.savefig("./results/ablation_alpha_plot.png", dpi=300)
    print("Ablation plot saved to results/ablation_alpha_plot.png")

if __name__ == "__main__":
    main()
