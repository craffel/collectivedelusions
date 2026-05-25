import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.func import functional_call
import os
import copy
import numpy as np

# Set random seed
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.enabled = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define base ResNet-18 and wrapper
class ResNetBackbone(nn.Module):
    def __init__(self, original_resnet):
        super().__init__()
        self.resnet_layers = nn.Sequential(*list(original_resnet.children())[:-1])
    def forward(self, x):
        x = self.resnet_layers(x)
        x = torch.flatten(x, 1)
        return x

def get_base_model():
    original_resnet = models.resnet18(weights=None)
    backbone = ResNetBackbone(original_resnet)
    return backbone

# Projection to simplex
def project_to_simplex(v):
    # Sort v in descending order
    u, _ = torch.sort(v, descending=True)
    cssv = torch.cumsum(u, dim=0) - 1
    ind = torch.arange(1, len(v) + 1, device=v.device)
    cond = u - cssv / ind > 0
    # Find the last index where condition is true
    rho = torch.max(ind[cond])
    theta = cssv[rho - 1] / rho
    w = torch.clamp(v - theta, min=0)
    return w

# Transformation
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Get dataset loaders
def get_task_loader(name, train=False, subset_size=1600, batch_size=32):
    if name == "mnist":
        dataset = torchvision.datasets.MNIST(root="data", train=train, transform=transform, download=False)
    elif name == "fashionmnist":
        dataset = torchvision.datasets.FashionMNIST(root="data", train=train, transform=transform, download=False)
    elif name == "kmnist":
        dataset = torchvision.datasets.KMNIST(root="data", train=train, transform=transform, download=False)
    else:
        raise ValueError("Unknown dataset")
    
    if subset_size is not None and subset_size < len(dataset):
        indices = np.random.choice(len(dataset), subset_size, replace=False)
        dataset = Subset(dataset, indices)
        
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return loader

# Apply test-time corruptions
def apply_corruption(images, corruption_type):
    if corruption_type == "clean":
        return images
    elif corruption_type == "noise":
        # Gaussian Noise
        return images + 0.15 * torch.randn_like(images)
    elif corruption_type == "blur":
        # Gaussian Blur
        blur = transforms.GaussianBlur(kernel_size=5, sigma=1.6)
        return blur(images)
    elif corruption_type == "contrast":
        # Reduce contrast safely by un-normalizing, adjusting contrast, and re-normalizing
        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
        images_unnorm = images * std + mean
        images_unnorm_corr = transforms.functional.adjust_contrast(images_unnorm, 0.3)
        return (images_unnorm_corr - mean) / std
    else:
        raise ValueError("Unknown corruption type")

# Load models
base_backbone = get_base_model().to(device)
base_backbone.load_state_dict(torch.load("checkpoints/base_backbone.pt", map_location=device))

task_names = ["mnist", "fashionmnist", "kmnist"]
expert_backbones = {}
expert_heads = {}
expert_buffers = {}
task_vectors = {}

for k, task in enumerate(task_names):
    # Load expert backbone
    eb = get_base_model().to(device)
    eb.load_state_dict(torch.load(f"checkpoints/{task}_backbone.pt", map_location=device))
    expert_backbones[task] = eb
    expert_buffers[task] = {name: buf.clone() for name, buf in eb.named_buffers()}
    
    # Compute task vector
    tv = {}
    for name, param in eb.named_parameters():
        base_param = base_backbone.state_dict()[name]
        tv[name] = param.data - base_param.data
    task_vectors[task] = tv
    
    # Load expert head
    head = nn.Linear(512, 10).to(device)
    head.load_state_dict(torch.load(f"checkpoints/{task}_head.pt", map_location=device))
    expert_heads[task] = head

# Pre-compute Fisher Information for EWC-TTA
print("Pre-computing diagonal Fisher Information Matrix for heads (EWC-TTA prior)...")
fisher_priors = {}
for task in task_names:
    head = expert_heads[task]
    eb = expert_backbones[task]
    loader = get_task_loader(task, train=True, subset_size=200, batch_size=32)
    
    fisher = {name: torch.zeros_like(param) for name, param in head.named_parameters()}
    head.eval()
    eb.eval()
    
    for images, _ in loader:
        images = images.to(device)
        with torch.no_grad():
            features = eb(images)
        features.requires_grad = True
        outputs = head(features)
        log_probs = torch.log_softmax(outputs, dim=1)
        
        for i in range(images.size(0)):
            pred_class = torch.multinomial(torch.softmax(outputs[i].detach(), dim=0), 1)
            loss = log_probs[i, pred_class]
            loss.backward(retain_graph=True)
            for name, param in head.named_parameters():
                if param.grad is not None:
                    fisher[name] += (param.grad.data ** 2) / 200
            head.zero_grad()
            
    fisher_priors[task] = fisher
print("Fisher priors computed!")

# Load pre-computed class prototypes for CPA-Merge
prototypes = {}
for task in task_names:
    prototypes[task] = torch.load(f"prototypes/{task}_prototypes.pt", map_location=device)

# Prepare Test Streams
def build_test_stream(stream_type):
    # 50 batches of size 32 per task (1600 samples per task)
    loaders = {
        "mnist": get_task_loader("mnist", train=False, subset_size=1600, batch_size=32),
        "fashionmnist": get_task_loader("fashionmnist", train=False, subset_size=1600, batch_size=32),
        "kmnist": get_task_loader("kmnist", train=False, subset_size=1600, batch_size=32),
    }
    
    iters = {k: iter(v) for k, v in loaders.items()}
    batches = []
    
    if stream_type == "alternating":
        # Alternate: 0, 1, 2, 0, 1, 2...
        for b_idx in range(50):
            for task_idx, task in enumerate(task_names):
                images, labels = next(iters[task])
                batches.append((task_idx, task, images, labels))
    elif stream_type == "sequential":
        # Sequential: MNIST (50), FashionMNIST (50), KMNIST (50)
        for task_idx, task in enumerate(task_names):
            for b_idx in range(50):
                images, labels = next(iters[task])
                batches.append((task_idx, task, images, labels))
    else:
        raise ValueError("Unknown stream type")
        
    return batches

# Main TTA Evaluation Function
def evaluate_tta(method_name, stream_type, corruption_type, lr=0.01, tau=0.02, beta=0.1, mask_threshold=0.85):
    batches = build_test_stream(stream_type)
    
    # Initialize merging coefficients lambda to [1/3, 1/3, 1/3]
    lambda_val = torch.tensor([1/3, 1/3, 1/3], device=device, requires_grad=True)
    optimizer_lambda = optim.Adam([lambda_val], lr=lr)
    
    # Copy expert heads for adaptation (used by EWC-TTA and SyMerge)
    adapted_heads = {task: copy.deepcopy(expert_heads[task]) for task in task_names}
    optimizers_heads = {
        task: optim.Adam(adapted_heads[task].parameters(), lr=1e-4) for task in task_names
    }
    
    correct_predictions = 0
    total_samples = 0
    
    for batch_idx, (task_idx, task, images, labels) in enumerate(batches):
        images, labels = images.to(device), labels.to(device)
        images_corr = apply_corruption(images, corruption_type)
        
        # Adaptation steps
        if method_name != "static":
            # Perform TTA: 1 gradient step per batch
            # We construct virtual merged model
            # Reconstruct merged parameter dict
            merged_params = {}
            for name, param in base_backbone.named_parameters():
                merged_params[name] = param + sum(lambda_val[k] * task_vectors[task_names[k]][name] for k in range(3))
            for name, buf in base_backbone.named_buffers():
                merged_params[name] = sum(lambda_val[k].detach() * expert_buffers[task_names[k]][name] for k in range(3))
            
            optimizer_lambda.zero_grad()
            optimizers_heads[task].zero_grad()
            
            # Forward pass to compute representations and predictions
            features = functional_call(base_backbone, merged_params, images_corr)
            outputs = adapted_heads[task](features)
            
            loss = 0.0
            
            if method_name == "adamerging":
                # Entropy minimization on prediction
                probs = torch.softmax(outputs, dim=1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
                loss = entropy
                
            elif method_name == "s2c_merge":
                # Entropy minimization + consistency
                probs = torch.softmax(outputs, dim=1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
                
                # Apply random horizontal flip for consistency
                images_aug = transforms.functional.hflip(images_corr)
                features_aug = functional_call(base_backbone, merged_params, images_aug)
                outputs_aug = adapted_heads[task](features_aug)
                probs_aug = torch.softmax(outputs_aug, dim=1)
                
                # KL divergence consistency loss
                kl_consistency = nn.functional.kl_div(torch.log(probs_aug + 1e-8), probs, reduction='batchmean')
                loss = entropy + 0.5 * kl_consistency
                
            elif method_name in ["sata_symerge", "ewc_tta"]:
                # Requires expert teacher in memory
                with torch.no_grad():
                    teacher_features = expert_backbones[task](images_corr)
                    teacher_outputs = expert_heads[task](teacher_features)
                    teacher_probs = torch.softmax(teacher_outputs, dim=1)
                
                probs = torch.softmax(outputs, dim=1)
                kl_loss = nn.functional.kl_div(torch.log(probs + 1e-8), teacher_probs, reduction='batchmean')
                
                if method_name == "ewc_tta":
                    # EWC regularization on classification head parameters
                    ewc_loss = 0.0
                    for p_name, param in adapted_heads[task].named_parameters():
                        prior_mean = expert_heads[task].state_dict()[p_name]
                        prior_fisher = fisher_priors[task][p_name]
                        ewc_loss += (prior_fisher * (param - prior_mean) ** 2).sum()
                    loss = kl_loss + 1000.0 * ewc_loss
                else:
                    loss = kl_loss
                    
            elif method_name == "cpa_merge": # OUR METHOD
                # 1. Unsupervised Prototype-driven Dynamic Prior (PD-Routing)
                with torch.no_grad():
                    # Compute anchor features using static uniform merge
                    anchor_params = {}
                    for name, param in base_backbone.named_parameters():
                        anchor_params[name] = param + sum(1/3 * task_vectors[task_names[k]][name] for k in range(3))
                    for name, buf in base_backbone.named_buffers():
                        anchor_params[name] = sum(1/3 * expert_buffers[task_names[k]][name] for k in range(3))
                    
                    anchor_features = functional_call(base_backbone, anchor_params, images_corr)
                    anchor_feat_norm = nn.functional.normalize(anchor_features, p=2, dim=1)
                    
                    scores = []
                    for k, t_name in enumerate(task_names):
                        t_protos = prototypes[t_name].to(device)
                        sims = torch.matmul(anchor_feat_norm, t_protos.t()) # [B, 10]
                        max_sims, _ = sims.max(dim=1)
                        scores.append(max_sims.mean())
                    
                    scores = torch.stack(scores)
                    lambda_prior = torch.softmax(scores / tau, dim=0) # Sharp prior
                
                # Reset lambda to the dynamic prior for this batch and clear optimizer state
                with torch.no_grad():
                    lambda_val.copy_(lambda_prior)
                optimizer_lambda.state.clear()
                
                # Re-compute merged params, features, and outputs with the reset lambda
                merged_params = {}
                for name, param in base_backbone.named_parameters():
                    merged_params[name] = param + sum(lambda_val[k] * task_vectors[task_names[k]][name] for k in range(3))
                for name, buf in base_backbone.named_buffers():
                    merged_params[name] = sum(lambda_val[k].detach() * expert_buffers[task_names[k]][name] for k in range(3))
                
                features = functional_call(base_backbone, merged_params, images_corr)
                outputs = adapted_heads[task](features)
                
                # 2. Dual Loss: Entropy minimization + Confidence-masked Contrastive Alignment
                probs = torch.softmax(outputs, dim=1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
                
                normalized_features = nn.functional.normalize(features, p=2, dim=1)
                task_prototypes = prototypes[task].to(device)
                sim_matrix = torch.matmul(normalized_features, task_prototypes.t())
                
                confidences, pseudo_labels = probs.max(dim=1)
                mask = confidences > mask_threshold
                
                if mask.any():
                    contrastive_loss = nn.functional.cross_entropy(sim_matrix[mask] / 0.1, pseudo_labels[mask])
                else:
                    contrastive_loss = 0.0
                
                loss = entropy + beta * contrastive_loss
                
            # Backward and Step
            loss.backward()
            
            optimizer_lambda.step()
            if method_name in ["sata_symerge", "ewc_tta"]:
                optimizers_heads[task].step()
                
            # Project lambda back to simplex
            with torch.no_grad():
                lambda_val.copy_(project_to_simplex(lambda_val))
                
        # Inference and performance collection
        with torch.no_grad():
            # Virtual merging
            merged_params = {}
            for name, param in base_backbone.named_parameters():
                merged_params[name] = param + sum(lambda_val[k] * task_vectors[task_names[k]][name] for k in range(3))
            for name, buf in base_backbone.named_buffers():
                merged_params[name] = sum(lambda_val[k] * expert_buffers[task_names[k]][name] for k in range(3))
                
            features = functional_call(base_backbone, merged_params, images_corr)
            outputs = adapted_heads[task](features)
            
            _, predicted = outputs.max(1)
            correct_predictions += predicted.eq(labels).sum().item()
            total_samples += labels.size(0)
            
    accuracy = 100. * correct_predictions / total_samples
    return accuracy

if __name__ == "__main__":
    print("\n=======================================================")
    print("RUNNING TTA EVALUATIONS")
    print("=======================================================")
    
    methods = ["static", "adamerging", "s2c_merge", "sata_symerge", "ewc_tta", "cpa_merge"]
    streams = ["alternating", "sequential"]
    corruptions = ["clean", "noise", "blur", "contrast"]
    
    results = {}
    
    # We will run all configurations and record accuracy
    for stream in streams:
        results[stream] = {}
        for corr in corruptions:
            results[stream][corr] = {}
            print(f"\n--- Stream: {stream.upper()} | Corruption: {corr.upper()} ---")
            for method in methods:
                acc = evaluate_tta(method, stream, corr)
                results[stream][corr][method] = acc
                print(f"[{method.upper()}] Accuracy: {acc:.2f}%")
                
    # Format and save the results as a beautiful Markdown table
    print("\n=======================================================")
    print("SUMMARY OF RESULTS")
    print("=======================================================")
    
    md_output = "# Experimental Results & Baselines Comparison\n\n"
    md_output += "We evaluate **CPA-Merge** against standard and recent test-time model merging baselines.\n\n"
    
    for stream in streams:
        md_output += f"## Stream Type: {stream.capitalize()} Stream\n\n"
        md_output += "| Method | Clean | Gaussian Noise | Gaussian Blur | Contrast Shift | Average |\n"
        md_output += "| :--- | :---: | :---: | :---: | :---: | :---: |\n"
        
        for method in methods:
            c_acc = results[stream]["clean"][method]
            n_acc = results[stream]["noise"][method]
            b_acc = results[stream]["blur"][method]
            co_acc = results[stream]["contrast"][method]
            avg_acc = (c_acc + n_acc + b_acc + co_acc) / 4.0
            
            method_display = method.replace("_", "-").upper()
            if method == "cpa_merge":
                method_display = f"**{method_display} (Ours)**"
            md_output += f"| {method_display} | {c_acc:.2f}% | {n_acc:.2f}% | {b_acc:.2f}% | {co_acc:.2f}% | **{avg_acc:.2f}%** |\n"
        md_output += "\n"
        
    print(md_output)
    
    with open("results_summary.md", "w") as f:
        f.write(md_output)
    print("Saved results to results_summary.md!")
