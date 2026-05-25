import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import copy
from torch.func import functional_call
from torch.utils.data import DataLoader, Subset

# Set random seed and disable cuDNN
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.enabled = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResNetBackbone(nn.Module):
    def __init__(self, original_resnet):
        super().__init__()
        self.resnet_layers = nn.Sequential(*list(original_resnet.children())[:-1])
    def forward(self, x):
        x = self.resnet_layers(x)
        x = torch.flatten(x, 1)
        return x

def get_base_model():
    original_resnet = torchvision.models.resnet18(weights=None)
    backbone = ResNetBackbone(original_resnet)
    return backbone

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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

def apply_corruption(images, corruption_type):
    if corruption_type == "clean":
        return images
    elif corruption_type == "noise":
        return images + 0.15 * torch.randn_like(images)
    elif corruption_type == "blur":
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

base_backbone = get_base_model().to(device)
base_backbone.load_state_dict(torch.load("checkpoints/base_backbone.pt", map_location=device))

task_names = ["mnist", "fashionmnist", "kmnist"]
expert_backbones = {}
expert_buffers = {}
task_vectors = {}

for k, task in enumerate(task_names):
    eb = get_base_model().to(device)
    eb.load_state_dict(torch.load(f"checkpoints/{task}_backbone.pt", map_location=device))
    expert_backbones[task] = eb
    expert_buffers[task] = {name: buf.clone() for name, buf in eb.named_buffers()}
    
    tv = {}
    for name, param in eb.named_parameters():
        base_param = base_backbone.state_dict()[name]
        tv[name] = param.data - base_param.data
    task_vectors[task] = tv

# Load prototypes
prototypes = {}
for task in task_names:
    prototypes[task] = torch.load(f"prototypes/{task}_prototypes.pt", map_location=device)

# Prepare alternating test stream
loaders = {
    "mnist": get_task_loader("mnist", train=False, subset_size=1600, batch_size=32),
    "fashionmnist": get_task_loader("fashionmnist", train=False, subset_size=1600, batch_size=32),
    "kmnist": get_task_loader("kmnist", train=False, subset_size=1600, batch_size=32),
}

# Target ImageNet stats to rescale standardized images back to correct norm space
target_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
target_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

for corr in ["clean", "noise", "blur", "contrast"]:
    iters = {k: iter(v) for k, v in loaders.items()}
    batches = []
    for b_idx in range(50):
        for task_idx, task in enumerate(task_names):
            images, labels = next(iters[task])
            batches.append((task_idx, task, images, labels))

    lambda_val = torch.tensor([1/3, 1/3, 1/3], device=device)
    merged_params = {}
    for name, param in base_backbone.named_parameters():
        merged_params[name] = param + sum(lambda_val[k] * task_vectors[task_names[k]][name] for k in range(3))
    for name, buf in base_backbone.named_buffers():
        merged_params[name] = sum(lambda_val[k] * expert_buffers[task_names[k]][name] for k in range(3))

    correct_detections = 0
    total_batches = len(batches)

    for batch_idx, (true_task_idx, true_task, images, labels) in enumerate(batches):
        images = images.to(device)
        images_corr = apply_corruption(images, corr)
        
        # ONLINE STANDARDIZATION & RE-NORMALIZATION
        if corr == "contrast":
            # standardise each image in batch independently
            batch_mean = images_corr.mean(dim=[2, 3], keepdim=True)
            batch_std = images_corr.std(dim=[2, 3], keepdim=True)
            # Re-scale back to target mean and std
            images_input = (images_corr - batch_mean) / (batch_std + 1e-5)
        else:
            images_input = images_corr

        with torch.no_grad():
            features = functional_call(base_backbone, merged_params, images_input)
            feat_norm = nn.functional.normalize(features, p=2, dim=1)
            
            scores = []
            for task in task_names:
                task_protos = prototypes[task].to(device)
                sim = torch.matmul(feat_norm, task_protos.t()) # [B, 10]
                max_sim, _ = sim.max(dim=1)
                scores.append(max_sim.mean().item())
                
            detected_task_idx = np.argmax(scores)
            if detected_task_idx == true_task_idx:
                correct_detections += 1

    detection_acc = 100. * correct_detections / total_batches
    print(f"Corruption: {corr.upper():10s} | Task Detection Accuracy: {detection_acc:.2f}% ({correct_detections}/{total_batches})")
