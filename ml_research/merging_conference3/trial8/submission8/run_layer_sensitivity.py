import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.utils.data import DataLoader, Subset
import timm
import numpy as np

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

TASKS = ["MNIST", "FashionMNIST", "CIFAR10", "SVHN"]

def prepare_dataloaders():
    transform_rgb = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_gray = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    loaders = {}
    for task in TASKS:
        if task == "MNIST":
            dataset = dsets.MNIST(root='./data', train=True, download=True, transform=transform_gray)
        elif task == "FashionMNIST":
            dataset = dsets.FashionMNIST(root='./data', train=True, download=True, transform=transform_gray)
        elif task == "CIFAR10":
            dataset = dsets.CIFAR10(root='./data', train=True, download=True, transform=transform_rgb)
        elif task == "SVHN":
            dataset = dsets.SVHN(root='./data', split='train', download=True, transform=transform_rgb)
        
        subset = Subset(dataset, list(range(64))) # Use 64 samples for fast extraction
        loaders[task] = DataLoader(subset, batch_size=64, shuffle=False)
    return loaders

class LayerFeatureExtractor:
    def __init__(self, layer_idx, device):
        self.device = device
        self.model = timm.create_model('vit_tiny_patch16_224', pretrained=True)
        self.model.eval()
        self.model.to(device)
        self.features = []
        self.hook = self.model.blocks[layer_idx].register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        self.features.append(output[:, 0].detach().cpu().clone())

    def extract(self, dataloader):
        self.features = []
        with torch.no_grad():
            for imgs, _ in dataloader:
                imgs = imgs.to(self.device)
                _ = self.model(imgs)
        return torch.cat(self.features, dim=0)

    def close(self):
        self.hook.remove()

def main():
    loaders = prepare_dataloaders()
    
    # We want to sweep layers 1 to 5 (block indices 0 to 4)
    results = {}
    for layer in range(1, 6):
        block_idx = layer - 1
        print(f"\nExtracting features from Layer {layer} (block index {block_idx})...")
        extractor = LayerFeatureExtractor(block_idx, device)
        
        task_features = {}
        for task in TASKS:
            task_features[task] = extractor.extract(loaders[task])
        extractor.close()
        
        # Compute centroids for each task
        centroids = {}
        for task in TASKS:
            centroids[task] = torch.mean(task_features[task], dim=0)
            
        # Compute average coordinate separation (ACS)
        # ACS = mean_over_tasks ( similarity of task_i features to task_i centroid - mean(similarity of task_i features to other centroids) )
        task_acs = []
        for i, task in enumerate(TASKS):
            features = task_features[task] # [N, D]
            # Normalize features and centroids for cosine similarity
            features_norm = features / (torch.norm(features, dim=1, keepdim=True) + 1e-8)
            
            centroid_self = centroids[task]
            centroid_self_norm = centroid_self / (torch.norm(centroid_self) + 1e-8)
            
            sim_self = torch.matmul(features_norm, centroid_self_norm) # [N]
            
            sim_others = []
            for other_task in TASKS:
                if other_task == task:
                    continue
                centroid_other = centroids[other_task]
                centroid_other_norm = centroid_other / (torch.norm(centroid_other) + 1e-8)
                sim_other = torch.matmul(features_norm, centroid_other_norm) # [N]
                sim_others.append(sim_other)
            
            sim_others_mean = torch.mean(torch.stack(sim_others, dim=0), dim=0) # [N]
            acs = torch.mean(sim_self - sim_others_mean).item()
            task_acs.append(acs)
            
        avg_acs = np.mean(task_acs)
        
        # Compute average pairwise centroid cosine similarity (Inter-Task Centroid Cosine Similarity)
        centroid_norms = [centroids[t] / (torch.norm(centroids[t]) + 1e-8) for t in TASKS]
        pairwise_sims = []
        for i in range(len(TASKS)):
            for j in range(i+1, len(TASKS)):
                sim = torch.dot(centroid_norms[i], centroid_norms[j]).item()
                pairwise_sims.append(sim)
        avg_centroid_sim = np.mean(pairwise_sims)
        
        results[layer] = {
            "avg_acs": avg_acs,
            "avg_centroid_sim": avg_centroid_sim
        }
        print(f"Layer {layer} -> ACS: {avg_acs:.4f}, Centroid Similarity: {avg_centroid_sim:.4f}")

    print("\nFinal Results Summary:")
    print("Layer | Average Coordinate Separation (ACS) | Inter-Task Centroid Cosine Similarity")
    print("------|------------------------------------|--------------------------------------")
    for layer in sorted(results.keys()):
        print(f"  {layer}   |               {results[layer]['avg_acs']:.4f}               |                {results[layer]['avg_centroid_sim']:.4f}")

if __name__ == "__main__":
    main()
