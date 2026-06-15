import os
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.decomposition import PCA
import numpy as np

def main():
    print("Initializing real-world feature extraction...")
    
    # Check if data directory exists
    os.makedirs("data", exist_ok=True)
    
    # Setup transforms
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load the 4 datasets
    print("Loading datasets...")
    datasets = []
    datasets.append(dset.MNIST(root='./data', train=True, download=False, transform=transform))
    datasets.append(dset.FashionMNIST(root='./data', train=True, download=False, transform=transform))
    datasets.append(dset.KMNIST(root='./data', train=True, download=False, transform=transform))
    datasets.append(dset.USPS(root='./data', train=True, download=False, transform=transform))
    
    # Initialize feature extractor
    print("Loading pre-trained ResNet18...")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.eval()
    feature_extractor = torch.nn.Sequential(*(list(model.children())[:-1]))
    
    # For each dataset, extract exactly 50 samples per class (500 total per task)
    all_features = []
    all_tasks = []
    all_classes = []
    
    for task_idx, dataset in enumerate(datasets):
        print(f"Extracting features for Task {task_idx}...")
        class_counts = {c: 0 for c in range(10)}
        task_features = []
        task_classes = []
        
        # Use a data loader for batching
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        
        finished = False
        with torch.no_grad():
            for images, labels in loader:
                feats = feature_extractor(images).squeeze(-1).squeeze(-1) # (B, 512)
                for f, l in zip(feats, labels):
                    class_label = l.item()
                    if class_label in class_counts and class_counts[class_label] < 50:
                        class_counts[class_label] += 1
                        task_features.append(f)
                        task_classes.append(class_label)
                        
                    # Check if we have 50 of all classes
                    if all(count >= 50 for count in class_counts.values()):
                        finished = True
                        break
                if finished:
                    break
        
        task_features = torch.stack(task_features) # (500, 512)
        task_classes = torch.tensor(task_classes) # (500,)
        
        all_features.append(task_features)
        all_tasks.append(torch.full((500,), task_idx, dtype=torch.long))
        all_classes.append(task_classes)
        print(f"Task {task_idx} extracted successfully.")
        
    all_features = torch.cat(all_features, dim=0) # (2000, 512)
    all_tasks = torch.cat(all_tasks, dim=0) # (2000,)
    all_classes = torch.cat(all_classes, dim=0) # (2000,)
    
    # Project features to 192 dimensions via PCA
    print("Projecting 512-dimensional features to 192 dimensions via PCA...")
    pca = PCA(n_components=192, random_state=42)
    features_np = all_features.numpy()
    features_reduced_np = pca.fit_transform(features_np)
    
    features_reduced = torch.tensor(features_reduced_np, dtype=torch.float32)
    
    # Center and normalize features to have mean norm 1.0 (matching R_h = 1.0 domain bound)
    features_mean = features_reduced.mean(dim=0, keepdim=True)
    features_centered = features_reduced - features_mean
    norms = torch.norm(features_centered, dim=1, keepdim=True)
    mean_norm = norms.mean()
    features_normalized = features_centered / mean_norm
    
    print(f"Extraction complete! Final features shape: {features_normalized.shape}")
    print(f"Mean norm after normalization: {torch.norm(features_normalized, dim=1).mean().item():.4f}")
    
    # Save checkpoint
    checkpoint = {
        "features": features_normalized,
        "tasks": all_tasks,
        "classes": all_classes
    }
    torch.save(checkpoint, "data/real_world_features.pt")
    print("Saved features to data/real_world_features.pt")

if __name__ == "__main__":
    main()
