import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset

# Disable cuDNN to prevent initialization errors on NVIDIA H100
torch.backends.cudnn.enabled = False

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define transform (same as training)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Datasets
mnist_train = datasets.MNIST("data", train=True, download=False, transform=transform)
fashion_train = datasets.FashionMNIST("data", train=True, download=False, transform=transform)
kmnist_train = datasets.KMNIST("data", train=True, download=False, transform=transform)

datasets_dict = {
    "mnist": mnist_train,
    "fashionmnist": fashion_train,
    "kmnist": kmnist_train
}

def get_feature_extractor(model):
    # For ResNet-18, the feature extractor is everything except the final fc layer
    # We can wrap it or modify forward to return the pooled features
    class FeatureExtractor(nn.Module):
        def __init__(self, resnet):
            super().__init__()
            self.features = nn.Sequential(*list(resnet.children())[:-1])
        def forward(self, x):
            x = self.features(x)
            return torch.flatten(x, 1)
    return FeatureExtractor(model)

def compute_priors_for_expert(name, dataset):
    print(f"\n--- Computing Priors for Expert: {name} ---")
    
    # Load model
    model = models.resnet18()
    model.fc = nn.Linear(512, 10)
    model.load_state_dict(torch.load(f"models/expert_{name}.pth", map_location=device))
    model = model.to(device)
    model.eval()
    
    feature_extractor = get_feature_extractor(model).to(device)
    feature_extractor.eval()
    
    # 1. Compute Class Prototypes
    # We use a calibration subset of 1000 samples
    indices = list(range(1000))
    calib_set = Subset(dataset, indices)
    calib_loader = DataLoader(calib_set, batch_size=64, shuffle=False)
    
    features_list = []
    labels_list = []
    with torch.no_grad():
        for images, labels in calib_loader:
            images = images.to(device)
            feats = feature_extractor(images)
            features_list.append(feats.cpu())
            labels_list.append(labels)
            
    features = torch.cat(features_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    
    prototypes = {}
    for c in range(10):
        class_idx = (labels == c)
        if class_idx.sum() > 0:
            class_feats = features[class_idx]
            # Normalize each feature vector
            class_feats_norm = class_feats / (class_feats.norm(dim=1, keepdim=True) + 1e-8)
            # Average normalized features
            mean_feat = class_feats_norm.mean(dim=0)
            # Normalize the mean vector to unit length
            proto = mean_feat / (mean_feat.norm() + 1e-8)
            prototypes[c] = proto
        else:
            # Fallback to zeros if class is missing in subset
            prototypes[c] = torch.zeros(512)
            
    # Convert prototypes dict to tensor [10, 512]
    proto_tensor = torch.stack([prototypes[c] for c in range(10)], dim=0)
    torch.save(proto_tensor, f"models/prototypes_{name}.pt")
    print(f"Saved prototypes to models/prototypes_{name}.pt")
    
    # 2. Compute Diagonal Fisher Information
    # We use a calibration loader and compute gradients on backpropagation
    # We will accumulate squared gradients for each parameter
    fisher_dict = {}
    for p_name, p in model.named_parameters():
        fisher_dict[p_name] = torch.zeros_like(p.data)
        
    model.eval()  # set to eval to avoid BatchNorm batch size 1 crash
    criterion = nn.CrossEntropyLoss()
    
    calib_loader_fisher = DataLoader(calib_set, batch_size=1, shuffle=False) # batch_size=1 for standard empirical Fisher
    
    count = 0
    for images, labels in calib_loader_fisher:
        images, labels = images.to(device), labels.to(device)
        model.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        with torch.no_grad():
            for p_name, p in model.named_parameters():
                if p.grad is not None:
                    fisher_dict[p_name] += p.grad.data ** 2
        count += 1
        if count >= 200: # 200 samples is plenty for a good diagonal Fisher estimate
            break
            
    # Average and save
    layer_fisher_scalars = {}
    with torch.no_grad():
        for p_name in fisher_dict:
            fisher_dict[p_name] /= count
            # Calculate mean scalar for this layer
            layer_fisher_scalars[p_name] = fisher_dict[p_name].mean().item()
            
    torch.save(fisher_dict, f"models/fisher_diag_{name}.pt")
    torch.save(layer_fisher_scalars, f"models/fisher_layer_scalars_{name}.pt")
    print(f"Saved parameter-level and layer-wise Fisher priors for {name}")

for name in ["mnist", "fashionmnist", "kmnist"]:
    compute_priors_for_expert(name, datasets_dict[name])
