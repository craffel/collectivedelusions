import os
import torch
torch.backends.cudnn.enabled = False
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from pretrain import ExpertModel

def check_bn_eval_vs_train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    expert1 = ExpertModel()
    expert1.load_state_dict(torch.load("expert_cifar10.pth", map_location=device))
    expert1.to(device)

    expert2 = ExpertModel()
    expert2.load_state_dict(torch.load("expert_svhn.pth", map_location=device))
    expert2.to(device)

    transform_clean = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_fmnist = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    cifar_train = torchvision.datasets.CIFAR10(root="./data", train=True, download=False, transform=transform_clean)
    svhn_train = torchvision.datasets.SVHN(root="./data", split="train", download=False, transform=transform_clean)
    fmnist_train = torchvision.datasets.FashionMNIST(root="./data", train=True, download=False, transform=transform_fmnist)

    def compute_offline_prototypes(model, dataset):
        model.eval()
        class_features = {c: [] for c in range(10)}
        loader = DataLoader(dataset, batch_size=64, shuffle=False)
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(device)
                features = model.get_features(inputs)
                for f, l in zip(features, labels):
                    l_item = l.item()
                    if len(class_features[l_item]) < 100:
                        class_features[l_item].append(f.cpu())
                if all(len(class_features[c]) >= 100 for c in range(10)):
                    break
        prototypes = {}
        for c in range(10):
            prototypes[c] = torch.stack(class_features[c]).mean(dim=0)
        return prototypes

    prototypes1 = compute_offline_prototypes(expert1, cifar_train)
    prototypes2 = compute_offline_prototypes(expert2, svhn_train)

    centroid1 = torch.stack(list(prototypes1.values())).mean(dim=0).to(device)
    centroid2 = torch.stack(list(prototypes2.values())).mean(dim=0).to(device)

    # Center prototypes
    for c in range(10):
        prototypes1[c] = prototypes1[c] - centroid1.cpu()
        prototypes2[c] = prototypes2[c] - centroid2.cpu()

    def add_corruption(x):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        x_raw = x * std + mean
        noise = torch.randn_like(x_raw) * 0.2
        x_corrupted = torch.clamp(x_raw + noise, 0.0, 1.0)
        x_normalized = (x_corrupted - mean) / std
        return x_normalized

    def get_cohesion(inputs, expert, centroid, prototypes, mode="eval", corrupt=False):
        if corrupt:
            inputs = add_corruption(inputs)
        
        # Set BN layers to the desired mode
        if mode == "train":
            expert.train() # use batch statistics
        else:
            expert.eval() # use running statistics
            
        with torch.no_grad():
            features = expert.get_features(inputs)
            features_cent = features - centroid
            features_norm = features_cent / (features_cent.norm(dim=1, keepdim=True) + 1e-8)
            sims = []
            for c in range(10):
                p_norm = prototypes[c].to(device)
                p_norm = p_norm / (p_norm.norm() + 1e-8)
                sims.append(torch.matmul(features_norm, p_norm))
            sims = torch.stack(sims, dim=1)
            cohesion = sims.max(dim=1)[0].mean().item()
        return cohesion

    # Check on a few batches
    loader_cifar = DataLoader(Subset(cifar_train, list(range(192))), batch_size=64, shuffle=False)
    loader_svhn = DataLoader(Subset(svhn_train, list(range(192))), batch_size=64, shuffle=False)
    loader_fmnist = DataLoader(Subset(fmnist_train, list(range(192))), batch_size=64, shuffle=False)

    print("--- Corrupted CIFAR-10 Batches (using BN Train Mode) ---")
    for i, (x, _) in enumerate(loader_cifar):
        x = x.to(device)
        coh1 = get_cohesion(x, expert1, centroid1, prototypes1, mode="train", corrupt=True)
        coh2 = get_cohesion(x, expert2, centroid2, prototypes2, mode="train", corrupt=True)
        print(f"Batch {i}: Coh1 (CIFAR10 expert) = {coh1:.4f}, Coh2 (SVHN expert) = {coh2:.4f}")

    print("--- Corrupted SVHN Batches (using BN Train Mode) ---")
    for i, (x, _) in enumerate(loader_svhn):
        x = x.to(device)
        coh1 = get_cohesion(x, expert1, centroid1, prototypes1, mode="train", corrupt=True)
        coh2 = get_cohesion(x, expert2, centroid2, prototypes2, mode="train", corrupt=True)
        print(f"Batch {i}: Coh1 (CIFAR10 expert) = {coh1:.4f}, Coh2 (SVHN expert) = {coh2:.4f}")

    print("--- Corrupted FashionMNIST Batches (using BN Train Mode) ---")
    for i, (x, _) in enumerate(loader_fmnist):
        x = x.to(device)
        coh1 = get_cohesion(x, expert1, centroid1, prototypes1, mode="train", corrupt=True)
        coh2 = get_cohesion(x, expert2, centroid2, prototypes2, mode="train", corrupt=True)
        print(f"Batch {i}: Coh1 (CIFAR10 expert) = {coh1:.4f}, Coh2 (SVHN expert) = {coh2:.4f}")

if __name__ == "__main__":
    check_bn_eval_vs_train()
