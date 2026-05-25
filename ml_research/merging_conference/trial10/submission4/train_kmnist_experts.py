import os
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from main import SimpleCNN, train_expert, set_seed

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training KMNIST experts on device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    os.makedirs("./data", exist_ok=True)
    kmnist_train = datasets.KMNIST("./data", train=True, download=True, transform=transform)
    kmnist_train_sub = Subset(kmnist_train, list(range(10000)))
    loader_kmnist_train = DataLoader(kmnist_train_sub, batch_size=64, shuffle=True)

    os.makedirs("./checkpoints", exist_ok=True)

    # 1. Train Standard KMNIST Expert
    std_kmnist_path = "./checkpoints/standard_expert_kmnist.pt"
    standard_expert = SimpleCNN(is_cosface=False)
    if os.path.exists(std_kmnist_path):
        print("Standard KMNIST Expert already exists, skipping training.")
    else:
        print("Training Standard KMNIST Expert...")
        standard_expert = train_expert(standard_expert, loader_kmnist_train, epochs=2, device=device)
        torch.save(standard_expert.state_dict(), std_kmnist_path)
        print("Saved Standard KMNIST Expert.")

    # 2. Train CosFace KMNIST Expert
    cos_kmnist_path = "./checkpoints/cosface_expert_kmnist.pt"
    cosface_expert = SimpleCNN(is_cosface=True)
    if os.path.exists(cos_kmnist_path):
        print("CosFace KMNIST Expert already exists, skipping training.")
    else:
        print("Training CosFace KMNIST Expert...")
        cosface_expert = train_expert(cosface_expert, loader_kmnist_train, epochs=2, device=device)
        torch.save(cosface_expert.state_dict(), cos_kmnist_path)
        print("Saved CosFace KMNIST Expert.")

if __name__ == "__main__":
    main()
