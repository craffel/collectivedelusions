import torch
import os
import timm
import torchvision.transforms as transforms
import torchvision.datasets as datasets

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = os.path.expanduser("~/data")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

grayscale_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_task_dataset(name, train=False):
    if name == "MNIST":
        return datasets.MNIST(root=DATA_DIR, train=train, download=False, transform=grayscale_transform)
    elif name == "FashionMNIST":
        return datasets.FashionMNIST(root=DATA_DIR, train=train, download=False, transform=grayscale_transform)
    elif name == "CIFAR10":
        return datasets.CIFAR10(root=DATA_DIR, train=train, download=False, transform=transform)
    elif name == "SVHN":
        split = "train" if train else "test"
        return datasets.SVHN(root=DATA_DIR, split=split, download=False, transform=transform)

tasks = ["MNIST", "FashionMNIST", "CIFAR10", "SVHN"]

for task in tasks:
    for suffix in ["_expert.pt", "_expert_converged.pt"]:
        path = f"checkpoints/{task}{suffix}"
        if os.path.exists(path):
            model = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=10)
            model.load_state_dict(torch.load(path, map_location=DEVICE))
            model.to(DEVICE)
            model.eval()
            
            dataset = get_task_dataset(task, train=False)
            indices = list(range(250))
            imgs_list = []
            labels_list = []
            for idx in indices:
                img, label = dataset[idx]
                imgs_list.append(img)
                labels_list.append(label)
            imgs = torch.stack(imgs_list).to(DEVICE)
            labels = torch.tensor(labels_list).to(DEVICE)
            
            with torch.no_grad():
                outputs = model(imgs)
                _, preds = outputs.max(1)
                acc = (preds.eq(labels).sum().item() / 250.0) * 100.0
            print(f"{task} {suffix}: {acc:.2f}%")
        else:
            print(f"{path} does not exist")
