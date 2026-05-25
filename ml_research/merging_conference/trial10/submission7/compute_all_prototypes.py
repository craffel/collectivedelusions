import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from download_and_pretrain import SimpleCNN

def get_datasets():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=transform)
    fmnist_train = torchvision.datasets.FashionMNIST(root='./data', train=True, download=False, transform=transform)
    return mnist_train, fmnist_train

def compute_prototypes(model, dataset, num_samples=10000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    indices = list(range(num_samples))
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=128, shuffle=False)
    
    class_features = {c: [] for c in range(10)}
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            feats = model.forward_features(x)
            for i in range(x.size(0)):
                label = y[i].item()
                class_features[label].append(feats[i].cpu())
                
    unnorm_prototypes = torch.zeros(10, 128)
    norm_prototypes = torch.zeros(10, 128)
    
    for c in range(10):
        feats_c = torch.stack(class_features[c])
        mean_feat = feats_c.mean(dim=0)
        unnorm_prototypes[c] = mean_feat
        
        norm = mean_feat.norm(p=2)
        if norm > 0:
            norm_prototypes[c] = mean_feat / norm
        else:
            norm_prototypes[c] = mean_feat
            
    return unnorm_prototypes, norm_prototypes

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load experts
    expert_mnist = SimpleCNN().to(device)
    expert_mnist.load_state_dict(torch.load("expert_mnist.pth", map_location=device))
    
    expert_fashion = SimpleCNN().to(device)
    expert_fashion.load_state_dict(torch.load("expert_fashion.pth", map_location=device))
    
    mnist_train, fmnist_train = get_datasets()
    
    mnist_unnorm, mnist_norm = compute_prototypes(expert_mnist, mnist_train)
    fashion_unnorm, fashion_norm = compute_prototypes(expert_fashion, fmnist_train)
    
    torch.save({
        "mnist_unnorm": mnist_unnorm,
        "mnist_norm": mnist_norm,
        "fashion_unnorm": fashion_unnorm,
        "fashion_norm": fashion_norm
    }, "prototypes.pth")
    
    print("Saved all prototypes to prototypes.pth:")
    print("mnist_unnorm norm mean:", mnist_unnorm.norm(dim=1).mean().item())
    print("mnist_norm norm mean:", mnist_norm.norm(dim=1).mean().item())
    print("fashion_unnorm norm mean:", fashion_unnorm.norm(dim=1).mean().item())
    print("fashion_norm norm mean:", fashion_norm.norm(dim=1).mean().item())
