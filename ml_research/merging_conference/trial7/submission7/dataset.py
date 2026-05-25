import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np

class MixedStreamDataset(Dataset):
    def __init__(self, samples, labels, task_ids):
        self.samples = samples
        self.labels = labels
        self.task_ids = task_ids  # 0: MNIST, 1: KMNIST, 2: FashionMNIST
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx], self.task_ids[idx]

def get_test_streams(batch_size=32, corruption="clean"):
    # Define basic transforms (just ToTensor, normalization is done inside if needed or here)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Load test datasets
    mnist_test = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    kmnist_test = datasets.KMNIST(root="./data", train=False, download=True, transform=transform)
    fmnist_test = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
    
    # We want 1600 samples from each task
    num_samples_per_task = 1600
    
    mnist_loader = DataLoader(mnist_test, batch_size=num_samples_per_task, shuffle=False)
    kmnist_loader = DataLoader(kmnist_test, batch_size=num_samples_per_task, shuffle=False)
    fmnist_loader = DataLoader(fmnist_test, batch_size=num_samples_per_task, shuffle=False)
    
    mnist_x, mnist_y = next(iter(mnist_loader))
    kmnist_x, kmnist_y = next(iter(kmnist_loader))
    fmnist_x, fmnist_y = next(iter(fmnist_loader))
    
    # Apply corruptions if requested
    # Clean: unaltered
    # Gaussian noise: add zero-mean Gaussian noise with std=0.2
    # Contrast shift: scale by 0.3
    def apply_corruption(x):
        if corruption == "gaussian":
            noise = torch.randn_like(x) * 0.2
            return torch.clamp(x + noise, -1.0, 1.0)
        elif corruption == "contrast":
            return x * 0.3
        return x
        
    mnist_x = apply_corruption(mnist_x)
    kmnist_x = apply_corruption(kmnist_x)
    fmnist_x = apply_corruption(fmnist_x)
    
    # Task mappings: MNIST -> 0, KMNIST -> 1, FashionMNIST -> 2
    
    # 1. Sequential Stream: 50 batches of MNIST, 50 of KMNIST, 50 of FashionMNIST
    seq_samples = torch.cat([mnist_x, kmnist_x, fmnist_x], dim=0)
    seq_labels = torch.cat([mnist_y, kmnist_y, fmnist_y], dim=0)
    seq_task_ids = torch.cat([
        torch.zeros(num_samples_per_task, dtype=torch.long),
        torch.ones(num_samples_per_task, dtype=torch.long),
        torch.ones(num_samples_per_task, dtype=torch.long) * 2
    ], dim=0)
    seq_stream = DataLoader(MixedStreamDataset(seq_samples, seq_labels, seq_task_ids), batch_size=batch_size, shuffle=False)
    
    # 2. Alternating Stream: Alternate batches of MNIST, KMNIST, FashionMNIST
    alt_samples = []
    alt_labels = []
    alt_task_ids = []
    
    num_batches = num_samples_per_task // batch_size
    for b in range(num_batches):
        # MNIST batch
        alt_samples.append(mnist_x[b*batch_size : (b+1)*batch_size])
        alt_labels.append(mnist_y[b*batch_size : (b+1)*batch_size])
        alt_task_ids.append(torch.zeros(batch_size, dtype=torch.long))
        
        # KMNIST batch
        alt_samples.append(kmnist_x[b*batch_size : (b+1)*batch_size])
        alt_labels.append(kmnist_y[b*batch_size : (b+1)*batch_size])
        alt_task_ids.append(torch.ones(batch_size, dtype=torch.long))
        
        # FashionMNIST batch
        alt_samples.append(fmnist_x[b*batch_size : (b+1)*batch_size])
        alt_labels.append(fmnist_y[b*batch_size : (b+1)*batch_size])
        alt_task_ids.append(torch.ones(batch_size, dtype=torch.long) * 2)
        
    alt_samples = torch.cat(alt_samples, dim=0)
    alt_labels = torch.cat(alt_labels, dim=0)
    alt_task_ids = torch.cat(alt_task_ids, dim=0)
    alt_stream = DataLoader(MixedStreamDataset(alt_samples, alt_labels, alt_task_ids), batch_size=batch_size, shuffle=False)
    
    # 3. Heterogeneous Stream: Fully random mixture of samples within each batch
    # Combine everything and shuffle
    np.random.seed(42)
    indices = np.random.permutation(len(seq_samples))
    het_samples = seq_samples[indices]
    het_labels = seq_labels[indices]
    het_task_ids = seq_task_ids[indices]
    het_stream = DataLoader(MixedStreamDataset(het_samples, het_labels, het_task_ids), batch_size=batch_size, shuffle=False)
    
    return seq_stream, alt_stream, het_stream
