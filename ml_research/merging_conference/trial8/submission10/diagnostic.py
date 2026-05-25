import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import json

torch.manual_seed(42)
np.random.seed(42)

# Same architecture as run_experiments.py
class SharedBackbone(nn.Module):
    def __init__(self):
        super(SharedBackbone, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(32 * 7 * 7, 128)
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class ExpertHead(nn.Module):
    def __init__(self):
        super(ExpertHead, self).__init__()
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, feats):
        return self.fc2(self.relu(feats))

# Train helper
def train_shared_moe(mnist_dataset, kmnist_dataset, epochs=2):
    backbone = SharedBackbone()
    mnist_head = ExpertHead()
    kmnist_head = ExpertHead()
    optimizer = optim.Adam(
        list(backbone.parameters()) + list(mnist_head.parameters()) + list(kmnist_head.parameters()),
        lr=0.003
    )
    criterion = nn.CrossEntropyLoss()
    indices = list(range(12000))
    mnist_loader = DataLoader(Subset(mnist_dataset, indices), batch_size=256, shuffle=True)
    kmnist_loader = DataLoader(Subset(kmnist_dataset, indices), batch_size=256, shuffle=True)
    backbone.train()
    mnist_head.train()
    kmnist_head.train()
    for epoch in range(epochs):
        for (x_m, y_m), (x_k, y_k) in zip(mnist_loader, kmnist_loader):
            optimizer.zero_grad()
            feats_m = backbone(x_m)
            out_m = mnist_head(feats_m)
            loss_m = criterion(out_m, y_m)
            feats_k = backbone(x_k)
            out_k = kmnist_head(feats_k)
            loss_k = criterion(out_k, y_k)
            loss = loss_m + loss_k
            loss.backward()
            optimizer.step()
    return backbone, mnist_head, kmnist_head

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    kmnist_train = datasets.KMNIST(root='./data', train=True, download=True, transform=transform)
    fmnist_test = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    kmnist_test = datasets.KMNIST(root='./data', train=False, download=True, transform=transform)
    
    backbone, mnist_head, kmnist_head = train_shared_moe(mnist_train, kmnist_train, epochs=2)
    backbone.eval()
    mnist_head.eval()
    kmnist_head.eval()
    
    # Compute initial prototypes
    mnist_cal_loader = DataLoader(Subset(mnist_train, list(range(500))), batch_size=500, shuffle=False)
    kmnist_cal_loader = DataLoader(Subset(kmnist_train, list(range(500))), batch_size=500, shuffle=False)
    with torch.no_grad():
        for x, _ in mnist_cal_loader:
            mnist_feats = backbone(x)
            proto_mnist = mnist_feats.mean(dim=0)
            cal_dist_mnist = torch.mean(torch.norm(mnist_feats - proto_mnist, dim=1)).item()
        for x, _ in kmnist_cal_loader:
            kmnist_feats = backbone(x)
            proto_kmnist = kmnist_feats.mean(dim=0)
            cal_dist_kmnist = torch.mean(torch.norm(kmnist_feats - proto_kmnist, dim=1)).item()
            
    global_cal_dist = (cal_dist_mnist + cal_dist_kmnist) / 2.0
    print(f"MNIST Prototype Norm: {torch.norm(proto_mnist).item():.4f} | KMNIST Prototype Norm: {torch.norm(proto_kmnist).item():.4f}")
    print(f"Global Calibration Distance: {global_cal_dist:.4f}")
    
    mnist_test_loader = DataLoader(mnist_test, batch_size=128, shuffle=False)
    kmnist_test_loader = DataLoader(kmnist_test, batch_size=128, shuffle=False)
    fmnist_test_loader = DataLoader(fmnist_test, batch_size=128, shuffle=False)
    
    stream_batches = []
    stream_labels = []
    stream_domains = []
    mnist_iter = iter(mnist_test_loader)
    kmnist_iter = iter(kmnist_test_loader)
    fmnist_iter = iter(fmnist_test_loader)
    
    for _ in range(10):
        x, y = next(mnist_iter)
        stream_batches.append(x)
        stream_labels.append(y)
        stream_domains.append('MNIST')
    for _ in range(10):
        x, y = next(mnist_iter)
        stream_batches.append(x + 0.6 * torch.randn_like(x))
        stream_labels.append(y)
        stream_domains.append('Noisy_MNIST')
    for _ in range(10):
        x, y = next(kmnist_iter)
        stream_batches.append(x)
        stream_labels.append(y)
        stream_domains.append('KMNIST')
        
    alpha = 0.15
    entropy_threshold = 1.45
    dist_threshold = 1.5
    temperature = 0.15
    
    print("--- DIAGNOSTIC: Gated EMA-Proto Internal Tracking ---")
    curr_proto_mnist = proto_mnist.clone()
    curr_proto_kmnist = proto_kmnist.clone()
    
    for t, (x, y_true, domain) in enumerate(zip(stream_batches, stream_labels, stream_domains)):
        with torch.no_grad():
            feats = backbone(x)
            
            dist_mnist = torch.mean(torch.norm(feats - curr_proto_mnist, dim=1)).item()
            dist_kmnist = torch.mean(torch.norm(feats - curr_proto_kmnist, dim=1)).item()
            
            norm_dist_mnist = dist_mnist / global_cal_dist
            norm_dist_kmnist = dist_kmnist / global_cal_dist
            
            out_mnist = mnist_head(feats)
            out_kmnist = kmnist_head(feats)
            
            probs_mnist = torch.softmax(out_mnist, dim=1)
            probs_kmnist = torch.softmax(out_kmnist, dim=1)
            
            conf_mnist = probs_mnist.max(dim=1)[0].mean().item()
            conf_kmnist = probs_kmnist.max(dim=1)[0].mean().item()
            
            logit_mnist = -norm_dist_mnist / temperature + 2.0 * conf_mnist
            logit_kmnist = -norm_dist_kmnist / temperature + 2.0 * conf_kmnist
            
            routing_weights = torch.softmax(torch.tensor([logit_mnist, logit_kmnist]), dim=0)
            w_mnist = routing_weights[0].item()
            w_kmnist = routing_weights[1].item()
            
            probs = w_mnist * probs_mnist + w_kmnist * probs_kmnist
            entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1).mean().item()
            
            _, pred = probs.max(1)
            acc = 100.0 * pred.eq(y_true).sum().item() / x.size(0)
            
            active_expert = torch.argmax(routing_weights).item()
            min_norm_dist = min(norm_dist_mnist, norm_dist_kmnist)
            
            is_ood = (entropy > entropy_threshold or min_norm_dist > dist_threshold)
            
            gated_by_confidence = False
            if active_expert == 0 and conf_mnist < conf_kmnist:
                gated_by_confidence = True
            elif active_expert == 1 and conf_kmnist < conf_mnist:
                gated_by_confidence = True
                
            proto_distance = torch.norm(curr_proto_mnist - curr_proto_kmnist).item()
            
            print(f"Batch {t:02d} ({domain:<11}) | Dist_M/K: {dist_mnist:.2f}/{dist_kmnist:.2f} | NormM/K: {norm_dist_mnist:.2f}/{norm_dist_kmnist:.2f} | GlobalCal: {global_cal_dist:.2f} | Proto_Dist: {proto_distance:.2f} | WM: {w_mnist:.3f} | Acc: {acc:.1f}%")
            
            # Gated Update
            if not is_ood and not gated_by_confidence:
                if active_expert == 0:
                    curr_proto_mnist = (1 - alpha) * curr_proto_mnist + alpha * feats.mean(dim=0)
                else:
                    curr_proto_kmnist = (1 - alpha) * curr_proto_kmnist + alpha * feats.mean(dim=0)
