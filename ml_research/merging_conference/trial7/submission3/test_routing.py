import torch
from torch.func import functional_call
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

def get_resnet18_grayscale():
    model = models.resnet18()
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.fc = nn.Linear(512, 10)
    return model

def compute_prototypes(model, calibration_set, device="cuda"):
    model.eval()
    loader = DataLoader(calibration_set, batch_size=32, shuffle=False)
    
    all_features = []
    all_targets = []
    
    def extract_features(x):
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
        x = model.avgpool(x)
        x = torch.flatten(x, 1)
        return x
        
    num_samples = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            feats = extract_features(inputs)
            all_features.append(feats.cpu())
            all_targets.append(targets.clone())
            num_samples += inputs.size(0)
            if num_samples >= 500:
                break
                
    all_features = torch.cat(all_features, dim=0)[:500]
    all_targets = torch.cat(all_targets, dim=0)[:500]
    
    centroid = all_features.mean(dim=0)
    prototypes = torch.zeros(10, 512)
    for c in range(10):
        c_mask = (all_targets == c)
        if c_mask.sum() > 0:
            class_feats = all_features[c_mask]
            centered_feats = class_feats - centroid
            mean_feat = centered_feats.mean(dim=0)
            prototypes[c] = mean_feat / (mean_feat.norm(p=2) + 1e-8)
            
    return centroid, prototypes

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    mnist_cal = torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=transform)
    kmnist_cal = torchvision.datasets.KMNIST(root='./data', train=True, download=False, transform=transform)
    
    model1 = get_resnet18_grayscale().to(device)
    model1.load_state_dict(torch.load("./checkpoints/expert_mnist.pth", map_location=device))
    
    model2 = get_resnet18_grayscale().to(device)
    model2.load_state_dict(torch.load("./checkpoints/expert_kmnist.pth", map_location=device))
    
    print("Computing prototypes...")
    centroid1, prototypes1 = compute_prototypes(model1, mnist_cal, device=device)
    centroid2, prototypes2 = compute_prototypes(model2, kmnist_cal, device=device)
    
    # Let's test the first batch of MNIST
    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform)
    loader = DataLoader(mnist_test, batch_size=64, shuffle=False)
    inputs, targets = next(iter(loader))
    inputs, targets = inputs.to(device), targets.to(device)
    
    # Merged model initially with uniform coefficients [0.5, 0.5]
    merged_model = get_resnet18_grayscale().to(device)
    
    model1_params = {name: param.clone() for name, param in model1.named_parameters()}
    model2_params = {name: param.clone() for name, param in model2.named_parameters()}
    model1_buffers = {name: buf.clone() for name, buf in model1.named_buffers()}
    
    merged_params = {}
    for name in model1_params:
        merged_params[name] = 0.5 * model1_params[name] + 0.5 * model2_params[name]
        
    merged_buffers = {}
    for name in model1_buffers:
        merged_buffers[name] = model1_buffers[name]
        
    features_list = []
    def hook_fn(module, input, output):
        features_list.append(torch.flatten(output, 1))
        
    hook = merged_model.avgpool.register_forward_hook(hook_fn)
    params_and_buffers = {**merged_params, **merged_buffers}
    logits = functional_call(merged_model, params_and_buffers, inputs)
    feats = features_list[0]
    hook.remove()
    
    # Centering
    mu_t = 0.5 * centroid1.to(device) + 0.5 * centroid2.to(device)
    centered_feats = feats - mu_t
    
    # Compute cohesion to known experts
    c1 = torch.zeros(inputs.size(0), device=device)
    c2 = torch.zeros(inputs.size(0), device=device)
    
    for i in range(inputs.size(0)):
        zi = centered_feats[i]
        sim1 = torch.zeros(10, device=device)
        sim2 = torch.zeros(10, device=device)
        for c in range(10):
            sim1[c] = F.cosine_similarity(zi.unsqueeze(0), prototypes1[c].to(device).unsqueeze(0))
            sim2[c] = F.cosine_similarity(zi.unsqueeze(0), prototypes2[c].to(device).unsqueeze(0))
        c1[i] = sim1.max()
        c2[i] = sim2.max()
        
    cohesion1 = c1.mean().item()
    cohesion2 = c2.mean().item()
    print(f"MNIST cohesion: {cohesion1:.4f}")
    print(f"KMNIST cohesion: {cohesion2:.4f}")

if __name__ == "__main__":
    main()
