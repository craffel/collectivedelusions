import os
import random
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import open_clip
import json

# Setup seed for strict reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load CLIP model and preprocess
print("Loading pre-trained CLIP model (ViT-B-32)...")
clip_model, _, val_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
image_encoder = clip_model.visual.to(device)
feature_dim = 512

pretrained_state_dict = {k: v.cpu().clone() for k, v in image_encoder.state_dict().items()}

transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
])

datasets_dict = {
    'MNIST': (torchvision.datasets.MNIST, 10),
    'FashionMNIST': (torchvision.datasets.FashionMNIST, 10),
    'CIFAR10': (torchvision.datasets.CIFAR10, 10),
    'SVHN': (torchvision.datasets.SVHN, 10)
}

param_to_group = {}
for k in pretrained_state_dict.keys():
    group_idx = 12
    for i in range(12):
        if f'transformer.resblocks.{i}.' in k:
            group_idx = i
            break
    param_to_group[k] = group_idx

train_loaders = {}
test_loaders = {}

for name, (dataset_cls, num_classes) in datasets_dict.items():
    if name == 'SVHN':
        train_data = dataset_cls(root='~/data', split='train', download=True, transform=transform)
        test_data = dataset_cls(root='~/data', split='test', download=True, transform=transform)
    else:
        train_data = dataset_cls(root='~/data', train=True, download=True, transform=transform)
        test_data = dataset_cls(root='~/data', train=False, download=True, transform=transform)
    
    train_indices = list(range(seed, seed + 512))
    test_indices = list(range(seed, seed + 512))
    
    train_subset = torch.utils.data.Subset(train_data, train_indices)
    test_subset = torch.utils.data.Subset(test_data, test_indices)
    
    train_loaders[name] = torch.utils.data.DataLoader(train_subset, batch_size=64, shuffle=True)
    test_loaders[name] = torch.utils.data.DataLoader(test_subset, batch_size=64, shuffle=False)

# Fine-tune experts
experts_state_dicts = {}
heads = {}

print("\n--- Fine-tuning Task Experts ---")
for name, (dataset_cls, num_classes) in datasets_dict.items():
    print(f"Training expert for {name}...")
    head = torch.nn.Linear(feature_dim, num_classes).to(device)
    image_encoder.load_state_dict({k: v.to(device) for k, v in pretrained_state_dict.items()})
    
    optimizer = torch.optim.Adam([
        {'params': image_encoder.parameters(), 'lr': 1e-5},
        {'params': head.parameters(), 'lr': 1e-3}
    ])
    criterion = torch.nn.CrossEntropyLoss()
    
    image_encoder.train()
    head.train()
    for epoch in range(5):
        for imgs, labels in train_loaders[name]:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            feats = image_encoder(imgs)
            logits = head(feats)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
    experts_state_dicts[name] = {k: v.cpu().clone() for k, v in image_encoder.state_dict().items()}
    heads[name] = head.cpu()

# Construct Task Vectors
task_vectors = {}
for name in datasets_dict.keys():
    task_vectors[name] = {}
    for k in pretrained_state_dict.keys():
        if pretrained_state_dict[k].dtype in [torch.int64, torch.uint8]:
            continue
        task_vectors[name][k] = experts_state_dicts[name][k] - pretrained_state_dict[k]

task_names = list(datasets_dict.keys())
task_vectors_gpu = {t_name: {k: v.to(device) for k, v in task_vectors[t_name].items()} for t_name in task_names}
pretrained_gpu = {k: v.to(device) for k, v in pretrained_state_dict.items()}

def apply_merged_weights(encoder, task_coeffs):
    new_state_dict = {}
    for k, v in pretrained_state_dict.items():
        if k not in task_vectors[task_names[0]]:
            new_state_dict[k] = v.to(device)
            continue
        
        merged_update = torch.zeros_like(v)
        for name in task_names:
            coeffs = task_coeffs[name]
            if isinstance(coeffs, list) or isinstance(coeffs, np.ndarray):
                g_idx = param_to_group[k]
                coeff = coeffs[g_idx]
            else:
                coeff = coeffs
            merged_update += coeff * task_vectors[name][k]
        
        new_state_dict[k] = (v + merged_update).to(device)
    encoder.load_state_dict(new_state_dict, strict=False)

def evaluate_merged_model(encoder, task_coeffs):
    apply_merged_weights(encoder, task_coeffs)
    encoder.eval()
    accuracies = {}
    for name, loader in test_loaders.items():
        head = heads[name].to(device).eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                feats = encoder(imgs)
                logits = head(feats)
                correct += (logits.argmax(dim=-1) == labels).sum().item()
                total += labels.size(0)
        accuracies[name] = correct / total
    return accuracies

calib_imgs = {}
for name, loader in train_loaders.items():
    imgs, _ = next(iter(loader))
    calib_imgs[name] = imgs[:64].to(device)

# Sweeping Learning Rate values in Adam optimizer
lr_values = [1e-4, 1e-3, 1e-2, 1e-1]
lr_results = []

print("\n--- Sweeping Optimizer Learning Rate (Adam) ---")
for lr in lr_values:
    print(f"Optimizing with Learning Rate = {lr}...")
    task_coeffs_tensor = torch.full((len(task_names), 13), 0.3, dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([task_coeffs_tensor], lr=lr)
    
    initial_loss = 0.0
    for step in range(200):
        optimizer.zero_grad()
        differentiable_state_dict = {}
        for k, v in pretrained_gpu.items():
            if k not in task_vectors_gpu[task_names[0]]:
                differentiable_state_dict[k] = v
                continue
            g_idx = param_to_group[k]
            merged_update = torch.zeros_like(v)
            for task_idx, t_name in enumerate(task_names):
                coeff = task_coeffs_tensor[task_idx, g_idx]
                merged_update = merged_update + coeff * task_vectors_gpu[t_name][k]
            differentiable_state_dict[k] = v + merged_update
            
        loss = 0.0
        for task_idx, t_name in enumerate(task_names):
            inputs = calib_imgs[t_name]
            feats = torch.func.functional_call(image_encoder, differentiable_state_dict, (inputs,))
            head = heads[t_name].to(device)
            logits = head(feats)
            probs = torch.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
            loss = loss + entropy
            
        if step == 0:
            initial_loss = loss.item()
            
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            task_coeffs_tensor.clamp_(0.0, 1.0)
            
    final_coeffs = {}
    for task_idx, t_name in enumerate(task_names):
        final_coeffs[t_name] = task_coeffs_tensor[task_idx].detach().cpu().numpy().tolist()
        
    accs = evaluate_merged_model(image_encoder, final_coeffs)
    avg_acc = np.mean(list(accs.values()))
    print(f"  LR {lr}: Init Loss = {initial_loss:.4f} -> Final Loss = {loss.item():.4f} | Avg Acc = {100*avg_acc:.2f}% | SVHN = {100*accs['SVHN']:.2f}% | CIFAR10 = {100*accs['CIFAR10']:.2f}%")
    
    lr_results.append({
        'learning_rate': lr,
        'initial_loss': initial_loss,
        'final_loss': loss.item(),
        'mnist': accs['MNIST'],
        'fashion': accs['FashionMNIST'],
        'cifar10': accs['CIFAR10'],
        'svhn': accs['SVHN'],
        'average': avg_acc,
        'coeffs': final_coeffs
    })

# Save results to JSON
os.makedirs('results', exist_ok=True)
with open('results/learning_rate_comparison.json', 'w') as f:
    json.dump(lr_results, f, indent=2)

print("\nLearning Rate Pilot complete! Saved to results/learning_rate_comparison.json")
