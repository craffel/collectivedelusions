import os
import argparse
import random
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, ConcatDataset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = False

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Define Model Architecture
class MultiTaskResNet18(nn.Module):
    def __init__(self, num_tasks=3, num_classes=10):
        super().__init__()
        # Load standard ResNet18
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_features = self.backbone.fc.in_features
        # Replace the fc layer with identity
        self.backbone.fc = nn.Identity()
        # Task-specific linear heads
        self.heads = nn.ModuleList([
            nn.Linear(in_features, num_classes) for _ in range(num_tasks)
        ])
        
    def forward(self, x, task_id):
        features = self.backbone(x)
        out = self.heads[task_id](features)
        return out

def get_datasets():
    # Transforms
    transform_mnist_train = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_mnist_test = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_cifar_train = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_cifar_test = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Download and load datasets
    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)
    
    # MNIST
    mnist_train = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=transform_mnist_train)
    mnist_test = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=transform_mnist_test)
    
    # Fashion-MNIST
    fmnist_train = torchvision.datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform_mnist_train)
    fmnist_test = torchvision.datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform_mnist_test)
    
    # CIFAR-10
    cifar_train = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_cifar_train)
    cifar_test = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_cifar_test)
    
    return (mnist_train, mnist_test), (fmnist_train, fmnist_test), (cifar_train, cifar_test)

def train_expert(task_id, train_dataset, test_dataset, epochs=5, lr=5e-4, wd=1e-4, batch_size=128):
    print(f"\n--- Training Expert for Task {task_id} ---")
    model = MultiTaskResNet18(num_tasks=3, num_classes=10).to(device)
    
    # Save pretrained base if not already saved
    if not os.path.exists("pretrained_base.pt"):
        torch.save(model.backbone.state_dict(), "pretrained_base.pt")
        print("Saved pre-trained ImageNet backbone.")
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x, task_id)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * x.size(0)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            
        scheduler.step()
        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")
        
    # Evaluate
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x, task_id)
            _, predicted = outputs.max(1)
            test_total += y.size(0)
            test_correct += predicted.eq(y).sum().item()
            
    test_acc = 100.0 * test_correct / test_total
    print(f"Task {task_id} Expert Test Accuracy: {test_acc:.2f}%")
    return model, test_acc

# Helper to merge backbones
def merge_backbones(expert_states, pre_state, method="WA", lambda_val=0.4):
    merged_state = copy.deepcopy(expert_states[0])
    
    if method == "WA":
        # Average weight states of backbones
        for key in merged_state.keys():
            tensors = [states[key] for states in expert_states]
            if torch.is_floating_point(tensors[0]):
                merged_state[key] = torch.stack(tensors).mean(dim=0)
            else:
                merged_state[key] = expert_states[0][key]
    elif method == "TA":
        # Task Arithmetic
        for key in merged_state.keys():
            if torch.is_floating_point(expert_states[0][key]):
                task_vectors = [states[key] - pre_state[key] for states in expert_states]
                merged_state[key] = pre_state[key] + lambda_val * torch.stack(task_vectors).sum(dim=0)
            else:
                merged_state[key] = pre_state[key]
            
    return merged_state

# Evaluation function
def evaluate_model(model, test_loaders):
    model.eval()
    accuracies = []
    with torch.no_grad():
        for task_id, loader in enumerate(test_loaders):
            correct = 0
            total = 0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x, task_id)
                _, predicted = outputs.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()
            acc = 100.0 * correct / total
            accuracies.append(acc)
    return accuracies

# N-TAAC Calibration Method
def run_ntaac(model, joint_loader, momentum=1.0):
    model.train()
    # Freeze learnable params
    for p in model.parameters():
        p.requires_grad = False
        
    # Set momentum of BatchNorm layers
    bn_modules = []
    original_moments = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            bn_modules.append(m)
            original_moments.append(m.momentum)
            m.momentum = momentum
            
    # Run forward pass on joint calibration set (single batch)
    with torch.no_grad():
        for x, _ in joint_loader:
            x = x.to(device)
            # Run forward pass through the backbone only
            _ = model.backbone(x)
            break # N-TAAC is a single batch pass
            
    # Restore moments and return to eval
    for m, orig_mom in zip(bn_modules, original_moments):
        m.momentum = orig_mom
    model.eval()

# LSC Calibrator Module
class LSCCalibrator:
    def __init__(self, bn_layers):
        self.bn_layers = bn_layers
        self.orig_stds = {}
        self.merged_stds = {}
        self.gammas = {}
        
    def collect_stds(self, model, loader, task_id, is_merged=False):
        stds = {}
        hooks = []
        
        def make_hook(name):
            def hook(module, input, output):
                # Compute global layer std (unbiased=False)
                std = torch.std(output, unbiased=False)
                if name not in stds:
                    stds[name] = []
                stds[name].append(std.item())
            return hook
            
        for name, module in self.bn_layers:
            hooks.append(module.register_forward_hook(make_hook(name)))
            
        model.eval()
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device)
                _ = model(x, task_id)
                
        for h in hooks:
            h.remove()
            
        for name, val_list in stds.items():
            avg_std = sum(val_list) / len(val_list)
            if is_merged:
                self.merged_stds[(name, task_id)] = avg_std
            else:
                self.orig_stds[(name, task_id)] = avg_std
                
    def compute_gammas(self, task_id, epsilon=1e-5):
        for name, _ in self.bn_layers:
            orig = self.orig_stds.get((name, task_id), 1.0)
            merged = self.merged_stds.get((name, task_id), 1.0)
            self.gammas[(name, task_id)] = (orig + epsilon) / (merged + epsilon)
            
    def apply_hooks(self, model, task_id):
        hooks = []
        def make_scale_hook(name):
            gamma = self.gammas.get((name, task_id), 1.0)
            def hook(module, input, output):
                return output * gamma
            return hook
            
        for name, module in self.bn_layers:
            hooks.append(module.register_forward_hook(make_scale_hook(name)))
        return hooks

def get_bn_layers(model):
    bn_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            bn_layers.append((name, module))
    return bn_layers

# Helper function to configure trainable backbone parameters
def configure_trainable_params(model, adapt_depth="heads"):
    if adapt_depth == "heads":
        for p in model.backbone.parameters():
            p.requires_grad = False
    elif adapt_depth == "layer4":
        for name, p in model.backbone.named_parameters():
            if "layer4" in name:
                p.requires_grad = True
            else:
                p.requires_grad = False
    elif adapt_depth == "full":
        for p in model.backbone.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"Unknown adapt_depth: {adapt_depth}")

# Head Adaptation (Supervised SFT)
def run_head_sft(model, cal_loaders, lr=1e-3, epochs=15, adapt_depth="heads"):
    # Clone heads to optimize them cleanly
    model_sft = copy.deepcopy(model)
    model_sft.eval()
    
    # Configure backbone parameters' requires_grad based on depth
    configure_trainable_params(model_sft, adapt_depth)
        
    # Optimizers for task specific heads + any trainable backbone parameters
    optimizers = []
    backbone_trainable = [p for p in model_sft.backbone.parameters() if p.requires_grad]
    for head in model_sft.heads:
        for p in head.parameters():
            p.requires_grad = True
        # Optimizer includes task head and trainable backbone parameters
        optimizers.append(optim.AdamW(list(head.parameters()) + backbone_trainable, lr=lr))
        
    criterion = nn.CrossEntropyLoss()
    
    # Train each head on its corresponding calibration set
    for task_id, loader in enumerate(cal_loaders):
        optimizer = optimizers[task_id]
        head = model_sft.heads[task_id]
        
        for epoch in range(epochs):
            head.train()
            # If we are training backbone layers, we must keep those layers in train mode
            if adapt_depth != "heads":
                model_sft.backbone.train()
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                outputs = model_sft(x, task_id)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                
    model_sft.eval()
    return model_sft

# Head Adaptation with N-TAAC active (SPJA-SFT)
def run_spja_sft(model, cal_loaders, joint_loader, lr=1e-3, epochs=15, adapt_depth="heads"):
    model_sft = copy.deepcopy(model)
    # First, run N-TAAC on the cloned model to calibrate its backbone
    run_ntaac(model_sft, joint_loader, momentum=1.0)
    
    # Configure backbone parameters' requires_grad based on depth
    configure_trainable_params(model_sft, adapt_depth)
        
    optimizers = []
    backbone_trainable = [p for p in model_sft.backbone.parameters() if p.requires_grad]
    for head in model_sft.heads:
        for p in head.parameters():
            p.requires_grad = True
        optimizers.append(optim.AdamW(list(head.parameters()) + backbone_trainable, lr=lr))
        
    criterion = nn.CrossEntropyLoss()
    
    for task_id, loader in enumerate(cal_loaders):
        optimizer = optimizers[task_id]
        head = model_sft.heads[task_id]
        
        for epoch in range(epochs):
            head.train()
            if adapt_depth != "heads":
                model_sft.backbone.train()
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                outputs = model_sft(x, task_id)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                
    model_sft.eval()
    return model_sft

# Head Adaptation (Unsupervised TTA)
def run_head_tta(model, cal_loaders, expert_models, lr=1e-3, epochs=15, adapt_depth="heads"):
    model_tta = copy.deepcopy(model)
    model_tta.eval()
    
    # Configure backbone parameters' requires_grad based on depth
    configure_trainable_params(model_tta, adapt_depth)
        
    optimizers = []
    backbone_trainable = [p for p in model_tta.backbone.parameters() if p.requires_grad]
    for head in model_tta.heads:
        for p in head.parameters():
            p.requires_grad = True
        optimizers.append(optim.AdamW(list(head.parameters()) + backbone_trainable, lr=lr))
        
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    
    for task_id, loader in enumerate(cal_loaders):
        optimizer = optimizers[task_id]
        head = model_tta.heads[task_id]
        expert = expert_models[task_id]
        expert.eval()
        
        for epoch in range(epochs):
            head.train()
            if adapt_depth != "heads":
                model_tta.backbone.train()
            for x, _ in loader:
                x = x.to(device)
                # Teacher expert logits
                with torch.no_grad():
                    teacher_logits = expert(x, task_id)
                    teacher_probs = torch.softmax(teacher_logits, dim=-1)
                    
                optimizer.zero_grad()
                student_logits = model_tta(x, task_id)
                student_log_probs = torch.log_softmax(student_logits, dim=-1)
                
                loss = kl_loss(student_log_probs, teacher_probs)
                loss.backward()
                optimizer.step()
                
    model_tta.eval()
    return model_tta

# Head Adaptation with N-TAAC active (SPJA-TTA)
def run_spja_tta(model, cal_loaders, expert_models, joint_loader, lr=1e-3, epochs=15, adapt_depth="heads"):
    model_tta = copy.deepcopy(model)
    # First, run N-TAAC on the cloned model to calibrate its backbone
    run_ntaac(model_tta, joint_loader, momentum=1.0)
    
    # Configure backbone parameters' requires_grad based on depth
    configure_trainable_params(model_tta, adapt_depth)
        
    optimizers = []
    backbone_trainable = [p for p in model_tta.backbone.parameters() if p.requires_grad]
    for head in model_tta.heads:
        for p in head.parameters():
            p.requires_grad = True
        optimizers.append(optim.AdamW(list(head.parameters()) + backbone_trainable, lr=lr))
        
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    
    for task_id, loader in enumerate(cal_loaders):
        optimizer = optimizers[task_id]
        head = model_tta.heads[task_id]
        expert = expert_models[task_id]
        expert.eval()
        
        for epoch in range(epochs):
            head.train()
            if adapt_depth != "heads":
                model_tta.backbone.train()
            for x, _ in loader:
                x = x.to(device)
                with torch.no_grad():
                    teacher_logits = expert(x, task_id)
                    teacher_probs = torch.softmax(teacher_logits, dim=-1)
                    
                optimizer.zero_grad()
                student_logits = model_tta(x, task_id)
                student_log_probs = torch.log_softmax(student_logits, dim=-1)
                
                loss = kl_loss(student_log_probs, teacher_probs)
                loss.backward()
                optimizer.step()
                
    model_tta.eval()
    return model_tta

def get_balanced_indices(dataset, cal_size, num_classes=10, seed=42):
    samples_per_class = cal_size // num_classes
    if samples_per_class == 0:
        samples_per_class = 1
        
    class_indices = {c: [] for c in range(num_classes)}
    
    if hasattr(dataset, "targets"):
        targets = dataset.targets
        if torch.is_tensor(targets):
            targets = targets.tolist()
        elif isinstance(targets, np.ndarray):
            targets = targets.tolist()
        for idx, label in enumerate(targets):
            class_indices[label].append(idx)
    else:
        for idx in range(len(dataset)):
            _, y = dataset[idx]
            class_indices[int(y)].append(idx)
            
    rng = random.Random(seed)
    selected_indices = []
    for c in range(num_classes):
        selected_indices.extend(rng.sample(class_indices[c], min(samples_per_class, len(class_indices[c]))))
        
    if len(selected_indices) < cal_size:
        remaining = cal_size - len(selected_indices)
        all_indices = set(range(len(dataset)))
        unused = list(all_indices - set(selected_indices))
        selected_indices.extend(rng.sample(unused, remaining))
        
    rng.shuffle(selected_indices)
    return selected_indices

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_epochs", type=int, default=5)
    parser.add_argument("--cal_size", type=int, default=128)
    parser.add_argument("--head_lr", type=type(1e-3), default=1e-3)
    parser.add_argument("--head_epochs", type=int, default=15)
    parser.add_argument("--method", type=str, default="WA", choices=["WA", "TA"])
    parser.add_argument("--lambda_val", type=float, default=0.4)
    parser.add_argument("--ours_only", action="store_true", help="Only run and evaluate SPJA methods")
    parser.add_argument("--adapt_depth", type=str, default="heads", choices=["heads", "layer4", "full"])
    parser.add_argument("--cal_strategy", type=str, default="sequential", choices=["sequential", "random", "balanced"])
    parser.add_argument("--cal_ratios", type=str, default="1.0,1.0,1.0", help="Comma-separated float ratios for each task's calibration size")
    args = parser.parse_args()
    
    set_seed(args.seed)
    print(f"Configurations: Seed={args.seed}, CalSize={args.cal_size}, HeadLR={args.head_lr}, HeadEpochs={args.head_epochs}, Merging={args.method}, Lambda={args.lambda_val}, AdaptDepth={args.adapt_depth}, CalStrategy={args.cal_strategy}, CalRatios={args.cal_ratios}")
    
    # Get datasets
    mnist, fmnist, cifar = get_datasets()
    
    # Train/Load experts
    expert_checkpoints = ["mnist_expert.pt", "fashion_expert.pt", "cifar_expert.pt"]
    expert_models = []
    expert_accs = []
    
    # MNIST Expert
    if os.path.exists(expert_checkpoints[0]):
        print("Loading MNIST expert from checkpoint...")
        model = MultiTaskResNet18().to(device)
        model.load_state_dict(torch.load(expert_checkpoints[0], map_location=device))
        model.eval()
        expert_models.append(model)
        # Fast evaluation of expert to check accuracy
        test_loader = DataLoader(mnist[1], batch_size=128, shuffle=False)
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x, 0)
                _, pred = out.max(1)
                total += y.size(0)
                correct += pred.eq(y).sum().item()
        expert_accs.append(100.0 * correct / total)
        print(f"MNIST expert loaded. Test Acc: {expert_accs[0]:.2f}%")
    else:
        model, acc = train_expert(0, mnist[0], mnist[1], epochs=args.train_epochs)
        torch.save(model.state_dict(), expert_checkpoints[0])
        expert_models.append(model)
        expert_accs.append(acc)
        
    # Fashion-MNIST Expert
    if os.path.exists(expert_checkpoints[1]):
        print("Loading Fashion-MNIST expert from checkpoint...")
        model = MultiTaskResNet18().to(device)
        model.load_state_dict(torch.load(expert_checkpoints[1], map_location=device))
        model.eval()
        expert_models.append(model)
        test_loader = DataLoader(fmnist[1], batch_size=128, shuffle=False)
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x, 1)
                _, pred = out.max(1)
                total += y.size(0)
                correct += pred.eq(y).sum().item()
        expert_accs.append(100.0 * correct / total)
        print(f"Fashion-MNIST expert loaded. Test Acc: {expert_accs[1]:.2f}%")
    else:
        model, acc = train_expert(1, fmnist[0], fmnist[1], epochs=args.train_epochs)
        torch.save(model.state_dict(), expert_checkpoints[1])
        expert_models.append(model)
        expert_accs.append(acc)
        
    # CIFAR-10 Expert
    if os.path.exists(expert_checkpoints[2]):
        print("Loading CIFAR-10 expert from checkpoint...")
        model = MultiTaskResNet18().to(device)
        model.load_state_dict(torch.load(expert_checkpoints[2], map_location=device))
        model.eval()
        expert_models.append(model)
        test_loader = DataLoader(cifar[1], batch_size=128, shuffle=False)
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x, 2)
                _, pred = out.max(1)
                total += y.size(0)
                correct += pred.eq(y).sum().item()
        expert_accs.append(100.0 * correct / total)
        print(f"CIFAR-10 expert loaded. Test Acc: {expert_accs[2]:.2f}%")
    else:
        model, acc = train_expert(2, cifar[0], cifar[1], epochs=args.train_epochs)
        torch.save(model.state_dict(), expert_checkpoints[2])
        expert_models.append(model)
        expert_accs.append(acc)
        
    print(f"\nExpert Accuracies: MNIST={expert_accs[0]:.2f}%, Fashion={expert_accs[1]:.2f}%, CIFAR10={expert_accs[2]:.2f}%")
    print(f"Average Expert Accuracy: {sum(expert_accs)/3:.2f}%")
    
    # Setup test dataloaders
    test_loaders = [
        DataLoader(mnist[1], batch_size=256, shuffle=False, num_workers=2, pin_memory=True),
        DataLoader(fmnist[1], batch_size=256, shuffle=False, num_workers=2, pin_memory=True),
        DataLoader(cifar[1], batch_size=256, shuffle=False, num_workers=2, pin_memory=True),
    ]
    
    # Parse ratios
    ratios = [float(r) for r in args.cal_ratios.split(",")]
    assert len(ratios) == 3, "Must provide exactly 3 ratios for the 3 tasks"
    cal_sizes = [max(1, int(round(r * args.cal_size))) for r in ratios]
    
    # Create Calibration subsets
    print(f"\nCreating Calibration datasets (Sizes MNIST={cal_sizes[0]}, Fashion={cal_sizes[1]}, CIFAR10={cal_sizes[2]} with strategy={args.cal_strategy})...")
    if args.cal_strategy == "sequential":
        cal_indices_mnist = list(range(cal_sizes[0]))
        cal_indices_fmnist = list(range(cal_sizes[1]))
        cal_indices_cifar = list(range(cal_sizes[2]))
    elif args.cal_strategy == "random":
        set_seed(args.seed + 1000) # derivative seed for sampling
        cal_indices_mnist = random.sample(range(len(mnist[0])), cal_sizes[0])
        cal_indices_fmnist = random.sample(range(len(fmnist[0])), cal_sizes[1])
        cal_indices_cifar = random.sample(range(len(cifar[0])), cal_sizes[2])
        set_seed(args.seed) # restore main seed
    elif args.cal_strategy == "balanced":
        cal_indices_mnist = get_balanced_indices(mnist[0], cal_sizes[0], seed=args.seed)
        cal_indices_fmnist = get_balanced_indices(fmnist[0], cal_sizes[1], seed=args.seed)
        cal_indices_cifar = get_balanced_indices(cifar[0], cal_sizes[2], seed=args.seed)
    else:
        raise ValueError(f"Unknown cal_strategy: {args.cal_strategy}")
        
    cal_mnist_ds = Subset(mnist[0], cal_indices_mnist)
    cal_fmnist_ds = Subset(fmnist[0], cal_indices_fmnist)
    cal_cifar_ds = Subset(cifar[0], cal_indices_cifar)
    
    cal_loaders = [
        DataLoader(cal_mnist_ds, batch_size=cal_sizes[0], shuffle=False, num_workers=2),
        DataLoader(cal_fmnist_ds, batch_size=cal_sizes[1], shuffle=False, num_workers=2),
        DataLoader(cal_cifar_ds, batch_size=cal_sizes[2], shuffle=False, num_workers=2)
    ]
    
    # Joint calibration dataset (for N-TAAC)
    joint_cal_ds = ConcatDataset([cal_mnist_ds, cal_fmnist_ds, cal_cifar_ds])
    # Batch size is sum of cal_sizes
    joint_cal_loader = DataLoader(joint_cal_ds, batch_size=sum(cal_sizes), shuffle=False, num_workers=2)
    
    # Merging backbones
    expert_backbone_states = [expert_models[k].backbone.state_dict() for k in range(3)]
    pre_backbone_state = torch.load("pretrained_base.pt", map_location=device)
    
    merged_backbone_state = merge_backbones(expert_backbone_states, pre_backbone_state, method=args.method, lambda_val=args.lambda_val)
    
    # Construct Merged Model
    # It gets the merged backbone and retains the task heads of their respective experts
    merged_model = MultiTaskResNet18().to(device)
    merged_model.backbone.load_state_dict(merged_backbone_state)
    for k in range(3):
        merged_model.heads[k].load_state_dict(expert_models[k].heads[k].state_dict())
        
    # --- EVALUATE BASELINES ---
    
    if args.ours_only:
        print("\nSkipping baselines evaluation (ours_only=True)")
        accs_none = [0.0, 0.0, 0.0]
        accs_ntaac = [0.0, 0.0, 0.0]
        accs_lsc = [0.0, 0.0, 0.0]
        accs_sft = [0.0, 0.0, 0.0]
        accs_tta = [0.0, 0.0, 0.0]
    else:
        # 1. Uncalibrated Baseline (NONE)
        accs_none = evaluate_model(merged_model, test_loaders)
        print(f"\n1. Baseline (NONE) Accuracies: MNIST={accs_none[0]:.2f}%, Fashion={accs_none[1]:.2f}%, CIFAR10={accs_none[2]:.2f}% | Avg={sum(accs_none)/3:.2f}%")
        
        # 2. N-TAAC Calibration (Paper 3)
        model_ntaac = copy.deepcopy(merged_model)
        run_ntaac(model_ntaac, joint_cal_loader, momentum=1.0)
        accs_ntaac = evaluate_model(model_ntaac, test_loaders)
        print(f"2. N-TAAC Accuracies: MNIST={accs_ntaac[0]:.2f}%, Fashion={accs_ntaac[1]:.2f}%, CIFAR10={accs_ntaac[2]:.2f}% | Avg={sum(accs_ntaac)/3:.2f}%")
        
        # 3. LSC Calibration (Paper 9)
        bn_layers = get_bn_layers(merged_model)
        lsc_calibrator = LSCCalibrator(bn_layers)
        
        # Collect stds on original experts
        for k in range(3):
            lsc_calibrator.collect_stds(expert_models[k], cal_loaders[k], task_id=k, is_merged=False)
            
        # Collect stds on merged model
        for k in range(3):
            lsc_calibrator.collect_stds(merged_model, cal_loaders[k], task_id=k, is_merged=True)
            lsc_calibrator.compute_gammas(task_id=k)
            
        # Evaluate LSC (Applying task-specific scaling hooks during test)
        accs_lsc = []
        for k in range(3):
            hooks = lsc_calibrator.apply_hooks(merged_model, task_id=k)
            # Evaluate task k only
            loader = test_loaders[k]
            merged_model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    out = merged_model(x, k)
                    _, pred = out.max(1)
                    total += y.size(0)
                    correct += pred.eq(y).sum().item()
            accs_lsc.append(100.0 * correct / total)
            for h in hooks:
                h.remove()
        print(f"3. LSC Accuracies: MNIST={accs_lsc[0]:.2f}%, Fashion={accs_lsc[1]:.2f}%, CIFAR10={accs_lsc[2]:.2f}% | Avg={sum(accs_lsc)/3:.2f}%")
        
        # 4. Head-only Supervised SFT (Paper 10)
        model_sft = run_head_sft(merged_model, cal_loaders, lr=args.head_lr, epochs=args.head_epochs, adapt_depth=args.adapt_depth)
        accs_sft = evaluate_model(model_sft, test_loaders)
        print(f"4. Head-SFT Accuracies: MNIST={accs_sft[0]:.2f}%, Fashion={accs_sft[1]:.2f}%, CIFAR10={accs_sft[2]:.2f}% | Avg={sum(accs_sft)/3:.2f}%")
        
        # 5. Head-only Unsupervised TTA (Paper 10)
        model_tta = run_head_tta(merged_model, cal_loaders, expert_models, lr=args.head_lr, epochs=args.head_epochs, adapt_depth=args.adapt_depth)
        accs_tta = evaluate_model(model_tta, test_loaders)
        print(f"5. Head-TTA Accuracies: MNIST={accs_tta[0]:.2f}%, Fashion={accs_tta[1]:.2f}%, CIFAR10={accs_tta[2]:.2f}% | Avg={sum(accs_tta)/3:.2f}%")
    
    # 6. SPJA-SFT (Our Proposed Method: N-TAAC + Head SFT)
    model_spja_sft = run_spja_sft(merged_model, cal_loaders, joint_cal_loader, lr=args.head_lr, epochs=args.head_epochs, adapt_depth=args.adapt_depth)
    accs_spja_sft = evaluate_model(model_spja_sft, test_loaders)
    print(f"6. SPJA-SFT Accuracies (Ours): MNIST={accs_spja_sft[0]:.2f}%, Fashion={accs_spja_sft[1]:.2f}%, CIFAR10={accs_spja_sft[2]:.2f}% | Avg={sum(accs_spja_sft)/3:.2f}%")
    
    # 7. SPJA-TTA (Our Proposed Method: N-TAAC + Head TTA)
    model_spja_tta = run_spja_tta(merged_model, cal_loaders, expert_models, joint_cal_loader, lr=args.head_lr, epochs=args.head_epochs, adapt_depth=args.adapt_depth)
    accs_spja_tta = evaluate_model(model_spja_tta, test_loaders)
    print(f"7. SPJA-TTA Accuracies (Ours): MNIST={accs_spja_tta[0]:.2f}%, Fashion={accs_spja_tta[1]:.2f}%, CIFAR10={accs_spja_tta[2]:.2f}% | Avg={sum(accs_spja_tta)/3:.2f}%")

    # Output CSV format of results to console for parsing
    print("\nRESULTS_CSV_START")
    print("Method,MNIST,Fashion,CIFAR10,Average")
    print(f"NONE,{accs_none[0]:.4f},{accs_none[1]:.4f},{accs_none[2]:.4f},{sum(accs_none)/3:.4f}")
    print(f"N-TAAC,{accs_ntaac[0]:.4f},{accs_ntaac[1]:.4f},{accs_ntaac[2]:.4f},{sum(accs_ntaac)/3:.4f}")
    print(f"LSC,{accs_lsc[0]:.4f},{accs_lsc[1]:.4f},{accs_lsc[2]:.4f},{sum(accs_lsc)/3:.4f}")
    print(f"Head-SFT,{accs_sft[0]:.4f},{accs_sft[1]:.4f},{accs_sft[2]:.4f},{sum(accs_sft)/3:.4f}")
    print(f"Head-TTA,{accs_tta[0]:.4f},{accs_tta[1]:.4f},{accs_tta[2]:.4f},{sum(accs_tta)/3:.4f}")
    print(f"SPJA-SFT,{accs_spja_sft[0]:.4f},{accs_spja_sft[1]:.4f},{accs_spja_sft[2]:.4f},{sum(accs_spja_sft)/3:.4f}")
    print(f"SPJA-TTA,{accs_spja_tta[0]:.4f},{accs_spja_tta[1]:.4f},{accs_spja_tta[2]:.4f},{sum(accs_spja_tta)/3:.4f}")
    print("RESULTS_CSV_END")

if __name__ == "__main__":
    main()
