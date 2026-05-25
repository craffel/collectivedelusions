import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
import os
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Disable cuDNN to bypass CUDNN_STATUS_NOT_INITIALIZED on cluster nodes
    torch.backends.cudnn.enabled = False

set_seed(42)

# SimpleCNN Architecture
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.pool3 = nn.AdaptiveAvgPool2d((3, 3))
        self.fc = nn.Linear(64 * 3 * 3, 10)
        self.relu = nn.ReLU()

    def forward(self, x, return_features=False):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        feat = self.pool3(x)
        feat = feat.view(feat.size(0), -1)
        out = self.fc(feat)
        if return_features:
            return out, feat
        return out

# Helper to merge parameters on-the-fly
def merge_parameters(expert0, expert1, merged_model, global_coef, layer_offsets):
    state0 = expert0.state_dict()
    state1 = expert1.state_dict()
    merged_state = {}
    
    for name, param in state0.items():
        if param.dtype.is_floating_point and name in layer_offsets:
            offset = layer_offsets[name]
            lambda_val = torch.sigmoid(global_coef + offset)
            merged_state[name] = (1.0 - lambda_val) * state0[name] + lambda_val * state1[name]
        elif param.dtype.is_floating_point:
            lambda_val = torch.sigmoid(global_coef)
            merged_state[name] = (1.0 - lambda_val) * state0[name] + lambda_val * state1[name]
        else:
            merged_state[name] = state0[name]
            
    merged_model.load_state_dict(merged_state)

# Helper to perform differentiable model forwarding on merged parameters
def differentiable_forward(expert0, expert1, merged_model, global_coef, layer_offsets, inputs, return_features=False):
    from torch.func import functional_call
    state0 = expert0.state_dict()
    state1 = expert1.state_dict()
    merged_params_and_buffers = {}
    
    for name, param in state0.items():
        if param.dtype.is_floating_point and name in layer_offsets:
            offset = layer_offsets[name]
            lambda_val = torch.sigmoid(global_coef + offset)
            merged_params_and_buffers[name] = (1.0 - lambda_val) * state0[name] + lambda_val * state1[name]
        elif param.dtype.is_floating_point:
            lambda_val = torch.sigmoid(global_coef)
            merged_params_and_buffers[name] = ((1.0 - lambda_val) * state0[name] + lambda_val * state1[name]).detach()
        else:
            merged_params_and_buffers[name] = state0[name]
            
    if return_features:
        return functional_call(merged_model, merged_params_and_buffers, args=(inputs,), kwargs={"return_features": True})
    else:
        return functional_call(merged_model, merged_params_and_buffers, args=(inputs,))

# Helper to calculate entropy
def softmax_entropy(x):
    probs = torch.softmax(x, dim=-1)
    return -torch.sum(probs * torch.log(probs + 1e-6), dim=-1).mean()

# Test-time augmentations for SSCPA
def apply_test_time_augmentations(x):
    # Augmentation 1: translation + small noise
    shift_x = random.randint(-2, 2)
    shift_y = random.randint(-2, 2)
    x_aug1 = torch.roll(x, shifts=(shift_x, shift_y), dims=(2, 3))
    x_aug1 = x_aug1 + 0.05 * torch.randn_like(x_aug1)
    
    # Augmentation 2: small rotation or scale (implemented as scaling + padding)
    scale = random.uniform(0.9, 1.1)
    h, w = x.shape[2], x.shape[3]
    new_h, new_w = int(h * scale), int(w * scale)
    x_resized = nn.functional.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
    if scale < 1.0:
        pad_h = (h - new_h) // 2
        pad_w = (w - new_w) // 2
        x_aug2 = nn.functional.pad(x_resized, (pad_w, h - new_h - pad_w, pad_h, w - new_w - pad_h), mode='constant', value=-1.0)
    else:
        crop_h = (new_h - h) // 2
        crop_w = (new_w - w) // 2
        x_aug2 = x_resized[:, :, crop_h:crop_h+h, crop_w:crop_w+w]
        
    x_aug2 = x_aug2 + 0.05 * torch.randn_like(x_aug2)
    return x_aug1, x_aug2

def train_expert(model, train_loader, epochs=2, device="cpu"):
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(train_loader):.4f} | Acc: {100.*correct/total:.2f}%")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    
    # 1. Download/Load datasets
    print("Loading Datasets...")
    mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    fashion_train = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    fashion_test = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    
    kmnist_test = torchvision.datasets.KMNIST(root='./data', train=False, download=True, transform=transform)
    
    # DataLoaders for Training
    mnist_train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=128, shuffle=True)
    fashion_train_loader = torch.utils.data.DataLoader(fashion_train, batch_size=128, shuffle=True)
    
    # 2. Train Expert 0 (MNIST) and Expert 1 (FashionMNIST) starting from a shared base initialization
    base_model_path = "base_model.pt"
    expert0_path = "expert0.pt"
    expert1_path = "expert1.pt"
    
    base_model = SimpleCNN().to(device)
    
    if os.path.exists(base_model_path) and os.path.exists(expert0_path) and os.path.exists(expert1_path):
        print("Loading pre-trained experts...")
        base_model.load_state_dict(torch.load(base_model_path, map_location=device))
        expert0 = SimpleCNN().to(device)
        expert0.load_state_dict(torch.load(expert0_path, map_location=device))
        expert1 = SimpleCNN().to(device)
        expert1.load_state_dict(torch.load(expert1_path, map_location=device))
    else:
        print("Pre-trained experts not found. Training from scratch...")
        torch.save(base_model.state_dict(), base_model_path)
        
        expert0 = SimpleCNN().to(device)
        expert0.load_state_dict(torch.load(base_model_path, map_location=device))
        print("Training Expert 0 (MNIST)...")
        train_expert(expert0, mnist_train_loader, epochs=2, device=device)
        torch.save(expert0.state_dict(), expert0_path)
        
        expert1 = SimpleCNN().to(device)
        expert1.load_state_dict(torch.load(base_model_path, map_location=device))
        print("Training Expert 1 (FashionMNIST)...")
        train_expert(expert1, fashion_train_loader, epochs=2, device=device)
        torch.save(expert1.state_dict(), expert1_path)
        
    expert0.eval()
    expert1.eval()
    
    # Set experts' BN layers to training mode with momentum=0 for test-time adaptation
    for m in expert0.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.train()
            m.momentum = 0.0
    for m in expert1.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.train()
            m.momentum = 0.0
            
    # Verify Standalone expert accuracy with BN adapt
    def eval_accuracy(model, dataset_loader):
        model.eval()
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.train()
                m.momentum = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in dataset_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        return 100.0 * correct / total

    mnist_test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=256, shuffle=False)
    fashion_test_loader = torch.utils.data.DataLoader(fashion_test, batch_size=256, shuffle=False)
    
    acc_exp0_mnist = eval_accuracy(expert0, mnist_test_loader)
    acc_exp1_fashion = eval_accuracy(expert1, fashion_test_loader)
    print(f"Expert 0 Accuracy on MNIST (with BN Adapt): {acc_exp0_mnist:.2f}%")
    print(f"Expert 1 Accuracy on FashionMNIST (with BN Adapt): {acc_exp1_fashion:.2f}%")
    
    # 3. Construct Non-stationary vision stream of 50 batches of size 64
    batch_size = 64
    
    mnist_eval_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=True)
    fashion_eval_loader = torch.utils.data.DataLoader(fashion_test, batch_size=batch_size, shuffle=True)
    kmnist_eval_loader = torch.utils.data.DataLoader(kmnist_test, batch_size=batch_size, shuffle=True)
    
    mnist_iter = iter(mnist_eval_loader)
    fashion_iter = iter(fashion_eval_loader)
    kmnist_iter = iter(kmnist_eval_loader)
    
    stream_batches = []
    
    # Segment 1: Clean MNIST (batches 0-9)
    for _ in range(10):
        inputs, targets = next(mnist_iter)
        stream_batches.append((inputs, targets, "Clean MNIST"))
        
    # Segment 2: Noisy MNIST with Gaussian noise (std=0.6, batches 10-19)
    for _ in range(10):
        inputs, targets = next(mnist_iter)
        noisy_inputs = inputs + 0.6 * torch.randn_like(inputs)
        noisy_inputs = torch.clamp(noisy_inputs, -1.0, 1.0)
        stream_batches.append((noisy_inputs, targets, "Noisy MNIST"))
        
    # Segment 3: Clean FashionMNIST (batches 20-29)
    for _ in range(10):
        inputs, targets = next(fashion_iter)
        stream_batches.append((inputs, targets, "Clean Fashion"))
        
    # Segment 4: Noisy FashionMNIST with Gaussian noise (std=0.6, batches 30-39)
    for _ in range(10):
        inputs, targets = next(fashion_iter)
        noisy_inputs = inputs + 0.6 * torch.randn_like(inputs)
        noisy_inputs = torch.clamp(noisy_inputs, -1.0, 1.0)
        stream_batches.append((noisy_inputs, targets, "Noisy Fashion"))
        
    # Segment 5: Novel KMNIST (batches 40-49)
    for _ in range(10):
        inputs, targets = next(kmnist_iter)
        stream_batches.append((inputs, targets, "Novel KMNIST"))
        
    print(f"Successfully constructed non-stationary stream with {len(stream_batches)} batches.")
    
    # 4. Precompute Static Prototypes for Expert 0 and Expert 1 (with BN adapt)
    print("Precomputing prototypes...")
    def compute_prototype(expert_model, dataset, num_samples=512):
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
        features = []
        with torch.no_grad():
            for inputs, _ in loader:
                inputs = inputs.to(device)
                _, feat = expert_model(inputs, return_features=True)
                features.append(feat)
                if len(features) * 64 >= num_samples:
                    break
        features = torch.cat(features, dim=0)[:num_samples]
        norm_features = features / (torch.norm(features, dim=-1, keepdim=True) + 1e-6)
        prototype = norm_features.mean(dim=0)
        prototype = prototype / (torch.norm(prototype) + 1e-6)
        return prototype
        
    prototype_0 = compute_prototype(expert0, mnist_train)
    prototype_1 = compute_prototype(expert1, fashion_train)
    
    print("Prototype 0 precomputed shape:", prototype_0.shape)
    print("Prototype 1 precomputed shape:", prototype_1.shape)
    
    # Define layer structure for layer-wise merging offsets
    layer_names = [name for name, _ in expert0.named_parameters() if _.dtype.is_floating_point]
    
    # 5. Define evaluation methods
    methods = ["Static Merging", "TTA (Entropy Min)", "BK-CoMerge (Approx)", "SSCPA (Ours)"]
    method_accuracies = {m: [] for m in methods}
    method_coefficients = {m: [] for m in methods}
    
    # Let's run each method across the stream
    for m_idx, method in enumerate(methods):
        print(f"\nEvaluating method: {method}...")
        
        # Set up dynamic prototypes for methods that adapt them
        dyn_proto_0 = prototype_0.clone()
        dyn_proto_1 = prototype_1.clone()
        
        # Evaluate batch-by-batch
        for b_idx, (inputs, targets, seg_name) in enumerate(stream_batches):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Reset merging coefficients and offsets per batch to avoid adaptation inertia
            global_coef = nn.Parameter(torch.tensor(0.0, device=device))
            layer_offsets = {name: nn.Parameter(torch.tensor(0.0, device=device)) for name in layer_names}
            all_params = [global_coef] + list(layer_offsets.values())
            optimizer = optim.AdamW(all_params, lr=1.0, weight_decay=1e-4)
            
            # Setup base merged model
            merged_model = SimpleCNN().to(device)
            merged_model.eval()
            
            # Enable BN adaptation with momentum=0 for merged model
            for m in merged_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.train()
                    m.momentum = 0.0
            
            if method == "Static Merging":
                # Fixed 0.5 / 0.5 merging
                with torch.no_grad():
                    g_coef = torch.tensor(0.0, device=device)
                    offsets = {name: torch.tensor(0.0, device=device) for name in layer_names}
                    outputs = differentiable_forward(expert0, expert1, merged_model, g_coef, offsets, inputs)
                    
            elif method == "TTA (Entropy Min)":
                # Standard TTA: optimize merging coefficients on current batch via entropy
                for step in range(5):
                    optimizer.zero_grad()
                    outputs = differentiable_forward(expert0, expert1, merged_model, global_coef, layer_offsets, inputs)
                    loss = softmax_entropy(outputs)
                    loss.backward()
                    optimizer.step()
                
                with torch.no_grad():
                    outputs = differentiable_forward(expert0, expert1, merged_model, global_coef, layer_offsets, inputs)
                    
            elif method == "BK-CoMerge (Approx)":
                for step in range(5):
                    optimizer.zero_grad()
                    outputs = differentiable_forward(expert0, expert1, merged_model, global_coef, layer_offsets, inputs)
                    entropy_loss = softmax_entropy(outputs)
                    entropy_loss.backward(retain_graph=True)
                    
                    with torch.no_grad():
                        for name in layer_names:
                            param = layer_offsets[name]
                            if param.grad is not None:
                                sens = torch.norm(param.grad) + 1e-4
                                param.grad = param.grad / sens
                                
                    optimizer.step()
                    
                with torch.no_grad():
                    outputs = differentiable_forward(expert0, expert1, merged_model, global_coef, layer_offsets, inputs)
                    
            elif method == "SSCPA (Ours)":
                # Self-Supervised Contrastive Prototype Alignment (SSCPA) with BN Adapt
                for step in range(5):
                    optimizer.zero_grad()
                    
                    # 1. Dynamic routing via mismatch-free expert prototype similarity
                    with torch.no_grad():
                        _, feat0 = expert0(inputs, return_features=True)
                        _, feat1 = expert1(inputs, return_features=True)
                        feat0_norm = feat0 / (torch.norm(feat0, dim=-1, keepdim=True) + 1e-6)
                        feat1_norm = feat1 / (torch.norm(feat1, dim=-1, keepdim=True) + 1e-6)
                        avg_feat0 = feat0_norm.mean(dim=0)
                        avg_feat0 = avg_feat0 / (torch.norm(avg_feat0) + 1e-6)
                        avg_feat1 = feat1_norm.mean(dim=0)
                        avg_feat1 = avg_feat1 / (torch.norm(avg_feat1) + 1e-6)
                        
                        sim0 = torch.dot(avg_feat0, dyn_proto_0)
                        sim1 = torch.dot(avg_feat1, dyn_proto_1)
                        routing_scores = torch.stack([sim0, sim1]) / 0.1
                        pi = torch.softmax(routing_scores, dim=0)
                    
                    # 2. Forward raw batch to get predictions for entropy loss
                    outputs_raw = differentiable_forward(expert0, expert1, merged_model, global_coef, layer_offsets, inputs)
                    
                    # 3. Get test-time augmented views
                    inputs_aug1, inputs_aug2 = apply_test_time_augmentations(inputs)
                    
                    # Forward augmented views through experts to get target representations
                    with torch.no_grad():
                        _, feat_exp0_aug1 = expert0(inputs_aug1, return_features=True)
                        _, feat_exp1_aug1 = expert1(inputs_aug1, return_features=True)
                        _, feat_exp0_aug2 = expert0(inputs_aug2, return_features=True)
                        _, feat_exp1_aug2 = expert1(inputs_aug2, return_features=True)
                        
                        feat_exp0_aug1 = feat_exp0_aug1 / (torch.norm(feat_exp0_aug1, dim=-1, keepdim=True) + 1e-6)
                        feat_exp1_aug1 = feat_exp1_aug1 / (torch.norm(feat_exp1_aug1, dim=-1, keepdim=True) + 1e-6)
                        feat_exp0_aug2 = feat_exp0_aug2 / (torch.norm(feat_exp0_aug2, dim=-1, keepdim=True) + 1e-6)
                        feat_exp1_aug2 = feat_exp1_aug2 / (torch.norm(feat_exp1_aug2, dim=-1, keepdim=True) + 1e-6)
                    
                    # Forward augmented views through merged model to get features
                    _, feat_aug1 = differentiable_forward(expert0, expert1, merged_model, global_coef, layer_offsets, inputs_aug1, return_features=True)
                    _, feat_aug2 = differentiable_forward(expert0, expert1, merged_model, global_coef, layer_offsets, inputs_aug2, return_features=True)
                    
                    # Normalize features
                    feat_aug1_norm = feat_aug1 / (torch.norm(feat_aug1, dim=-1, keepdim=True) + 1e-6)
                    feat_aug2_norm = feat_aug2 / (torch.norm(feat_aug2, dim=-1, keepdim=True) + 1e-6)
                    
                    avg_aug1 = feat_aug1_norm.mean(dim=0)
                    avg_aug1 = avg_aug1 / (torch.norm(avg_aug1) + 1e-6)
                    
                    avg_aug2 = feat_aug2_norm.mean(dim=0)
                    avg_aug2 = avg_aug2 / (torch.norm(avg_aug2) + 1e-6)
                    
                    # Sample-wise alignment loss
                    loss_align = - (
                        pi[0] * (feat_aug1_norm * feat_exp0_aug1).sum(dim=-1).mean() +
                        pi[1] * (feat_aug1_norm * feat_exp1_aug1).sum(dim=-1).mean() +
                        pi[0] * (feat_aug2_norm * feat_exp0_aug2).sum(dim=-1).mean() +
                        pi[1] * (feat_aug2_norm * feat_exp1_aug2).sum(dim=-1).mean()
                    )
                    
                    # Consistency loss (between views)
                    loss_consistency = - torch.dot(avg_aug1, avg_aug2)
                    
                    # Selective entropy loss
                    ent = softmax_entropy(outputs_raw)
                    
                    # Total Loss
                    loss_total = loss_align + 0.5 * loss_consistency + 0.1 * ent
                    loss_total.backward()
                    optimizer.step()
                    
                # Dynamic prototype update on high-confidence predictions using expert features
                with torch.no_grad():
                    outputs = differentiable_forward(expert0, expert1, merged_model, global_coef, layer_offsets, inputs)
                    
                    _, feat0 = expert0(inputs, return_features=True)
                    _, feat1 = expert1(inputs, return_features=True)
                    
                    feat0_norm = feat0 / (torch.norm(feat0, dim=-1, keepdim=True) + 1e-6)
                    feat1_norm = feat1 / (torch.norm(feat1, dim=-1, keepdim=True) + 1e-6)
                    
                    avg_feat0 = feat0_norm.mean(dim=0)
                    avg_feat0 = avg_feat0 / (torch.norm(avg_feat0) + 1e-6)
                    
                    avg_feat1 = feat1_norm.mean(dim=0)
                    avg_feat1 = avg_feat1 / (torch.norm(avg_feat1) + 1e-6)
                    
                    sim0 = torch.dot(avg_feat0, dyn_proto_0)
                    sim1 = torch.dot(avg_feat1, dyn_proto_1)
                    routing_scores = torch.stack([sim0, sim1]) / 0.1
                    pi = torch.softmax(routing_scores, dim=0)
                    
                    # Update active prototype with momentum 0.9 if confidence > 0.85
                    if pi[0] > 0.85:
                        dyn_proto_0 = 0.9 * dyn_proto_0 + 0.1 * avg_feat0
                        dyn_proto_0 = dyn_proto_0 / (torch.norm(dyn_proto_0) + 1e-6)
                    elif pi[1] > 0.85:
                        dyn_proto_1 = 0.9 * dyn_proto_1 + 0.1 * avg_feat1
                        dyn_proto_1 = dyn_proto_1 / (torch.norm(dyn_proto_1) + 1e-6)
            
            # Compute accuracy on the batch
            with torch.no_grad():
                _, predicted = outputs.max(1)
                correct = predicted.eq(targets).sum().item()
                acc = 100.0 * correct / targets.size(0)
                method_accuracies[method].append(acc)
                m_coef = torch.sigmoid(global_coef).item()
                method_coefficients[method].append(m_coef)
                
            if (b_idx + 1) % 10 == 0:
                seg_acc = np.mean(method_accuracies[method][b_idx-9:b_idx+1])
                print(f"Batch {b_idx+1}/50 ({seg_name}) | Accuracy: {seg_acc:.2f}% | Merging Coef: {m_coef:.3f}")

    # 6. Print Summary Metrics by Segment
    segment_names = ["Clean MNIST", "Noisy MNIST", "Clean Fashion", "Noisy Fashion", "Novel KMNIST"]
    segment_results = {m: {} for m in methods}
    
    print("\n" + "="*50)
    print("FINAL SEGMENT RESULTS COMPARISON WITH BN ADAPT")
    print("="*50)
    
    for method in methods:
        print(f"\nMethod: {method}")
        for s_idx, seg in enumerate(segment_names):
            start = s_idx * 10
            end = (s_idx + 1) * 10
            seg_acc = np.mean(method_accuracies[method][start:end])
            segment_results[method][seg] = seg_acc
            print(f"  {seg}: {seg_acc:.2f}%")
        overall = np.mean(method_accuracies[method])
        segment_results[method]["Overall"] = overall
        print(f"  Overall Stream: {overall:.2f}%")
        
    # Generate Line Plot
    plt.figure(figsize=(12, 6))
    for method in methods:
        accs = method_accuracies[method]
        smoothed_accs = np.convolve(accs, np.ones(3)/3, mode='valid')
        plt.plot(np.arange(len(smoothed_accs)) + 1, smoothed_accs, label=method, linewidth=2)
        
    # Mark Segments
    for i in range(1, 5):
        plt.axvline(x=i*10, color='gray', linestyle='--', alpha=0.5)
        
    # Annotate segments
    for s_idx, seg in enumerate(segment_names):
        plt.text(s_idx*10 + 3, 5, seg, fontsize=10, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))
        
    plt.title("Streaming Test-Time Model Merging Accuracy (with BN Adapt)", fontsize=14)
    plt.xlabel("Batch Number in Test Stream", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.ylim(0, 105)
    plt.legend(fontsize=10, loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("accuracy_comparison.png", dpi=300)
    print("\nComparison plot saved as 'accuracy_comparison.png'")
    
    # Generate LaTeX Table
    print("\n" + "="*50)
    print("LATEX TABLE OF EXPERIMENTAL RESULTS WITH BN ADAPT")
    print("="*50)
    latex = """\\begin{table*}[t]
\\centering
\\caption{Classification accuracy (\\%) of test-time model merging methods under test-time Batch Normalization adaptation (BN Adapt) across five non-stationary stream segments. The best performing method in each column is highlighted in \\textbf{bold}.}
\\label{table:bn_adapt_results}
\\begin{tabular}{lcccccc}
\\toprule
\\textbf{Method} & \\textbf{Clean MNIST} & \\textbf{Noisy MNIST} & \\textbf{Clean Fashion} & \\textbf{Noisy Fashion} & \\textbf{Novel KMNIST} & \\textbf{Overall} \\\\
\\midrule
"""
    for m in methods:
        row = f"{m} "
        best_val = {}
        for seg in segment_names + ["Overall"]:
            best_val[seg] = max([segment_results[other][seg] for other in methods])
            
        for seg in segment_names + ["Overall"]:
            val = segment_results[m][seg]
            if abs(val - best_val[seg]) < 1e-5:
                row += f"& \\textbf{{{val:.2f}\\%}} "
            else:
                row += f"& {val:.2f}\\% "
        row += "\\\\\n"
        latex += row
        
    latex += """\\bottomrule
\\end{tabular}
\\end{table*}
"""
    print(latex)

if __name__ == "__main__":
    main()
