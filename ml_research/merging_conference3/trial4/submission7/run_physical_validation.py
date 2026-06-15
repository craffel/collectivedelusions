import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import scipy.optimize

# Set seed for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)

# Define network architecture
class SharedFeatureExtractor(nn.Module):
    def __init__(self):
        super(SharedFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2) # 14x14
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2) # 7x7
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2) # 3x3 -> flattened: 64*3*3 = 576
        
        self.fc1 = nn.Linear(576, 128)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu4(self.fc1(x)))
        return x

class MultiTaskModel(nn.Module):
    def __init__(self, feature_extractor, head):
        super(MultiTaskModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.head = head

    def forward(self, x):
        feats = self.feature_extractor(x)
        return self.head(feats)

def get_datasets():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Download MNIST & FMNIST
    mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    fmnist_train = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    fmnist_test = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    
    # Extract small subsets for rapid CPU training & evaluation
    set_seed(42)
    mnist_train_sub = Subset(mnist_train, np.random.choice(len(mnist_train), 2000, replace=False))
    mnist_test_sub = Subset(mnist_test, np.random.choice(len(mnist_test), 500, replace=False))
    fmnist_train_sub = Subset(fmnist_train, np.random.choice(len(fmnist_train), 2000, replace=False))
    fmnist_test_sub = Subset(fmnist_test, np.random.choice(len(fmnist_test), 500, replace=False))
    
    return mnist_train_sub, mnist_test_sub, fmnist_train_sub, fmnist_test_sub

def pretrain_base(base_fe, mnist_train, fmnist_train, epochs=2):
    print("Pre-training shared feature extractor on combined dataset (MNIST + FMNIST)...")
    base_fe.train()
    # Create joint 20-class classification head (classes 0-9 for MNIST, 10-19 for FMNIST)
    pretrain_head = nn.Linear(128, 20)
    pretrain_head.train()
    
    # Combine datasets
    combined_data = []
    for idx in range(len(mnist_train)):
        d, y = mnist_train[idx]
        combined_data.append((d, y))
    for idx in range(len(fmnist_train)):
        d, y = fmnist_train[idx]
        combined_data.append((d, y + 10))
        
    loader = DataLoader(combined_data, batch_size=64, shuffle=True)
    
    optimizer = optim.Adam(list(base_fe.parameters()) + list(pretrain_head.parameters()), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        for idx, (data, targets) in enumerate(loader):
            optimizer.zero_grad()
            feats = base_fe(data)
            outputs = pretrain_head(feats)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
    print("Pre-training complete.")
    base_fe.eval()

def train_expert(base_fe, head, train_loader, epochs=3):
    # Deep copy the feature extractor to fine-tune from the shared base
    import copy
    fe = copy.deepcopy(base_fe)
    model = MultiTaskModel(fe, head)
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        for idx, (data, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
    return fe, head

def evaluate_model(fe, head, test_loader):
    model = MultiTaskModel(fe, head)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, targets in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
    return (correct / total) * 100.0

def merge_models(base_fe, fe1, fe2, alpha1, alpha2):
    # Layers mapping (7 layers)
    layers_keys = [
        ["conv1.weight", "conv1.bias"],
        ["bn1.weight", "bn1.bias", "bn1.running_mean", "bn1.running_var"],
        ["conv2.weight", "conv2.bias"],
        ["bn2.weight", "bn2.bias", "bn2.running_mean", "bn2.running_var"],
        ["conv3.weight", "conv3.bias"],
        ["bn3.weight", "bn3.bias", "bn3.running_mean", "bn3.running_var"],
        ["fc1.weight", "fc1.bias"]
    ]
    
    import copy
    merged_fe = copy.deepcopy(base_fe)
    base_state = base_fe.state_dict()
    fe1_state = fe1.state_dict()
    fe2_state = fe2.state_dict()
    merged_state = merged_fe.state_dict()
    
    for l_idx, keys in enumerate(layers_keys):
        a1 = alpha1[l_idx]
        a2 = alpha2[l_idx]
        for key in keys:
            if key in base_state:
                tau1 = fe1_state[key] - base_state[key]
                tau2 = fe2_state[key] - base_state[key]
                # Cast alpha values to tensor of matching device and type
                a1_t = torch.tensor(a1, dtype=base_state[key].dtype, device=base_state[key].device)
                a2_t = torch.tensor(a2, dtype=base_state[key].dtype, device=base_state[key].device)
                merged_state[key] = base_state[key] + a1_t * tau1 + a2_t * tau2
                if key.endswith("running_var"):
                    merged_state[key] = torch.clamp(merged_state[key], min=1e-5)
                
    merged_fe.load_state_dict(merged_state)
    return merged_fe

# Prediction entropy for TTA
def calculate_entropy(outputs):
    probs = torch.softmax(outputs, dim=1)
    probs = torch.clamp(probs, min=1e-8)
    entropy = -torch.sum(probs * torch.log(probs), dim=1)
    return torch.mean(entropy)

# Soft merged feature extractor for exact backpropagation
class SoftMergedFeatureExtractor(nn.Module):
    def __init__(self, base_fe, fe1, fe2):
        super(SoftMergedFeatureExtractor, self).__init__()
        self.base_fe = base_fe
        self.fe1 = fe1
        self.fe2 = fe2
        
    def forward(self, x, alpha1, alpha2):
        # Helper to merge weight/bias
        def merge_param(base_p, p1, p2, a1, a2):
            if base_p is None:
                return None
            return base_p + a1 * (p1 - base_p) + a2 * (p2 - base_p)
        
        # Conv1
        w = merge_param(self.base_fe.conv1.weight, self.fe1.conv1.weight, self.fe2.conv1.weight, alpha1[0], alpha2[0])
        b = merge_param(self.base_fe.conv1.bias, self.fe1.conv1.bias, self.fe2.conv1.bias, alpha1[0], alpha2[0])
        x = nn.functional.conv2d(x, w, b, padding=1)
        
        # BN1
        w = merge_param(self.base_fe.bn1.weight, self.fe1.bn1.weight, self.fe2.bn1.weight, alpha1[1], alpha2[1])
        b = merge_param(self.base_fe.bn1.bias, self.fe1.bn1.bias, self.fe2.bn1.bias, alpha1[1], alpha2[1])
        rm = merge_param(self.base_fe.bn1.running_mean, self.fe1.bn1.running_mean, self.fe2.bn1.running_mean, alpha1[1], alpha2[1]).detach()
        rv = merge_param(self.base_fe.bn1.running_var, self.fe1.bn1.running_var, self.fe2.bn1.running_var, alpha1[1], alpha2[1]).detach()
        rv = torch.clamp(rv, min=1e-5)
        x = nn.functional.batch_norm(x, rm, rv, w, b, training=False)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        
        # Conv2
        w = merge_param(self.base_fe.conv2.weight, self.fe1.conv2.weight, self.fe2.conv2.weight, alpha1[2], alpha2[2])
        b = merge_param(self.base_fe.conv2.bias, self.fe1.conv2.bias, self.fe2.conv2.bias, alpha1[2], alpha2[2])
        x = nn.functional.conv2d(x, w, b, padding=1)
        
        # BN2
        w = merge_param(self.base_fe.bn2.weight, self.fe1.bn2.weight, self.fe2.bn2.weight, alpha1[3], alpha2[3])
        b = merge_param(self.base_fe.bn2.bias, self.fe1.bn2.bias, self.fe2.bn2.bias, alpha1[3], alpha2[3])
        rm = merge_param(self.base_fe.bn2.running_mean, self.fe1.bn2.running_mean, self.fe2.bn2.running_mean, alpha1[3], alpha2[3]).detach()
        rv = merge_param(self.base_fe.bn2.running_var, self.fe1.bn2.running_var, self.fe2.bn2.running_var, alpha1[3], alpha2[3]).detach()
        rv = torch.clamp(rv, min=1e-5)
        x = nn.functional.batch_norm(x, rm, rv, w, b, training=False)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        
        # Conv3
        w = merge_param(self.base_fe.conv3.weight, self.fe1.conv3.weight, self.fe2.conv3.weight, alpha1[4], alpha2[4])
        b = merge_param(self.base_fe.conv3.bias, self.fe1.conv3.bias, self.fe2.conv3.bias, alpha1[4], alpha2[4])
        x = nn.functional.conv2d(x, w, b, padding=1)
        
        # BN3
        w = merge_param(self.base_fe.bn3.weight, self.fe1.bn3.weight, self.fe2.bn3.weight, alpha1[5], alpha2[5])
        b = merge_param(self.base_fe.bn3.bias, self.fe1.bn3.bias, self.fe2.bn3.bias, alpha1[5], alpha2[5])
        rm = merge_param(self.base_fe.bn3.running_mean, self.fe1.bn3.running_mean, self.fe2.bn3.running_mean, alpha1[5], alpha2[5]).detach()
        rv = merge_param(self.base_fe.bn3.running_var, self.fe1.bn3.running_var, self.fe2.bn3.running_var, alpha1[5], alpha2[5]).detach()
        rv = torch.clamp(rv, min=1e-5)
        x = nn.functional.batch_norm(x, rm, rv, w, b, training=False)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        
        # FC1
        x = x.view(x.size(0), -1)
        w = merge_param(self.base_fe.fc1.weight, self.fe1.fc1.weight, self.fe2.fc1.weight, alpha1[6], alpha2[6])
        b = merge_param(self.base_fe.fc1.bias, self.fe1.fc1.bias, self.fe2.fc1.bias, alpha1[6], alpha2[6])
        x = nn.functional.linear(x, w, b)
        x = nn.functional.relu(x)
        return x

def run_evaluation_suite(base_fe, fe_mnist, fe_fmnist, head_mnist, head_fmnist,
                         mnist_test_loader, fmnist_test_loader, combined_test_data, combined_test_tasks,
                         mnist_train_sub, fmnist_train_sub, name="Regime"):
    print(f"\n--- Running Evaluation Suite for: {name} ---")
    
    # 1. Uniform Merging
    alpha_uniform_1 = np.full(7, 0.5)
    alpha_uniform_2 = np.full(7, 0.5)
    fe_uniform = merge_models(base_fe, fe_mnist, fe_fmnist, alpha_uniform_1, alpha_uniform_2)
    acc_mnist_unif = evaluate_model(fe_uniform, head_mnist, mnist_test_loader)
    acc_fmnist_unif = evaluate_model(fe_uniform, head_fmnist, fmnist_test_loader)
    avg_unif = np.mean([acc_mnist_unif, acc_fmnist_unif])
    print(f"Uniform Merging: MNIST={acc_mnist_unif:.2f}%, FMNIST={acc_fmnist_unif:.2f}%, Avg={avg_unif:.2f}%")
    
    soft_fe = SoftMergedFeatureExtractor(base_fe, fe_mnist, fe_fmnist)
    layers_depth = torch.arange(1, 8, dtype=torch.float32) / 7.0
    beta_ema = 0.90
    
    # 2. Online AdaMerging (Unsupervised TTA - No privileged task routing)
    alpha1_t = torch.full((7,), 0.5, dtype=torch.float32, requires_grad=True)
    alpha2_t = torch.full((7,), 0.5, dtype=torch.float32, requires_grad=True)
    optimizer = optim.Adam([alpha1_t, alpha2_t], lr=0.01)
    ema_alpha1 = torch.full((7,), 0.5, dtype=torch.float32)
    ema_alpha2 = torch.full((7,), 0.5, dtype=torch.float32)
    
    set_seed(42)
    for step in range(100):
        optimizer.zero_grad()
        indices = np.random.choice(len(combined_test_data), 64, replace=False)
        batch_x = combined_test_data[indices]
        
        feats = soft_fe(batch_x, alpha1_t, alpha2_t)
        out1 = head_mnist(feats)
        out2 = head_fmnist(feats)
        
        entropy1 = calculate_entropy(out1)
        entropy2 = calculate_entropy(out2)
        
        loss = entropy1 + entropy2 # Mathematically flawed joint OOD minimization
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            alpha1_t.clamp_(0.0, 1.0)
            alpha2_t.clamp_(0.0, 1.0)
            ema_alpha1 = beta_ema * ema_alpha1 + (1 - beta_ema) * alpha1_t
            ema_alpha2 = beta_ema * ema_alpha2 + (1 - beta_ema) * alpha2_t
            
    fe_ada_unsup = merge_models(base_fe, fe_mnist, fe_fmnist, ema_alpha1.detach().numpy(), ema_alpha2.detach().numpy())
    acc_mnist_ada_unsup = evaluate_model(fe_ada_unsup, head_mnist, mnist_test_loader)
    acc_fmnist_ada_unsup = evaluate_model(fe_ada_unsup, head_fmnist, fmnist_test_loader)
    avg_ada_unsup = np.mean([acc_mnist_ada_unsup, acc_fmnist_ada_unsup])
    print(f"Online AdaMerging (Unsupervised TTA - Mixed Stream): MNIST={acc_mnist_ada_unsup:.2f}%, FMNIST={acc_fmnist_ada_unsup:.2f}%, Avg={avg_ada_unsup:.2f}%")
    
    # 3. Online AdaMerging (Privileged TTA - With task routing mask)
    alpha1_t = torch.full((7,), 0.5, dtype=torch.float32, requires_grad=True)
    alpha2_t = torch.full((7,), 0.5, dtype=torch.float32, requires_grad=True)
    optimizer = optim.Adam([alpha1_t, alpha2_t], lr=0.01)
    ema_alpha1 = torch.full((7,), 0.5, dtype=torch.float32)
    ema_alpha2 = torch.full((7,), 0.5, dtype=torch.float32)
    
    set_seed(42)
    for step in range(100):
        optimizer.zero_grad()
        indices = np.random.choice(len(combined_test_data), 64, replace=False)
        batch_x = combined_test_data[indices]
        batch_tasks = combined_test_tasks[indices]
        
        feats = soft_fe(batch_x, alpha1_t, alpha2_t)
        out1 = head_mnist(feats)
        out2 = head_fmnist(feats)
        
        probs1 = torch.softmax(out1, dim=1)
        probs1 = torch.clamp(probs1, min=1e-8)
        entropy1 = -torch.sum(probs1 * torch.log(probs1), dim=1)
        
        probs2 = torch.softmax(out2, dim=1)
        probs2 = torch.clamp(probs2, min=1e-8)
        entropy2 = -torch.sum(probs2 * torch.log(probs2), dim=1)
        
        loss_mnist = entropy1[batch_tasks == 0].mean() if (batch_tasks == 0).any() else torch.tensor(0.0, device=feats.device)
        loss_fmnist = entropy2[batch_tasks == 1].mean() if (batch_tasks == 1).any() else torch.tensor(0.0, device=feats.device)
        
        loss = loss_mnist + loss_fmnist
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            alpha1_t.clamp_(0.0, 1.0)
            alpha2_t.clamp_(0.0, 1.0)
            ema_alpha1 = beta_ema * ema_alpha1 + (1 - beta_ema) * alpha1_t
            ema_alpha2 = beta_ema * ema_alpha2 + (1 - beta_ema) * alpha2_t
            
    fe_ada_priv = merge_models(base_fe, fe_mnist, fe_fmnist, ema_alpha1.detach().numpy(), ema_alpha2.detach().numpy())
    acc_mnist_ada_priv = evaluate_model(fe_ada_priv, head_mnist, mnist_test_loader)
    acc_fmnist_ada_priv = evaluate_model(fe_ada_priv, head_fmnist, fmnist_test_loader)
    avg_ada_priv = np.mean([acc_mnist_ada_priv, acc_fmnist_ada_priv])
    print(f"Online AdaMerging (Privileged TTA - Task Routed): MNIST={acc_mnist_ada_priv:.2f}%, FMNIST={acc_fmnist_ada_priv:.2f}%, Avg={avg_ada_priv:.2f}%")
    
    # 4. Online PolyMerge (Unsupervised TTA)
    c1_t = torch.zeros((3,), dtype=torch.float32, requires_grad=True)
    c2_t = torch.zeros((3,), dtype=torch.float32, requires_grad=True)
    with torch.no_grad():
        c1_t[0] = 0.5
        c2_t[0] = 0.5
    optimizer_poly = optim.Adam([c1_t, c2_t], lr=0.01)
    ema_c1 = torch.zeros((3,), dtype=torch.float32)
    ema_c2 = torch.zeros((3,), dtype=torch.float32)
    with torch.no_grad():
        ema_c1[0] = 0.5
        ema_c2[0] = 0.5
        
    set_seed(42)
    for step in range(100):
        optimizer_poly.zero_grad()
        indices = np.random.choice(len(combined_test_data), 64, replace=False)
        batch_x = combined_test_data[indices]
        
        alpha1 = c1_t[0] + c1_t[1] * layers_depth + c1_t[2] * (layers_depth ** 2)
        alpha2 = c2_t[0] + c2_t[1] * layers_depth + c2_t[2] * (layers_depth ** 2)
        
        feats = soft_fe(batch_x, alpha1, alpha2)
        out1 = head_mnist(feats)
        out2 = head_fmnist(feats)
        
        loss = calculate_entropy(out1) + calculate_entropy(out2)
        loss.backward()
        optimizer_poly.step()
        
        with torch.no_grad():
            ema_c1 = beta_ema * ema_c1 + (1 - beta_ema) * c1_t
            ema_c2 = beta_ema * ema_c2 + (1 - beta_ema) * c2_t
            
    a1_poly_unsup = (ema_c1[0] + ema_c1[1] * layers_depth + ema_c1[2] * (layers_depth ** 2)).detach().numpy()
    a2_poly_unsup = (ema_c2[0] + ema_c2[1] * layers_depth + ema_c2[2] * (layers_depth ** 2)).detach().numpy()
    fe_poly_unsup = merge_models(base_fe, fe_mnist, fe_fmnist, np.clip(a1_poly_unsup, 0.0, 1.0), np.clip(a2_poly_unsup, 0.0, 1.0))
    acc_mnist_poly_unsup = evaluate_model(fe_poly_unsup, head_mnist, mnist_test_loader)
    acc_fmnist_poly_unsup = evaluate_model(fe_poly_unsup, head_fmnist, fmnist_test_loader)
    avg_poly_unsup = np.mean([acc_mnist_poly_unsup, acc_fmnist_poly_unsup])
    print(f"Online PolyMerge (Unsupervised TTA): MNIST={acc_mnist_poly_unsup:.2f}%, FMNIST={acc_fmnist_poly_unsup:.2f}%, Avg={avg_poly_unsup:.2f}%")
    
    # 5. Online PolyMerge (Privileged TTA)
    c1_t = torch.zeros((3,), dtype=torch.float32, requires_grad=True)
    c2_t = torch.zeros((3,), dtype=torch.float32, requires_grad=True)
    with torch.no_grad():
        c1_t[0] = 0.5
        c2_t[0] = 0.5
    optimizer_poly = optim.Adam([c1_t, c2_t], lr=0.01)
    ema_c1 = torch.zeros((3,), dtype=torch.float32)
    ema_c2 = torch.zeros((3,), dtype=torch.float32)
    with torch.no_grad():
        ema_c1[0] = 0.5
        ema_c2[0] = 0.5
        
    set_seed(42)
    for step in range(100):
        optimizer_poly.zero_grad()
        indices = np.random.choice(len(combined_test_data), 64, replace=False)
        batch_x = combined_test_data[indices]
        batch_tasks = combined_test_tasks[indices]
        
        alpha1 = c1_t[0] + c1_t[1] * layers_depth + c1_t[2] * (layers_depth ** 2)
        alpha2 = c2_t[0] + c2_t[1] * layers_depth + c2_t[2] * (layers_depth ** 2)
        
        feats = soft_fe(batch_x, alpha1, alpha2)
        out1 = head_mnist(feats)
        out2 = head_fmnist(feats)
        
        probs1 = torch.softmax(out1, dim=1)
        probs1 = torch.clamp(probs1, min=1e-8)
        entropy1 = -torch.sum(probs1 * torch.log(probs1), dim=1)
        
        probs2 = torch.softmax(out2, dim=1)
        probs2 = torch.clamp(probs2, min=1e-8)
        entropy2 = -torch.sum(probs2 * torch.log(probs2), dim=1)
        
        loss_mnist = entropy1[batch_tasks == 0].mean() if (batch_tasks == 0).any() else torch.tensor(0.0, device=feats.device)
        loss_fmnist = entropy2[batch_tasks == 1].mean() if (batch_tasks == 1).any() else torch.tensor(0.0, device=feats.device)
        loss = loss_mnist + loss_fmnist
        loss.backward()
        optimizer_poly.step()
        
        with torch.no_grad():
            ema_c1 = beta_ema * ema_c1 + (1 - beta_ema) * c1_t
            ema_c2 = beta_ema * ema_c2 + (1 - beta_ema) * c2_t
            
    a1_poly_priv = (ema_c1[0] + ema_c1[1] * layers_depth + ema_c1[2] * (layers_depth ** 2)).detach().numpy()
    a2_poly_priv = (ema_c2[0] + ema_c2[1] * layers_depth + ema_c2[2] * (layers_depth ** 2)).detach().numpy()
    fe_poly_priv = merge_models(base_fe, fe_mnist, fe_fmnist, np.clip(a1_poly_priv, 0.0, 1.0), np.clip(a2_poly_priv, 0.0, 1.0))
    acc_mnist_poly_priv = evaluate_model(fe_poly_priv, head_mnist, mnist_test_loader)
    acc_fmnist_poly_priv = evaluate_model(fe_poly_priv, head_fmnist, fmnist_test_loader)
    avg_poly_priv = np.mean([acc_mnist_poly_priv, acc_fmnist_poly_priv])
    print(f"Online PolyMerge (Privileged TTA): MNIST={acc_mnist_poly_priv:.2f}%, FMNIST={acc_fmnist_poly_priv:.2f}%, Avg={avg_poly_priv:.2f}%")
    
    # 6. Offline Few-Shot Validation Tuning (OFS-Tune, d=1)
    set_seed(42)
    val_mnist_indices = np.random.choice(len(mnist_train_sub), 10, replace=False)
    val_fmnist_indices = np.random.choice(len(fmnist_train_sub), 10, replace=False)
    val_mnist_x = []
    val_mnist_y = []
    for idx in val_mnist_indices:
        d, y = mnist_train_sub[idx]
        val_mnist_x.append(d)
        val_mnist_y.append(y)
    val_fmnist_x = []
    val_fmnist_y = []
    for idx in val_fmnist_indices:
        d, y = fmnist_train_sub[idx]
        val_fmnist_x.append(d)
        val_fmnist_y.append(y)
    val_mnist_x = torch.stack(val_mnist_x)
    val_mnist_y = torch.tensor(val_mnist_y)
    val_fmnist_x = torch.stack(val_fmnist_x)
    val_fmnist_y = torch.tensor(val_fmnist_y)
    criterion_ce = nn.CrossEntropyLoss()
    
    def validation_loss(theta):
        c1_0, c1_1, c2_0, c2_1 = theta
        a1 = c1_0 + c1_1 * layers_depth.numpy()
        a2 = c2_0 + c2_1 * layers_depth.numpy()
        a1 = np.clip(a1, 0.0, 1.0)
        a2 = np.clip(a2, 0.0, 1.0)
        with torch.no_grad():
            a1_t = torch.tensor(a1, dtype=torch.float32)
            a2_t = torch.tensor(a2, dtype=torch.float32)
            feats_m = soft_fe(val_mnist_x, a1_t, a2_t)
            out_m = head_mnist(feats_m)
            loss_m = criterion_ce(out_m, val_mnist_y)
            feats_f = soft_fe(val_fmnist_x, a1_t, a2_t)
            out_f = head_fmnist(feats_f)
            loss_f = criterion_ce(out_f, val_fmnist_y)
        return (loss_m.item() + loss_f.item())

    init_theta = [0.5, 0.0, 0.5, 0.0]
    res = scipy.optimize.minimize(validation_loss, init_theta, method='Nelder-Mead', options={'maxiter': 500})
    opt_theta = res.x
    a1_ofs = opt_theta[0] + opt_theta[1] * layers_depth.numpy()
    a2_ofs = opt_theta[2] + opt_theta[3] * layers_depth.numpy()
    a1_ofs = np.clip(a1_ofs, 0.0, 1.0)
    a2_ofs = np.clip(a2_ofs, 0.0, 1.0)
    fe_ofs = merge_models(base_fe, fe_mnist, fe_fmnist, a1_ofs, a2_ofs)
    acc_mnist_ofs = evaluate_model(fe_ofs, head_mnist, mnist_test_loader)
    acc_fmnist_ofs = evaluate_model(fe_ofs, head_fmnist, fmnist_test_loader)
    avg_ofs = np.mean([acc_mnist_ofs, acc_fmnist_ofs])
    print(f"Offline OFS-Tune (Ours): MNIST={acc_mnist_ofs:.2f}%, FMNIST={acc_fmnist_ofs:.2f}%, Avg={avg_ofs:.2f}%")
    
    return {
        "Uniform": (acc_mnist_unif, acc_fmnist_unif, avg_unif),
        "AdaMerging_Unsup": (acc_mnist_ada_unsup, acc_fmnist_ada_unsup, avg_ada_unsup),
        "AdaMerging_Priv": (acc_mnist_ada_priv, acc_fmnist_ada_priv, avg_ada_priv),
        "PolyMerge_Unsup": (acc_mnist_poly_unsup, acc_fmnist_poly_unsup, avg_poly_unsup),
        "PolyMerge_Priv": (acc_mnist_poly_priv, acc_fmnist_poly_priv, avg_poly_priv),
        "OFS_Tune": (acc_mnist_ofs, acc_fmnist_ofs, avg_ofs)
    }

def main():
    print("=== SuiteMerge: Starting Physical Weight-Space Validation ===")
    set_seed(42)
    
    # 1. Load data
    mnist_train_sub, mnist_test_sub, fmnist_train_sub, fmnist_test_sub = get_datasets()
    
    mnist_train_loader = DataLoader(mnist_train_sub, batch_size=64, shuffle=True)
    mnist_test_loader = DataLoader(mnist_test_sub, batch_size=64, shuffle=False)
    fmnist_train_loader = DataLoader(fmnist_train_sub, batch_size=64, shuffle=True)
    fmnist_test_loader = DataLoader(fmnist_test_sub, batch_size=64, shuffle=False)
    
    # Create combined test set for TTA with task labels for task routing
    combined_test_data = []
    combined_test_tasks = [] # 0 for MNIST, 1 for FMNIST
    
    for d, _ in mnist_test_sub:
        combined_test_data.append(d)
        combined_test_tasks.append(0)
    for d, _ in fmnist_test_sub:
        combined_test_data.append(d)
        combined_test_tasks.append(1)
        
    combined_test_data = torch.stack(combined_test_data)
    combined_test_tasks = torch.tensor(combined_test_tasks, dtype=torch.long)
    
    # -------------------------------------------------------------
    # REGIME A: Scratch-Trained (High Representational Conflict)
    # -------------------------------------------------------------
    print("\n=============================================================")
    print("REGIME A: Scratch-Trained Experts (High Representation Conflict)")
    print("=============================================================")
    base_fe_scratch = SharedFeatureExtractor()
    mnist_head_scratch = nn.Linear(128, 10)
    fmnist_head_scratch = nn.Linear(128, 10)
    
    print("Training MNIST Expert from random initialization (3 epochs)...")
    fe_mnist_scratch, head_mnist_scratch = train_expert(base_fe_scratch, mnist_head_scratch, mnist_train_loader, epochs=3)
    acc_mnist_exp_scratch = evaluate_model(fe_mnist_scratch, head_mnist_scratch, mnist_test_loader)
    print(f"MNIST Expert test accuracy: {acc_mnist_exp_scratch:.2f}%")
    
    print("Training FashionMNIST Expert from random initialization (3 epochs)...")
    # To resolve the shared-initialization confound and evaluate truly disjoint basins,
    # we instantiate a separate base feature extractor with a different random seed initialization.
    set_seed(100)
    base_fe_scratch_2 = SharedFeatureExtractor()
    set_seed(42)  # Restore the default seed for reproducibility of subsequent sections
    fe_fmnist_scratch, head_fmnist_scratch = train_expert(base_fe_scratch_2, fmnist_head_scratch, fmnist_train_loader, epochs=3)
    acc_fmnist_exp_scratch = evaluate_model(fe_fmnist_scratch, head_fmnist_scratch, fmnist_test_loader)
    print(f"FashionMNIST Expert test accuracy: {acc_fmnist_exp_scratch:.2f}%")
    
    results_scratch = run_evaluation_suite(
        base_fe_scratch, fe_mnist_scratch, fe_fmnist_scratch, head_mnist_scratch, head_fmnist_scratch,
        mnist_test_loader, fmnist_test_loader, combined_test_data, combined_test_tasks,
        mnist_train_sub, fmnist_train_sub, name="Scratch-Trained (Disjoint Basins)"
    )
    
    # -------------------------------------------------------------
    # REGIME B: Pre-trained Shared Basin (Low Representational Conflict)
    # -------------------------------------------------------------
    print("\n=============================================================")
    print("REGIME B: Pre-trained Shared Basin (Low Representation Conflict)")
    print("=============================================================")
    base_fe_pretrain = SharedFeatureExtractor()
    pretrain_base(base_fe_pretrain, mnist_train_sub, fmnist_train_sub, epochs=2)
    
    mnist_head_pretrain = nn.Linear(128, 10)
    fmnist_head_pretrain = nn.Linear(128, 10)
    
    print("\nTraining MNIST Expert by fine-tuning from pre-trained shared base (3 epochs)...")
    fe_mnist_pretrain, head_mnist_pretrain = train_expert(base_fe_pretrain, mnist_head_pretrain, mnist_train_loader, epochs=3)
    acc_mnist_exp_pretrain = evaluate_model(fe_mnist_pretrain, head_mnist_pretrain, mnist_test_loader)
    print(f"MNIST Expert fine-tuned test accuracy: {acc_mnist_exp_pretrain:.2f}%")
    
    print("\nTraining FashionMNIST Expert by fine-tuning from pre-trained shared base (3 epochs)...")
    fe_fmnist_pretrain, head_fmnist_pretrain = train_expert(base_fe_pretrain, fmnist_head_pretrain, fmnist_train_loader, epochs=3)
    acc_fmnist_exp_pretrain = evaluate_model(fe_fmnist_pretrain, head_fmnist_pretrain, fmnist_test_loader)
    print(f"FashionMNIST Expert fine-tuned test accuracy: {acc_fmnist_exp_pretrain:.2f}%")
    
    results_pretrain = run_evaluation_suite(
        base_fe_pretrain, fe_mnist_pretrain, fe_fmnist_pretrain, head_mnist_pretrain, head_fmnist_pretrain,
        mnist_test_loader, fmnist_test_loader, combined_test_data, combined_test_tasks,
        mnist_train_sub, fmnist_train_sub, name="Pre-trained (Shared Basin)"
    )
    
    # --- Save physical results to physical_results.txt ---
    print("\nSaving physical validation results...")
    with open("physical_results.txt", "w") as f:
        f.write("=== SuiteMerge Physical Weight-Space Validation Accuracies ===\n\n")
        f.write("-------------------------------------------------------------\n")
        f.write("REGIME A: Scratch-Trained Experts (Disjoint Basins)\n")
        f.write("-------------------------------------------------------------\n")
        f.write(f"MNIST Expert: {acc_mnist_exp_scratch:.2f}%, FMNIST Expert: {acc_fmnist_exp_scratch:.2f}%\n")
        f.write(f"Uniform Baseline:               MNIST={results_scratch['Uniform'][0]:.2f}%, FMNIST={results_scratch['Uniform'][1]:.2f}%, Avg={results_scratch['Uniform'][2]:.2f}%\n")
        f.write(f"Online AdaMerging (Unsupervised): MNIST={results_scratch['AdaMerging_Unsup'][0]:.2f}%, FMNIST={results_scratch['AdaMerging_Unsup'][1]:.2f}%, Avg={results_scratch['AdaMerging_Unsup'][2]:.2f}%\n")
        f.write(f"Online AdaMerging (Privileged):   MNIST={results_scratch['AdaMerging_Priv'][0]:.2f}%, FMNIST={results_scratch['AdaMerging_Priv'][1]:.2f}%, Avg={results_scratch['AdaMerging_Priv'][2]:.2f}%\n")
        f.write(f"Online PolyMerge (Unsupervised):  MNIST={results_scratch['PolyMerge_Unsup'][0]:.2f}%, FMNIST={results_scratch['PolyMerge_Unsup'][1]:.2f}%, Avg={results_scratch['PolyMerge_Unsup'][2]:.2f}%\n")
        f.write(f"Online PolyMerge (Privileged):    MNIST={results_scratch['PolyMerge_Priv'][0]:.2f}%, FMNIST={results_scratch['PolyMerge_Priv'][1]:.2f}%, Avg={results_scratch['PolyMerge_Priv'][2]:.2f}%\n")
        f.write(f"Offline OFS-Tune:               MNIST={results_scratch['OFS_Tune'][0]:.2f}%, FMNIST={results_scratch['OFS_Tune'][1]:.2f}%, Avg={results_scratch['OFS_Tune'][2]:.2f}%\n")
        
        f.write("\n-------------------------------------------------------------\n")
        f.write("REGIME B: Pre-trained Shared Basin (Linear Mode Connected)\n")
        f.write("-------------------------------------------------------------\n")
        f.write(f"MNIST Expert: {acc_mnist_exp_pretrain:.2f}%, FMNIST Expert: {acc_fmnist_exp_pretrain:.2f}%\n")
        f.write(f"Uniform Baseline:               MNIST={results_pretrain['Uniform'][0]:.2f}%, FMNIST={results_pretrain['Uniform'][1]:.2f}%, Avg={results_pretrain['Uniform'][2]:.2f}%\n")
        f.write(f"Online AdaMerging (Unsupervised): MNIST={results_pretrain['AdaMerging_Unsup'][0]:.2f}%, FMNIST={results_pretrain['AdaMerging_Unsup'][1]:.2f}%, Avg={results_pretrain['AdaMerging_Unsup'][2]:.2f}%\n")
        f.write(f"Online AdaMerging (Privileged):   MNIST={results_pretrain['AdaMerging_Priv'][0]:.2f}%, FMNIST={results_pretrain['AdaMerging_Priv'][1]:.2f}%, Avg={results_pretrain['AdaMerging_Priv'][2]:.2f}%\n")
        f.write(f"Online PolyMerge (Unsupervised):  MNIST={results_pretrain['PolyMerge_Unsup'][0]:.2f}%, FMNIST={results_pretrain['PolyMerge_Unsup'][1]:.2f}%, Avg={results_pretrain['PolyMerge_Unsup'][2]:.2f}%\n")
        f.write(f"Online PolyMerge (Privileged):    MNIST={results_pretrain['PolyMerge_Priv'][0]:.2f}%, FMNIST={results_pretrain['PolyMerge_Priv'][1]:.2f}%, Avg={results_pretrain['PolyMerge_Priv'][2]:.2f}%\n")
        f.write(f"Offline OFS-Tune:               MNIST={results_pretrain['OFS_Tune'][0]:.2f}%, FMNIST={results_pretrain['OFS_Tune'][1]:.2f}%, Avg={results_pretrain['OFS_Tune'][2]:.2f}%\n")
        
    print("Physical validation completed and results saved!")

if __name__ == '__main__':
    main()
