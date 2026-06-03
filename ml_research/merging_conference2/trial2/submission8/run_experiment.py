import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import math
import os

# --- Mathematical ReLU propagation ---
def relu_propagation(mu, var, eps=1e-8):
    sigma = torch.sqrt(torch.clamp(var, min=eps))
    alpha = -mu / sigma
    
    # Standard normal PDF
    phi = torch.exp(-0.5 * alpha**2) / math.sqrt(2 * math.pi)
    
    # Standard normal CDF
    cdf = 0.5 * (1.0 + torch.erf(alpha / math.sqrt(2.0)))
    
    mean_relu = mu * (1.0 - cdf) + sigma * phi
    m2_relu = (mu**2 + var) * (1.0 - cdf) + mu * sigma * phi
    var_relu = m2_relu - mean_relu**2
    
    # Handle small variance case
    mean_relu = torch.where(sigma < 1e-5, torch.clamp(mu, min=0.0), mean_relu)
    var_relu = torch.where(sigma < 1e-5, torch.where(mu >= 0.0, var, torch.zeros_like(var)), var_relu)
    
    return mean_relu, torch.clamp(var_relu, min=0.0)

# --- ResNet-18 Analytical Propagation ---
def propagate_stats(model, m_in, v_in):
    stats = {}
    m = m_in
    v = v_in
    
    # 1. conv1
    w_sum = model.conv1.weight.sum(dim=(2, 3))
    w_sq_sum = (model.conv1.weight**2).sum(dim=(2, 3))
    m = w_sum @ m
    v = w_sq_sum @ v
    stats['bn1'] = (m.clone(), v.clone())
    
    # 2. bn1 output
    m = model.bn1.bias
    v = model.bn1.weight**2
    
    # 3. relu
    m, v = relu_propagation(m, v)
    
    # 4. maxpool (identity mapping for statistics)
    
    # 5. layer1, layer2, layer3, layer4
    def propagate_basic_block(block, m_x, v_x, prefix):
        # conv1
        w1_sum = block.conv1.weight.sum(dim=(2, 3))
        w1_sq_sum = (block.conv1.weight**2).sum(dim=(2, 3))
        m_c1 = w1_sum @ m_x
        v_c1 = w1_sq_sum @ v_x
        stats[prefix + '.bn1'] = (m_c1.clone(), v_c1.clone())
        
        # bn1
        m_bn1 = block.bn1.bias
        v_bn1 = block.bn1.weight**2
        
        # relu
        m_r1, v_r1 = relu_propagation(m_bn1, v_bn1)
        
        # conv2
        w2_sum = block.conv2.weight.sum(dim=(2, 3))
        w2_sq_sum = (block.conv2.weight**2).sum(dim=(2, 3))
        m_c2 = w2_sum @ m_r1
        v_c2 = w2_sq_sum @ v_r1
        stats[prefix + '.bn2'] = (m_c2.clone(), v_c2.clone())
        
        # bn2
        m_bn2 = block.bn2.bias
        v_bn2 = block.bn2.weight**2
        
        # downsample
        if block.downsample is not None:
            conv_ds = block.downsample[0]
            bn_ds = block.downsample[1]
            w_ds_sum = conv_ds.weight.sum(dim=(2, 3))
            w_ds_sq_sum = (conv_ds.weight**2).sum(dim=(2, 3))
            m_ds_c = w_ds_sum @ m_x
            v_ds_c = w_ds_sq_sum @ v_x
            stats[prefix + '.downsample.1'] = (m_ds_c.clone(), v_ds_c.clone())
            
            m_ds_out = bn_ds.bias
            v_ds_out = bn_ds.weight**2
        else:
            m_ds_out = m_x
            v_ds_out = v_x
            
        m_sum = m_bn2 + m_ds_out
        v_sum = v_bn2 + v_ds_out
        
        m_out, v_out = relu_propagation(m_sum, v_sum)
        return m_out, v_out

    for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
        layer = getattr(model, layer_name)
        for i, block in enumerate(layer):
            m, v = propagate_basic_block(block, m, v, f"{layer_name}.{i}")
            
    return stats

# --- Sequential Empirical TCAC Calibrator ---
def calibrate_model_empirical_tcac(merged_model, expert_model, dataloader, num_samples=128, device='cuda'):
    merged_model.eval()
    expert_model.eval()
    
    bn_names = []
    for name, module in merged_model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            bn_names.append(name)
            
    for name in bn_names:
        # 1. Collect stats for expert_model at this layer
        expert_module = None
        for n, m in expert_model.named_modules():
            if n == name:
                expert_module = m
                break
                
        expert_stats = []
        def expert_hook_fn(module, input, output):
            x = input[0].detach()
            mean = x.mean(dim=(0, 2, 3))
            var = x.var(dim=(0, 2, 3), unbiased=False)
            expert_stats.append((mean, var, len(x)))
            
        hook_expert = expert_module.register_forward_hook(expert_hook_fn)
        
        samples = 0
        with torch.no_grad():
            for inputs, _ in dataloader:
                inputs = inputs.to(device)
                expert_model(inputs)
                samples += len(inputs)
                if samples >= num_samples:
                    break
        hook_expert.remove()
        
        total_n = sum(n for _, _, n in expert_stats)
        mu_exp = sum(m * (n / total_n) for m, _, n in expert_stats)
        m2_exp = sum((v + m**2) * (n / total_n) for m, v, n in expert_stats)
        var_exp = m2_exp - mu_exp**2
        
        # 2. Collect stats for merged_model at this layer (current state)
        merged_module = None
        for n, m in merged_model.named_modules():
            if n == name:
                merged_module = m
                break
                
        merged_stats = []
        def merged_hook_fn(module, input, output):
            x = input[0].detach()
            mean = x.mean(dim=(0, 2, 3))
            var = x.var(dim=(0, 2, 3), unbiased=False)
            merged_stats.append((mean, var, len(x)))
            
        hook_merged = merged_module.register_forward_hook(merged_hook_fn)
        
        samples = 0
        with torch.no_grad():
            for inputs, _ in dataloader:
                inputs = inputs.to(device)
                merged_model(inputs)
                samples += len(inputs)
                if samples >= num_samples:
                    break
        hook_merged.remove()
        
        total_n = sum(n for _, _, n in merged_stats)
        mu_m = sum(m * (n / total_n) for m, _, n in merged_stats)
        m2_m = sum((v + m**2) * (n / total_n) for m, v, n in merged_stats)
        var_m = m2_m - mu_m**2
        
        # 3. Calibrate this layer in-place
        with torch.no_grad():
            exp_weight = expert_module.weight
            exp_bias = expert_module.bias
            
            merged_module.running_mean.copy_(mu_m)
            merged_module.running_var.copy_(var_m)
            merged_module.weight.copy_(exp_weight)
            merged_module.bias.copy_(exp_bias)

# --- REPAIR Calibrator ---
def calibrate_model_repair(merged_model, expert_model, dataloader, num_samples=128, device='cuda'):
    # Rescales variance to match expert but does NOT align the mean (REPAIR baseline)
    merged_model.eval()
    expert_model.eval()
    
    bn_names = []
    for name, module in merged_model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            bn_names.append(name)
            
    for name in bn_names:
        expert_module = None
        for n, m in expert_model.named_modules():
            if n == name:
                expert_module = m
                break
                
        merged_module = None
        for n, m in merged_model.named_modules():
            if n == name:
                merged_module = m
                break
                
        # Collect empirical statistics for merged model at this layer
        merged_stats = []
        def merged_hook_fn(module, input, output):
            x = input[0].detach()
            mean = x.mean(dim=(0, 2, 3))
            var = x.var(dim=(0, 2, 3), unbiased=False)
            merged_stats.append((mean, var, len(x)))
            
        hook_merged = merged_module.register_forward_hook(merged_hook_fn)
        
        samples = 0
        with torch.no_grad():
            for inputs, _ in dataloader:
                inputs = inputs.to(device)
                merged_model(inputs)
                samples += len(inputs)
                if samples >= num_samples:
                    break
        hook_merged.remove()
        
        total_n = sum(n for _, _, n in merged_stats)
        mu_m = sum(m * (n / total_n) for m, _, n in merged_stats)
        m2_m = sum((v + m**2) * (n / total_n) for m, v, n in merged_stats)
        var_m = m2_m - mu_m**2
        
        # Calibrate: set running_var to var_m, but keep expert_module's running_mean!
        with torch.no_grad():
            exp_weight = expert_module.weight
            exp_bias = expert_module.bias
            
            merged_module.running_mean.copy_(expert_module.running_mean)
            merged_module.running_var.copy_(var_m)
            merged_module.weight.copy_(exp_weight)
            merged_module.bias.copy_(exp_bias)

# --- Naive ReLU Propagation ---
def relu_propagation_naive(mu, var):
    mean_relu = torch.clamp(mu, min=0.0)
    var_relu = 0.5 * var
    return mean_relu, var_relu

# --- Sequential Analytical AAC Calibrator ---
def calibrate_model_analytical_aac(merged_model, expert_model, device='cuda', use_naive_relu=False):
    merged_model.eval()
    expert_model.eval()
    
    eps = 1e-5
    relu_prop_fn = relu_propagation_naive if use_naive_relu else relu_propagation
    
    # We start with the input statistics (which are identical for both)
    m_exp = torch.zeros(3).to(device)
    v_exp = torch.ones(3).to(device)
    
    m_mrg = torch.zeros(3).to(device)
    v_mrg = torch.ones(3).to(device)
    
    # 1. conv1
    # Expert
    w_sum_exp = expert_model.conv1.weight.sum(dim=(2, 3))
    w_sq_sum_exp = (expert_model.conv1.weight**2).sum(dim=(2, 3))
    m_exp = w_sum_exp @ m_exp
    v_exp = w_sq_sum_exp @ v_exp
    
    # Merged
    w_sum_mrg = merged_model.conv1.weight.sum(dim=(2, 3))
    w_sq_sum_mrg = (merged_model.conv1.weight**2).sum(dim=(2, 3))
    m_mrg = w_sum_mrg @ m_mrg
    v_mrg = w_sq_sum_mrg @ v_mrg
    
    # Calibrate bn1
    with torch.no_grad():
        ratio_v = torch.clamp(v_mrg / torch.clamp(v_exp, min=eps), min=0.1, max=10.0)
        diff_m = m_mrg - m_exp
        
        est_mean = expert_model.bn1.running_mean + diff_m
        est_var = expert_model.bn1.running_var * ratio_v
        
        merged_model.bn1.running_mean.copy_(est_mean)
        merged_model.bn1.running_var.copy_(est_var)
        merged_model.bn1.weight.copy_(expert_model.bn1.weight)
        merged_model.bn1.bias.copy_(expert_model.bn1.bias)
        
        # After BN, output stats of both expert and merged are identical to the BN affine parameters
        m_exp = expert_model.bn1.bias
        v_exp = expert_model.bn1.weight**2
        
        m_mrg = merged_model.bn1.bias
        v_mrg = merged_model.bn1.weight**2
        
    # 3. relu
    m_exp, v_exp = relu_prop_fn(m_exp, v_exp)
    m_mrg, v_mrg = relu_prop_fn(m_mrg, v_mrg)
    
    # 4. maxpool (identity mapping)
    
    # 5. layer1, layer2, layer3, layer4
    def calibrate_basic_block(merged_block, expert_block, m_exp_x, v_exp_x, m_mrg_x, v_mrg_x):
        # conv1
        # Expert
        w1_sum_exp = expert_block.conv1.weight.sum(dim=(2, 3))
        w1_sq_sum_exp = (expert_block.conv1.weight**2).sum(dim=(2, 3))
        m_c1_exp = w1_sum_exp @ m_exp_x
        v_c1_exp = w1_sq_sum_exp @ v_exp_x
        
        # Merged
        w1_sum_mrg = merged_block.conv1.weight.sum(dim=(2, 3))
        w1_sq_sum_mrg = (merged_block.conv1.weight**2).sum(dim=(2, 3))
        m_c1_mrg = w1_sum_mrg @ m_mrg_x
        v_c1_mrg = w1_sq_sum_mrg @ v_mrg_x
        
        # Calibrate bn1
        with torch.no_grad():
            ratio_v = torch.clamp(v_c1_mrg / torch.clamp(v_c1_exp, min=eps), min=0.1, max=10.0)
            diff_m = m_c1_mrg - m_c1_exp
            
            est_mean = expert_block.bn1.running_mean + diff_m
            est_var = expert_block.bn1.running_var * ratio_v
            
            merged_block.bn1.running_mean.copy_(est_mean)
            merged_block.bn1.running_var.copy_(est_var)
            merged_block.bn1.weight.copy_(expert_block.bn1.weight)
            merged_block.bn1.bias.copy_(expert_block.bn1.bias)
            
            m_bn1_exp = expert_block.bn1.bias
            v_bn1_exp = expert_block.bn1.weight**2
            
            m_bn1_mrg = merged_block.bn1.bias
            v_bn1_mrg = merged_block.bn1.weight**2
            
        # relu1
        m_r1_exp, v_r1_exp = relu_prop_fn(m_bn1_exp, v_bn1_exp)
        m_r1_mrg, v_r1_mrg = relu_prop_fn(m_bn1_mrg, v_bn1_mrg)
        
        # conv2
        # Expert
        w2_sum_exp = expert_block.conv2.weight.sum(dim=(2, 3))
        w2_sq_sum_exp = (expert_block.conv2.weight**2).sum(dim=(2, 3))
        m_c2_exp = w2_sum_exp @ m_r1_exp
        v_c2_exp = w2_sq_sum_exp @ v_r1_exp
        
        # Merged
        w2_sum_mrg = merged_block.conv2.weight.sum(dim=(2, 3))
        w2_sq_sum_mrg = (merged_block.conv2.weight**2).sum(dim=(2, 3))
        m_c2_mrg = w2_sum_mrg @ m_r1_mrg
        v_c2_mrg = w2_sq_sum_mrg @ v_r1_mrg
        
        # Calibrate bn2
        with torch.no_grad():
            ratio_v = torch.clamp(v_c2_mrg / torch.clamp(v_c2_exp, min=eps), min=0.1, max=10.0)
            diff_m = m_c2_mrg - m_c2_exp
            
            est_mean = expert_block.bn2.running_mean + diff_m
            est_var = expert_block.bn2.running_var * ratio_v
            
            merged_block.bn2.running_mean.copy_(est_mean)
            merged_block.bn2.running_var.copy_(est_var)
            merged_block.bn2.weight.copy_(expert_block.bn2.weight)
            merged_block.bn2.bias.copy_(expert_block.bn2.bias)
            
            m_bn2_exp = expert_block.bn2.bias
            v_bn2_exp = expert_block.bn2.weight**2
            
            m_bn2_mrg = merged_block.bn2.bias
            v_bn2_mrg = merged_block.bn2.weight**2
            
        # downsample
        if merged_block.downsample is not None:
            conv_ds_exp = expert_block.downsample[0]
            bn_ds_exp = expert_block.downsample[1]
            
            conv_ds_mrg = merged_block.downsample[0]
            bn_ds_mrg = merged_block.downsample[1]
            
            w_ds_sum_exp = conv_ds_exp.weight.sum(dim=(2, 3))
            w_ds_sq_sum_exp = (conv_ds_exp.weight**2).sum(dim=(2, 3))
            m_ds_c_exp = w_ds_sum_exp @ m_exp_x
            v_ds_c_exp = w_ds_sq_sum_exp @ v_exp_x
            
            w_ds_sum_mrg = conv_ds_mrg.weight.sum(dim=(2, 3))
            w_ds_sq_sum_mrg = (conv_ds_mrg.weight**2).sum(dim=(2, 3))
            m_ds_c_mrg = w_ds_sum_mrg @ m_mrg_x
            v_ds_c_mrg = w_ds_sq_sum_mrg @ v_mrg_x
            
            with torch.no_grad():
                ratio_v_ds = torch.clamp(v_ds_c_mrg / torch.clamp(v_ds_c_exp, min=eps), min=0.1, max=10.0)
                diff_m_ds = m_ds_c_mrg - m_ds_c_exp
                
                est_mean_ds = bn_ds_exp.running_mean + diff_m_ds
                est_var_ds = bn_ds_exp.running_var * ratio_v_ds
                
                bn_ds_mrg.running_mean.copy_(est_mean_ds)
                bn_ds_mrg.running_var.copy_(est_var_ds)
                bn_ds_mrg.weight.copy_(bn_ds_exp.weight)
                bn_ds_mrg.bias.copy_(bn_ds_exp.bias)
                
                m_ds_out_exp = bn_ds_exp.bias
                v_ds_out_exp = bn_ds_exp.weight**2
                
                m_ds_out_mrg = bn_ds_mrg.bias
                v_ds_out_mrg = bn_ds_mrg.weight**2
        else:
            m_ds_out_exp = m_exp_x
            v_ds_out_exp = v_exp_x
            
            m_ds_out_mrg = m_mrg_x
            v_ds_out_mrg = v_mrg_x
            
        m_sum_exp = m_bn2_exp + m_ds_out_exp
        v_sum_exp = v_bn2_exp + v_ds_out_exp
        
        m_sum_mrg = m_bn2_mrg + m_ds_out_mrg
        v_sum_mrg = v_bn2_mrg + v_ds_out_mrg
        
        m_out_exp, v_out_exp = relu_prop_fn(m_sum_exp, v_sum_exp)
        m_out_mrg, v_out_mrg = relu_prop_fn(m_sum_mrg, v_sum_mrg)
        
        return m_out_exp, v_out_exp, m_out_mrg, v_out_mrg

    for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
        merged_layer = getattr(merged_model, layer_name)
        expert_layer = getattr(expert_model, layer_name)
        for i in range(len(merged_layer)):
            m_exp, v_exp, m_mrg, v_mrg = calibrate_basic_block(
                merged_layer[i], expert_layer[i], m_exp, v_exp, m_mrg, v_mrg
            )

# --- Evaluate function ---
def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100.0 * correct / total

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        torch.backends.cudnn.enabled = False
        print("Disabled cuDNN to bypass initialization errors")
    
    # 1. Prepare Datasets & Dataloaders
    print("Preparing datasets...")
    
    # Normalizations
    transform_rgb = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_gray = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Download datasets
    mnist_train_full = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_gray)
    mnist_test_full = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_gray)
    
    fmnist_train_full = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_gray)
    fmnist_test_full = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_gray)
    
    cifar_train_full = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_rgb)
    cifar_test_full = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_rgb)
    
    # Create subsets: 2000 for training, 500 for testing, 128 for calibration
    train_size = 2000
    test_size = 500
    calib_size = 128
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    mnist_train = Subset(mnist_train_full, np.random.choice(len(mnist_train_full), train_size, replace=False))
    mnist_test = Subset(mnist_test_full, np.random.choice(len(mnist_test_full), test_size, replace=False))
    mnist_calib = Subset(mnist_train_full, np.random.choice(len(mnist_train_full), calib_size, replace=False))
    
    fmnist_train = Subset(fmnist_train_full, np.random.choice(len(fmnist_train_full), train_size, replace=False))
    fmnist_test = Subset(fmnist_test_full, np.random.choice(len(fmnist_test_full), test_size, replace=False))
    fmnist_calib = Subset(fmnist_train_full, np.random.choice(len(fmnist_train_full), calib_size, replace=False))
    
    cifar_train = Subset(cifar_train_full, np.random.choice(len(cifar_train_full), train_size, replace=False))
    cifar_test = Subset(cifar_test_full, np.random.choice(len(cifar_test_full), test_size, replace=False))
    cifar_calib = Subset(cifar_train_full, np.random.choice(len(cifar_train_full), calib_size, replace=False))
    
    # Dataloaders
    train_loaders = [
        DataLoader(mnist_train, batch_size=128, shuffle=True),
        DataLoader(fmnist_train, batch_size=128, shuffle=True),
        DataLoader(cifar_train, batch_size=128, shuffle=True)
    ]
    
    test_loaders = [
        DataLoader(mnist_test, batch_size=128, shuffle=False),
        DataLoader(fmnist_test, batch_size=128, shuffle=False),
        DataLoader(cifar_test, batch_size=128, shuffle=False)
    ]
    
    calib_loaders = [
        DataLoader(mnist_calib, batch_size=128, shuffle=False),
        DataLoader(fmnist_calib, batch_size=128, shuffle=False),
        DataLoader(cifar_calib, batch_size=128, shuffle=False)
    ]
    
    task_names = ["MNIST", "FashionMNIST", "CIFAR-10"]
    
    # 2. Train Experts
    expert_paths = [f"expert_{k}.pt" for k in range(3)]
    pretrained_path = "pretrained_base.pt"
    
    # Save the pretrained base
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    torch.save(model.state_dict(), pretrained_path)
    
    expert_states = []
    expert_accuracies = []
    
    for k in range(3):
        path = expert_paths[k]
        if os.path.exists(path):
            print(f"Loading expert {task_names[k]} from {path}...")
            state = torch.load(path, map_location=device)
            expert_states.append(state)
            
            # Evaluate expert accuracy
            model = models.resnet18()
            model.fc = nn.Linear(512, 10)
            model.load_state_dict(state)
            model = model.to(device)
            acc = evaluate(model, test_loaders[k], device)
            expert_accuracies.append(acc)
            print(f"Expert {task_names[k]} accuracy: {acc:.2f}%")
        else:
            print(f"\n--- Training Expert {task_names[k]} ---")
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            model.fc = nn.Linear(512, 10)
            model = model.to(device)
            
            optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
            criterion = nn.CrossEntropyLoss()
            
            for epoch in range(2):
                model.train()
                epoch_loss = 0.0
                for inputs, targets in train_loaders[k]:
                    inputs, targets = inputs.to(device), targets.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                print(f"Epoch {epoch+1}/2, Loss: {epoch_loss/len(train_loaders[k]):.4f}")
                
            acc = evaluate(model, test_loaders[k], device)
            expert_accuracies.append(acc)
            print(f"Expert {task_names[k]} test accuracy: {acc:.2f}%")
            
            # Save state dict
            torch.save(model.state_dict(), path)
            expert_states.append(model.state_dict())
            
    print(f"\nAverage Expert Accuracy: {sum(expert_accuracies)/3:.2f}%")
    
    pretrained_state = torch.load(pretrained_path, map_location=device)
    
    # 3. Model Merging and Evaluations
    print("\n--- Starting Model Merging Evaluations ---")
    
    lambdas = np.linspace(0.1, 1.5, 15)
    ta_accs = {name: [] for name in task_names}
    repair_accs = {name: [] for name in task_names}
    tcac_accs = {name: [] for name in task_names}
    aac_accs = {name: [] for name in task_names}
    aac_norelu_accs = {name: [] for name in task_names}
    
    for lmbda in lambdas:
        print(f"\nEvaluating scaling factor lambda = {lmbda:.2f}")
        
        # Construct merged backbone state dict
        merged_state = {}
        for key in pretrained_state.keys():
            if 'fc' in key:
                continue
            param_0 = pretrained_state[key]
            if not torch.is_tensor(param_0):
                merged_state[key] = param_0
                continue
            sum_tau = torch.zeros_like(param_0)
            for state in expert_states:
                tau = state[key] - param_0
                sum_tau += tau
            merged_state[key] = param_0 + lmbda * sum_tau
            
        # Evaluation
        for k in range(3):
            # Load base ResNet-18
            eval_model = models.resnet18()
            eval_model.fc = nn.Linear(512, 10)
            
            # Prepare state dict: merged backbone + expert k's head & batchnorm running stats
            eval_state = eval_model.state_dict()
            for key in merged_state.keys():
                eval_state[key] = merged_state[key]
            for key in eval_state.keys():
                if 'fc' in key or 'running_mean' in key or 'running_var' in key or 'num_batches_tracked' in key:
                    eval_state[key] = expert_states[k][key]
                    
            eval_model.load_state_dict(eval_state)
            eval_model = eval_model.to(device)
            
            # A. Task Arithmetic (No Calibration)
            ta_acc = evaluate(eval_model, test_loaders[k], device)
            ta_accs[task_names[k]].append(ta_acc)
            
            # B. REPAIR (Empirical Variance Rescaling only, Data-Assisted)
            eval_model_for_repair = models.resnet18()
            eval_model_for_repair.fc = nn.Linear(512, 10)
            eval_model_for_repair.load_state_dict(eval_state)
            eval_model_for_repair = eval_model_for_repair.to(device)
            
            expert_model = models.resnet18()
            expert_model.fc = nn.Linear(512, 10)
            expert_model.load_state_dict(expert_states[k])
            expert_model = expert_model.to(device)
            
            calibrate_model_repair(eval_model_for_repair, expert_model, calib_loaders[k], num_samples=calib_size, device=device)
            repair_acc = evaluate(eval_model_for_repair, test_loaders[k], device)
            repair_accs[task_names[k]].append(repair_acc)
            
            # C. Empirical TCAC (Sequential Calibration, Data-Assisted)
            eval_model_for_calib = models.resnet18()
            eval_model_for_calib.fc = nn.Linear(512, 10)
            eval_model_for_calib.load_state_dict(eval_state)
            eval_model_for_calib = eval_model_for_calib.to(device)
            
            calibrate_model_empirical_tcac(eval_model_for_calib, expert_model, calib_loaders[k], num_samples=calib_size, device=device)
            tcac_acc = evaluate(eval_model_for_calib, test_loaders[k], device)
            tcac_accs[task_names[k]].append(tcac_acc)
            
            # D. Proposed Analytical Activation Calibration (AAC) (Sequential, Data-Free, Ours)
            eval_model_for_aac = models.resnet18()
            eval_model_for_aac.fc = nn.Linear(512, 10)
            eval_model_for_aac.load_state_dict(eval_state)
            eval_model_for_aac = eval_model_for_aac.to(device)
            
            calibrate_model_analytical_aac(eval_model_for_aac, expert_model, device=device, use_naive_relu=False)
            aac_acc = evaluate(eval_model_for_aac, test_loaders[k], device)
            aac_accs[task_names[k]].append(aac_acc)
            
            # E. Analytical AAC Ablation: Naive ReLU (Sequential, Data-Free, Ablation)
            eval_model_for_aac_norelu = models.resnet18()
            eval_model_for_aac_norelu.fc = nn.Linear(512, 10)
            eval_model_for_aac_norelu.load_state_dict(eval_state)
            eval_model_for_aac_norelu = eval_model_for_aac_norelu.to(device)
            
            calibrate_model_analytical_aac(eval_model_for_aac_norelu, expert_model, device=device, use_naive_relu=True)
            aac_norelu_acc = evaluate(eval_model_for_aac_norelu, test_loaders[k], device)
            aac_norelu_accs[task_names[k]].append(aac_norelu_acc)
            
            print(f"Task {task_names[k]} -> TA: {ta_acc:.2f}%, REPAIR: {repair_acc:.2f}%, TCAC: {tcac_acc:.2f}%, AAC: {aac_acc:.2f}%, AAC-NoReLU: {aac_norelu_acc:.2f}%")

    # 4. Compile Results
    print("\n=== Merging Summary (Average Accuracies) ===")
    print(f"{'Lambda':<10}{'TA (Uncal)':<15}{'REPAIR (Emp)':<15}{'TCAC (Emp)':<15}{'AAC (Ours)':<15}{'AAC-NoReLU':<15}")
    for idx, lmbda in enumerate(lambdas):
        ta_avg = sum(ta_accs[name][idx] for name in task_names) / 3
        repair_avg = sum(repair_accs[name][idx] for name in task_names) / 3
        tcac_avg = sum(tcac_accs[name][idx] for name in task_names) / 3
        aac_avg = sum(aac_accs[name][idx] for name in task_names) / 3
        aac_norelu_avg = sum(aac_norelu_accs[name][idx] for name in task_names) / 3
        print(f"{lmbda:<10.2f}{ta_avg:<15.2f}{repair_avg:<15.2f}{tcac_avg:<15.2f}{aac_avg:<15.2f}{aac_norelu_avg:<15.2f}")
        
    # Find peak results
    ta_avgs = [sum(ta_accs[name][idx] for name in task_names) / 3 for idx in range(len(lambdas))]
    repair_avgs = [sum(repair_accs[name][idx] for name in task_names) / 3 for idx in range(len(lambdas))]
    tcac_avgs = [sum(tcac_accs[name][idx] for name in task_names) / 3 for idx in range(len(lambdas))]
    aac_avgs = [sum(aac_accs[name][idx] for name in task_names) / 3 for idx in range(len(lambdas))]
    aac_norelu_avgs = [sum(aac_norelu_accs[name][idx] for name in task_names) / 3 for idx in range(len(lambdas))]
    
    print("\n--- Peak Performance ---")
    print(f"Task Arithmetic (TA) Peak:      {max(ta_avgs):.2f}% at lambda = {lambdas[np.argmax(ta_avgs)]:.2f}")
    print(f"REPAIR (Empirical SD) Peak:      {max(repair_avgs):.2f}% at lambda = {lambdas[np.argmax(repair_avgs)]:.2f}")
    print(f"Empirical TCAC Peak:            {max(tcac_avgs):.2f}% at lambda = {lambdas[np.argmax(tcac_avgs)]:.2f}")
    print(f"Analytical AAC (Ours) Peak:     {max(aac_avgs):.2f}% at lambda = {lambdas[np.argmax(aac_avgs)]:.2f}")
    print(f"Analytical AAC-NoReLU Peak:     {max(aac_norelu_avgs):.2f}% at lambda = {lambdas[np.argmax(aac_norelu_avgs)]:.2f}")
    
    # Save a plot of the results
    plt.figure(figsize=(11, 7))
    plt.plot(lambdas, ta_avgs, 'o-', label='Task Arithmetic (Uncalibrated)', color='red')
    plt.plot(lambdas, repair_avgs, 'v:', label='REPAIR (Empirical SD only, Data-Assisted)', color='purple')
    plt.plot(lambdas, tcac_avgs, 's--', label='Empirical TCAC (Data-Assisted)', color='blue')
    plt.plot(lambdas, aac_avgs, '^-.', label='Analytical AAC (Ours, 100% Data-Free)', color='green')
    plt.plot(lambdas, aac_norelu_avgs, 'x-', label='Analytical AAC-NoReLU (Ablation, 100% Data-Free)', color='orange')
    plt.axhline(y=sum(expert_accuracies)/3, color='black', linestyle=':', label='Individual Experts (Upper Bound)')
    plt.xlabel('Scaling Factor (lambda)')
    plt.ylabel('Average Multi-Task Accuracy (%)')
    plt.title('Comparison of Model Merging and Activation Calibration Methods')
    plt.legend()
    plt.grid(True)
    plt.savefig('merging_comparison.png', dpi=300)
    print("Saved plot to merging_comparison.png")
    
    # Save raw data to npz
    np.savez('merging_results.npz', 
             lambdas=lambdas, 
             ta_avgs=ta_avgs, 
             repair_avgs=repair_avgs,
             tcac_avgs=tcac_avgs, 
             aac_avgs=aac_avgs, 
             aac_norelu_avgs=aac_norelu_avgs,
             expert_accuracies=expert_accuracies)
