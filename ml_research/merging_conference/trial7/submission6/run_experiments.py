import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib.pyplot as plt

# Set reproducibility seeds
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class ResNet18Custom(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)
        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)
        x = self.base_model.avgpool(x)
        feat = torch.flatten(x, 1)
        logits = self.base_model.fc(feat)
        return feat, logits

def get_resnet18_1channel():
    # Load pre-trained ResNet18
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    # Modify conv1 to accept 1 channel instead of 3
    conv1_old = model.conv1
    conv1_new = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # Sum weights along the input channel dimension
    with torch.no_grad():
        conv1_new.weight.copy_(conv1_old.weight.sum(dim=1, keepdim=True))
    model.conv1 = conv1_new
    # Modify fc to output 10 classes
    model.fc = nn.Linear(512, 10)
    return model

def train_expert(dataset_name, model_path, train_loader, epochs=3, device='cuda'):
    print(f"Training expert for {dataset_name}...")
    model = get_resnet18_1channel().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            _, outputs = model(x) if isinstance(model, ResNet18Custom) else (None, model(x))
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * x.size(0)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")
    
    torch.save(model.state_dict(), model_path)
    print(f"Saved checkpoint to {model_path}")
    return model

def project_simplex(v):
    v_sorted, _ = torch.sort(v, descending=True)
    j = torch.arange(1, len(v) + 1, device=v.device)
    cumsum = torch.cumsum(v_sorted, dim=0)
    rho_cond = v_sorted - (cumsum - 1.0) / j > 0
    rho = torch.max(torch.where(rho_cond)[0]) + 1
    theta = (cumsum[rho - 1] - 1.0) / rho
    return torch.clamp(v - theta, min=0.0)

# Fast batch Fisher approximation
def compute_batch_fisher(expert_model, batch_X, device):
    fisher = {name: torch.zeros_like(p) for name, p in expert_model.named_parameters() if p.requires_grad}
    expert_model.eval()
    
    _, logits = expert_model(batch_X)
    pseudo_labels = torch.argmax(logits, dim=-1)
    
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, pseudo_labels)
    expert_model.zero_grad()
    loss.backward()
    
    for name, param in expert_model.named_parameters():
        if param.requires_grad and param.grad is not None:
            fisher[name] = param.grad.data ** 2
            
    return fisher

def get_joint_fisher(expert_models, batch_X, device):
    joint_fisher = {}
    K = len(expert_models)
    for k in range(K):
        fisher_k = compute_batch_fisher(expert_models[k], batch_X, device)
        for name, val in fisher_k.items():
            clean_name = name.replace('base_model.', '')
            tensor_avg = val.mean().item()
            if clean_name not in joint_fisher:
                joint_fisher[clean_name] = 0.0
            joint_fisher[clean_name] += tensor_avg / K
    return joint_fisher

def compute_preconditioned_lrs(joint_fisher, base_lr=1e-3, eps=1e-6, alpha=1.0):
    sensitivities = list(joint_fisher.values())
    mean_sens = sum(sensitivities) / len(sensitivities)
    
    preconditioned_lrs = {}
    for name, sens in joint_fisher.items():
        norm_sens = sens / (mean_sens + 1e-12)
        lr_mult = (norm_sens + eps) ** (-alpha)
        lr_mult = torch.clamp(torch.tensor(lr_mult), 0.01, 10.0).item()
        preconditioned_lrs[name] = base_lr * lr_mult
    return preconditioned_lrs

def main():
    set_seed(42)
    torch.backends.cudnn.enabled = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Running on device:", device)
    
    # 1. Load datasets
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    print("Loading datasets...")
    mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    kmnist_train = torchvision.datasets.KMNIST(root='./data', train=True, download=True, transform=transform)
    kmnist_test = torchvision.datasets.KMNIST(root='./data', train=False, download=True, transform=transform)
    
    fmnist_train = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    fmnist_test = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loaders = {
        'mnist': DataLoader(mnist_train, batch_size=256, shuffle=True),
        'kmnist': DataLoader(kmnist_train, batch_size=256, shuffle=True),
        'fashionmnist': DataLoader(fmnist_train, batch_size=256, shuffle=True)
    }
    
    # 2. Get/Train expert models
    expert_paths = {
        'mnist': 'mnist_expert.pt',
        'kmnist': 'kmnist_expert.pt',
        'fashionmnist': 'fashionmnist_expert.pt'
    }
    
    experts = {}
    for name, path in expert_paths.items():
        model = get_resnet18_1channel()
        if os.path.exists(path):
            print(f"Loading checkpoint for {name} expert from {path}...")
            model.load_state_dict(torch.load(path, map_location=device))
        else:
            model = train_expert(name, path, train_loaders[name], epochs=3, device=device)
        model = ResNet18Custom(model).to(device)
        model.eval()
        experts[name] = model
        
    expert_list = [experts['mnist'], experts['kmnist'], experts['fashionmnist']]
    expert_names = ['mnist', 'kmnist', 'fashionmnist']
    K = len(expert_list)
    
    # 3. Precompute static model parameters and mean features for known tasks
    # Known domains are MNIST (k=0) and KMNIST (k=1)
    # Static model is uniform average of all 3 experts
    print("Precomputing static model...")
    static_model = get_resnet18_1channel()
    static_state_dict = {}
    for name in static_model.state_dict().keys():
        w_sum = sum(experts[e_name].base_model.state_dict()[name].float() for e_name in expert_names)
        static_state_dict[name] = w_sum / K
    static_model.load_state_dict(static_state_dict)
    static_model = ResNet18Custom(static_model).to(device)
    static_model.eval()
    
    # Precompute dataset mean feature vector (mu_k) on calibration sets (500 samples)
    print("Computing domain mean features and class prototypes...")
    calib_size = 500
    calib_subsets = {
        'mnist': Subset(mnist_train, list(range(calib_size))),
        'kmnist': Subset(kmnist_train, list(range(calib_size))),
        'fashionmnist': Subset(fmnist_train, list(range(calib_size)))
    }
    
    mu_static_domain = {}
    for e_name in expert_names:
        loader = DataLoader(calib_subsets[e_name], batch_size=100, shuffle=False)
        feats_list = []
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device)
                feat, _ = static_model(x)
                feats_list.append(feat)
        mu_static_domain[e_name] = torch.cat(feats_list, dim=0).mean(dim=0)
    
    # Precompute offline class prototypes for known tasks in Unified Static Space
    # Class prototypes: P_k = {pi_k,c} for k in {0, 1} (mnist, kmnist), c in {0..9}
    prototypes = {0: {}, 1: {}}
    known_datasets = ['mnist', 'kmnist']
    for k, e_name in enumerate(known_datasets):
        loader = DataLoader(calib_subsets[e_name], batch_size=100, shuffle=False)
        feats_all = []
        labels_all = []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                feat, _ = static_model(x)
                feats_all.append(feat)
                labels_all.append(y.to(device))
        feats_all = torch.cat(feats_all, dim=0)
        labels_all = torch.cat(labels_all, dim=0)
        
        # Center features with its own domain mean extracted from static_model
        z_all = feats_all - mu_static_domain[e_name]
        
        for c in range(10):
            mask = (labels_all == c)
            if mask.sum() > 0:
                pi_c = z_all[mask].mean(dim=0)
                # Normalize to unit length
                pi_c = pi_c / (pi_c.norm(p=2) + 1e-12)
                prototypes[k][c] = pi_c
            else:
                prototypes[k][c] = torch.zeros_like(mu_static_domain[e_name])
                
    # Precompute Source Fisher sensitivities (S-Fisher) on clean calibration data (500 samples each)
    # We take 500 samples total, e.g., 250 MNIST and 250 KMNIST
    print("Computing Source Fisher (S-Fisher)...")
    sf_subset_mnist = Subset(mnist_train, list(range(250)))
    sf_subset_kmnist = Subset(kmnist_train, list(range(250)))
    sf_loader = DataLoader(Subset(mnist_train, list(range(250))), batch_size=250, shuffle=False)
    for x_m, _ in sf_loader:
        sf_x_mnist = x_m
    sf_loader = DataLoader(Subset(kmnist_train, list(range(250))), batch_size=250, shuffle=False)
    for x_k, _ in sf_loader:
        sf_x_kmnist = x_k
    sf_X = torch.cat([sf_x_mnist, sf_x_kmnist], dim=0).to(device)
    s_fisher = get_joint_fisher(expert_list, sf_X, device)
    
    # 4. Construct test-time streams
    # Batch size = 64
    # Sequential Stream: Batches 1-30 MNIST, 31-60 KMNIST, 61-90 FashionMNIST
    # Alternating Stream: Batches 1-30 alternate between MNIST, KMNIST, FashionMNIST
    print("Constructing streams...")
    test_size_per_task = 30 * 64 # 1920 samples each
    mnist_test_subset = Subset(mnist_test, list(range(test_size_per_task)))
    kmnist_test_subset = Subset(kmnist_test, list(range(test_size_per_task)))
    fmnist_test_subset = Subset(fmnist_test, list(range(test_size_per_task)))
    
    mnist_loader = DataLoader(mnist_test_subset, batch_size=64, shuffle=False)
    kmnist_loader = DataLoader(kmnist_test_subset, batch_size=64, shuffle=False)
    fmnist_loader = DataLoader(fmnist_test_subset, batch_size=64, shuffle=False)
    
    mnist_batches = [b for b in mnist_loader]
    kmnist_batches = [b for b in kmnist_loader]
    fmnist_batches = [b for b in fmnist_loader]
    
    # Sequential Stream
    seq_stream_clean = mnist_batches[:30] + kmnist_batches[:30] + fmnist_batches[:30]
    # Keep track of ground truth domains: 0=MNIST, 1=KMNIST, 2=FashionMNIST
    seq_domains = [0]*30 + [1]*30 + [2]*30
    
    # Alternating Stream
    alt_stream_clean = []
    alt_domains = []
    for i in range(30):
        alt_stream_clean.append(mnist_batches[i])
        alt_domains.append(0)
        alt_stream_clean.append(kmnist_batches[i])
        alt_domains.append(1)
        alt_stream_clean.append(fmnist_batches[i])
        alt_domains.append(2)
        
    # Corruptions helper
    def apply_noise(batch_X):
        return torch.clamp(batch_X + torch.randn_like(batch_X) * 0.2, -1.0, 1.0)
        
    def apply_contrast(batch_X):
        # Scale contrast (around mean 0, since normalization is applied, or scale pixel values)
        # Assuming normalized [-1, 1], let's scale by 0.3
        return torch.clamp(batch_X * 0.3, -1.0, 1.0)
        
    # Prepare streams dictionary
    streams = {
        'Sequential_Clean': (seq_stream_clean, seq_domains),
        'Sequential_Noise': ([(apply_noise(x), y) for x, y in seq_stream_clean], seq_domains),
        'Sequential_Contrast': ([(apply_contrast(x), y) for x, y in seq_stream_clean], seq_domains),
        'Alternating_Clean': (alt_stream_clean, alt_domains),
        'Alternating_Noise': ([(apply_noise(x), y) for x, y in alt_stream_clean], alt_domains),
        'Alternating_Contrast': ([(apply_contrast(x), y) for x, y in alt_stream_clean], alt_domains)
    }
    
    # Parameters to optimize
    # We wrap everything function-style. To do functional_call, we need base_model parameters and names
    base_model = get_resnet18_1channel().to(device)
    parameter_names = [name for name, p in base_model.named_parameters() if p.requires_grad]
    buffer_names = [name for name, b in base_model.named_buffers()]
    
    expert_params = []
    expert_buffers = []
    for k in range(K):
        expert_params.append({name: p.clone().detach() for name, p in expert_list[k].base_model.named_parameters()})
        expert_buffers.append({name: b.clone().detach() for name, b in expert_list[k].base_model.named_buffers()})
        
    base_buffers_dict = {name: b for name, b in base_model.named_buffers()}
    
    # Evaluator function
    def evaluate_framework(stream, domains, method_name, alpha_ema=0.1, tau_N=0.65, gamma_fisher=0.1, alpha_damping=1.0):
        # Initialize coefficients for each layer
        # Lambda_w is shape (3,)
        coefficients = {name: torch.tensor([1/3, 1/3, 1/3], device=device, requires_grad=True) for name in parameter_names}
        
        # Track accuracy, NDR, FPR
        correct_total = 0
        samples_total = 0
        novel_detected = 0
        novel_actual = 0
        known_detected_novel = 0
        known_actual = 0
        
        # History of coefficients over batches for plotting
        coeff_history = []
        
        # For D-TT-Fisher, we maintain a running EMA of the test-time Fisher sensitivities
        # Initialize running fisher with S-Fisher
        running_fisher = {name: torch.tensor(s_fisher[name], device=device) for name in parameter_names}
        
        # To simulate "TT-Fisher" (computed once on calibration window), we compute it on the first 15 batches and freeze it
        tt_fisher_frozen = None
        
        for t, (batch_X, batch_y) in enumerate(stream):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            true_dom = domains[t]
            is_novel_actual = (true_dom == 2) # FashionMNIST is the novel domain
            
            # --- Step 1: Anchor Pass and Unbiased Routing ---
            with torch.no_grad():
                anchor_feats, _ = static_model(batch_X)
            
            # Compute cohesion score for known experts (0=MNIST, 1=KMNIST) by centering with each corresponding mean
            cohesions = []
            for k in range(2):
                e_name = known_datasets[k]
                if method_name == 'D-TT-Fisher':
                    z_anchor_k = anchor_feats - anchor_feats.mean(dim=0)
                else:
                    z_anchor_k = anchor_feats - mu_static_domain[e_name]
                # Cosine similarity to class prototypes
                max_sims = []
                for i in range(batch_X.size(0)):
                    z_i = z_anchor_k[i]
                    z_i_norm = z_i / (z_i.norm(p=2) + 1e-12)
                    max_sim = max(torch.dot(z_i_norm, prototypes[k][c]) for c in range(10))
                    max_sims.append(max_sim)
                cohesion_k = sum(max_sims) / len(max_sims)
                cohesions.append(cohesion_k.item())
                
            max_cohesion = max(cohesions)
            local_tau_N = 0.51 if method_name == 'D-TT-Fisher' else tau_N
            is_novel_pred = (max_cohesion < local_tau_N)
            
            # Record routing metrics
            if is_novel_actual:
                novel_actual += 1
                if is_novel_pred:
                    novel_detected += 1
            else:
                known_actual += 1
                if is_novel_pred:
                    known_detected_novel += 1
            
            # Compute predictive entropy of individual experts on batch
            entropies = []
            with torch.no_grad():
                for k in range(K):
                    _, logits_k = expert_list[k](batch_X)
                    probs_k = torch.softmax(logits_k, dim=-1)
                    ent_k = -torch.mean(torch.sum(probs_k * torch.log(probs_k + 1e-12), dim=-1)).item()
                    entropies.append(ent_k)
            
            # --- Step 2: Determine Target Routing & Perform Update ---
            if method_name in ['DR-Fisher', 'D-TT-Fisher']:
                # 1. EBER: Route batch to active expert with lowest predictive entropy among all K experts
                k_star = np.argmin(entropies)
                
                # 2. Reset coefficients to Lambda_prior
                Lambda_prior = torch.tensor([0.005, 0.005, 0.005], device=device)
                Lambda_prior[k_star] = 0.99
                
                with torch.no_grad():
                    for name in parameter_names:
                        coefficients[name].copy_(Lambda_prior)
                
                # Enable gradient tracking on coefficients
                for name in parameter_names:
                    coefficients[name].requires_grad_(True)
                    if coefficients[name].grad is not None:
                        coefficients[name].grad.zero_()
                
                # 3. Differentiably merge weights
                params_dict = {}
                for name in parameter_names:
                    coeff = coefficients[name]
                    params_dict[name] = (
                        coeff[0] * expert_params[0][name] +
                        coeff[1] * expert_params[1][name] +
                        coeff[2] * expert_params[2][name]
                    )
                
                # 4. Unsupervised forward pass & compute prediction entropy loss (Lent)
                logits = torch.func.functional_call(base_model, params_dict, (batch_X,))
                probs = torch.softmax(logits, dim=-1)
                loss = -torch.mean(torch.sum(probs * torch.log(probs + 1e-12), dim=-1))
                
                # 5. Backpropagate
                loss.backward()
                
                # 6. Determine preconditioning learning rates and perform Riemannian update
                if method_name == 'DR-Fisher':
                    if t < 15:
                        batch_fisher = get_joint_fisher(expert_list, batch_X, device)
                        if tt_fisher_frozen is None:
                            tt_fisher_frozen = {name: batch_fisher[name] for name in parameter_names}
                        else:
                            for name in parameter_names:
                                tt_fisher_frozen[name] = (tt_fisher_frozen[name] * t + batch_fisher[name]) / (t + 1)
                    
                    lrs = compute_preconditioned_lrs(tt_fisher_frozen if tt_fisher_frozen is not None else s_fisher, base_lr=1e-3, alpha=alpha_damping)
                else: # D-TT-Fisher (Ours)
                    batch_fisher = get_joint_fisher(expert_list, batch_X, device)
                    for name in parameter_names:
                        running_fisher[name] = (1 - gamma_fisher) * running_fisher[name] + gamma_fisher * batch_fisher[name]
                    
                    lrs = compute_preconditioned_lrs({name: val.item() for name, val in running_fisher.items()}, base_lr=1e-3, alpha=alpha_damping)
                
                with torch.no_grad():
                    for name in parameter_names:
                        grad = coefficients[name].grad
                        if grad is not None:
                            updated = coefficients[name] - lrs[name] * grad
                            coefficients[name].copy_(project_simplex(updated))
                            
            else:
                # Other methods (Static, Closed-World, PROTO-TTMM, IGGS-OW)
                if not is_novel_pred:
                    k_star = np.argmax(cohesions)
                else:
                    k_star = np.argmin(entropies)
                
                Y_t = torch.zeros(K, device=device)
                Y_t[k_star] = 1.0
                
                if method_name == 'Static':
                    pass
                elif method_name == 'Closed-World':
                    with torch.no_grad():
                        for name in parameter_names:
                            coefficients[name].copy_((1 - alpha_ema) * coefficients[name] + alpha_ema * Y_t)
                elif method_name == 'PROTO-TTMM':
                    if not is_novel_pred:
                        # Known domain: update via EMA
                        with torch.no_grad():
                            for name in parameter_names:
                                coefficients[name].copy_((1 - alpha_ema) * coefficients[name] + alpha_ema * Y_t)
                    else:
                        # Novel domain: flat Euclidean gradient step
                        lr = 1e-3
                        with torch.no_grad():
                            for name in parameter_names:
                                grad = coefficients[name] - Y_t
                                updated = coefficients[name] - lr * grad
                                coefficients[name].copy_(project_simplex(updated))
                elif method_name == 'IGGS-OW':
                    if not is_novel_pred:
                        # Known domain: update via EMA
                        with torch.no_grad():
                            for name in parameter_names:
                                coefficients[name].copy_((1 - alpha_ema) * coefficients[name] + alpha_ema * Y_t)
                    else:
                        # Novel domain: Fisher-preconditioned Riemannian update step
                        lrs = compute_preconditioned_lrs(s_fisher, base_lr=1e-3, alpha=alpha_damping)
                        with torch.no_grad():
                            for name in parameter_names:
                                grad = coefficients[name] - Y_t
                                updated = coefficients[name] - lrs[name] * grad
                                coefficients[name].copy_(project_simplex(updated))
                        
            # --- Step 4: Differentiable Weight and BN Buffer Merging ---
            # Compute average coefficients across all layers
            with torch.no_grad():
                coeff_sum = torch.zeros(K, device=device)
                for name in parameter_names:
                    coeff_sum += coefficients[name]
                coeff_avg = coeff_sum / len(parameter_names)
                
                # Merge Batch Normalization running statistics (buffers)
                for name in buffer_names:
                    b_val = 0.0
                    for k in range(K):
                        b_val += coeff_avg[k].item() * expert_buffers[k][name]
                    base_buffers_dict[name].copy_(b_val)
                    
            # Compute merged parameters in params_dict
            params_dict = {}
            for name in parameter_names:
                coeff = coefficients[name]
                params_dict[name] = coeff[0] * expert_params[0][name] + coeff[1] * expert_params[1][name] + coeff[2] * expert_params[2][name]
                
            # --- Step 5: Evaluate Merged Model on the Batch ---
            with torch.no_grad():
                # Differentiable forward pass
                logits = torch.func.functional_call(base_model, params_dict, (batch_X,))
                _, preds = logits.max(1)
                correct_total += preds.eq(batch_y).sum().item()
                samples_total += batch_X.size(0)
                
            # Log coefficient state for tracking
            coeff_history.append(coeff_avg.cpu().numpy())
            
        # Compute final metrics
        accuracy = (correct_total / samples_total) * 100.0
        ndr = (novel_detected / novel_actual) * 100.0 if novel_actual > 0 else 100.0
        fpr = (known_detected_novel / known_actual) * 100.0 if known_actual > 0 else 0.0
        
        return accuracy, ndr, fpr, np.array(coeff_history)

    # 5. Execute all experiments across streams and methods
    methods = ['Static', 'Closed-World', 'PROTO-TTMM', 'IGGS-OW', 'DR-Fisher', 'D-TT-Fisher']
    
    results = {}
    history = {}
    for stream_name, (stream, domains) in streams.items():
        print(f"\nEvaluating stream: {stream_name}...")
        results[stream_name] = {}
        history[stream_name] = {}
        for method in methods:
            acc, ndr, fpr, coeff_hist = evaluate_framework(stream, domains, method)
            results[stream_name][method] = {'Acc': acc, 'NDR': ndr, 'FPR': fpr}
            history[stream_name][method] = coeff_hist
            print(f"[{method}] Acc: {acc:.2f}%, NDR: {ndr:.2f}%, FPR: {fpr:.2f}%")
            
    # Save results to a text report
    with open("results_report.txt", "w") as f:
        f.write("EXPERIMENT RESULTS REPORT\n")
        f.write("=========================\n\n")
        for stream_name, res in results.items():
            f.write(f"Stream: {stream_name}\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'Method':<15} | {'Acc (%)':<10} | {'NDR (%)':<10} | {'FPR (%)':<10}\n")
            f.write("-" * 50 + "\n")
            for method in methods:
                metrics = res[method]
                f.write(f"{method:<15} | {metrics['Acc']:<10.2f} | {metrics['NDR']:<10.2f} | {metrics['FPR']:<10.2f}\n")
            f.write("\n")
            
    print("\nResults report written to results_report.txt.")
    
    # 6. Generate Figures
    print("Generating figures...")
    
    # Figure 1: Accuracy comparison across all methods on clean/noisy streams
    # (Bar chart with groups for Sequential Clean, Noise, Contrast)
    labels = ['Clean', 'Noise', 'Contrast']
    x = np.arange(len(labels))
    width = 0.12
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, method in enumerate(methods):
        accs = [
            results['Sequential_Clean'][method]['Acc'],
            results['Sequential_Noise'][method]['Acc'],
            results['Sequential_Contrast'][method]['Acc']
        ]
        ax.bar(x + (i - len(methods)/2)*width + width/2, accs, width, label=method)
        
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Test-Time Model Merging Accuracy Comparison (Sequential Streams)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='lower left')
    ax.set_ylim(40, 100)
    plt.tight_layout()
    plt.savefig('fig_accuracy_sequential.png')
    plt.close()
    
    # Figure 2: Coefficient evolution for D-TT-Fisher vs. DR-Fisher on Sequential Clean
    # Show how the average merging coefficient changes over 90 batches
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # DR-Fisher
    dr_hist = history['Sequential_Clean']['DR-Fisher']
    ax1.plot(dr_hist[:, 0], label='MNIST Expert', color='blue')
    ax1.plot(dr_hist[:, 1], label='KMNIST Expert', color='green')
    ax1.plot(dr_hist[:, 2], label='FashionMNIST Expert', color='red')
    ax1.set_title('Coefficient Evolution: DR-Fisher (Frozen TT-Fisher)')
    ax1.set_xlabel('Batch')
    ax1.set_ylabel('Coefficient value')
    ax1.legend()
    ax1.grid(True)
    
    # D-TT-Fisher (Ours)
    dtt_hist = history['Sequential_Clean']['D-TT-Fisher']
    ax2.plot(dtt_hist[:, 0], label='MNIST Expert', color='blue')
    ax2.plot(dtt_hist[:, 1], label='KMNIST Expert', color='green')
    ax2.plot(dtt_hist[:, 2], label='FashionMNIST Expert', color='red')
    ax2.set_title('Coefficient Evolution: D-TT-Fisher (Dynamic EMA, Ours)')
    ax2.set_xlabel('Batch')
    ax2.set_ylabel('Coefficient value')
    ax2.legend()
    ax2.grid(True)
    
    plt.suptitle('Merging Coefficients Dynamic Evolution over Sequential Stream')
    plt.tight_layout()
    plt.savefig('fig_coefficient_evolution.png')
    plt.close()
    
    print("Figures generated and saved successfully!")

if __name__ == '__main__':
    main()
