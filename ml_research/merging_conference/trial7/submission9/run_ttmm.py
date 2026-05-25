import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import torchvision.models as models
import numpy as np

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cudnn.enabled = False
print(f"Using device: {device}")

# Helper: Simplex projection
def project_simplex(v):
    """
    Project vector v onto the unit simplex (sum to 1, all elements >= 0).
    v: tensor of shape (..., K)
    """
    shape = v.shape
    K = shape[-1]
    v_flat = v.view(-1, K)
    
    s, _ = torch.sort(v_flat, dim=-1, descending=True)
    css = torch.cumsum(s, dim=-1)
    
    indices = torch.arange(1, K + 1, device=v.device, dtype=v.dtype)
    cond = s - (css - 1.0) / indices > 0.0
    
    rho = (torch.max(indices * cond, dim=-1).values - 1).long()
    theta = (css[torch.arange(css.size(0), device=v.device), rho] - 1.0) / (rho + 1)
    
    w = torch.clamp(v_flat - theta.unsqueeze(-1), min=0.0)
    return w.view(shape)

# Helper: Load modified ResNet18 structure
def get_modified_resnet18():
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    
    conv1 = model.conv1
    new_conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=conv1.out_channels,
        kernel_size=conv1.kernel_size,
        stride=conv1.stride,
        padding=conv1.padding,
        bias=(conv1.bias is not None)
    )
    with torch.no_grad():
        new_conv1.weight.copy_(conv1.weight.sum(dim=1, keepdim=True))
        if conv1.bias is not None:
            new_conv1.bias.copy_(conv1.bias)
    model.conv1 = new_conv1
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model.to(device)

# Load datasets and create a continuous test stream
def get_dataset(dataset_name, train=False):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    if dataset_name == "mnist":
        return datasets.MNIST(root="./data", train=train, download=True, transform=transform)
    elif dataset_name == "kmnist":
        return datasets.KMNIST(root="./data", train=train, download=True, transform=transform)
    elif dataset_name == "fashionmnist":
        return datasets.FashionMNIST(root="./data", train=train, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def construct_test_stream(batch_size=64, num_batches_per_task=30):
    print("Constructing continuous test stream...")
    mnist_test = get_dataset("mnist", train=False)
    kmnist_test = get_dataset("kmnist", train=False)
    fmnist_test = get_dataset("fashionmnist", train=False)
    
    # We want num_batches_per_task of each dataset
    mnist_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)
    kmnist_loader = DataLoader(kmnist_test, batch_size=batch_size, shuffle=False)
    fmnist_loader = DataLoader(fmnist_test, batch_size=batch_size, shuffle=False)
    
    stream_batches = []
    task_labels = [] # To track which task each batch belongs to (0: mnist, 1: kmnist, 2: fashionmnist)
    
    # Collect batches
    for i, (imgs, lbls) in enumerate(mnist_loader):
        if i >= num_batches_per_task: break
        stream_batches.append((imgs, lbls))
        task_labels.append(0)
        
    for i, (imgs, lbls) in enumerate(kmnist_loader):
        if i >= num_batches_per_task: break
        stream_batches.append((imgs, lbls))
        task_labels.append(1)
        
    for i, (imgs, lbls) in enumerate(fmnist_loader):
        if i >= num_batches_per_task: break
        stream_batches.append((imgs, lbls))
        task_labels.append(2)
        
    print(f"Total batches in test stream: {len(stream_batches)} (each batch size {batch_size})")
    return stream_batches, task_labels

# Weight and BN stats merging
def merge_experts(model, expert_state_dicts, lambdas):
    """
    model: standard ResNet18 model
    expert_state_dicts: list of 3 state dicts [mnist, kmnist, fashionmnist]
    lambdas: dict mapping param name to tensor of shape (3,) [mnist, kmnist, fashionmnist]
    """
    merged_state_dict = {}
    keys = expert_state_dicts[0].keys()
    for key in keys:
        if "num_batches_tracked" in key:
            merged_state_dict[key] = expert_state_dicts[0][key].clone()
            continue
            
        if key in lambdas:
            l = lambdas[key] # tensor of shape (3,) on device
            merged_param = torch.zeros_like(expert_state_dicts[0][key])
            for k, state_dict in enumerate(expert_state_dicts):
                merged_param += l[k] * state_dict[key].to(l.device)
            merged_state_dict[key] = merged_param
        elif "running_mean" in key or "running_var" in key:
            # Merge BN running buffers using average lambdas across all parameters
            device = lambdas[list(lambdas.keys())[0]].device
            avg_lambda = torch.stack(list(lambdas.values())).mean(dim=0).to(device)
            merged_buffer = torch.zeros_like(expert_state_dicts[0][key])
            for k, state_dict in enumerate(expert_state_dicts):
                merged_buffer += avg_lambda[k] * state_dict[key].to(avg_lambda.device)
            merged_state_dict[key] = merged_buffer
        else:
            merged_state_dict[key] = expert_state_dicts[0][key].clone()
            
    model.load_state_dict(merged_state_dict)

# KT-Fisher Tracker
class KTFisherTracker:
    def __init__(self, model):
        self.model = model
        self.activation_norms = {}
        self.gradient_norms = {}
        self.hooks = []
        self._register_hooks()
        
    def _register_hooks(self):
        for name, module in self.model.named_modules():
            # We precondition both convolution and linear layers (ignoring downsamples and short-cuts for simplicity, or preconditioning everything)
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                self.hooks.append(module.register_forward_hook(self._make_forward_hook(name)))
                self.hooks.append(module.register_full_backward_hook(self._make_backward_hook(name)))
                
    def _make_forward_hook(self, name):
        def forward_hook(module, input, output):
            act = input[0].detach()
            self.activation_norms[name] = (act ** 2).sum().item() / act.size(0)
        return forward_hook
        
    def _make_backward_hook(self, name):
        def backward_hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                grad = grad_output[0].detach()
                self.gradient_norms[name] = (grad ** 2).sum().item() / grad.size(0)
        return backward_hook
        
    def compute_sensitivities(self):
        sensitivities = {}
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # weight name is name + ".weight"
                weight_name = name + ".weight"
                num_params = module.weight.numel()
                act_norm = self.activation_norms.get(name, 0.0)
                grad_norm = self.gradient_norms.get(name, 0.0)
                # Avoid division by zero
                sensitivity = (act_norm * grad_norm) / (num_params + 1e-8)
                sensitivities[weight_name] = sensitivity
                if hasattr(module, "bias") and module.bias is not None:
                    bias_name = name + ".bias"
                    sensitivities[bias_name] = sensitivity # Share same sensitivity factor
        return sensitivities
        
    def close(self):
        for hook in self.hooks:
            hook.remove()

# Main evaluation loop
def evaluate_method(method_name, stream_batches, task_labels, expert_state_dicts, source_fisher=None, damping_factor=0.5, lr_eta=0.005, threshold_N=0.58, precondition_interval=1):
    print(f"\nEvaluating Method: {method_name.upper()}...")
    
    # Initialize model and coefficients
    model = get_modified_resnet18()
    
    # Known experts: MNIST (index 0) and KMNIST (index 1). Novel expert: FashionMNIST (index 2).
    # Initialize coefficients lambdas for each parameter key
    lambdas = {}
    for name, param in model.named_parameters():
        # [mnist, kmnist, fashionmnist]
        lambdas[name] = torch.tensor([0.5, 0.5, 0.0], device=device)
        
    # Set up static uniformly merged model for feature/prototype extraction
    # Using all 3 experts for the static anchor model
    anchor_model = get_modified_resnet18()
    merge_experts(anchor_model, expert_state_dicts, {name: torch.tensor([1/3, 1/3, 1/3], device=device) for name, _ in anchor_model.named_parameters()})
    anchor_model.eval()
    
    # Extract feature dimension and class prototypes
    # To do this, we pass a small validation set of known classes
    print("Pre-computing offline prototypes in Unified Static Space...")
    mnist_cal = get_dataset("mnist", train=True)
    kmnist_cal = get_dataset("kmnist", train=True)
    
    # Extract class-wise features using anchor model
    def get_prototypes(dataset, num_samples=500):
        loader = DataLoader(dataset, batch_size=64, shuffle=False)
        features = []
        labels = []
        
        # We hook the feature extractor (before fc)
        feat_list = []
        def hook_fn(module, input, output):
            feat_list.append(output.detach())
            
        hook = anchor_model.avgpool.register_forward_hook(hook_fn)
        
        count = 0
        with torch.no_grad():
            for imgs, lbls in loader:
                imgs = imgs.to(device)
                anchor_model(imgs)
                labels.append(lbls)
                count += imgs.size(0)
                if count >= num_samples:
                    break
                    
        hook.remove()
        
        feats = torch.cat(feat_list, dim=0)[:num_samples].squeeze(-1).squeeze(-1) # (500, 512)
        lbls = torch.cat(labels, dim=0)[:num_samples].to(device)
        
        # Mean centroid
        mu = feats.mean(dim=0)
        
        # Centered class prototypes
        centered_feats = feats - mu
        class_prototypes = {}
        for c in range(10):
            mask = (lbls == c)
            if mask.sum() > 0:
                class_prototypes[c] = centered_feats[mask].mean(dim=0)
            else:
                class_prototypes[c] = torch.zeros_like(mu)
                
        return mu, class_prototypes

    mu_mnist, proto_mnist = get_prototypes(mnist_cal)
    mu_kmnist, proto_kmnist = get_prototypes(kmnist_cal)
    
    # Unified Static Space mean
    mu_static = (mu_mnist + mu_kmnist) / 2.0
    
    # Pre-computed class prototypes for each expert
    # proto_mnist: dict of class -> 512-dim vector
    # proto_kmnist: dict of class -> 512-dim vector
    
    # Evaluate continuous test stream
    task_accuracies = {0: [], 1: [], 2: []}
    overall_correct = 0
    overall_total = 0
    
    # Novelty routing metrics
    novel_detected_correctly = 0 # Batch is novel and detected as novel (NDR)
    false_positives = 0 # Batch is known but detected as novel (FPR)
    total_known_batches = 0
    total_novel_batches = 0
    
    threshold_N = threshold_N # Novelty detection threshold (cohesion < threshold_N)
    ema_alpha = 0.1 # Coefficient EMA update rate on known tasks
    lr_eta = lr_eta # Learning rate for novel task adaptation
    epsilon_scale = 1e-5
    # Use passed damping_factor parameter
    damping_factor = damping_factor # Damping exponent beta
    
    batch_times = []
    
    # Instantiate KT-Fisher Tracker on active model
    kt_tracker = KTFisherTracker(model)
    
    sensitivities_cache = None
    novel_batch_count = 0
    
    for t, (images, labels) in enumerate(stream_batches):
        images, labels = images.to(device), labels.to(device)
        active_task = task_labels[t]
        
        t_start = time.time()
        
        # Step 1: Extract test features using anchor model to compute cohesion scores
        feat_list_test = []
        def hook_fn_test(module, input, output):
            feat_list_test.append(output.detach())
        hook_t = anchor_model.avgpool.register_forward_hook(hook_fn_test)
        
        with torch.no_grad():
            anchor_model(images)
        hook_t.remove()
        
        # Center test features
        feats_test = torch.cat(feat_list_test, dim=0).squeeze(-1).squeeze(-1) # (B, 512)
        z_anchor = feats_test - mu_static # (B, 512)
        
        # Compute cohesion scores to known experts (0: MNIST, 1: KMNIST)
        def compute_cohesion(z, prototypes):
            # z: (B, 512), prototypes: dict of 10 class vectors
            # sim(z_i, pi_c) = cosine_similarity
            cohesion_sum = 0.0
            for i in range(z.size(0)):
                max_sim = -1.0
                zi = z[i]
                for c, proto in prototypes.items():
                    sim = torch.dot(zi, proto) / (torch.norm(zi) * torch.norm(proto) + 1e-8)
                    if sim.item() > max_sim:
                        max_sim = sim.item()
                cohesion_sum += max_sim
            return cohesion_sum / z.size(0)
            
        c_mnist = compute_cohesion(z_anchor, proto_mnist)
        c_kmnist = compute_cohesion(z_anchor, proto_kmnist)
        max_cohesion = max(c_mnist, c_kmnist)
        
        # Novelty detection decision
        is_novel = (max_cohesion < threshold_N)
        
        if active_task == 2:
            total_novel_batches += 1
            if is_novel:
                novel_detected_correctly += 1
        else:
            total_known_batches += 1
            if is_novel:
                false_positives += 1
                
        # Merge experts and evaluate accuracy BEFORE adaptation on this batch
        merge_experts(model, expert_state_dicts, lambdas)
        model.eval()
        
        # Evaluate accuracy on current batch
        with torch.no_grad():
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct_cnt = predicted.eq(labels).sum().item()
            acc = 100.0 * correct_cnt / images.size(0)
            task_accuracies[active_task].append(acc)
            overall_correct += correct_cnt
            overall_total += images.size(0)
            
        # Update coefficients lambdas
        if method_name == "static":
            # No update
            pass
            
        elif not is_novel:
            # Route to best known expert and update via EMA
            k_star = 0 if c_mnist >= c_kmnist else 1
            # Update lambdas
            for name in lambdas:
                target = torch.zeros(3, device=device)
                target[k_star] = 1.0
                lambdas[name] = (1.0 - ema_alpha) * lambdas[name] + ema_alpha * target
                
        else:
            # Batch is flagged as novel. Perform active adaptation.
            # 1. Determine target expert via entropy on the batch
            # Evaluate predictive entropy of each expert model
            entropies = []
            for k in range(3):
                # Temporary load expert weights to model to evaluate its entropy
                temp_lambdas = {name: torch.zeros(3, device=device) for name in lambdas}
                for name in temp_lambdas:
                    temp_lambdas[name][k] = 1.0
                merge_experts(model, expert_state_dicts, temp_lambdas)
                model.eval()
                
                with torch.no_grad():
                    outputs = model(images)
                    probs = torch.softmax(outputs, dim=-1)
                    ent = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean().item()
                entropies.append(ent)
                
            k_star = np.argmin(entropies) # The expert with the lowest predictive entropy (which should target FashionMNIST!)
            target = torch.zeros(3, device=device)
            target[k_star] = 1.0
            
            # Restore model merged state
            merge_experts(model, expert_state_dicts, lambdas)
            
            if method_name == "proto_ttmm":
                # Uniform learning rate, no preconditioning
                # Optimize lambdas to minimize entropy
                # In standard PROTO-TTMM, update direction is directly towards the targeted expert Y_t
                for name in lambdas:
                    grad = lambdas[name] - target # Update direction
                    lambdas[name] = project_simplex(lambdas[name] - lr_eta * grad)
                    
            elif method_name in ["iggs_ow", "fp_ow"]:
                # Preconditioned learning rate using pre-computed source Fisher
                for name in lambdas:
                    f_w = source_fisher.get(name, 1.0)
                    eta_w = lr_eta * ((f_w + epsilon_scale) ** (-damping_factor))
                    grad = lambdas[name] - target
                    lambdas[name] = project_simplex(lambdas[name] - eta_w * grad)
                    
            elif method_name == "tt_diag_fisher":
                # Test-Time Diagonal Fisher on-the-fly preconditioning!
                model.train() # Enable gradient computation
                # Forward pass
                outputs = model(images)
                probs = torch.softmax(outputs, dim=-1)
                entropy_loss = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()
                
                # Zero out parameter gradients
                model.zero_grad()
                entropy_loss.backward()
                
                # Compute diagonal Fisher as average squared gradient of each parameter
                sensitivities = {}
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            sensitivities[name] = torch.mean(param.grad ** 2).item()
                        else:
                            sensitivities[name] = 0.0
                
                # Apply preconditioned updates to lambdas
                for name in lambdas:
                    f_w = sensitivities.get(name, 1.0)
                    eta_w = lr_eta * ((f_w + epsilon_scale) ** (-damping_factor))
                    grad = lambdas[name] - target
                    lambdas[name] = project_simplex(lambdas[name] - eta_w * grad)
                    
            elif method_name == "kt_fisher":
                # KT-Fisher on-the-fly preconditioning!
                # If precondition_interval > 1, we can reuse the sensitivity cache
                novel_batch_count += 1
                if sensitivities_cache is None or (novel_batch_count - 1) % precondition_interval == 0:
                    model.train() # Enable gradient computation
                    # Forward pass
                    outputs = model(images)
                    probs = torch.softmax(outputs, dim=-1)
                    entropy_loss = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()
                    
                    # Zero out parameter gradients
                    model.zero_grad()
                    entropy_loss.backward()
                    
                    # Compute KT-Fisher sensitivities on-the-fly and cache them
                    sensitivities_cache = kt_tracker.compute_sensitivities()
                
                sensitivities = sensitivities_cache
                
                # Apply preconditioned updates to lambdas
                for name in lambdas:
                    f_w = sensitivities.get(name, 1.0)
                    eta_w = lr_eta * ((f_w + epsilon_scale) ** (-damping_factor))
                    grad = lambdas[name] - target
                    lambdas[name] = project_simplex(lambdas[name] - eta_w * grad)
                    
        t_end = time.time()
        batch_times.append(t_end - t_start)
        
    # Clean up hooks
    kt_tracker.close()
    
    # Compute metrics
    mnist_acc = np.mean(task_accuracies[0]) if len(task_accuracies[0]) > 0 else 0.0
    kmnist_acc = np.mean(task_accuracies[1]) if len(task_accuracies[1]) > 0 else 0.0
    fmnist_acc = np.mean(task_accuracies[2]) if len(task_accuracies[2]) > 0 else 0.0
    overall_acc = 100.0 * overall_correct / overall_total
    
    ndr = 100.0 * novel_detected_correctly / total_novel_batches if total_novel_batches > 0 else 0.0
    fpr = 100.0 * false_positives / total_known_batches if total_known_batches > 0 else 0.0
    avg_batch_time = np.mean(batch_times) * 1000.0 # ms
    
    print(f"MNIST (Known) Acc: {mnist_acc:.2f}%")
    print(f"KMNIST (Known) Acc: {kmnist_acc:.2f}%")
    print(f"FashionMNIST (Novel) Acc: {fmnist_acc:.2f}%")
    print(f"Overall Acc: {overall_acc:.2f}%")
    print(f"Novelty Detection Rate (NDR): {ndr:.2f}%")
    print(f"False Positive Rate (FPR): {fpr:.2f}%")
    print(f"Average Batch Processing Time: {avg_batch_time:.2f} ms")
    
    return {
        "mnist": mnist_acc,
        "kmnist": kmnist_acc,
        "fashionmnist": fmnist_acc,
        "overall": overall_acc,
        "ndr": ndr,
        "fpr": fpr,
        "time": avg_batch_time
    }

if __name__ == "__main__":
    # Check if expert models exist
    if not (os.path.exists("expert_mnist.pth") and os.path.exists("expert_kmnist.pth") and os.path.exists("expert_fashionmnist.pth")):
        print("Expert models not found! Please make sure train_experts.py runs first.")
        exit(1)
        
    # Load expert model weights
    print("Loading expert model weights...")
    expert_mnist = torch.load("expert_mnist.pth", map_location=device)
    expert_kmnist = torch.load("expert_kmnist.pth", map_location=device)
    expert_fashionmnist = torch.load("expert_fashionmnist.pth", map_location=device)
    expert_state_dicts = [expert_mnist, expert_kmnist, expert_fashionmnist]
    
    # Construct test stream
    stream_batches, task_labels = construct_test_stream(batch_size=64, num_batches_per_task=30)
    
    # Pre-compute source Fisher for FP-OW / IGGS-OW baseline
    # S-Fisher is computed on MNIST and KMNIST calibration sets
    print("\nPre-computing Source Fisher sensitivities (FP-OW/IGGS-OW)...")
    model = get_modified_resnet18()
    source_fisher = {}
    for name, param in model.named_parameters():
        source_fisher[name] = 0.0
        
    for k, dataset_name in enumerate(["mnist", "kmnist"]):
        cal_dataset = get_dataset(dataset_name, train=True)
        loader = DataLoader(cal_dataset, batch_size=64, shuffle=False)
        model.load_state_dict(expert_state_dicts[k])
        model.eval()
        
        # Standard diagonal Fisher precomputation
        # We accumulate squared gradients of log probabilities
        count = 0
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            
            # Accumulate squared gradients
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        source_fisher[name] += (param.grad ** 2).sum().item() / 500.0 # Normalized by calibration size
            count += images.size(0)
            if count >= 500:
                break
                
    # Average across the experts
    for name in source_fisher:
        source_fisher[name] = source_fisher[name] / 2.0
        
    # Now evaluate all methods
    methods = ["static", "proto_ttmm", "iggs_ow", "tt_diag_fisher", "kt_fisher"]
    results = {}
    
    for method in methods:
        results[method] = evaluate_method(method, stream_batches, task_labels, expert_state_dicts, source_fisher)
        
    # Print comparison table
    print("\n" + "="*80)
    print(f"{'Method':<15} | {'MNIST':<8} | {'KMNIST':<8} | {'F-MNIST':<8} | {'Overall':<8} | {'NDR (%)':<8} | {'FPR (%)':<8} | {'Time (ms)':<9}")
    print("="*80)
    for m in methods:
        r = results[m]
        print(f"{m:<15} | {r['mnist']:<8.2f} | {r['kmnist']:<8.2f} | {r['fashionmnist']:<8.2f} | {r['overall']:<8.2f} | {r['ndr']:<8.2f} | {r['fpr']:<8.2f} | {r['time']:<9.2f}")
    print("="*80)
    
    # Save results to a file for writing the paper
    import json
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Results saved to results.json")
