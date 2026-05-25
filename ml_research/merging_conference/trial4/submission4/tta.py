import os
import argparse
import copy
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torch.func import functional_call
from torchvision.models import resnet18, ResNet18_Weights

# Set random seed for reproducibility
torch.manual_seed(42)
torch.backends.cudnn.enabled = False
import builtins
print = lambda *args, **kwargs: builtins.print(*args, **kwargs, flush=True)

class ResNetBackbone(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.avgpool = base_model.avgpool

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

def get_dataset(name, train=False):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if name == 'mnist':
        return torchvision.datasets.MNIST(root='./data', train=train, download=False, transform=transform)
    elif name == 'fashion':
        return torchvision.datasets.FashionMNIST(root='./data', train=train, download=False, transform=transform)
    elif name == 'kmnist':
        return torchvision.datasets.KMNIST(root='./data', train=train, download=False, transform=transform)
    else:
        raise ValueError(f"Unknown dataset {name}")

def project_to_simplex(v):
    # Duchi et al. (2008) projection onto the probability simplex
    n_features = v.shape[0]
    u, _ = torch.sort(v, descending=True)
    cssv = torch.cumsum(u, dim=0) - 1.0
    ind = torch.arange(n_features, device=v.device) + 1
    cond = u - cssv / ind > 0
    nonzero_indices = torch.nonzero(cond)
    if len(nonzero_indices) == 0:
        return torch.ones_like(v) / n_features
    rho = nonzero_indices[-1].item()
    theta = cssv[rho] / (rho + 1)
    w = torch.clamp(v - theta, min=0)
    return w

def run_evaluation(
    stream_name, 
    batches, 
    method, 
    base_backbone, 
    base_backbone_params, 
    expert_backbones, 
    expert_heads, 
    task_vectors, 
    fisher_priors, 
    lr_head, 
    lr_lambda, 
    gamma, 
    dts_alpha, 
    dts_beta, 
    device
):
    # Clone classification heads and task-vectors to prevent modification across runs
    adapted_heads = [copy.deepcopy(h).to(device) for h in expert_heads]
    
    # Initialize coefficients
    lambda_val = torch.tensor([1/3, 1/3, 1/3], device=device, requires_grad=True)
    
    # Running entropy for DTS
    dts_ema_entropy = 0.0
    if method == 'tf_ewc_dts':
        # Initialize running entropy with a reasonable estimate
        dts_ema_entropy = 1.0
        
    correct_count = 0
    total_count = 0
    
    # Track coefficients over the stream
    lambda_history = []
    
    for step, ((inputs, targets), task_idx) in enumerate(batches):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Enable grads for head of the active task
        active_head = adapted_heads[task_idx]
        for p in active_head.parameters():
            p.requires_grad = True
            
        if method == 'static':
            # No adaptation: use uniform coefficients
            static_lambda = torch.tensor([1/3, 1/3, 1/3], device=device)
            merged_params = {}
            for k in base_backbone_params.keys():
                merged_params[k] = base_backbone_params[k] + sum(
                    static_lambda[i] * task_vectors[i][k] for i in range(3)
                )
            with torch.no_grad():
                features = functional_call(base_backbone, merged_params, inputs)
                logits = active_head(features)
                _, predicted = logits.max(1)
                correct_count += predicted.eq(targets).sum().item()
                total_count += targets.size(0)
            lambda_history.append(static_lambda.tolist())
            continue
            
        # For adaptive methods, we perform an online optimization step on the current batch
        # before making final predictions (as in Algorithm 1 of EWC-TTA)
        
        # Step 1: Compute loss
        if method == 'unconstrained_tta':
            # Differentiable Virtual Merging
            merged_params = {}
            for k in base_backbone_params.keys():
                merged_params[k] = base_backbone_params[k] + sum(
                    lambda_val[i] * task_vectors[i][k] for i in range(3)
                )
            
            features = functional_call(base_backbone, merged_params, inputs)
            logits = active_head(features)
            
            # Unsupervised Prediction Entropy Loss
            probs = F.softmax(logits, dim=-1)
            loss = -(probs * torch.log(probs + 1e-12)).sum(dim=-1).mean()
            
            # Step 2: Backward pass
            loss.backward()
            
            # Step 3: Gradient Updates
            with torch.no_grad():
                lambda_val.data -= lr_lambda * lambda_val.grad
                lambda_val.data = project_to_simplex(lambda_val.data)
                
                for p in active_head.parameters():
                    if p.grad is not None:
                        p.data -= lr_head * p.grad
                        
            # Zero gradients
            lambda_val.grad = None
            active_head.zero_grad()
            
            # Step 4: Final Inference using updated parameters
            with torch.no_grad():
                merged_params = {}
                for k in base_backbone_params.keys():
                    merged_params[k] = base_backbone_params[k] + sum(
                        lambda_val[i] * task_vectors[i][k] for i in range(3)
                    )
                features = functional_call(base_backbone, merged_params, inputs)
                logits = active_head(features)
                _, predicted = logits.max(1)
                correct_count += predicted.eq(targets).sum().item()
                total_count += targets.size(0)
                
        elif method == 'ewc_tta':
            # Expert-guided test-time adaptation with EWC
            # Get soft predictions from the original expert k (teacher)
            with torch.no_grad():
                expert_features = expert_backbones[task_idx](inputs)
                expert_logits = expert_heads[task_idx](expert_features)
                expert_probs = F.softmax(expert_logits, dim=-1)
                
            # Differentiable Virtual Merging
            merged_params = {}
            for k in base_backbone_params.keys():
                merged_params[k] = base_backbone_params[k] + sum(
                    lambda_val[i] * task_vectors[i][k] for i in range(3)
                )
                
            features = functional_call(base_backbone, merged_params, inputs)
            logits = active_head(features)
            merged_probs = F.softmax(logits, dim=-1)
            
            # KL divergence loss
            loss_kl = (expert_probs * (torch.log(expert_probs + 1e-12) - torch.log(merged_probs + 1e-12))).sum(dim=-1).mean()
            
            # EWC Penalty
            loss_ewc = 0.0
            fim = fisher_priors[task_idx]
            init_head = expert_heads[task_idx]
            for p_name, p in active_head.named_parameters():
                init_p = getattr(init_head, p_name)
                f_weight = fim[p_name]
                loss_ewc += 0.5 * (f_weight * (p - init_p) ** 2).sum()
                
            loss = loss_kl + gamma * loss_ewc
            
            # Backward pass
            loss.backward()
            
            # Gradient Updates
            with torch.no_grad():
                lambda_val.data -= lr_lambda * lambda_val.grad
                lambda_val.data = project_to_simplex(lambda_val.data)
                
                for p in active_head.parameters():
                    if p.grad is not None:
                        p.data -= lr_head * p.grad
                        
            # Zero gradients
            lambda_val.grad = None
            active_head.zero_grad()
            
            # Final Inference
            with torch.no_grad():
                merged_params = {}
                for k in base_backbone_params.keys():
                    merged_params[k] = base_backbone_params[k] + sum(
                        lambda_val[i] * task_vectors[i][k] for i in range(3)
                    )
                features = functional_call(base_backbone, merged_params, inputs)
                logits = active_head(features)
                _, predicted = logits.max(1)
                correct_count += predicted.eq(targets).sum().item()
                total_count += targets.size(0)
                
        elif method == 'tf_ewc_dts':
            # Our proposed Teacher-Free EWC with Dynamic Temperature Scaling
            # Differentiable Virtual Merging
            merged_params = {}
            for k in base_backbone_params.keys():
                merged_params[k] = base_backbone_params[k] + sum(
                    lambda_val[i] * task_vectors[i][k] for i in range(3)
                )
                
            features = functional_call(base_backbone, merged_params, inputs)
            logits = active_head(features)
            
            # 1. Compute standard predictions & batch entropy to update running estimate
            with torch.no_grad():
                probs_std = F.softmax(logits, dim=-1)
                batch_entropy = -(probs_std * torch.log(probs_std + 1e-12)).sum(dim=-1).mean().item()
                
            # Update running entropy EMA
            dts_ema_entropy = dts_alpha * dts_ema_entropy + (1 - dts_alpha) * batch_entropy
            
            # 2. Determine temperature and scale logits
            T = 1.0 + dts_beta * dts_ema_entropy
            probs_scaled = F.softmax(logits / T, dim=-1)
            
            # 3. Compute losses: scaled entropy minimization + head EWC penalty
            loss_ent = -(probs_scaled * torch.log(probs_scaled + 1e-12)).sum(dim=-1).mean()
            
            loss_ewc = 0.0
            fim = fisher_priors[task_idx]
            init_head = expert_heads[task_idx]
            for p_name, p in active_head.named_parameters():
                init_p = getattr(init_head, p_name)
                f_weight = fim[p_name]
                loss_ewc += 0.5 * (f_weight * (p - init_p) ** 2).sum()
                
            loss = loss_ent + gamma * loss_ewc
            
            # Backward pass
            loss.backward()
            
            # Gradient Updates
            with torch.no_grad():
                lambda_val.data -= lr_lambda * lambda_val.grad
                lambda_val.data = project_to_simplex(lambda_val.data)
                
                for p in active_head.parameters():
                    if p.grad is not None:
                        p.data -= lr_head * p.grad
                        
            # Zero gradients
            lambda_val.grad = None
            active_head.zero_grad()
            
            # Final Inference
            with torch.no_grad():
                merged_params = {}
                for k in base_backbone_params.keys():
                    merged_params[k] = base_backbone_params[k] + sum(
                        lambda_val[i] * task_vectors[i][k] for i in range(3)
                    )
                features = functional_call(base_backbone, merged_params, inputs)
                logits = active_head(features)
                _, predicted = logits.max(1)
                correct_count += predicted.eq(targets).sum().item()
                total_count += targets.size(0)
                
        lambda_history.append(lambda_val.tolist())
        
    accuracy = 100.0 * correct_count / total_count
    return accuracy, lambda_history

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr_head', type=float, default=1e-4)
    parser.add_argument('--lr_lambda_sweep', type=str, default='0.05,0.1,0.2,0.5')
    parser.add_argument('--gamma_sweep', type=str, default='10.0,100.0')
    parser.add_argument('--dts_alpha', type=float, default=0.9)
    parser.add_argument('--dts_beta', type=float, default=1.0)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_known_args()[0]
    
    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print("Using device for evaluation:", device)
    
    lr_lambdas = [float(x) for x in args.lr_lambda_sweep.split(',')]
    gammas = [float(x) for x in args.gamma_sweep.split(',')]
    
    # 1. Load pre-trained base backbone
    base_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    base_backbone = ResNetBackbone(base_model).to(device)
    base_backbone_params = {k: v.clone().detach() for k, v in base_backbone.named_parameters()}
    
    # 2. Load experts and compute task vectors
    expert_backbones = []
    expert_heads = []
    task_vectors = []
    fisher_priors = []
    
    for i in range(3):
        ckpt_path = f'checkpoints/expert_{i}.pt'
        fim_path = f'checkpoints/fim_{i}.pt'
        if not os.path.exists(ckpt_path) or not os.path.exists(fim_path):
            raise FileNotFoundError(f"Expert or FIM checkpoints missing for task {i}. Please run train_experts.py first.")
            
        print(f"Loading Expert {i} from {ckpt_path}...")
        checkpoint = torch.load(ckpt_path, map_location=device)
        
        # Expert Backbone - use fresh resnet18 instance to prevent shared reference bugs
        m_expert = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        eb = ResNetBackbone(m_expert).to(device)
        eb.load_state_dict(checkpoint['backbone_state_dict'])
        eb.eval()
        expert_backbones.append(eb)
        
        # Expert Head
        eh = nn.Linear(512, 10).to(device)
        eh.load_state_dict(checkpoint['head_state_dict'])
        eh.eval()
        expert_heads.append(eh)
        
        # Compute Task Vector for this expert
        eb_params = {k: v.clone().detach() for k, v in eb.named_parameters()}
        tv = {k: eb_params[k] - base_backbone_params[k] for k in base_backbone_params.keys()}
        task_vectors.append(tv)
        
        # Load FIM prior
        fim = torch.load(fim_path, map_location=device)
        fisher_priors.append(fim)
        
    # 3. Load Datasets and Construct Streams
    print("\nLoading test datasets and building streams...")
    mnist_test = get_dataset('mnist', train=False)
    fashion_test = get_dataset('fashion', train=False)
    kmnist_test = get_dataset('kmnist', train=False)
    
    mnist_loader = DataLoader(mnist_test, batch_size=32, shuffle=False, num_workers=0)
    fashion_loader = DataLoader(fashion_test, batch_size=32, shuffle=False, num_workers=0)
    kmnist_loader = DataLoader(kmnist_test, batch_size=32, shuffle=False, num_workers=0)
    
    # Extract 50 batches of size 32 for each
    mnist_batches = []
    for i, batch in enumerate(mnist_loader):
        if i >= 50:
            break
        mnist_batches.append((batch, 0))
        
    fashion_batches = []
    for i, batch in enumerate(fashion_loader):
        if i >= 50:
            break
        fashion_batches.append((batch, 1))
        
    kmnist_batches = []
    for i, batch in enumerate(kmnist_loader):
        if i >= 50:
            break
        kmnist_batches.append((batch, 2))
        
    # Build Sequential Stream (150 batches)
    sequential_batches = mnist_batches + fashion_batches + kmnist_batches
    
    # Build Alternating Stream (150 batches)
    alternating_batches = []
    for i in range(50):
        alternating_batches.append(mnist_batches[i])
        alternating_batches.append(fashion_batches[i])
        alternating_batches.append(kmnist_batches[i])
        
    streams = {
        'Sequential': sequential_batches,
        'Alternating': alternating_batches
    }
    
    # 4. Running evaluations across sweep parameters
    results = {}
    
    # Static model merging (uniform coefficients) - same for any lr/gamma
    static_results = {}
    for s_name, s_batches in streams.items():
        acc, _ = run_evaluation(
            s_name, s_batches, 'static', base_backbone, base_backbone_params, 
            expert_backbones, expert_heads, task_vectors, fisher_priors, 
            args.lr_head, 0.0, 0.0, args.dts_alpha, args.dts_beta, device
        )
        static_results[s_name] = acc
    print("\n--- STATIC MERGING (UNIFORM) ---")
    for s_name, acc in static_results.items():
        print(f"  {s_name} Accuracy: {acc:.2f}%")
        
    # SATA-TTA / Unconstrained TTA sweep
    unconstrained_results = {}
    print("\n--- RUNNING SWEEP FOR UNCONSTRAINED TTA ---")
    for lr_lam in lr_lambdas:
        unconstrained_results[lr_lam] = {}
        for s_name, s_batches in streams.items():
            acc, _ = run_evaluation(
                s_name, s_batches, 'unconstrained_tta', base_backbone, base_backbone_params,
                expert_backbones, expert_heads, task_vectors, fisher_priors,
                args.lr_head, lr_lam, 0.0, args.dts_alpha, args.dts_beta, device
            )
            unconstrained_results[lr_lam][s_name] = acc
            print(f"  lr_lambda={lr_lam} | {s_name} Accuracy: {acc:.2f}%")
            
    # EWC-TTA (Baseline) sweep
    ewc_results = {}
    print("\n--- RUNNING SWEEP FOR EWC-TTA (BASELINE) ---")
    for gamma in gammas:
        ewc_results[gamma] = {}
        for lr_lam in lr_lambdas:
            ewc_results[gamma][lr_lam] = {}
            for s_name, s_batches in streams.items():
                acc, _ = run_evaluation(
                    s_name, s_batches, 'ewc_tta', base_backbone, base_backbone_params,
                    expert_backbones, expert_heads, task_vectors, fisher_priors,
                    args.lr_head, lr_lam, gamma, args.dts_alpha, args.dts_beta, device
                )
                ewc_results[gamma][lr_lam][s_name] = acc
                print(f"  gamma={gamma} | lr_lambda={lr_lam} | {s_name} Accuracy: {acc:.2f}%")
                
    # TF-EWC-DTS (Ours) sweep
    tf_ewc_dts_results = {}
    print("\n--- RUNNING SWEEP FOR TF-EWC-DTS (OUR PROPOSED TEACHER-FREE) ---")
    for gamma in gammas:
        tf_ewc_dts_results[gamma] = {}
        for lr_lam in lr_lambdas:
            tf_ewc_dts_results[gamma][lr_lam] = {}
            for s_name, s_batches in streams.items():
                acc, _ = run_evaluation(
                    s_name, s_batches, 'tf_ewc_dts', base_backbone, base_backbone_params,
                    expert_backbones, expert_heads, task_vectors, fisher_priors,
                    args.lr_head, lr_lam, gamma, args.dts_alpha, args.dts_beta, device
                )
                tf_ewc_dts_results[gamma][lr_lam][s_name] = acc
                print(f"  gamma={gamma} | lr_lambda={lr_lam} | {s_name} Accuracy: {acc:.2f}%")
                
    # Write results to JSON
    summary = {
        'static': static_results,
        'unconstrained': unconstrained_results,
        'ewc_tta': ewc_results,
        'tf_ewc_dts': tf_ewc_dts_results
    }
    with open('tta_results.json', 'w') as f:
        json.dump(summary, f, indent=4)
    print("\nSaved all evaluation results to tta_results.json!")

if __name__ == "__main__":
    main()
