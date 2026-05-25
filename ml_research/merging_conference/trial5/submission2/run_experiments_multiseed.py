import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.func
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

# -------------------------------------------------------------
# 1. SETUP & UTILS
# -------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Task normalizations
MEANS = {
    0: [0.1307, 0.1307, 0.1307], # MNIST
    1: [0.2860, 0.2860, 0.2860], # FashionMNIST
    2: [0.1918, 0.1918, 0.1918]  # KMNIST
}
STDS = {
    0: [0.3081, 0.3081, 0.3081], # MNIST
    1: [0.3530, 0.3530, 0.3530], # FashionMNIST
    2: [0.3483, 0.3483, 0.3483]  # KMNIST
}

def normalize_batch(imgs, task_idx):
    dtype = imgs.dtype
    device = imgs.device
    mean = torch.tensor(MEANS[task_idx], dtype=dtype, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STDS[task_idx], dtype=dtype, device=device).view(1, 3, 1, 1)
    return (imgs - mean) / std

def corrupt_batch(imgs, corruption_type):
    if corruption_type == 'gaussian_noise':
        noise = torch.randn_like(imgs) * 0.2
        return torch.clamp(imgs + noise, 0.0, 1.0)
    elif corruption_type == 'contrast':
        corrupted = []
        for img in imgs:
            c_img = torchvision.transforms.functional.adjust_contrast(img, 0.3)
            corrupted.append(c_img)
        return torch.clamp(torch.stack(corrupted), 0.0, 1.0)
    else:
        return imgs

# ResNet-18 wrapper to return both features and logits
class ResNet18Wrapper(nn.Module):
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
        features = torch.flatten(x, 1)
        logits = self.base_model.fc(features)
        return features, logits

# Helper to merge state dict using layer-wise coefficients
def get_merged_state_dict(lambdas_dict, base_state, task_vectors):
    merged = {}
    lambdas_tensors = torch.stack([lambdas_dict[name].detach() for name in lambdas_dict])
    avg_lambda = lambdas_tensors.mean(dim=0)
    
    for name in base_state:
        if name in lambdas_dict:
            coefs = lambdas_dict[name]
            merged[name] = base_state[name] + sum(coefs[k] * task_vectors[k][name] for k in range(3))
        elif base_state[name].is_floating_point():
            merged[name] = base_state[name] + sum(avg_lambda[k] * task_vectors[k][name] for k in range(3))
        else:
            merged[name] = base_state[name].clone()
    return merged

# Standard Simplex Projection
def project_simplex(v):
    v_sorted, _ = torch.sort(v, descending=True)
    cssv = torch.cumsum(v_sorted, dim=0)
    ind = torch.arange(1, len(v) + 1, dtype=v.dtype, device=v.device)
    cond = v_sorted - (cssv - 1.0) / ind > 0
    idx = torch.nonzero(cond).max().item()
    theta = (cssv[idx] - 1.0) / (idx + 1)
    return torch.clamp(v - theta, min=0.0)

def main():
    set_seed(42)
    torch.backends.cudnn.enabled = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Transform: replicate to 3 channels but keep raw range [0, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1))
    ])
    
    print("Loading datasets...")
    mnist_test_full = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    fmnist_test_full = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    kmnist_test_full = torchvision.datasets.KMNIST(root='./data', train=False, download=True, transform=transform)
    
    # Calibration subsets (500 samples)
    mnist_train_full = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    fmnist_train_full = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    kmnist_train_full = torchvision.datasets.KMNIST(root='./data', train=True, download=True, transform=transform)
    
    # Use standard 500 samples for offline calibration
    mnist_calib = Subset(mnist_train_full, list(range(10000, 10500)))
    fmnist_calib = Subset(fmnist_train_full, list(range(10000, 10500)))
    kmnist_calib = Subset(kmnist_train_full, list(range(10000, 10500)))
    
    calib_loaders = [
        DataLoader(mnist_calib, batch_size=32, shuffle=False),
        DataLoader(fmnist_calib, batch_size=32, shuffle=False),
        DataLoader(kmnist_calib, batch_size=32, shuffle=False)
    ]
    
    # Base Model (Pre-trained)
    print("Initializing base pre-trained ResNet-18 model...")
    base_model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    base_model.fc = nn.Linear(512, 10)
    base_model.load_state_dict(torch.load('checkpoints/pretrained_base.pth', map_location=device))
    base_model.to(device)
    
    experts_paths = [
        'checkpoints/expert_mnist.pth',
        'checkpoints/expert_fashionmnist.pth',
        'checkpoints/expert_kmnist.pth'
    ]
    
    experts_models = []
    for k, path in enumerate(experts_paths):
        model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(512, 10)
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        experts_models.append(model)
        
    trainable_names = [name for name, param in base_model.named_parameters() if param.requires_grad]
    
    # -------------------------------------------------------------
    # COMPUTE CO-ESTIMATED DIAGONAL FISHER & CLASS PROTOTYPES
    # -------------------------------------------------------------
    print("\n--- Computing Parameter Fisher Sensitivity & Class Prototypes ---")
    fisher_experts = []
    criterion_fisher = nn.CrossEntropyLoss()
    
    for k, model in enumerate(experts_models):
        model.eval()
        fisher_dict = {name: torch.zeros_like(param) for name, param in model.named_parameters() if param.requires_grad}
        num_samples = 0
        
        for imgs, labels in calib_loaders[k]:
            imgs, labels = imgs.to(device), labels.to(device)
            imgs_norm = normalize_batch(imgs, k)
            B = imgs.size(0)
            num_samples += B
            
            for i in range(B):
                img = imgs_norm[i:i+1]
                label = labels[i:i+1]
                model.zero_grad()
                output = model(img)
                loss = criterion_fisher(output, label)
                loss.backward()
                
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if param.requires_grad and param.grad is not None:
                            fisher_dict[name] += param.grad.data ** 2
                            
        for name in fisher_dict:
            fisher_dict[name] /= num_samples
        fisher_experts.append(fisher_dict)
        
    joint_fisher_scalar = {}
    for name in trainable_names:
        avg_fisher_across_experts = []
        for k in range(3):
            avg_f_layer = fisher_experts[k][name].mean().item()
            avg_fisher_across_experts.append(avg_f_layer)
        joint_fisher_scalar[name] = np.mean(avg_fisher_across_experts)
        
    # Extract Class Prototypes
    prototypes_experts = []
    for k, model in enumerate(experts_models):
        model.eval()
        embeddings_class = {c: [] for c in range(10)}
        
        with torch.no_grad():
            for imgs, labels in calib_loaders[k]:
                imgs = imgs.to(device)
                imgs_norm = normalize_batch(imgs, k)
                features = model.conv1(imgs_norm)
                features = model.bn1(features)
                features = model.relu(features)
                features = model.maxpool(features)
                features = model.layer1(features)
                features = model.layer2(features)
                features = model.layer3(features)
                features = model.layer4(features)
                features = model.avgpool(features)
                features = torch.flatten(features, 1)
                
                for feat, label in zip(features, labels):
                    embeddings_class[label.item()].append(feat)
                    
        proto_list = []
        for c in range(10):
            if len(embeddings_class[c]) > 0:
                stacked = torch.stack(embeddings_class[c])
                mean_emb = stacked.mean(dim=0)
                norm_emb = mean_emb / (mean_emb.norm(p=2) + 1e-8)
                proto_list.append(norm_emb)
            else:
                proto_list.append(torch.zeros(512, device=device))
        prototypes_experts.append(torch.stack(proto_list))
        
    # Prepare task vectors
    base_state = {name: param.clone().to(device) for name, param in base_model.named_parameters()}
    task_vectors = []
    for k in range(3):
        tv = {}
        expert_state = experts_models[k].state_dict()
        for name in base_state:
            if base_state[name].is_floating_point():
                tv[name] = expert_state[name].to(device) - base_state[name].to(device)
            else:
                tv[name] = expert_state[name].to(device).clone()
        task_vectors.append(tv)
        
    wrapper = ResNet18Wrapper(base_model).to(device)
    
    # Evaluation configuration
    batch_size = 32
    num_batches = 150
    seeds = [42, 100, 2026]
    methods = ['Static', 'AdaMerging', 'LFWA', 'CPA-Merge', 'FP-CA']
    corruptions = ['clean', 'gaussian_noise', 'contrast']
    streams_keys = ['Alternating', 'Sequential']
    
    # Store all runs results: results_all[seed][stream][corruption][method] = acc
    results_all = {seed: {s: {c: {} for c in corruptions} for s in streams_keys} for seed in seeds}
    
    for seed in seeds:
        print(f"\n=======================================================")
        print(f"RUNNING EVALUATION SWEET WITH RANDOM SEED {seed}")
        print(f"=======================================================")
        set_seed(seed)
        
        # Draw randomized test subsets
        g_sub = torch.Generator().manual_seed(seed)
        mnist_indices = torch.randperm(10000, generator=g_sub)[:1600].tolist()
        fmnist_indices = torch.randperm(10000, generator=g_sub)[:1600].tolist()
        kmnist_indices = torch.randperm(10000, generator=g_sub)[:1600].tolist()
        
        mnist_test_subset = Subset(mnist_test_full, mnist_indices)
        fmnist_test_subset = Subset(fmnist_test_full, fmnist_indices)
        kmnist_test_subset = Subset(kmnist_test_full, kmnist_indices)
        
        # Build streams
        stream_alternating = []
        stream_sequential = []
        
        indices_mnist = 0
        indices_fmnist = 0
        indices_kmnist = 0
        for b in range(num_batches):
            task_idx_alt = b % 3
            if task_idx_alt == 0:
                batch_mnist = [mnist_test_subset[indices_mnist + i] for i in range(batch_size)]
                indices_mnist += batch_size
                stream_alternating.append((batch_mnist, 0))
            elif task_idx_alt == 1:
                batch_fmnist = [fmnist_test_subset[indices_fmnist + i] for i in range(batch_size)]
                indices_fmnist += batch_size
                stream_alternating.append((batch_fmnist, 1))
            else:
                batch_kmnist = [kmnist_test_subset[indices_kmnist + i] for i in range(batch_size)]
                indices_kmnist += batch_size
                stream_alternating.append((batch_kmnist, 2))
                
        indices_mnist = 0
        indices_fmnist = 0
        indices_kmnist = 0
        for b in range(num_batches):
            task_idx_seq = b // 50
            if task_idx_seq == 0:
                batch_mnist = [mnist_test_subset[indices_mnist + i] for i in range(batch_size)]
                indices_mnist += batch_size
                stream_sequential.append((batch_mnist, 0))
            elif task_idx_seq == 1:
                batch_fmnist = [fmnist_test_subset[indices_fmnist + i] for i in range(batch_size)]
                indices_fmnist += batch_size
                stream_sequential.append((batch_fmnist, 1))
            else:
                batch_kmnist = [kmnist_test_subset[indices_kmnist + i] for i in range(batch_size)]
                indices_kmnist += batch_size
                stream_sequential.append((batch_kmnist, 2))
                
        streams = {
            'Alternating': stream_alternating,
            'Sequential': stream_sequential
        }
        
        # Evaluate methods
        for stream_name, stream_data in streams.items():
            for corruption in corruptions:
                for method_name in methods:
                    # Initialize lambdas
                    lambdas_dict = {}
                    if method_name in ['Static', 'AdaMerging', 'LFWA']:
                        lambdas_dict = {name: torch.tensor([1/3, 1/3, 1/3], device=device, requires_grad=True) for name in trainable_names}
                    elif method_name in ['CPA-Merge']:
                        global_lambda = torch.tensor([1/3, 1/3, 1/3], device=device, requires_grad=True)
                        lambdas_dict = {name: global_lambda for name in trainable_names}
                    elif method_name in ['FP-CA']:
                        lambdas_dict = {name: torch.tensor([1/3, 1/3, 1/3], device=device, requires_grad=True) for name in trainable_names}
                        
                    correct_predictions = 0
                    total_predictions = 0
                    
                    for b_idx, (batch_data, true_task) in enumerate(stream_data):
                        batch_imgs = torch.stack([x[0] for x in batch_data]).to(device)
                        batch_labels = torch.tensor([x[1] for x in batch_data]).to(device)
                        corrupted_imgs = corrupt_batch(batch_imgs, corruption)
                        
                        # Task Detection for CPA and FP-CA
                        detected_task = true_task
                        if method_name in ['CPA-Merge', 'FP-CA']:
                            uniform_lambdas = {name: torch.tensor([1/3, 1/3, 1/3], device=device) for name in trainable_names}
                            uniform_state = get_merged_state_dict(uniform_lambdas, base_state, task_vectors)
                            
                            with torch.no_grad():
                                wrapper.base_model.load_state_dict(uniform_state, strict=False)
                                anchor_imgs_norm = normalize_batch(corrupted_imgs, 0)
                                anchor_feats, _ = wrapper(anchor_imgs_norm)
                                anchor_feats_norm = anchor_feats / (anchor_feats.norm(p=2, dim=1, keepdim=True) + 1e-8)
                                
                                scores = []
                                for k in range(3):
                                    sim_k = torch.matmul(anchor_feats_norm, prototypes_experts[k].t())
                                    max_sim_k, _ = sim_k.max(dim=1)
                                    score_k = max_sim_k.mean().item()
                                    scores.append(score_k)
                                    
                                scores_tensor = torch.tensor(scores, device=device)
                                tau = 0.02
                                lambda_prior = torch.softmax(scores_tensor / tau, dim=0)
                                detected_task = torch.argmax(lambda_prior).item()
                                
                                if method_name == 'CPA-Merge':
                                    global_lambda = lambda_prior.clone().requires_grad_(True)
                                    lambdas_dict = {name: global_lambda for name in trainable_names}
                                elif method_name == 'FP-CA':
                                    lambdas_dict = {name: lambda_prior.clone().requires_grad_(True) for name in trainable_names}
                                    
                        # Optimize
                        active_merged_state = get_merged_state_dict(lambdas_dict, base_state, task_vectors)
                        active_head_idx = detected_task if method_name in ['CPA-Merge', 'FP-CA'] else true_task
                        norm_imgs = normalize_batch(corrupted_imgs, active_head_idx)
                        
                        active_merged_state_prefixed = {f"base_model.{name}": val for name, val in active_merged_state.items()}
                        features, logits = torch.func.functional_call(wrapper, active_merged_state_prefixed, norm_imgs)
                        
                        probs = torch.softmax(logits, dim=-1)
                        loss_ent = -torch.mean(torch.sum(probs * torch.log(probs + 1e-8), dim=-1))
                        
                        loss_contra = torch.tensor(0.0, device=device)
                        if method_name in ['CPA-Merge', 'FP-CA']:
                            max_probs, pred_classes = probs.max(dim=1)
                            conf_mask = max_probs > 0.85
                            high_conf_indices = torch.nonzero(conf_mask).squeeze(1)
                            
                            if len(high_conf_indices) > 0:
                                z_norm = features / (features.norm(p=2, dim=1, keepdim=True) + 1e-8)
                                z_high_conf = z_norm[high_conf_indices]
                                pred_classes_high_conf = pred_classes[high_conf_indices]
                                sim_matrix = torch.matmul(z_high_conf, prototypes_experts[active_head_idx].t())
                                kappa = 0.1
                                logits_contra = sim_matrix / kappa
                                loss_contra = nn.CrossEntropyLoss()(logits_contra, pred_classes_high_conf)
                                
                        beta = 0.1
                        loss_total = loss_ent + beta * loss_contra if method_name in ['CPA-Merge', 'FP-CA'] else loss_ent
                        
                        # Optimization update
                        if method_name in ['AdaMerging', 'LFWA', 'FP-CA']:
                            grad_tensors = list(lambdas_dict.values())
                            grads = torch.autograd.grad(loss_total, grad_tensors, allow_unused=True)
                            with torch.no_grad():
                                for name, grad in zip(lambdas_dict.keys(), grads):
                                    if grad is not None:
                                        eta = 0.1 if method_name in ['AdaMerging', 'LFWA'] else 0.01
                                        alpha = 1.0 if method_name == 'FP-CA' else (0.5 if method_name == 'LFWA' else 0.0)
                                        eps_scale = 1e-6
                                        lr_w = eta * (joint_fisher_scalar[name] + eps_scale) ** (-alpha)
                                        lambdas_dict[name] -= lr_w * grad
                                        lambdas_dict[name].copy_(project_simplex(lambdas_dict[name]))
                        elif method_name == 'CPA-Merge':
                            grads = torch.autograd.grad(loss_total, [global_lambda])
                            if grads[0] is not None:
                                with torch.no_grad():
                                    eta = 0.01
                                    global_lambda -= eta * grads[0]
                                    global_lambda.copy_(project_simplex(global_lambda))
                                    
                        # Eval True Head
                        with torch.no_grad():
                            final_merged_state = get_merged_state_dict(lambdas_dict, base_state, task_vectors)
                            final_merged_state_prefixed = {f"base_model.{name}": val for name, val in final_merged_state.items()}
                            norm_imgs_eval = normalize_batch(corrupted_imgs, true_task)
                            _, eval_logits = torch.func.functional_call(wrapper, final_merged_state_prefixed, norm_imgs_eval)
                            _, preds_eval = eval_logits.max(dim=1)
                            
                            correct_predictions += preds_eval.eq(batch_labels).sum().item()
                            total_predictions += batch_imgs.size(0)
                            
                    acc = (correct_predictions / total_predictions) * 100
                    results_all[seed][stream_name][corruption][method_name] = acc
                    print(f"Seed: {seed} | Stream: {stream_name} | Corr: {corruption} | Method: {method_name} | Acc: {acc:.2f}%")

    # -------------------------------------------------------------
    # STATISTICAL ANALYSIS
    # -------------------------------------------------------------
    print("\n=======================================================")
    print("FINAL AGGREGATED STATISTICAL RESULTS (MEAN ± STD)")
    print("=======================================================")
    
    # Store aggregated stats: stats[stream][corruption][method] = (mean, std)
    stats = {s: {c: {} for c in corruptions} for s in streams_keys}
    
    for stream in streams_keys:
        for corr in corruptions:
            for m in methods:
                accs = [results_all[seed][stream][corr][m] for seed in seeds]
                mean_acc = np.mean(accs)
                std_acc = np.std(accs)
                stats[stream][corr][m] = (mean_acc, std_acc)
                print(f"Aggregated -> Stream: {stream:12s} | Corruption: {corr:14s} | Method: {m:10s} | Accuracy: {mean_acc:6.2f} ± {std_acc:4.2f}%")

    # Output LaTeX and Markdown representations
    print("\n--- LaTeX Table Snippet ---")
    for stream in streams_keys:
        for corr in corruptions:
            corr_label = corr.replace('_', ' ').title() if corr != 'clean' else 'Clean'
            row_str = f"{stream[:3]} & {corr_label:14s}"
            for m in methods:
                mean, std = stats[stream][corr][m]
                row_str += f" & {mean:.1f}\\% \\scalebox{{0.8}}{{\\tiny$\\pm$ {std:.1f}\\%}}"
            print(row_str + " \\\\")

    # Save details to file
    with open('multiseed_results.txt', 'w') as f:
        f.write("Multi-seed results across seeds: " + str(seeds) + "\n\n")
        for stream in streams_keys:
            for corr in corruptions:
                for m in methods:
                    accs = [results_all[seed][stream][corr][m] for seed in seeds]
                    f.write(f"{stream} | {corr} | {m} | Mean: {np.mean(accs):.4f} | Std: {np.std(accs):.4f} | Runs: {accs}\n")
                    
    print("\nSuccessfully saved multi-seed aggregated results to multiseed_results.txt")

if __name__ == '__main__':
    main()
