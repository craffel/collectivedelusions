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
    
    mnist_calib = Subset(mnist_train_full, list(range(10000, 10500)))
    fmnist_calib = Subset(fmnist_train_full, list(range(10000, 10500)))
    kmnist_calib = Subset(kmnist_train_full, list(range(10000, 10500)))
    
    # Test subsets (1600 samples for evaluation stream)
    mnist_test = Subset(mnist_test_full, list(range(1600)))
    fmnist_test = Subset(fmnist_test_full, list(range(1600)))
    kmnist_test = Subset(kmnist_test_full, list(range(1600)))
    
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
        
    # -------------------------------------------------------------
    # COMPUTE FISHER AND CLASS PROTOTYPES
    # -------------------------------------------------------------
    print("\n--- Computing Parameter Fisher Sensitivity & Class Prototypes ---")
    fisher_experts = []
    criterion_fisher = nn.CrossEntropyLoss()
    
    for k, model in enumerate(experts_models):
        print(f"Computing Fisher sensitivity for Expert {k}...")
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
    trainable_names = [name for name, param in base_model.named_parameters() if param.requires_grad]
    
    for name in trainable_names:
        avg_fisher_across_experts = []
        for k in range(3):
            avg_f_layer = fisher_experts[k][name].mean().item()
            avg_fisher_across_experts.append(avg_f_layer)
        joint_fisher_scalar[name] = np.mean(avg_fisher_across_experts)
        
    print("Extracting L2-normalized class prototypes...")
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
    
    # -------------------------------------------------------------
    # DEFINE EVALUATION FOR A GIVEN BATCH SIZE
    # -------------------------------------------------------------
    def get_streams_for_batch_size(b_size):
        tot_samples = 1600
        n_batches_per_task = tot_samples // b_size
        tot_batches = n_batches_per_task * 3
        
        # Alternating stream
        stream_alt = []
        indices_m = 0
        indices_f = 0
        indices_k = 0
        for b in range(tot_batches):
            task_idx = b % 3
            if task_idx == 0:
                batch_data = [mnist_test[indices_m + i] for i in range(b_size)]
                indices_m += b_size
                stream_alt.append((batch_data, 0))
            elif task_idx == 1:
                batch_data = [fmnist_test[indices_f + i] for i in range(b_size)]
                indices_f += b_size
                stream_alt.append((batch_data, 1))
            else:
                batch_data = [kmnist_test[indices_k + i] for i in range(b_size)]
                indices_k += b_size
                stream_alt.append((batch_data, 2))
                
        # Sequential stream
        stream_seq = []
        indices_m = 0
        indices_f = 0
        indices_k = 0
        for b in range(tot_batches):
            task_idx = b // n_batches_per_task
            if task_idx == 0:
                batch_data = [mnist_test[indices_m + i] for i in range(b_size)]
                indices_m += b_size
                stream_seq.append((batch_data, 0))
            elif task_idx == 1:
                batch_data = [fmnist_test[indices_f + i] for i in range(b_size)]
                indices_f += b_size
                stream_seq.append((batch_data, 1))
            else:
                batch_data = [kmnist_test[indices_k + i] for i in range(b_size)]
                indices_k += b_size
                stream_seq.append((batch_data, 2))
                
        return stream_alt, stream_seq

    def evaluate_fpca_batch_size(stream, b_size, corruption='gaussian_noise'):
        alpha = 1.0
        gamma = 0.85
        beta = 0.1
        # Set base learning rate eta
        eta = 0.01
        
        lambdas_dict = {name: torch.tensor([1/3, 1/3, 1/3], device=device, requires_grad=True) for name in trainable_names}
        correct_predictions = 0
        total_predictions = 0
        
        for b_idx, (batch_data, true_task) in enumerate(stream):
            batch_imgs = torch.stack([x[0] for x in batch_data]).to(device)
            batch_labels = torch.tensor([x[1] for x in batch_data]).to(device)
            corrupted_imgs = corrupt_batch(batch_imgs, corruption)
            
            # PD-Routing Task Detection
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
                
                lambdas_dict = {name: lambda_prior.clone().requires_grad_(True) for name in trainable_names}
                    
            active_merged_state = get_merged_state_dict(lambdas_dict, base_state, task_vectors)
            norm_imgs = normalize_batch(corrupted_imgs, detected_task)
            
            active_merged_state_prefixed = {f"base_model.{name}": val for name, val in active_merged_state.items()}
            features, logits = torch.func.functional_call(wrapper, active_merged_state_prefixed, norm_imgs)
            
            probs = torch.softmax(logits, dim=-1)
            loss_ent = -torch.mean(torch.sum(probs * torch.log(probs + 1e-8), dim=-1))
            
            # Contrastive loss
            loss_contra = torch.tensor(0.0, device=device)
            max_probs, pred_classes = probs.max(dim=1)
            conf_mask = max_probs > gamma
            high_conf_indices = torch.nonzero(conf_mask).squeeze(1)
            
            if len(high_conf_indices) > 0:
                z_norm = features / (features.norm(p=2, dim=1, keepdim=True) + 1e-8)
                z_high_conf = z_norm[high_conf_indices]
                pred_classes_high_conf = pred_classes[high_conf_indices]
                sim_matrix = torch.matmul(z_high_conf, prototypes_experts[detected_task].t())
                kappa = 0.1
                logits_contra = sim_matrix / kappa
                loss_contra = nn.CrossEntropyLoss()(logits_contra, pred_classes_high_conf)
                
            loss_total = loss_ent + beta * loss_contra
            
            grad_tensors = list(lambdas_dict.values())
            grads = torch.autograd.grad(loss_total, grad_tensors, allow_unused=True)
            
            with torch.no_grad():
                for name, grad in zip(lambdas_dict.keys(), grads):
                    if grad is not None:
                        eps_scale = 1e-6
                        lr_w = eta * (joint_fisher_scalar[name] + eps_scale) ** (-alpha)
                        lambdas_dict[name] -= lr_w * grad
                        lambdas_dict[name].copy_(project_simplex(lambdas_dict[name]))
                        
            # Inference Accuracy
            with torch.no_grad():
                final_merged_state = get_merged_state_dict(lambdas_dict, base_state, task_vectors)
                final_merged_state_prefixed = {f"base_model.{name}": val for name, val in final_merged_state.items()}
                norm_imgs_eval = normalize_batch(corrupted_imgs, true_task)
                _, eval_logits = torch.func.functional_call(wrapper, final_merged_state_prefixed, norm_imgs_eval)
                _, preds_eval = eval_logits.max(dim=1)
                correct_predictions += preds_eval.eq(batch_labels).sum().item()
                total_predictions += batch_imgs.size(0)
                
        return (correct_predictions / total_predictions) * 100

    # Running batch size sweep
    batch_sizes = [8, 16, 32, 64]
    results_alt = {}
    results_seq = {}
    
    print("\n--- Running Test-Time Batch Size Sweep (Gaussian Noise Corruption) ---")
    for bs in batch_sizes:
        print(f"\nEvaluating batch_size={bs}...")
        stream_alt, stream_seq = get_streams_for_batch_size(bs)
        
        acc_alt = evaluate_fpca_batch_size(stream_alt, bs, 'gaussian_noise')
        acc_seq = evaluate_fpca_batch_size(stream_seq, bs, 'gaussian_noise')
        
        results_alt[bs] = acc_alt
        results_seq[bs] = acc_seq
        print(f"Batch Size {bs} -> Alternating Stream: {acc_alt:.2f}%, Sequential Stream: {acc_seq:.2f}%")
        
    print("\n--- Final Results Summary (Gaussian Noise) ---")
    print("| Batch Size ($B$) | Alternating Stream Accuracy | Sequential Stream Accuracy |")
    print("| :---: | :---: | :---: |")
    for bs in batch_sizes:
        print(f"| {bs} | {results_alt[bs]:.2f}% | {results_seq[bs]:.2f}% |")

if __name__ == '__main__':
    main()
