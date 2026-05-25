import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.func import functional_call
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import os
import torch
torch.backends.cudnn.enabled = False

import argparse

# Define the exact grayscale ResNet-18 architecture as used in training
def get_resnet18_grayscale():
    model = models.resnet18()
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.fc = nn.Linear(512, 10)
    return model

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size(), device=tensor.device) * self.std + self.mean

def project_simplex(v):
    # v is a 1D tensor of shape [K]
    sorted_v, _ = torch.sort(v, descending=True)
    cssv = torch.cumsum(sorted_v, dim=-1)
    indices = torch.arange(1, len(v) + 1, device=v.device, dtype=v.dtype)
    cond = sorted_v - (cssv - 1.0) / indices > 0
    rho = torch.max(torch.where(cond)[0]) + 1
    theta = (cssv[rho - 1] - 1.0) / rho
    return torch.clamp(v - theta, min=0.0)

def compute_diagonal_fisher(model, calibration_set, device="cuda"):
    model.eval()
    fisher = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher[name] = torch.zeros_like(param.data)
            
    loader = DataLoader(calibration_set, batch_size=32, shuffle=False)
    num_samples = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        batch_size = inputs.size(0)
        
        # Forward
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)
        
        # Backward
        model.zero_grad()
        loss.backward()
        
        # Accumulate
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher[name] += (param.grad.data ** 2) * batch_size
                
        num_samples += batch_size
        if num_samples >= 500:
            break
            
    # Divide and average
    for name in fisher:
        fisher[name] /= num_samples
        
    return fisher

def compute_prototypes(model, calibration_set, device="cuda"):
    model.eval()
    loader = DataLoader(calibration_set, batch_size=32, shuffle=False)
    
    # 1. Extract all features and labels
    all_features = []
    all_targets = []
    
    # Feature extractor hook or manual forward (ResNet-18 features are output of avgpool)
    # Let's extract features by temporarily modifying the forward pass or using a feature extractor
    # A cleaner way is to define a helper function to extract features:
    def extract_features(x):
        # Forward up to avgpool
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
        x = model.avgpool(x)
        x = torch.flatten(x, 1)
        return x
        
    num_samples = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            feats = extract_features(inputs)
            all_features.append(feats.cpu())
            all_targets.append(targets.clone())
            num_samples += inputs.size(0)
            if num_samples >= 500:
                break
                
    all_features = torch.cat(all_features, dim=0)[:500]
    all_targets = torch.cat(all_targets, dim=0)[:500]
    
    # Compute centroid
    centroid = all_features.mean(dim=0)
    
    # Compute class prototypes
    prototypes = torch.zeros(10, 512)
    for c in range(10):
        c_mask = (all_targets == c)
        if c_mask.sum() > 0:
            class_feats = all_features[c_mask]
            # Center the features with the centroid
            centered_feats = class_feats - centroid
            # Average and normalize
            mean_feat = centered_feats.mean(dim=0)
            prototypes[c] = mean_feat / (mean_feat.norm(p=2) + 1e-8)
            
    return centroid, prototypes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="Rob-OW", choices=["Static", "PROTO-TTMM", "IGGS-OW", "Rob-OW"])
    parser.add_argument("--corruption", type=str, default="clean", choices=["clean", "corrupted"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=1.0, help="Fisher damping factor")
    parser.add_argument("--lr", type=float, default=5e-2, help="Base learning rate")
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Main Experiment: Method={args.method}, Corruption={args.corruption}, Seed={args.seed}, Device={device}")
    
    # 1. Load trained expert models
    model1 = get_resnet18_grayscale().to(device)
    model2 = get_resnet18_grayscale().to(device)
    
    model1.load_state_dict(torch.load("./checkpoints/expert_mnist.pth", map_location=device))
    model2.load_state_dict(torch.load("./checkpoints/expert_kmnist.pth", map_location=device))
    
    print("Experts loaded successfully.")
    
    # 2. Setup datasets for calibration
    transform_clean = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    mnist_cal_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=transform_clean)
    kmnist_cal_dataset = torchvision.datasets.KMNIST(root='./data', train=True, download=False, transform=transform_clean)
    
    # 3. Compute offline centroids, prototypes, and diagonal Fisher sensitivity
    print("Pre-computing expert centroids and prototypes...")
    centroid1, prototypes1 = compute_prototypes(model1, mnist_cal_dataset, device=device)
    centroid2, prototypes2 = compute_prototypes(model2, kmnist_cal_dataset, device=device)
    
    print("Pre-computing diagonal Fisher Information matrices...")
    fisher1 = compute_diagonal_fisher(model1, mnist_cal_dataset, device=device)
    fisher2 = compute_diagonal_fisher(model2, kmnist_cal_dataset, device=device)
    
    # Calculate tensor-level joint sensitivity
    joint_fisher = {}
    for name in fisher1:
        f1_avg = fisher1[name].mean().item()
        f2_avg = fisher2[name].mean().item()
        joint_fisher[name] = 0.5 * (f1_avg + f2_avg)
        
    # 4. Construct the test-time stream
    if args.corruption == "corrupted":
        transform_stream = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            AddGaussianNoise(0.0, 0.2)
        ])
    else:
        transform_stream = transform_clean
        
    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform_stream)
    kmnist_test = torchvision.datasets.KMNIST(root='./data', train=False, download=False, transform=transform_stream)
    fmnist_test = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_stream)
    
    print("Stream datasets loaded.")
    
    # Construct exactly 90 batches of size 64:
    # Batches 1-30: MNIST (known task 1)
    # Batches 31-60: KMNIST (known task 2)
    # Batches 61-90: FashionMNIST (novel domain)
    mnist_batches = []
    kmnist_batches = []
    fmnist_batches = []
    
    # Helper to build batches
    def get_batches(dataset, num_batches, batch_size=64):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        batches = []
        for idx, (inputs, targets) in enumerate(loader):
            if idx >= num_batches:
                break
            batches.append((inputs, targets))
        return batches
        
    mnist_batches = get_batches(mnist_test, 30, batch_size=64)
    kmnist_batches = get_batches(kmnist_test, 30, batch_size=64)
    fmnist_batches = get_batches(fmnist_test, 30, batch_size=64)
    
    stream_batches = mnist_batches + kmnist_batches + fmnist_batches
    print(f"Constructed stream with {len(stream_batches)} batches of size 64.")
    
    # Initialize coefficients for each named parameter
    coefficients = {}
    # Store them as PyTorch parameters to easily track gradients
    model_template = get_resnet18_grayscale()
    for name, param in model_template.named_parameters():
        if param.requires_grad:
            # Initialize with uniform coefficients [0.5, 0.5]
            coef_tensor = torch.tensor([0.5, 0.5], device=device, requires_grad=True)
            coefficients[name] = coef_tensor
            
    # Keep flat dicts of expert parameters/buffers
    model1_params = {name: param.clone() for name, param in model1.named_parameters()}
    model2_params = {name: param.clone() for name, param in model2.named_parameters()}
    model1_buffers = {name: buf.clone() for name, buf in model1.named_buffers()}
    model2_buffers = {name: buf.clone() for name, buf in model2.named_buffers()}
    
    # Unpack model template parameters/buffers to avoid mutating original expert models
    merged_model = get_resnet18_grayscale().to(device)
    
    # Online prototypes for the novel domain (initialized on first detection)
    novel_prototypes = None
    
    # Cohesion thresholds
    tau_N = 0.35 if args.method == "IGGS-OW" or args.method == "Rob-OW" else 0.5
    tau_p = 0.9
    temperature = 0.1
    
    # Keep track of results
    results_acc = []
    results_domain = [] # 'MNIST', 'KMNIST', 'FMNIST'
    results_routed = [] # Routed task: 0=MNIST, 1=KMNIST, 2=Novel
    
    # Run test-time adaptation stream
    for batch_idx, (inputs, targets) in enumerate(stream_batches):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Determine current ground-truth domain
        if batch_idx < 30:
            gt_domain = "MNIST"
        elif batch_idx < 60:
            gt_domain = "KMNIST"
        else:
            gt_domain = "FMNIST"
            
        # 1. Construct the merged model parameters and buffers
        merged_params = {}
        for name in model1_params:
            if name in coefficients:
                l1, l2 = coefficients[name][0], coefficients[name][1]
                merged_params[name] = l1 * model1_params[name] + l2 * model2_params[name]
            else:
                merged_params[name] = model1_params[name]
                
        merged_buffers = {}
        # Rob-OW does autograd-detached BN buffer merging!
        if args.method == "Rob-OW":
            for name in model1_buffers:
                if "running_mean" in name or "running_var" in name:
                    base_name = name.replace(".running_mean", "").replace(".running_var", "")
                    weight_name = base_name + ".weight"
                    if weight_name in coefficients:
                        l1, l2 = coefficients[weight_name][0], coefficients[weight_name][1]
                        l1, l2 = l1.detach(), l2.detach()
                        merged_buffers[name] = l1 * model1_buffers[name] + l2 * model2_buffers[name]
                else:
                    merged_buffers[name] = model1_buffers[name]
        else:
            # Standard methods do not merge BN statistics, they use expert 1's or default buffers
            for name in model1_buffers:
                merged_buffers[name] = model1_buffers[name]
                
        # 2. Extract features using a forward hook on avgpool combined with functional_call on merged_model
        features_list = []
        def hook_fn(module, input, output):
            features_list.append(torch.flatten(output, 1))
            
        hook = merged_model.avgpool.register_forward_hook(hook_fn)
        
        # Under Rob-OW, we activate AdaBN (train mode for BN layers) during forward passes of the novel domain!
        def set_bn_mode(model, training=False):
            model.eval()
            if training:
                for m in model.modules():
                    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                        m.train()
                        
        if args.method == "Rob-OW" and gt_domain == "FMNIST":
            set_bn_mode(merged_model, training=True)
        else:
            set_bn_mode(merged_model, training=False)
            
        params_and_buffers = {**merged_params, **merged_buffers}
        
        # 3. Extract features and outputs
        logits = functional_call(merged_model, params_and_buffers, inputs)
        feats = features_list[0]
        hook.remove()
        
        # Centering
        current_l1 = coefficients['fc.weight'][0].item()
        current_l2 = coefficients['fc.weight'][1].item()
        mu_t = current_l1 * centroid1.to(device) + current_l2 * centroid2.to(device)
        centered_feats = feats - mu_t
        
        # Compute EBER entropy on individual experts to perform robust routing and novelty detection
        def get_entropy(model, x):
            model.eval()
            with torch.no_grad():
                out = model(x)
                p = torch.softmax(out, dim=1)
                ent = -torch.sum(p * torch.log(p + 1e-8), dim=1)
            return ent.mean().item()
            
        entropy1 = get_entropy(model1, inputs)
        entropy2 = get_entropy(model2, inputs)
        
        min_entropy = min(entropy1, entropy2)
        tau_E = 0.40 # Entropy threshold
        
        # Unbiased Routing / Novelty Detection
        if min_entropy >= tau_E:
            # Detected Novel Domain
            detected_domain = "Novel"
            routed_expert_idx = 2
        else:
            if entropy1 <= entropy2:
                detected_domain = "MNIST"
                routed_expert_idx = 0
            else:
                detected_domain = "KMNIST"
                routed_expert_idx = 1
                
        # For logging, we still compute dummy cohesion to avoid breaking downstream variables
        cohesion1 = 1.0 - entropy1
        cohesion2 = 1.0 - entropy2
        max_cohesion = 1.0 - min_entropy
                
        if batch_idx < 5:
            print(f"[DEBUG Batch {batch_idx+1}] gt={gt_domain} cohesion1={cohesion1:.4f} cohesion2={cohesion2:.4f} max_cohesion={max_cohesion:.4f} tau_N={tau_N} routed_expert={routed_expert_idx}")
            print(f"    feats_norm={feats.norm().item():.4f} mu_t_norm={mu_t.norm().item():.4f} centered_feats_norm={centered_feats.norm().item():.4f}")
                
        results_routed.append(routed_expert_idx)
        results_domain.append(gt_domain)
        
        # 4. Handle adaptation step
        if detected_domain == "Novel":
            # Adapt coefficients on the novel domain!
            # Initialize novel prototypes on first detection
            if novel_prototypes is None:
                # Start from zero or the average of known prototypes
                novel_prototypes = torch.zeros(10, 512, device=device)
                novel_counts = torch.zeros(10, device=device)
                
            # Compute pseudo-labels and confidence
            probs = F.softmax(logits, dim=1)
            confidences, pseudo_labels = probs.max(dim=1)
            
            # Online Prototype Generation
            # Update online prototypes using high-confidence samples
            high_conf_mask = (confidences > tau_p)
            if high_conf_mask.sum() > 0:
                high_conf_feats = centered_feats[high_conf_mask]
                high_conf_labels = pseudo_labels[high_conf_mask]
                
                for i in range(high_conf_feats.size(0)):
                    c = high_conf_labels[i].item()
                    zi = high_conf_feats[i].detach()
                    if novel_counts[c] == 0:
                        novel_prototypes[c] = zi / (zi.norm(p=2) + 1e-8)
                    else:
                        # EMA update
                        novel_prototypes[c] = (1.0 - 0.1) * novel_prototypes[c].detach() + 0.1 * (zi / (zi.norm(p=2) + 1e-8))
                        novel_prototypes[c] = novel_prototypes[c] / (novel_prototypes[c].norm(p=2) + 1e-8)
                    novel_counts[c] += 1
                    
            # Optimize coefficients via contrastive alignment loss
            if args.method in ["PROTO-TTMM", "IGGS-OW", "Rob-OW"]:
                # Filter high-confidence samples for loss
                loss_mask = (confidences > tau_p)
                if loss_mask.sum() > 0:
                    # Extract active samples
                    active_feats = centered_feats[loss_mask]
                    active_labels = pseudo_labels[loss_mask]
                    
                    # Compute similarities to novel prototypes
                    # active_feats shape: [N_active, 512], novel_prototypes shape: [10, 512]
                    # Compute similarity matrix: shape [N_active, 10]
                    sim_matrix = torch.zeros(active_feats.size(0), 10, device=device)
                    for i in range(active_feats.size(0)):
                        for c in range(10):
                            sim_matrix[i, c] = F.cosine_similarity(active_feats[i].unsqueeze(0), novel_prototypes[c].unsqueeze(0))
                            
                    # Contrastive alignment loss
                    # Softmax over similarity / temperature
                    prob_alignment = F.softmax(sim_matrix / temperature, dim=1)
                    # Cross entropy of prob_alignment to active_labels
                    loss = -torch.log(prob_alignment[torch.arange(active_feats.size(0)), active_labels] + 1e-8).mean()
                    
                    # Backprop to update coefficients!
                    # Setup zero grad
                    for name in coefficients:
                        coefficients[name].grad = None
                            
                    loss.backward()
                    
                    # Update coefficients
                    with torch.no_grad():
                        for name in coefficients:
                            grad = coefficients[name].grad
                            if grad is not None:
                                # Apply Fisher preconditioning
                                if args.method in ["IGGS-OW", "Rob-OW"]:
                                    # η_w = η * (F_w + 1e-6)^(-α)
                                    f_w = joint_fisher.get(name, 1.0)
                                    lr_w = args.lr * ((f_w + 1e-6) ** (-args.alpha))
                                else:
                                    # PROTO-TTMM uses uniform learning rate
                                    lr_w = args.lr
                                    
                                # Gradient update
                                coefficients[name].data -= lr_w * grad.data
                                # Project back to simplex
                                coefficients[name].data = project_simplex(coefficients[name].data)
                                
        else:
            # Routed to known expert
            # EMA update towards the routed expert's one-hot
            if args.method in ["PROTO-TTMM", "IGGS-OW", "Rob-OW"]:
                target_coef = torch.zeros(2, device=device)
                target_coef[routed_expert_idx] = 1.0
                with torch.no_grad():
                    for name in coefficients:
                        coefficients[name].data = (1.0 - 0.1) * coefficients[name].data + 0.1 * target_coef
                        
        # 5. Evaluate accuracy on this batch
        # For prediction, we use the classification head of the routed domain, or standard model logits
        # In multi-task setup, MNIST and KMNIST both have 10-way classification, so we can evaluate directly using logits.
        _, preds = logits.max(dim=1)
        correct_preds = preds.eq(targets).sum().item()
        batch_acc = 100.0 * correct_preds / inputs.size(0)
        results_acc.append(batch_acc)
        
    # Summarize results
    results_acc = np.array(results_acc)
    results_domain = np.array(results_domain)
    results_routed = np.array(results_routed)
    
    # Task accuracies
    mnist_mask = (results_domain == "MNIST")
    kmnist_mask = (results_domain == "KMNIST")
    fmnist_mask = (results_domain == "FMNIST")
    
    acc_mnist = results_acc[mnist_mask].mean()
    acc_kmnist = results_acc[kmnist_mask].mean()
    acc_fmnist = results_acc[fmnist_mask].mean()
    acc_overall = results_acc.mean()
    
    # Novelty Detection Statistics
    # MNIST and KMNIST batches are "known"
    # FMNIST batches are "novel"
    # NDR: % of FMNIST batches correctly flagged as novel (routed_expert_idx == 2)
    ndr_batches = (results_routed[fmnist_mask] == 2)
    ndr = 100.0 * ndr_batches.sum() / fmnist_mask.sum()
    
    # FPR: % of known batches incorrectly flagged as novel
    fpr_batches = (results_routed[~fmnist_mask] == 2)
    fpr = 100.0 * fpr_batches.sum() / (~fmnist_mask).sum()
    
    print("\n==================== STREAM SUMMARY ====================")
    print(f"MNIST Accuracy (known):  {acc_mnist:.2f}%")
    print(f"KMNIST Accuracy (known): {acc_kmnist:.2f}%")
    print(f"FashionMNIST Acc (novel): {acc_fmnist:.2f}%")
    print(f"Overall Stream Accuracy: {acc_overall:.2f}%")
    print(f"Novelty Detection Rate (NDR): {ndr:.2f}%")
    print(f"False Positive Rate (FPR):    {fpr:.2f}%")
    print("========================================================")
    
    # Save results to txt file for parsing
    os.makedirs("results", exist_ok=True)
    filename = f"results/result_{args.method}_{args.corruption}_seed{args.seed}.txt"
    with open(filename, "w") as f:
        f.write(f"Method: {args.method}\n")
        f.write(f"Corruption: {args.corruption}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"MNIST_Acc: {acc_mnist:.4f}\n")
        f.write(f"KMNIST_Acc: {acc_kmnist:.4f}\n")
        f.write(f"FashionMNIST_Acc: {acc_fmnist:.4f}\n")
        f.write(f"Overall_Acc: {acc_overall:.4f}\n")
        f.write(f"NDR: {ndr:.4f}\n")
        f.write(f"FPR: {fpr:.4f}\n")
    print(f"Saved results to {filename}")

if __name__ == "__main__":
    main()
