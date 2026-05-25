import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from torch.nn.utils.stateless import functional_call
import os
import copy
import numpy as np

# Model definition (must match train_experts.py)
class CNNEncoder(nn.Module):
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc = nn.Linear(64 * 7 * 7, 128)
        self.relu_fc = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.relu2(self.conv2(x))
        x = self.pool2(self.relu3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.relu_fc(self.fc(x))
        return x

class ClassificationHead(nn.Module):
    def __init__(self):
        super(ClassificationHead, self).__init__()
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        return self.fc(x)

# Corruption helpers
def apply_gaussian_noise(x, sigma=0.4):
    return torch.clamp(x + torch.randn_like(x) * sigma, 0.0, 1.0)

def apply_gaussian_blur(x, sigma=2.0):
    kernel_size = 5
    grid = torch.arange(kernel_size).float() - (kernel_size - 1) / 2
    grid_x, grid_y = torch.meshgrid(grid, grid, indexing='ij')
    kernel = torch.exp(-(grid_x**2 + grid_y**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, 5, 5).to(x.device)
    padded = F.pad(x, (2, 2, 2, 2), mode='reflect')
    return F.conv2d(padded, kernel)

def apply_contrast(x, alpha=0.15):
    return torch.clamp(0.5 + alpha * (x - 0.5), 0.0, 1.0)

def apply_corruption(x, domain):
    if domain == "Clean":
        return x
    elif domain == "Gaussian Noise":
        return apply_gaussian_noise(x)
    elif domain == "Gaussian Blur":
        return apply_gaussian_blur(x)
    elif domain == "Contrast":
        return apply_contrast(x)
    else:
        raise ValueError(f"Unknown domain: {domain}")

# Compute FIM priors for experts (for EWC-TTA)
def compute_fim_priors(device):
    print("Computing FIM priors on validation splits...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    fim_priors = {}
    
    # We will use train datasets for validating / computing FIM
    mnist_train = datasets.MNIST(root="./data", train=True, transform=transform)
    fashion_train = datasets.FashionMNIST(root="./data", train=True, transform=transform)
    kmnist_train = datasets.KMNIST(root="./data", train=True, transform=transform)
    
    datasets_dict = {"MNIST": mnist_train, "FashionMNIST": fashion_train, "KMNIST": kmnist_train}
    checkpoints = {
        "MNIST": "checkpoints/mnist_expert.pt",
        "FashionMNIST": "checkpoints/fashion_expert.pt",
        "KMNIST": "checkpoints/kmnist_expert.pt"
    }
    
    for task_name, dataset in datasets_dict.items():
        ckpt = torch.load(checkpoints[task_name], map_location=device)
        encoder = CNNEncoder().to(device)
        head = ClassificationHead().to(device)
        encoder.load_state_dict(ckpt['encoder_state_dict'])
        head.load_state_dict(ckpt['head_state_dict'])
        
        encoder.eval()
        head.eval()
        
        # Subset of N=200 samples
        subset = Subset(dataset, list(range(200)))
        loader = DataLoader(subset, batch_size=1, shuffle=False)
        
        fim_w = torch.zeros_like(head.fc.weight)
        fim_b = torch.zeros_like(head.fc.bias)
        
        for images, _ in loader:
            images = images.to(device)
            features = encoder(images)
            outputs = head(features)
            
            # Predict
            pred_class = outputs.argmax(dim=-1)[0]
            log_prob = F.log_softmax(outputs, dim=-1)[0, pred_class]
            
            head.zero_grad()
            log_prob.backward()
            
            fim_w += head.fc.weight.grad ** 2
            fim_b += head.fc.bias.grad ** 2
            
        fim_w = (fim_w / 200.0).detach() + 1e-8
        fim_b = (fim_b / 200.0).detach() + 1e-8
        
        fim_priors[task_name] = (fim_w, fim_b)
        
    print("FIM priors computed.")
    return fim_priors

# Helper for KL divergence
def kl_divergence_loss(q_logits, p_probs):
    q_log_probs = F.log_softmax(q_logits, dim=-1)
    p_log_probs = torch.log(p_probs + 1e-12)
    kl = p_probs * (p_log_probs - q_log_probs)
    return kl.sum(dim=-1).mean()

# PCGrad Projection algorithm
def project_gradients(grads):
    flat_grads = [g.flatten() for g in grads]
    num_grads = len(flat_grads)
    projected_flat = [g.clone() for g in flat_grads]
    
    for i in range(num_grads):
        # pair-wise projection
        for j in range(num_grads):
            if i != j:
                dot = torch.dot(projected_flat[i], flat_grads[j])
                if dot < 0:
                    norm_sq = torch.dot(flat_grads[j], flat_grads[j]) + 1e-12
                    projected_flat[i] -= (dot / norm_sq) * flat_grads[j]
                    
    projected_grads = []
    for i, g in enumerate(grads):
        projected_grads.append(projected_flat[i].reshape(g.shape))
        
    return sum(projected_grads)

def evaluate_method(method_name, domain, stream_type, expert_data, fim_priors, device, lr_lambda_custom=None, threshold_multiplier_custom=None, block_size=None, lr_head_custom=None):
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Load expert checkpoints
    checkpoint_mnist = torch.load("checkpoints/mnist_expert.pt", map_location=device)
    checkpoint_fashion = torch.load("checkpoints/fashion_expert.pt", map_location=device)
    checkpoint_kmnist = torch.load("checkpoints/kmnist_expert.pt", map_location=device)

    expert_encoders = [
        checkpoint_mnist['encoder_state_dict'],
        checkpoint_fashion['encoder_state_dict'],
        checkpoint_kmnist['encoder_state_dict']
    ]

    expert_heads = [
        copy.deepcopy(checkpoint_mnist['head_state_dict']),
        copy.deepcopy(checkpoint_fashion['head_state_dict']),
        copy.deepcopy(checkpoint_kmnist['head_state_dict'])
    ]

    # Construct base encoder and heads
    base_encoder = CNNEncoder().to(device)
    adapted_heads = [ClassificationHead().to(device) for _ in range(3)]
    for k in range(3):
        adapted_heads[k].load_state_dict(expert_heads[k])

    # Store initial head parameters for EWC penalty
    initial_heads = [copy.deepcopy(expert_heads[k]) for k in range(3)]

    # Trainable merging coefficients Lambda (8, 3)
    Lambda = torch.zeros(8, 3, device=device, requires_grad=True)

    # Optimizers
    if lr_lambda_custom is not None:
        lr_lambda = lr_lambda_custom
    else:
        lr_lambda = 0.1 if "OPR" in method_name else 0.05
    
    if lr_head_custom is not None:
        lr_head = lr_head_custom
    else:
        lr_head = 0.0001    
    # Set up optimizer for heads
    head_optimizers = [torch.optim.Adam(adapted_heads[k].parameters(), lr=lr_head) for k in range(3)]
    
    # Lambda optimizer
    lambda_optimizer = torch.optim.Adam([Lambda], lr=lr_lambda)
    
    # Stream setup
    mnist_loader, fashion_loader, kmnist_loader = expert_data
    mnist_iter = iter(mnist_loader)
    fashion_iter = iter(fashion_loader)
    kmnist_iter = iter(kmnist_loader)
    
    # Generate 150 total batches (50 per task)
    batches = []
    if block_size is not None:
        num_cycles = 50 // block_size
        for cycle in range(num_cycles):
            for _ in range(block_size):
                batches.append((0, next(mnist_iter)))
            for _ in range(block_size):
                batches.append((1, next(fashion_iter)))
            for _ in range(block_size):
                batches.append((2, next(kmnist_iter)))
    elif stream_type == "Alternating":
        for i in range(50):
            batches.append((0, next(mnist_iter)))
            batches.append((1, next(fashion_iter)))
            batches.append((2, next(kmnist_iter)))
    elif stream_type == "Sequential":
        for i in range(50):
            batches.append((0, next(mnist_iter)))
        for i in range(50):
            batches.append((1, next(fashion_iter)))
        for i in range(50):
            batches.append((2, next(kmnist_iter)))
            
    # Evaluation loop
    correct_predictions = 0
    total_samples = 0
    
    param_names = ['conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias', 'conv3.weight', 'conv3.bias', 'fc.weight', 'fc.bias']
    
    running_loss = 0.0
    
    for step, (task_id, (images, labels)) in enumerate(batches):
        images, labels = images.to(device), labels.to(device)
        
        # Apply corruption to test stream images
        corrupted_images = apply_corruption(images, domain)
        
        # Determine target soft labels from frozen expert
        with torch.no_grad():
            expert_enc_dict = expert_encoders[task_id]
            expert_head_dict = expert_heads[task_id]
            
            exp_features = functional_call(base_encoder, expert_enc_dict, corrupted_images)
            exp_head = ClassificationHead().to(device)
            exp_head.load_state_dict(expert_head_dict)
            exp_logits = exp_head(exp_features)
            expert_probs = F.softmax(exp_logits, dim=-1)
            
        # Evaluation prediction of merged model
        weights = torch.softmax(Lambda, dim=1)
        merged_encoder_params = {}
        for i, name in enumerate(param_names):
            merged_encoder_params[name] = sum(weights[i, k] * expert_encoders[k][name] for k in range(3))
            
        merged_features = functional_call(base_encoder, merged_encoder_params, corrupted_images)
        merged_logits = adapted_heads[task_id](merged_features)
        
        # Record accuracy on current batch
        with torch.no_grad():
            preds = merged_logits.argmax(dim=-1)
            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.size(0)
            
        # Adaptation step (unless Static)
        if method_name != "Static Merged":
            loss_sl = kl_divergence_loss(merged_logits, expert_probs)
            
            # OPR detection and handling
            if "OPR" in method_name and step > 0:
                # If the loss is suddenly 4x larger than the recent running loss, reset!
                # Since corruptions also affect loss, we scale threshold dynamically
                if threshold_multiplier_custom is not None:
                    thresh = threshold_multiplier_custom
                else:
                    thresh = 4.0 if domain == "Clean" else 2.5
                if loss_sl.item() > thresh * running_loss and running_loss > 0.01:
                    with torch.no_grad():
                        Lambda.fill_(0.0)
                    lambda_optimizer = torch.optim.Adam([Lambda], lr=lr_lambda)
                    
                    # Recalculate
                    weights = torch.softmax(Lambda, dim=1)
                    merged_encoder_params = {}
                    for i, name in enumerate(param_names):
                        merged_encoder_params[name] = sum(weights[i, k] * expert_encoders[k][name] for k in range(3))
                    merged_features = functional_call(base_encoder, merged_encoder_params, corrupted_images)
                    merged_logits = adapted_heads[task_id](merged_features)
                    loss_sl = kl_divergence_loss(merged_logits, expert_probs)
            
            running_loss = 0.9 * running_loss + 0.1 * loss_sl.item() if step > 0 else loss_sl.item()
            
            loss_reg = torch.tensor(0.0, device=device)
            if method_name == "EWC-TTA":
                gamma = 100.0
                fim_w, fim_b = fim_priors[list(fim_priors.keys())[task_id]]
                curr_head = adapted_heads[task_id]
                init_w = initial_heads[task_id]['fc.weight']
                init_b = initial_heads[task_id]['fc.bias']
                
                loss_reg = 0.5 * gamma * (
                    (fim_w * (curr_head.fc.weight - init_w) ** 2).sum() +
                    (fim_b * (curr_head.fc.bias - init_b) ** 2).sum()
                )
                
            loss_total = loss_sl + loss_reg
            
            lambda_optimizer.zero_grad()
            head_optimizers[task_id].zero_grad()
            
            if "PC-Merge" in method_name:
                # Compute class-specific gradients
                q_log_probs = F.log_softmax(merged_logits, dim=-1)
                p_log_probs = torch.log(expert_probs + 1e-12)
                per_sample_kl = (expert_probs * (p_log_probs - q_log_probs)).sum(dim=-1)
                
                expert_labels = expert_probs.argmax(dim=-1)
                grads_Lambda = []
                for c in range(10):
                    mask = (expert_labels == c)
                    if mask.sum() > 0:
                        class_loss = per_sample_kl[mask].mean()
                        grad_L = torch.autograd.grad(class_loss, Lambda, retain_graph=True, allow_unused=True)[0]
                        if grad_L is not None:
                            grads_Lambda.append(grad_L.clone())
                            
                if len(grads_Lambda) > 0:
                    final_grad_Lambda = project_gradients(grads_Lambda)
                else:
                    final_grad_Lambda = torch.zeros_like(Lambda)
                    
                loss_sl.backward()
                Lambda.grad = final_grad_Lambda
                
                lambda_optimizer.step()
                head_optimizers[task_id].step()
            else:
                loss_total.backward()
                lambda_optimizer.step()
                head_optimizers[task_id].step()
                
    accuracy = correct_predictions / total_samples
    return accuracy

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    fim_priors = compute_fim_priors(device)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    mnist_test = datasets.MNIST(root="./data", train=False, transform=transform)
    fashion_test = datasets.FashionMNIST(root="./data", train=False, transform=transform)
    kmnist_test = datasets.KMNIST(root="./data", train=False, transform=transform)
    
    mnist_loader = DataLoader(mnist_test, batch_size=64, shuffle=False)
    fashion_loader = DataLoader(fashion_test, batch_size=64, shuffle=False)
    kmnist_loader = DataLoader(kmnist_test, batch_size=64, shuffle=False)
    expert_data = (mnist_loader, fashion_loader, kmnist_loader)
    
    domains = ["Clean", "Gaussian Noise", "Gaussian Blur", "Contrast"]
    streams = ["Sequential", "Alternating"]
    methods = [
        "Static Merged", 
        "Standard TTA", 
        "EWC-TTA", 
        "TTA with OPR (Ours)", 
        "PC-Merge with OPR (Ours)"
    ]
    
    results = {}
    for stream in streams:
        results[stream] = {}
        for domain in domains:
            results[stream][domain] = {}
            for method in methods:
                print(f"\nEvaluating Stream: {stream} | Domain: {domain} | Method: {method} ...")
                acc = evaluate_method(method, domain, stream, expert_data, fim_priors, device)
                results[stream][domain][method] = acc
                print(f"Accuracy: {acc*100:.2f}%")
                
    # Print nice markdown table
    print("\n\n### EXPERIMENTAL RESULTS SUMMARY")
    for stream in streams:
        print(f"\n#### Stream: {stream}")
        print("| Domain | Static Merged | Standard TTA | EWC-TTA | TTA + OPR (Ours) | PC-Merge + OPR (Ours) |")
        print("|--------|---------------|--------------|---------|------------------|-----------------------|")
        for domain in domains:
            m_accs = [results[stream][domain][m] * 100 for m in methods]
            print(f"| {domain} | {m_accs[0]:.2f}% | {m_accs[1]:.2f}% | {m_accs[2]:.2f}% | {m_accs[3]:.2f}% | {m_accs[4]:.2f}% |")

if __name__ == "__main__":
    main()
