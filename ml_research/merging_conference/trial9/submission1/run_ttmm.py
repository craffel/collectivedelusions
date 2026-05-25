import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import copy
import argparse
import os

# Patch BasicBlock.forward to be out-of-place to support backward hooks perfectly
def basic_block_forward_patched(self, x):
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
        identity = self.downsample(x)

    # Out-of-place addition instead of out += identity
    out = out + identity
    out = self.relu(out)

    return out

torchvision.models.resnet.BasicBlock.forward = basic_block_forward_patched

class ChannelReductionResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        # Load standard ResNet-18 pretrained weights
        self.resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Disable in-place ReLU to avoid any autograd conflicts
        for m in self.resnet.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = False
        
        # Surgery on conv1 to reduce from 3 channels to 1 channel
        old_conv1 = self.resnet.conv1
        new_conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv1.out_channels,
            kernel_size=old_conv1.kernel_size,
            stride=old_conv1.stride,
            padding=old_conv1.padding,
            bias=old_conv1.bias is not None
        )
        # Sum the pretrained weights across the channel dimension
        with torch.no_grad():
            new_conv1.weight.copy_(old_conv1.weight.sum(dim=1, keepdim=True))
        self.resnet.conv1 = new_conv1
        
        # Replace classification head
        self.resnet.fc = nn.Linear(512, 10)

    def forward(self, x):
        return self.resnet(x)

def get_test_stream():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    mnist_test = torchvision.datasets.MNIST("./data", train=False, download=True, transform=transform)
    fashion_test = torchvision.datasets.FashionMNIST("./data", train=False, download=True, transform=transform)
    kmnist_test = torchvision.datasets.KMNIST("./data", train=False, download=True, transform=transform)
    
    g = torch.Generator()
    g.manual_seed(42)
    
    mnist_loader = DataLoader(mnist_test, batch_size=64, shuffle=True, generator=g)
    fashion_loader = DataLoader(fashion_test, batch_size=64, shuffle=True, generator=g)
    kmnist_loader = DataLoader(kmnist_test, batch_size=64, shuffle=True, generator=g)
    
    mnist_iter = iter(mnist_loader)
    fashion_iter = iter(fashion_loader)
    kmnist_iter = iter(kmnist_loader)
    
    stream = []
    
    # Phase 0: Clean MNIST (batches 0-9)
    for _ in range(10):
        x, y = next(mnist_iter)
        stream.append((x, y, "Clean MNIST"))
        
    # Phase 1: Noisy MNIST (batches 10-19)
    for _ in range(10):
        x, y = next(mnist_iter)
        noise = torch.randn_like(x) * 0.6
        x_noisy = torch.clamp(x + noise, -1.0, 1.0)
        stream.append((x_noisy, y, "Noisy MNIST"))
        
    # Phase 2: Clean FashionMNIST (batches 20-29)
    for _ in range(10):
        x, y = next(fashion_iter)
        stream.append((x, y, "Clean Fashion"))
        
    # Phase 3: Noisy FashionMNIST (batches 30-39)
    for _ in range(10):
        x, y = next(fashion_iter)
        noise = torch.randn_like(x) * 0.6
        x_noisy = torch.clamp(x + noise, -1.0, 1.0)
        stream.append((x_noisy, y, "Noisy Fashion"))
        
    # Phase 4: Novel KMNIST (batches 40-49)
    for _ in range(10):
        x, y = next(kmnist_iter)
        stream.append((x, y, "Novel KMNIST"))
        
    return stream

def fuse_bn_buffers(model, expert0, expert1, w0, w1):
    for (name, m), (_, m0), (_, m1) in zip(model.named_modules(), expert0.named_modules(), expert1.named_modules()):
        if isinstance(m, nn.BatchNorm2d):
            with torch.no_grad():
                mu0 = m0.running_mean
                var0 = m0.running_var
                mu1 = m1.running_mean
                var1 = m1.running_var
                
                mu_fused = w0 * mu0 + w1 * mu1
                var_fused = w0 * (var0 + (mu0 - mu_fused)**2) + w1 * (var1 + (mu1 - mu_fused)**2)
                
                m.running_mean.copy_(mu_fused)
                m.running_var.copy_(var_fused)

def merge_parameters_in_place(model, expert0_state, expert1_state, w_global, deltas, mergeable_params):
    for name in mergeable_params:
        d = deltas[name]
        lambda_j = torch.sigmoid(w_global + d)
        
        # Blend the weights differentiably
        merged_val = (1 - lambda_j) * expert0_state[name] + lambda_j * expert1_state[name]
        
        # Call retain_grad() on the non-leaf tensor to preserve its gradients
        if merged_val.requires_grad:
            merged_val.retain_grad()
        
        # Get parent module and parameter name
        parts = name.split('.')
        submodule = model
        for part in parts[:-1]:
            submodule = getattr(submodule, part)
        param_name = parts[-1]
        
        # Re-assign as attribute to keep gradients flowing
        if hasattr(submodule, param_name):
            delattr(submodule, param_name)
        setattr(submodule, param_name, merged_val)

def compute_entropy(logits):
    probs = F.softmax(logits, dim=-1)
    return - (probs * torch.log(probs + 1e-5)).sum(dim=-1).mean()

def main():
    parser = argparse.ArgumentParser(description="Test-Time Model Merging Evaluation")
    parser.add_argument("--method", type=str, default="bk_comerge", choices=["static", "bk_comerge", "vakp_bc", "uniform"],
                        help="Merging and adaptation method")
    parser.add_argument("--lr", type=float, default=0.05, help="Test-time learning rate (eta)")
    parser.add_argument("--steps", type=int, default=5, help="Adaptation steps per batch")
    parser.add_argument("--coherence_weight", type=float, default=0.05, help="Consensus coherence regularization weight (gamma_c)")
    parser.add_argument("--variance_weight", type=float, default=0.1, help="Variance-aware scaling factor (gamma_v)")
    parser.add_argument("--beta", type=float, default=1.0, help="KL regularization loss coefficient")
    parser.add_argument("--novelty_threshold", type=float, default=1.6, help="Entropy threshold for novel domains")
    parser.add_argument("--confidence_scale", type=float, default=2.0, help="SCTS scale factor s")
    parser.add_argument("--ema_smoothing", type=float, default=0.9, help="EMA smoothing factor for entropy gap")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating {args.method.upper()} on device {device}...")

    # Load specialized expert models
    model0 = ChannelReductionResNet18().to(device)
    model1 = ChannelReductionResNet18().to(device)
    
    model0.load_state_dict(torch.load("models/expert_mnist.pt", map_location=device))
    model1.load_state_dict(torch.load("models/expert_fashion.pt", map_location=device))
    
    model0.eval()
    model1.eval()
    
    # Save original parameters as static tensors for blending
    expert0_state = {k: v.clone().detach() for k, v in model0.state_dict().items()}
    expert1_state = {k: v.clone().detach() for k, v in model1.state_dict().items()}

    # Initialize the merged model
    merged_model = ChannelReductionResNet18().to(device)
    
    # Identify mergeable parameters (Conv, Linear, BatchNorm parameters)
    mergeable_params = []
    for name, param in merged_model.named_parameters():
        if "weight" in name or "bias" in name:
            mergeable_params.append(name)
            
    # Initialize optimization variables
    w_global = torch.tensor(0.0, requires_grad=True, device=device)
    deltas = {name: torch.tensor(0.0, requires_grad=True, device=device) for name in mergeable_params}
    
    # Running sensitivities for preconditioning
    running_sens = {name: 1.0 for name in mergeable_params}
    
    # State variables for temporal routing
    delta_smoothed = None
    
    # Get test stream of batches
    test_stream = get_test_stream()
    
    phase_accuracies = {}
    phase_counts = {}
    
    overall_correct = 0
    overall_total = 0
    
    print("Starting streaming evaluation...")
    
    for b_idx, (images, labels, phase_name) in enumerate(test_stream):
        images, labels = images.to(device), labels.to(device)
        
        # 1. Compute expert prediction entropies (using static expert models)
        with torch.no_grad():
            outputs0 = model0(images)
            outputs1 = model1(images)
            H0 = compute_entropy(outputs0)
            H1 = compute_entropy(outputs1)
            H_bar = (H0 + H1) / 2.0
            
        # 2. Soft routing prior computation
        if H_bar > args.novelty_threshold:
            w = [0.5, 0.5]
        else:
            entropy_gap = abs(H0.item() - H1.item())
            if delta_smoothed is None:
                delta_smoothed = entropy_gap
            else:
                delta_smoothed = args.ema_smoothing * delta_smoothed + (1 - args.ema_smoothing) * entropy_gap
                
            tau_self = delta_smoothed / args.confidence_scale + 1e-5
            w0 = torch.exp(-H0 / tau_self)
            w1 = torch.exp(-H1 / tau_self)
            w_sum = w0 + w1
            w = [(w0 / w_sum).item(), (w1 / w_sum).item()]
            
        # 3. Fuse Batch Normalization running statistics
        fuse_bn_buffers(merged_model, model0, model1, w[0], w[1])
        target_p = w[1] # target probability of expert 1
        
        # If method is static or uniform, we don't adapt; we just use the weights to merge parameters
        if args.method in ["static", "uniform"]:
            if args.method == "uniform":
                fuse_bn_buffers(merged_model, model0, model1, 0.5, 0.5)
                merge_parameters_in_place(merged_model, expert0_state, expert1_state, torch.tensor(0.0, device=device), {name: torch.tensor(0.0, device=device) for name in mergeable_params}, mergeable_params)
            else:
                clamped_p = torch.clamp(torch.tensor(target_p, device=device), 1e-5, 1.0 - 1e-5)
                merge_parameters_in_place(merged_model, expert0_state, expert1_state, torch.tensor(0.0, device=device), {name: torch.logit(clamped_p) for name in mergeable_params}, mergeable_params)
            
            with torch.no_grad():
                merged_model.eval()
                final_outputs = merged_model(images)
                _, preds = final_outputs.max(1)
                correct = preds.eq(labels).sum().item()
                total = labels.size(0)
        else:
            # Test-time adaptation (BK-CoMerge or VAKP-BC)
            merged_model.train() # Set to train to enable gradient computation (autograd)
            
            # Reset and initialize optimization variables for this batch using target_p
            with torch.no_grad():
                clamped_p = torch.clamp(torch.tensor(target_p, device=device), 1e-4, 1.0 - 1e-4)
                w_global_init = torch.logit(clamped_p).item()
                w_global.copy_(torch.tensor(w_global_init, device=device))
                for name in mergeable_params:
                    deltas[name].zero_()
            
            for step in range(args.steps):
                # We will register hooks to capture exact input activations and output pre-activation gradients
                activations = {}
                grads = {}
                handles = []
                
                for name, module in merged_model.named_modules():
                    if isinstance(module, (nn.Conv2d, nn.Linear)):
                        def get_forward_hook(m_name):
                            def hook(m, inp, out):
                                activations[m_name] = inp[0].detach()
                            return hook
                        def get_backward_hook(m_name):
                            def hook(m, grad_in, grad_out):
                                if grad_out[0] is not None:
                                    grads[m_name] = grad_out[0].detach()
                            return hook
                        
                        h_f = module.register_forward_hook(get_forward_hook(name))
                        h_b = module.register_full_backward_hook(get_backward_hook(name))
                        handles.extend([h_f, h_b])
                
                # Construct merged model parameters with w_global and deltas
                merge_parameters_in_place(merged_model, expert0_state, expert1_state, w_global, deltas, mergeable_params)
                
                # Forward pass
                outputs = merged_model(images)
                
                # Compute prediction entropy loss
                L_entropy = compute_entropy(outputs)
                
                # Compute KL regularization to prior
                lambda_list = []
                for name in mergeable_params:
                    lambda_list.append(torch.sigmoid(w_global + deltas[name]))
                lambda_mean = torch.stack(lambda_list).mean()
                
                # Compute numerically stable KL
                lambda_mean_clamped = torch.clamp(lambda_mean, 1e-5, 1.0 - 1e-5)
                target_p_clamped = torch.clamp(torch.tensor(target_p, device=device), 1e-5, 1.0 - 1e-5)
                L_kl = (1.0 - lambda_mean_clamped) * torch.log((1.0 - lambda_mean_clamped) / (1.0 - target_p_clamped)) + \
                       lambda_mean_clamped * torch.log(lambda_mean_clamped / target_p_clamped)
                       
                # Normalize sensitivities globally
                sum_sens = sum(running_sens.values()) + 1e-8
                normalized_sens = {k: v / sum_sens for k, v in running_sens.items()}
                
                # Compute coherence regularization loss
                L_coherence = 0.0
                for name in mergeable_params:
                    L_coherence += normalized_sens[name] * deltas[name].pow(2)
                L_coherence = args.coherence_weight * L_coherence
                
                # Total adaptation loss
                loss = L_entropy + args.beta * L_kl + L_coherence
                
                # Backward pass
                loss.backward()
                
                # Update running sensitivities based on current parameter gradients, activations, and pre-activation grads
                with torch.no_grad():
                    new_sens = {}
                    for name in mergeable_params:
                        # Find the actual parameter tensor
                        parts = name.split('.')
                        submodule = merged_model
                        for part in parts[:-1]:
                            submodule = getattr(submodule, part)
                        param_name = parts[-1]
                        param = getattr(submodule, param_name)
                        
                        # Parent module name
                        parent_name = '.'.join(parts[:-1])
                        
                        if parent_name in activations and parent_name in grads:
                            act = activations[parent_name]
                            grad = grads[parent_name]
                            num_params = param.numel()
                            
                            # Compute sample-wise L2 norms
                            act_sample_norm2 = act.pow(2).sum(dim=list(range(1, act.dim())))
                            grad_sample_norm2 = grad.pow(2).sum(dim=list(range(1, grad.dim())))
                            
                            sample_Fj = (act_sample_norm2 * grad_sample_norm2) / num_params
                            mean_Fj = sample_Fj.mean().item()
                            
                            Fj = mean_Fj
                            
                            # VAKP-BC: Variance-Aware scaling using actual sample-wise Kronecker trace sensitivity
                            if args.method == "vakp_bc":
                                std_Fj = sample_Fj.std().item()
                                cv = std_Fj / (mean_Fj + 1e-5)
                                Fj = Fj * (1.0 + args.variance_weight * cv)
                                
                            new_sens[name] = Fj
                        else:
                            # Fallback for non-Conv/Linear layers (such as BatchNorm)
                            if param.grad is not None:
                                new_sens[name] = param.grad.pow(2).mean().item()
                            else:
                                new_sens[name] = running_sens[name]
                            
                    # Update running sensitivities
                    for name in mergeable_params:
                        running_sens[name] = 0.9 * running_sens[name] + 0.1 * new_sens[name]
                            
                # Remove hooks to free memory
                for h in handles:
                    h.remove()
                    
                # Perform preconditioned optimization step
                with torch.no_grad():
                    # Update w_global (non-preconditioned)
                    if w_global.grad is not None:
                        w_global -= args.lr * w_global.grad
                        w_global.grad.zero_()
                        
                    # Update deltas (preconditioned)
                    for name in mergeable_params:
                        if deltas[name].grad is not None:
                            d_grad = deltas[name].grad
                            preconditioner = normalized_sens[name] + 1e-5
                            deltas[name] -= args.lr * (1.0 / preconditioner) * d_grad
                            deltas[name].grad.zero_()
                            
            # Run final forward pass in eval mode for classification
            with torch.no_grad():
                merged_model.eval()
                # Apply optimized parameters
                merge_parameters_in_place(merged_model, expert0_state, expert1_state, w_global, deltas, mergeable_params)
                
                final_outputs = merged_model(images)
                _, preds = final_outputs.max(1)
                correct = preds.eq(labels).sum().item()
                total = labels.size(0)
                
        # Record results
        acc = 100.0 * correct / total
        overall_correct += correct
        overall_total += total
        
        if phase_name not in phase_accuracies:
            phase_accuracies[phase_name] = 0.0
            phase_counts[phase_name] = 0
        phase_accuracies[phase_name] += acc
        phase_counts[phase_name] += 1
        
        if (b_idx + 1) % 5 == 0 or b_idx == 49:
            print(f"Batch {b_idx+1}/50 | Phase: {phase_name} | Accuracy: {acc:.2f}% | w_global: {w_global.item():.4f}")
            
    print("\n" + "="*40)
    print(f"Evaluation Summary for {args.method.upper()}:")
    print("="*40)
    for p_name in phase_accuracies:
        avg_phase_acc = phase_accuracies[p_name] / phase_counts[p_name]
        print(f"Phase {p_name:<15} : {avg_phase_acc:.2f}%")
    overall_acc = 100.0 * overall_correct / overall_total
    print(f"Overall Accuracy      : {overall_acc:.2f}%")
    print("="*40 + "\n")
    
if __name__ == "__main__":
    main()
