import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from models import SimpleCNN
import numpy as np
import torch.func as tf

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def add_gaussian_noise(tensor, mean=0., std=0.6):
    return tensor + torch.randn(tensor.size()) * std + mean

def get_dataset_loader(dataset_class, is_train=False, subset_size=1000, noise=False):
    transform_list = [transforms.ToTensor()]
    if noise:
        transform_list.append(transforms.Normalize((0.1307,), (0.3081,)))
        transform_list.append(transforms.Lambda(lambda x: add_gaussian_noise(x, std=0.6)))
    else:
        transform_list.append(transforms.Normalize((0.1307,), (0.3081,)))
        
    transform = transforms.Compose(transform_list)
    dataset = dataset_class(root='data', train=is_train, download=True, transform=transform)
    subset = Subset(dataset, range(min(subset_size, len(dataset))))
    loader = DataLoader(subset, batch_size=64, shuffle=False)
    return loader

def compute_prototypes(model, loader):
    model.eval()
    features_list = [[] for _ in range(10)]
    with torch.no_grad():
        for data, target in loader:
            _, features = model(data, return_features=True)
            for f, t in zip(features, target):
                features_list[t.item()].append(f)
                
    prototypes = []
    for c in range(10):
        if len(features_list[c]) > 0:
            stacked = torch.stack(features_list[c])
            prototypes.append(stacked.mean(0))
        else:
            prototypes.append(torch.zeros(128))
    return torch.stack(prototypes)

def compute_batch_distance(batch_features, prototypes):
    dists = torch.cdist(batch_features, prototypes, p=2) ** 2
    min_dists, _ = dists.min(dim=1)
    return min_dists.mean().item()

def compute_joint_fisher(mnist_expert, fashion_expert, loader_mnist, loader_fashion):
    # Compute tensor-wise sensitivities on calibration data
    mnist_expert.eval()
    fashion_expert.eval()
    
    fisher_m = {name: torch.zeros_like(p) for name, p in mnist_expert.named_parameters()}
    fisher_f = {name: torch.zeros_like(p) for name, p in fashion_expert.named_parameters()}
    
    # MNIST expert Fisher
    count_m = 0
    for data, _ in loader_mnist:
        with torch.no_grad():
            outputs = mnist_expert(data)
            pseudo_labels = outputs.argmax(dim=1)
        for i in range(data.size(0)):
            mnist_expert.zero_grad()
            single_data = data[i:i+1]
            single_label = pseudo_labels[i:i+1]
            loss = F.cross_entropy(mnist_expert(single_data), single_label)
            loss.backward()
            for name, p in mnist_expert.named_parameters():
                if p.grad is not None:
                    fisher_m[name] += p.grad.data ** 2
            count_m += 1
            
    # Fashion expert Fisher
    count_f = 0
    for data, _ in loader_fashion:
        with torch.no_grad():
            outputs = fashion_expert(data)
            pseudo_labels = outputs.argmax(dim=1)
        for i in range(data.size(0)):
            fashion_expert.zero_grad()
            single_data = data[i:i+1]
            single_label = pseudo_labels[i:i+1]
            loss = F.cross_entropy(fashion_expert(single_data), single_label)
            loss.backward()
            for name, p in fashion_expert.named_parameters():
                if p.grad is not None:
                    fisher_f[name] += p.grad.data ** 2
            count_f += 1
            
    # Calculate average sensitivities
    joint_fisher = {}
    for name in fisher_m.keys():
        m_sens = (fisher_m[name] / count_m).mean().item()
        f_sens = (fisher_f[name] / count_f).mean().item()
        joint_fisher[name] = 0.5 * (m_sens + f_sens)
        
    # Global normalization: divide by the mean sensitivity across all tensors
    mean_sens = np.mean(list(joint_fisher.values()))
    norm_joint_fisher = {name: joint_fisher[name] / mean_sens for name in joint_fisher.keys()}
    
    return norm_joint_fisher

def stable_softmax(s_0, s_1, tau):
    val_0 = s_0 / tau
    val_1 = s_1 / tau
    max_val = max(val_0, val_1)
    exp_0 = np.exp(val_0 - max_val)
    exp_1 = np.exp(val_1 - max_val)
    prob_0 = exp_0 / (exp_0 + exp_1)
    return torch.tensor([prob_0, 1.0 - prob_0], dtype=torch.float32)

def entropy_loss(logits):
    probs = F.softmax(logits, dim=1)
    return -torch.mean(torch.sum(probs * torch.log(probs + 1e-12), dim=1))

def run_simulation(mnist_expert, fashion_expert, prototypes_mnist, prototypes_fashion, stream_batches, norm_joint_fisher, method_config):
    # Configurations:
    # method_config: dict containing "method" ("fixed" or "cpr-dts"), "use_init" (bool), "layer_wise" (bool), "use_fisher" (bool)
    
    method = method_config["method"]
    use_init = method_config["use_init"]
    layer_wise = method_config["layer_wise"]
    use_fisher = method_config["use_fisher"]
    
    accuracies = []
    lambdas_log = []
    
    # Extract base expert weights and buffers
    base_model = SimpleCNN()
    params_0 = dict(mnist_expert.named_parameters())
    params_1 = dict(fashion_expert.named_parameters())
    buffers_0 = dict(mnist_expert.named_buffers())
    buffers_1 = dict(fashion_expert.named_buffers())
    
    # We will run TTA over stream
    for batch_idx, (data, target, task_label, noise_flag) in enumerate(stream_batches):
        # 1. Compute prototype routing prior
        with torch.no_grad():
            _, features_mnist = mnist_expert(data, return_features=True)
            _, features_fashion = fashion_expert(data, return_features=True)
            
        d_mnist = compute_batch_distance(features_mnist, prototypes_mnist)
        d_fashion = compute_batch_distance(features_fashion, prototypes_fashion)
        
        s_mnist = -d_mnist
        s_fashion = -d_fashion
        
        if method == "fixed":
            tau = 1200.0
        elif method == "cpr-dts":
            d_min = min(d_mnist, d_fashion)
            d_max = max(d_mnist, d_fashion)
            ratio = d_min / (d_max + 1e-5)
            tau = 1200.0 * (ratio ** 2)
            tau = max(100.0, min(tau, 1200.0))
            
        w_prior = stable_softmax(s_mnist, s_fashion, tau)
        p = w_prior[0].item()
        p = max(1e-4, min(p, 1 - 1e-4))
        w_init = np.log(p / (1.0 - p))
        
        # 2. Define learnable logits for merging
        if layer_wise:
            # Separate logit for each parameter tensor
            # Initialized based on routing prior or 0.0
            logits_dict = {}
            for name in params_0.keys():
                init_val = w_init if use_init else 0.0
                logits_dict[name] = torch.tensor([init_val], dtype=torch.float32, requires_grad=True)
        else:
            # Single global logit
            init_val = w_init if use_init else 0.0
            global_logit = torch.tensor([init_val], dtype=torch.float32, requires_grad=True)
            
        # Optimization settings
        beta = 1.5
        n_steps = 5
        lr = 0.1 if not layer_wise else 0.05
        
        # Optimization loop
        for step in range(n_steps):
            # Form merged parameters and buffers
            merged_params = {}
            
            # Form the tensor-wise coefficients
            if layer_wise:
                lambdas = {name: torch.sigmoid(logits_dict[name]) for name in params_0.keys()}
                # Detach for BN buffers as in DR-Fisher
                lamb_vals = [lambdas[name] for name in params_0.keys()]
                avg_lambda = torch.stack(lamb_vals).mean()
            else:
                l_val = torch.sigmoid(global_logit)
                lambdas = {name: l_val for name in params_0.keys()}
                avg_lambda = l_val
                
            for name in params_0.keys():
                merged_params[name] = lambdas[name] * params_0[name] + (1.0 - lambdas[name]) * params_1[name]
                
            merged_buffers = {}
            avg_lambda_detached = avg_lambda.detach()
            for name in buffers_0.keys():
                if 'running_mean' in name or 'running_var' in name:
                    merged_buffers[name] = avg_lambda_detached * buffers_0[name] + (1.0 - avg_lambda_detached) * buffers_1[name]
                elif 'num_batches_tracked' in name:
                    merged_buffers[name] = buffers_0[name]
                else:
                    merged_buffers[name] = avg_lambda_detached * buffers_0[name] + (1.0 - avg_lambda_detached) * buffers_1[name]
                    
            # Differentiable forward pass
            outputs = tf.functional_call(base_model, (merged_params, merged_buffers), data)
            
            # Adaptation Loss = Entropy + KL regularizer
            loss_ent = entropy_loss(outputs)
            
            loss_kl = 0.0
            if layer_wise:
                for name in params_0.keys():
                    l_val = lambdas[name]
                    loss_kl += w_prior[0] * torch.log((w_prior[0] + 1e-12) / (l_val + 1e-12)) + w_prior[1] * torch.log((w_prior[1] + 1e-12) / ((1.0 - l_val) + 1e-12))
                loss_kl = loss_kl / len(params_0)
            else:
                l_val = lambdas[list(params_0.keys())[0]]
                loss_kl = w_prior[0] * torch.log((w_prior[0] + 1e-12) / (l_val + 1e-12)) + w_prior[1] * torch.log((w_prior[1] + 1e-12) / ((1.0 - l_val) + 1e-12))
                
            loss = loss_ent + beta * loss_kl
            
            # Compute analytical gradients
            if layer_wise:
                # Zero gradients manually
                grads = torch.autograd.grad(loss, list(logits_dict.values()))
                for idx, name in enumerate(params_0.keys()):
                    g = grads[idx]
                    # Fisher preconditioning
                    if use_fisher:
                        f_sens = norm_joint_fisher[name]
                        # Inversely scale learning rate
                        scaled_lr = lr / (f_sens + 1e-2)
                    else:
                        scaled_lr = lr
                        
                    # SGD Step
                    with torch.no_grad():
                        logits_dict[name] -= scaled_lr * g
            else:
                g = torch.autograd.grad(loss, global_logit)[0]
                with torch.no_grad():
                    global_logit -= lr * g
                    
        # Final Evaluation after adaptation
        with torch.no_grad():
            merged_params = {}
            if layer_wise:
                lambdas = {name: torch.sigmoid(logits_dict[name]).item() for name in params_0.keys()}
                avg_l = np.mean(list(lambdas.values()))
            else:
                l_val = torch.sigmoid(global_logit).item()
                lambdas = {name: l_val for name in params_0.keys()}
                avg_l = l_val
                
            for name in params_0.keys():
                merged_params[name] = lambdas[name] * params_0[name] + (1.0 - lambdas[name]) * params_1[name]
                
            merged_buffers = {}
            for name in buffers_0.keys():
                if 'running_mean' in name or 'running_var' in name:
                    merged_buffers[name] = avg_l * buffers_0[name] + (1.0 - avg_l) * buffers_1[name]
                elif 'num_batches_tracked' in name:
                    merged_buffers[name] = buffers_0[name]
                else:
                    merged_buffers[name] = avg_l * buffers_0[name] + (1.0 - avg_l) * buffers_1[name]
                    
            final_outputs = tf.functional_call(base_model, (merged_params, merged_buffers), data)
            _, predicted = final_outputs.max(dim=1)
            correct = predicted.eq(target).sum().item()
            acc = 100. * correct / target.size(0)
            
            accuracies.append(acc)
            lambdas_log.append(avg_l)
            
    return accuracies, lambdas_log

if __name__ == "__main__":
    # Load expert models
    mnist_expert = SimpleCNN()
    mnist_expert.load_state_dict(torch.load("expert_mnist.pt"))
    fashion_expert = SimpleCNN()
    fashion_expert.load_state_dict(torch.load("expert_fashion.pt"))
    
    # Precompute calibration prototypes
    cal_loader_mnist = get_dataset_loader(datasets.MNIST, is_train=True, subset_size=256)
    cal_loader_fashion = get_dataset_loader(datasets.FashionMNIST, is_train=True, subset_size=256)
    
    prototypes_mnist = compute_prototypes(mnist_expert, cal_loader_mnist)
    prototypes_fashion = compute_prototypes(fashion_expert, cal_loader_fashion)
    
    # Precompute joint Fisher
    print("Precomputing Joint Fisher sensitivities...")
    norm_joint_fisher = compute_joint_fisher(mnist_expert, fashion_expert, cal_loader_mnist, cal_loader_fashion)
    
    # Create test stream (50 batches)
    loader_mnist_clean = get_dataset_loader(datasets.MNIST, is_train=False, subset_size=640, noise=False)
    loader_mnist_noisy = get_dataset_loader(datasets.MNIST, is_train=False, subset_size=640, noise=True)
    loader_fashion_clean = get_dataset_loader(datasets.FashionMNIST, is_train=False, subset_size=640, noise=False)
    loader_fashion_noisy = get_dataset_loader(datasets.FashionMNIST, is_train=False, subset_size=640, noise=True)
    loader_kmnist = get_dataset_loader(datasets.KMNIST, is_train=False, subset_size=640, noise=False)
    
    stream_batches = []
    for i, (data, target) in enumerate(loader_mnist_clean):
        stream_batches.append((data, target, 0, False))
    for i, (data, target) in enumerate(loader_mnist_noisy):
        stream_batches.append((data, target, 0, True))
    for i, (data, target) in enumerate(loader_fashion_clean):
        stream_batches.append((data, target, 1, False))
    for i, (data, target) in enumerate(loader_fashion_noisy):
        stream_batches.append((data, target, 1, True))
    for i, (data, target) in enumerate(loader_kmnist):
        stream_batches.append((data, target, 2, False))
        
    print(f"\nCreated non-stationary stream with {len(stream_batches)} batches.")
    
    # Define methods to run
    methods = {
        "Fixed + Reset": {"method": "fixed", "use_init": False, "layer_wise": False, "use_fisher": False},
        "Fixed + PG-Init": {"method": "fixed", "use_init": True, "layer_wise": False, "use_fisher": False},
        "CPR-DTS + Reset": {"method": "cpr-dts", "use_init": False, "layer_wise": False, "use_fisher": False},
        "CPR-DTS + PG-Init": {"method": "cpr-dts", "use_init": True, "layer_wise": False, "use_fisher": False},
        "DLW-Fisher (Ours)": {"method": "cpr-dts", "use_init": True, "layer_wise": True, "use_fisher": True}
    }
    
    all_results = {}
    
    segments = {
        "Clean MNIST (0-9)": (0, 10),
        "Noisy MNIST (10-19)": (10, 20),
        "Clean Fashion (20-29)": (20, 30),
        "Noisy Fashion (30-39)": (30, 40),
        "Novel KMNIST (40-49)": (40, 50)
    }
    
    for name, config in methods.items():
        print(f"\nRunning {name}...")
        accs, lambdas = run_simulation(
            mnist_expert, fashion_expert, prototypes_mnist, prototypes_fashion, stream_batches, norm_joint_fisher, config
        )
        all_results[name] = {"accs": accs, "lambdas": lambdas}
        
    # Print comparison table
    print("\n" + "="*125)
    print(f"{'SEGMENT':<25} | {'FIXED+RESET':<13} | {'FIXED+INIT':<13} | {'CPR+RESET':<13} | {'CPR+INIT':<13} | {'DLW-FISHER':<13}")
    print("="*125)
    
    for seg_name, (start, end) in segments.items():
        row_str = f"{seg_name:<25}"
        for name in methods.keys():
            avg_acc = np.mean(all_results[name]["accs"][start:end])
            row_str += f" | {avg_acc:.2f}%      "
        print(row_str)
    print("="*125)
    
    # Print average lambda table to see convergence
    print("\n" + "="*125)
    print(f"{'SEGMENT':<25} | {'FIXED+RESET':<13} | {'FIXED+INIT':<13} | {'CPR+RESET':<13} | {'CPR+INIT':<13} | {'DLW-FISHER':<13}")
    print("="*125)
    
    for seg_name, (start, end) in segments.items():
        row_str = f"{seg_name:<25}"
        for name in methods.keys():
            avg_l = np.mean(all_results[name]["lambdas"][start:end])
            row_str += f" | {avg_l:.4f}       "
        print(row_str)
    print("="*125)
    
    # Save results to a file
    with open("results_dlw.txt", "w") as f:
        f.write("Results Comparison Table:\n")
        f.write(f"{'Segment':<25}," + ",".join(methods.keys()) + "\n")
        for seg_name, (start, end) in segments.items():
            acc_strs = [f"{np.mean(all_results[name]['accs'][start:end]):.2f}" for name in methods.keys()]
            f.write(f"{seg_name:<25}," + ",".join(acc_strs) + "\n")
        f.write("\nLambda Convergence Table:\n")
        f.write(f"{'Segment':<25}," + ",".join(methods.keys()) + "\n")
        for seg_name, (start, end) in segments.items():
            l_strs = [f"{np.mean(all_results[name]['lambdas'][start:end]):.4f}" for name in methods.keys()]
            f.write(f"{seg_name:<25}," + ",".join(l_strs) + "\n")
