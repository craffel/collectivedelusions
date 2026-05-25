import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from models import SimpleCNN
import numpy as np
import matplotlib.pyplot as plt
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
    mnist_expert.eval()
    fashion_expert.eval()
    fisher_m = {name: torch.zeros_like(p) for name, p in mnist_expert.named_parameters()}
    fisher_f = {name: torch.zeros_like(p) for name, p in fashion_expert.named_parameters()}
    
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
            
    joint_fisher = {}
    for name in fisher_m.keys():
        m_sens = (fisher_m[name] / count_m).mean().item()
        f_sens = (fisher_f[name] / count_f).mean().item()
        joint_fisher[name] = 0.5 * (m_sens + f_sens)
        
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

def get_merged_model(mnist_expert, fashion_expert, lambda_0):
    lambda_1 = 1.0 - lambda_0
    merged_model = SimpleCNN()
    merged_state = {}
    state_0 = mnist_expert.state_dict()
    state_1 = fashion_expert.state_dict()
    for key in state_0.keys():
        if 'running_mean' in key or 'running_var' in key:
            merged_state[key] = lambda_0 * state_0[key] + lambda_1 * state_1[key]
        elif 'num_batches_tracked' in key:
            merged_state[key] = state_0[key]
        else:
            merged_state[key] = lambda_0 * state_0[key] + lambda_1 * state_1[key]
    merged_model.load_state_dict(merged_state)
    return merged_model

def adapt_and_record(mnist_expert, fashion_expert, prototypes_mnist, prototypes_fashion, stream_batches, method="fixed", beta=1.5, tau_base=1200.0, use_init=True):
    recorded_lambdas = []
    recorded_taus = []
    
    for batch_idx, (data, target, task_label, noise_flag) in enumerate(stream_batches):
        with torch.no_grad():
            _, features_mnist = mnist_expert(data, return_features=True)
            _, features_fashion = fashion_expert(data, return_features=True)
            
        d_mnist = compute_batch_distance(features_mnist, prototypes_mnist)
        d_fashion = compute_batch_distance(features_fashion, prototypes_fashion)
        
        s_mnist = -d_mnist
        s_fashion = -d_fashion
        
        if method == "fixed":
            tau = tau_base
        elif method == "cpr-dts":
            d_min = min(d_mnist, d_fashion)
            d_max = max(d_mnist, d_fashion)
            ratio = d_min / (d_max + 1e-5)
            tau = tau_base * (ratio ** 2)
            tau = max(100.0, min(tau, tau_base))
            
        recorded_taus.append(tau)
        w_prior = stable_softmax(s_mnist, s_fashion, tau)
        
        if use_init:
            p = w_prior[0].item()
            p = max(1e-4, min(p, 1 - 1e-4))
            w_init = np.log(p / (1.0 - p))
            w_param = torch.tensor([w_init], requires_grad=True)
        else:
            w_param = torch.tensor([0.0], requires_grad=True)
            
        for step in range(5):
            eps = 1e-3
            with torch.no_grad():
                l_0_plus = torch.sigmoid(w_param + eps)
                m_plus = get_merged_model(mnist_expert, fashion_expert, l_0_plus.item())
                loss_plus = entropy_loss(m_plus(data)) + beta * torch.sum(w_prior * torch.log((w_prior + 1e-12) / (torch.cat([l_0_plus, 1.0 - l_0_plus]) + 1e-12)))
                
                l_0_minus = torch.sigmoid(w_param - eps)
                m_minus = get_merged_model(mnist_expert, fashion_expert, l_0_minus.item())
                loss_minus = entropy_loss(m_minus(data)) + beta * torch.sum(w_prior * torch.log((w_prior + 1e-12) / (torch.cat([l_0_minus, 1.0 - l_0_minus]) + 1e-12)))
                
                grad = (loss_plus - loss_minus) / (2 * eps)
            
            with torch.no_grad():
                w_param -= 0.1 * grad
                
        with torch.no_grad():
            final_lambda_0 = torch.sigmoid(w_param).item()
            recorded_lambdas.append(final_lambda_0)
            
    return recorded_lambdas, recorded_taus

def adapt_dlw_fisher(mnist_expert, fashion_expert, prototypes_mnist, prototypes_fashion, stream_batches, norm_joint_fisher):
    recorded_lambdas = []
    base_model = SimpleCNN()
    params_0 = dict(mnist_expert.named_parameters())
    params_1 = dict(fashion_expert.named_parameters())
    buffers_0 = dict(mnist_expert.named_buffers())
    buffers_1 = dict(fashion_expert.named_buffers())
    
    for batch_idx, (data, target, task_label, noise_flag) in enumerate(stream_batches):
        with torch.no_grad():
            _, features_mnist = mnist_expert(data, return_features=True)
            _, features_fashion = fashion_expert(data, return_features=True)
            
        d_mnist = compute_batch_distance(features_mnist, prototypes_mnist)
        d_fashion = compute_batch_distance(features_fashion, prototypes_fashion)
        
        s_mnist = -d_mnist
        s_fashion = -d_fashion
        
        d_min = min(d_mnist, d_fashion)
        d_max = max(d_mnist, d_fashion)
        ratio = d_min / (d_max + 1e-5)
        tau = 1200.0 * (ratio ** 2)
        tau = max(100.0, min(tau, 1200.0))
        
        w_prior = stable_softmax(s_mnist, s_fashion, tau)
        p = w_prior[0].item()
        p = max(1e-4, min(p, 1 - 1e-4))
        w_init = np.log(p / (1.0 - p))
        
        logits_dict = {}
        for name in params_0.keys():
            logits_dict[name] = torch.tensor([w_init], dtype=torch.float32, requires_grad=True)
            
        beta = 1.5
        lr = 0.05
        
        for step in range(5):
            merged_params = {}
            lambdas = {name: torch.sigmoid(logits_dict[name]) for name in params_0.keys()}
            lamb_vals = [lambdas[name] for name in params_0.keys()]
            avg_lambda = torch.stack(lamb_vals).mean()
            
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
                    
            outputs = tf.functional_call(base_model, (merged_params, merged_buffers), data)
            loss_ent = entropy_loss(outputs)
            
            loss_kl = 0.0
            for name in params_0.keys():
                l_val = lambdas[name]
                loss_kl += w_prior[0] * torch.log((w_prior[0] + 1e-12) / (l_val + 1e-12)) + w_prior[1] * torch.log((w_prior[1] + 1e-12) / ((1.0 - l_val) + 1e-12))
            loss_kl = loss_kl / len(params_0)
            
            loss = loss_ent + beta * loss_kl
            grads = torch.autograd.grad(loss, list(logits_dict.values()))
            
            for idx, name in enumerate(params_0.keys()):
                g = grads[idx]
                f_sens = norm_joint_fisher[name]
                scaled_lr = lr / (f_sens + 1e-2)
                with torch.no_grad():
                    logits_dict[name] -= scaled_lr * g
                    
        with torch.no_grad():
            lambdas_final = {name: torch.sigmoid(logits_dict[name]).item() for name in params_0.keys()}
            avg_l = np.mean(list(lambdas_final.values()))
            recorded_lambdas.append(avg_l)
            
    return recorded_lambdas

def adapt_clw_fisher(mnist_expert, fashion_expert, prototypes_mnist, prototypes_fashion, stream_batches, norm_joint_fisher):
    recorded_lambdas = []
    base_model = SimpleCNN()
    params_0 = dict(mnist_expert.named_parameters())
    params_1 = dict(fashion_expert.named_parameters())
    buffers_0 = dict(mnist_expert.named_buffers())
    buffers_1 = dict(fashion_expert.named_buffers())
    
    for batch_idx, (data, target, task_label, noise_flag) in enumerate(stream_batches):
        with torch.no_grad():
            _, features_mnist = mnist_expert(data, return_features=True)
            _, features_fashion = fashion_expert(data, return_features=True)
            
        d_mnist = compute_batch_distance(features_mnist, prototypes_mnist)
        d_fashion = compute_batch_distance(features_fashion, prototypes_fashion)
        
        s_mnist = -d_mnist
        s_fashion = -d_fashion
        
        d_min = min(d_mnist, d_fashion)
        d_max = max(d_mnist, d_fashion)
        ratio = d_min / (d_max + 1e-5)
        tau = 1200.0 * (ratio ** 2)
        tau = max(100.0, min(tau, 1200.0))
        
        w_prior = stable_softmax(s_mnist, s_fashion, tau)
        p = w_prior[0].item()
        p = max(1e-4, min(p, 1 - 1e-4))
        w_init = np.log(p / (1.0 - p))
        
        # CLW uses a global logit and layer-wise offsets
        global_logit = torch.tensor([w_init], dtype=torch.float32, requires_grad=True)
        offsets_dict = {name: torch.tensor([0.0], dtype=torch.float32, requires_grad=True) for name in params_0.keys()}
        
        beta = 1.5
        lr = 0.05
        
        for step in range(5):
            lambdas = {name: torch.sigmoid(global_logit + offsets_dict[name]) for name in params_0.keys()}
            lamb_vals = [lambdas[name] for name in params_0.keys()]
            avg_lambda = torch.stack(lamb_vals).mean()
            
            merged_params = {}
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
                    
            outputs = tf.functional_call(base_model, (merged_params, merged_buffers), data)
            loss_ent = entropy_loss(outputs)
            
            loss_kl = 0.0
            for name in params_0.keys():
                l_val = lambdas[name]
                loss_kl += w_prior[0] * torch.log((w_prior[0] + 1e-12) / (l_val + 1e-12)) + w_prior[1] * torch.log((w_prior[1] + 1e-12) / ((1.0 - l_val) + 1e-12))
            loss_kl = loss_kl / len(params_0)
            
            loss = loss_ent + beta * loss_kl
            coherence_penalty = 0.02 * sum((off**2).sum() for off in offsets_dict.values())
            loss = loss + coherence_penalty
            
            all_tensors = [global_logit] + list(offsets_dict.values())
            grads = torch.autograd.grad(loss, all_tensors)
            g_global = grads[0]
            grads_offsets = grads[1:]
            
            with torch.no_grad():
                global_logit -= lr * g_global
                
            for idx, name in enumerate(params_0.keys()):
                g_off = grads_offsets[idx]
                f_sens = norm_joint_fisher[name]
                scaled_lr = lr / (f_sens + 1e-2)
                with torch.no_grad():
                    offsets_dict[name] -= scaled_lr * g_off
                    
        with torch.no_grad():
            lambdas_final = {name: torch.sigmoid(global_logit + offsets_dict[name]).item() for name in params_0.keys()}
            avg_l = np.mean(list(lambdas_final.values()))
            recorded_lambdas.append(avg_l)
            
    return recorded_lambdas

def adapt_clw_fisher_scts(mnist_expert, fashion_expert, prototypes_mnist, prototypes_fashion, stream_batches, norm_joint_fisher):
    recorded_lambdas = []
    base_model = SimpleCNN()
    params_0 = dict(mnist_expert.named_parameters())
    params_1 = dict(fashion_expert.named_parameters())
    buffers_0 = dict(mnist_expert.named_buffers())
    buffers_1 = dict(fashion_expert.named_buffers())
    
    for batch_idx, (data, target, task_label, noise_flag) in enumerate(stream_batches):
        with torch.no_grad():
            _, features_mnist = mnist_expert(data, return_features=True)
            _, features_fashion = fashion_expert(data, return_features=True)
            
        d_mnist = compute_batch_distance(features_mnist, prototypes_mnist)
        d_fashion = compute_batch_distance(features_fashion, prototypes_fashion)
        
        s_mnist = -d_mnist
        s_fashion = -d_fashion
        
        # SCTS Formulation
        gap = abs(d_mnist - d_fashion)
        tau = gap / 3.0 + 150.0
        tau = max(10.0, tau)
        
        w_prior = stable_softmax(s_mnist, s_fashion, tau)
        p = w_prior[0].item()
        p = max(1e-4, min(p, 1 - 1e-4))
        w_init = np.log(p / (1.0 - p))
        
        global_logit = torch.tensor([w_init], dtype=torch.float32, requires_grad=True)
        offsets_dict = {name: torch.tensor([0.0], dtype=torch.float32, requires_grad=True) for name in params_0.keys()}
        
        beta = 1.5
        lr = 0.05
        
        for step in range(5):
            lambdas = {name: torch.sigmoid(global_logit + offsets_dict[name]) for name in params_0.keys()}
            lamb_vals = [lambdas[name] for name in params_0.keys()]
            avg_lambda = torch.stack(lamb_vals).mean()
            
            merged_params = {}
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
                    
            outputs = tf.functional_call(base_model, (merged_params, merged_buffers), data)
            loss_ent = entropy_loss(outputs)
            
            loss_kl = 0.0
            for name in params_0.keys():
                l_val = lambdas[name]
                loss_kl += w_prior[0] * torch.log((w_prior[0] + 1e-12) / (l_val + 1e-12)) + w_prior[1] * torch.log((w_prior[1] + 1e-12) / ((1.0 - l_val) + 1e-12))
            loss_kl = loss_kl / len(params_0)
            
            loss = loss_ent + beta * loss_kl
            coherence_penalty = 0.02 * sum((off**2).sum() for off in offsets_dict.values())
            loss = loss + coherence_penalty
            
            all_tensors = [global_logit] + list(offsets_dict.values())
            grads = torch.autograd.grad(loss, all_tensors)
            g_global = grads[0]
            grads_offsets = grads[1:]
            
            with torch.no_grad():
                global_logit -= lr * g_global
                
            for idx, name in enumerate(params_0.keys()):
                g_off = grads_offsets[idx]
                f_sens = norm_joint_fisher[name]
                scaled_lr = lr / (f_sens + 1e-2)
                with torch.no_grad():
                    offsets_dict[name] -= scaled_lr * g_off
                    
        with torch.no_grad():
            lambdas_final = {name: torch.sigmoid(global_logit + offsets_dict[name]).item() for name in params_0.keys()}
            avg_l = np.mean(list(lambdas_final.values()))
            recorded_lambdas.append(avg_l)
            
    return recorded_lambdas

if __name__ == "__main__":
    # Load models
    mnist_expert = SimpleCNN()
    mnist_expert.load_state_dict(torch.load("expert_mnist.pt"))
    fashion_expert = SimpleCNN()
    fashion_expert.load_state_dict(torch.load("expert_fashion.pt"))
    
    # Precompute prototypes
    cal_loader_mnist = get_dataset_loader(datasets.MNIST, is_train=True, subset_size=256)
    cal_loader_fashion = get_dataset_loader(datasets.FashionMNIST, is_train=True, subset_size=256)
    
    prototypes_mnist = compute_prototypes(mnist_expert, cal_loader_mnist)
    prototypes_fashion = compute_prototypes(fashion_expert, cal_loader_fashion)
    
    norm_joint_fisher = compute_joint_fisher(mnist_expert, fashion_expert, cal_loader_mnist, cal_loader_fashion)
    
    # Create test stream
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
        
    print("Simulating for plots...")
    
    fixed_reset_lambdas, _ = adapt_and_record(
        mnist_expert, fashion_expert, prototypes_mnist, prototypes_fashion, stream_batches, method="fixed", beta=1.5, use_init=False
    )
    
    cpr_init_lambdas, cpr_init_taus = adapt_and_record(
        mnist_expert, fashion_expert, prototypes_mnist, prototypes_fashion, stream_batches, method="cpr-dts", beta=1.5, use_init=True
    )
    
    dlw_fisher_lambdas = adapt_dlw_fisher(
        mnist_expert, fashion_expert, prototypes_mnist, prototypes_fashion, stream_batches, norm_joint_fisher
    )
    
    clw_fisher_lambdas = adapt_clw_fisher(
        mnist_expert, fashion_expert, prototypes_mnist, prototypes_fashion, stream_batches, norm_joint_fisher
    )
    
    clw_fisher_scts_lambdas = adapt_clw_fisher_scts(
        mnist_expert, fashion_expert, prototypes_mnist, prototypes_fashion, stream_batches, norm_joint_fisher
    )
    
    # Create plot of Lambdas
    plt.figure(figsize=(10, 3.5))
    plt.rcParams.update({'font.size': 11, 'font.family': 'sans-serif'})
    
    x = np.arange(len(stream_batches))
    
    plt.plot(x, fixed_reset_lambdas, label='Fixed Temp + Reset (Baseline)', color='#d62728', linestyle='--', alpha=0.8, linewidth=1.8)
    plt.plot(x, cpr_init_lambdas, label='CPR-DTS + PG-Init (Global)', color='#ff7f0e', linestyle='-.', alpha=0.9, linewidth=1.8)
    plt.plot(x, dlw_fisher_lambdas, label='DLW-Fisher', color='#1f77b4', linestyle=':', alpha=0.9, linewidth=2.0)
    plt.plot(x, clw_fisher_lambdas, label='CLW-Fisher (CPR-DTS)', color='#bcbd22', linestyle='-.', alpha=0.9, linewidth=2.0)
    plt.plot(x, clw_fisher_scts_lambdas, label='CLW-Fisher + SCTS (Ours)', color='#800080', linestyle='-', linewidth=2.5)
    
    plt.axvspan(0, 10, color='#e2f0d9', alpha=0.3, label='Clean MNIST')
    plt.axvspan(10, 20, color='#fce4d6', alpha=0.3, label='Noisy MNIST')
    plt.axvspan(20, 30, color='#fff2cc', alpha=0.3, label='Clean Fashion')
    plt.axvspan(30, 40, color='#eddcfc', alpha=0.3, label='Noisy Fashion')
    plt.axvspan(40, 50, color='#e1f5fe', alpha=0.3, label='Novel KMNIST')
    
    for boundary in [10, 20, 30, 40]:
        plt.axvline(boundary, color='gray', linestyle=':', linewidth=1)
        
    plt.xlabel('Test Stream Batch Index')
    plt.ylabel(r'MNIST Expert Merging Coefficient ($\lambda_0$)')
    plt.title('Weight-Space Merging Coefficient Trajectory Under Continuous Stream')
    plt.ylim(-0.05, 1.05)
    plt.xlim(0, 50)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), ncol=2, frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    plt.savefig('template/trajectory_lambda.png', dpi=300)
    plt.savefig('template/trajectory_lambda.pdf')
    plt.close()
    print("Saved trajectory_lambda.png and trajectory_lambda.pdf")
    
    plt.figure(figsize=(10, 2.2))
    plt.plot(x, cpr_init_taus, label=r'Dynamic Temperature $\tau(X^{(t)})$', color='#2ca02c', linewidth=2.5)
    
    plt.axvspan(0, 10, color='#e2f0d9', alpha=0.3)
    plt.axvspan(10, 20, color='#fce4d6', alpha=0.3)
    plt.axvspan(20, 30, color='#fff2cc', alpha=0.3)
    plt.axvspan(30, 40, color='#eddcfc', alpha=0.3)
    plt.axvspan(40, 50, color='#e1f5fe', alpha=0.3)
    
    for boundary in [10, 20, 30, 40]:
        plt.axvline(boundary, color='gray', linestyle=':', linewidth=1)
        
    plt.xlabel('Test Stream Batch Index')
    plt.ylabel('Routing Softmax Temperature ($\tau$)')
    plt.title('CPR-DTS Dynamic Temperature Adaptation')
    plt.ylim(0, 1300)
    plt.xlim(0, 50)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig('template/trajectory_tau.png', dpi=300)
    plt.savefig('template/trajectory_tau.pdf')
    plt.close()
    print("Saved trajectory_tau.png and trajectory_tau.pdf")
