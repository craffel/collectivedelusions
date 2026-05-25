import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torch.func import functional_call
import numpy as np
import argparse
from models import get_resnet18_model

# Constants
NUM_CLASSES = 10
BATCH_SIZE = 64
NUM_BATCHS_PER_TASK = 50
NUM_TASKS = 3
TOTAL_BATCHES = NUM_BATCHS_PER_TASK * NUM_TASKS
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    mnist_test = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    fashion_test = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
    kmnist_test = datasets.KMNIST(root="./data", train=False, download=True, transform=transform)
    
    mnist_loader = DataLoader(mnist_test, batch_size=BATCH_SIZE, shuffle=False)
    fashion_loader = DataLoader(fashion_test, batch_size=BATCH_SIZE, shuffle=False)
    kmnist_loader = DataLoader(kmnist_test, batch_size=BATCH_SIZE, shuffle=False)
    
    return mnist_loader, fashion_loader, kmnist_loader

def construct_streams(mnist_loader, fashion_loader, kmnist_loader):
    mnist_iter = iter(mnist_loader)
    fashion_iter = iter(fashion_loader)
    kmnist_iter = iter(kmnist_loader)
    
    mnist_batches = [next(mnist_iter) for _ in range(NUM_BATCHS_PER_TASK)]
    fashion_batches = [next(fashion_iter) for _ in range(NUM_BATCHS_PER_TASK)]
    kmnist_batches = [next(kmnist_iter) for _ in range(NUM_BATCHS_PER_TASK)]
    
    sequential_stream = []
    for b in mnist_batches:
        sequential_stream.append((b[0], b[1], 0))
    for b in fashion_batches:
        sequential_stream.append((b[0], b[1], 1))
    for b in kmnist_batches:
        sequential_stream.append((b[0], b[1], 2))
        
    alternating_stream = []
    for i in range(NUM_BATCHS_PER_TASK):
        alternating_stream.append((mnist_batches[i][0], mnist_batches[i][1], 0))
        alternating_stream.append((fashion_batches[i][0], fashion_batches[i][1], 1))
        alternating_stream.append((kmnist_batches[i][0], kmnist_batches[i][1], 2))
        
    return sequential_stream, alternating_stream

def run_ttmm_adaptation(stream, algorithm, base_state_dict, expert_state_dicts, fisher_dicts, 
                         lr_global, alpha_damping, fisher_floor, num_steps_per_batch, beta_momentum):
    
    model = get_resnet18_model(num_classes=NUM_CLASSES, pretrained=False)
    model.load_state_dict(base_state_dict)
    model = model.to(DEVICE)
    model.eval()
    
    trainable_names = {name for name, _ in model.named_parameters()}
    fp_keys = [k for k in base_state_dict.keys() if k in trainable_names and base_state_dict[k].is_floating_point()]
    
    # Pre-calculate G
    G = {}
    epsilon_scale = 1e-6
    for name in fp_keys:
        mean_fisher_mnist = fisher_dicts[0][name].mean().item()
        mean_fisher_fashion = fisher_dicts[1][name].mean().item()
        mean_fisher_kmnist = fisher_dicts[2][name].mean().item()
        joint_fisher_mean = (mean_fisher_mnist + mean_fisher_fashion + mean_fisher_kmnist) / 3.0
        
        # Apply damping and clamping/floor
        raw_G = (joint_fisher_mean + epsilon_scale) ** alpha_damping
        G[name] = max(raw_G, fisher_floor)

    c = {name: torch.zeros(3, device=DEVICE, requires_grad=True) for name in fp_keys}
    v = {name: torch.zeros(3, device=DEVICE) for name in fp_keys}
    
    accuracies = []
    
    for step, (inputs, targets, task_idx) in enumerate(stream):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        
        if algorithm != "Uniform":
            for _ in range(num_steps_per_batch):
                for name in c:
                    if c[name].grad is not None:
                        c[name].grad.zero_()
                        
                merged_params = {}
                lambdas = {}
                for name in base_state_dict.keys():
                    if name in fp_keys:
                        l = torch.softmax(c[name], dim=0)
                        lambdas[name] = l
                        merged_params[name] = base_state_dict[name].to(DEVICE) + \
                                              l[0] * (expert_state_dicts[0][name].to(DEVICE) - base_state_dict[name].to(DEVICE)) + \
                                              l[1] * (expert_state_dicts[1][name].to(DEVICE) - base_state_dict[name].to(DEVICE)) + \
                                              l[2] * (expert_state_dicts[2][name].to(DEVICE) - base_state_dict[name].to(DEVICE))
                    else:
                        merged_params[name] = base_state_dict[name].to(DEVICE)
                        
                outputs = functional_call(model, merged_params, inputs)
                p = torch.softmax(outputs, dim=1)
                entropy = -torch.sum(p * torch.log(p + 1e-8), dim=1).mean()
                entropy.backward(retain_graph=True)
                
                grads = {name: c[name].grad.clone() for name in fp_keys if c[name].grad is not None}
                
                if "IGGS" in algorithm:
                    with torch.no_grad():
                        preds = outputs.argmax(dim=1)
                    unique_classes = preds.unique()
                    class_grads = {}
                    
                    for cls in unique_classes:
                        mask = (preds == cls)
                        if mask.sum() == 0:
                            continue
                        cls_inputs = inputs[mask]
                        cls_outputs = functional_call(model, merged_params, cls_inputs)
                        cls_p = torch.softmax(cls_outputs, dim=1)
                        cls_entropy = -torch.sum(cls_p * torch.log(cls_p + 1e-8), dim=1).mean()
                        
                        model.zero_grad()
                        for name in c:
                            if c[name].grad is not None:
                                c[name].grad.zero_()
                        cls_entropy.backward(retain_graph=True)
                        class_grads[cls.item()] = {name: c[name].grad.clone() for name in fp_keys if c[name].grad is not None}
                    
                    classes = list(class_grads.keys())
                    projected_grads = {cls: {name: g.clone() for name, g in class_grads[cls].items()} for cls in class_grads}
                    
                    for i in range(len(classes)):
                        for j in range(len(classes)):
                            if i != j:
                                ca, cb = classes[i], classes[j]
                                inner_prod = 0.0
                                norm_b = 0.0
                                for name in fp_keys:
                                    if name in projected_grads[ca] and name in projected_grads[cb]:
                                        g_aw = projected_grads[ca][name]
                                        g_bw = projected_grads[cb][name]
                                        term = G[name] * torch.dot(g_aw, g_bw)
                                        inner_prod += term.item()
                                        norm_b += (G[name] * torch.dot(g_bw, g_bw)).item()
                                        
                                if inner_prod < 0:
                                    for name in fp_keys:
                                        if name in projected_grads[ca] and name in projected_grads[cb]:
                                            projected_grads[ca][name] -= (inner_prod / (norm_b + 1e-8)) * projected_grads[cb][name]
                                            
                    g_final = {name: torch.zeros_like(c[name]) for name in fp_keys}
                    for name in fp_keys:
                        for cls in classes:
                            if name in projected_grads[cls]:
                                g_final[name] += projected_grads[cls][name]
                else:
                    g_final = grads
                    
                if "TMS" in algorithm:
                    inner_prod_v_g = 0.0
                    norm_g = 0.0
                    for name in fp_keys:
                        if name in v and name in g_final:
                            term = G[name] * torch.dot(v[name], g_final[name])
                            inner_prod_v_g += term.item()
                            norm_g += (G[name] * torch.dot(g_final[name], g_final[name])).item()
                    
                    if inner_prod_v_g < 0:
                        for name in fp_keys:
                            if name in v and name in g_final:
                                v[name] = v[name] - (inner_prod_v_g / (norm_g + 1e-8)) * g_final[name]
                                
                with torch.no_grad():
                    for name in fp_keys:
                        if name in g_final:
                            v[name] = beta_momentum * v[name] + g_final[name]
                            if algorithm == "AdaMerging":
                                lr_w = lr_global
                            else:
                                lr_w = lr_global / G[name]
                            c[name].copy_(c[name] - lr_w * v[name])
                            
        with torch.no_grad():
            eval_params = {}
            for name in base_state_dict.keys():
                if name in fp_keys:
                    l = torch.softmax(c[name], dim=0)
                    eval_params[name] = base_state_dict[name].to(DEVICE) + \
                                        l[0] * (expert_state_dicts[0][name].to(DEVICE) - base_state_dict[name].to(DEVICE)) + \
                                        l[1] * (expert_state_dicts[1][name].to(DEVICE) - base_state_dict[name].to(DEVICE)) + \
                                        l[2] * (expert_state_dicts[2][name].to(DEVICE) - base_state_dict[name].to(DEVICE))
                else:
                    eval_params[name] = base_state_dict[name].to(DEVICE)
            
            eval_outputs = functional_call(model, eval_params, inputs)
            _, predicted = eval_outputs.max(1)
            correct_cnt = predicted.eq(targets).sum().item()
            total_cnt = targets.size(0)
            accuracies.append(correct_cnt / total_cnt * 100)
            
    return np.mean(accuracies)

def main():
    parser = argparse.ArgumentParser(description="Sweep TTMM adaptation hyperparameters.")
    parser.add_argument("--lr_global", type=float, default=1e-3)
    parser.add_argument("--alpha_damping", type=float, default=1.0)
    parser.add_argument("--fisher_floor", type=float, default=1e-6)
    parser.add_argument("--num_steps_per_batch", type=int, default=1)
    parser.add_argument("--beta_momentum", type=float, default=0.9)
    args = parser.parse_args()
    
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = False
        print("Disabled cuDNN to avoid initialization errors.")
        
    # Load checkpoints
    base_state_dict = torch.load("checkpoints/base_model.pt", map_location=DEVICE)
    expert_state_dicts = [
        torch.load("checkpoints/mnist_expert.pt", map_location=DEVICE),
        torch.load("checkpoints/fashion_expert.pt", map_location=DEVICE),
        torch.load("checkpoints/kmnist_expert.pt", map_location=DEVICE)
    ]
    fisher_dicts = [
        torch.load("checkpoints/mnist_fisher.pt", map_location=DEVICE),
        torch.load("checkpoints/fashion_fisher.pt", map_location=DEVICE),
        torch.load("checkpoints/kmnist_fisher.pt", map_location=DEVICE)
    ]
    
    mnist_loader, fashion_loader, kmnist_loader = load_data()
    sequential_stream, alternating_stream = construct_streams(mnist_loader, fashion_loader, kmnist_loader)
    
    streams = {
        "Sequential": sequential_stream,
        "Alternating": alternating_stream
    }
    
    algorithms = [
        "Uniform",
        "AdaMerging",
        "FP-CA",
        "IGGS-Merge",
        "FP-CA + TMS (Ours)",
        "IGGS-Merge + TMS (Ours)"
    ]
    
    print(f"\nEvaluating with: lr_global={args.lr_global}, alpha_damping={args.alpha_damping}, fisher_floor={args.fisher_floor}, steps_per_batch={args.num_steps_per_batch}, beta_momentum={args.beta_momentum}")
    
    results = {}
    for stream_name, stream in streams.items():
        results[stream_name] = {}
        for algo in algorithms:
            avg_acc = run_ttmm_adaptation(stream, algo, base_state_dict, expert_state_dicts, fisher_dicts,
                                         args.lr_global, args.alpha_damping, args.fisher_floor,
                                         args.num_steps_per_batch, args.beta_momentum)
            results[stream_name][algo] = avg_acc
            
    print("\n" + "="*80)
    print(f"{'Algorithm':<25} | {'Sequential Acc (%)':<18} | {'Alternating Acc (%)':<18}")
    print("="*80)
    for algo in algorithms:
        print(f"{algo:<25} | {results['Sequential'][algo]:<18.2f} | {results['Alternating'][algo]:<18.2f}")
    print("="*80)

if __name__ == "__main__":
    main()
