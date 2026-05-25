import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torch.func import functional_call
import matplotlib.pyplot as plt
import numpy as np
from models import get_resnet18_model

# Constants
NUM_CLASSES = 10
BATCH_SIZE = 64
NUM_BATCHS_PER_TASK = 50
NUM_TASKS = 3
TOTAL_BATCHES = NUM_BATCHS_PER_TASK * NUM_TASKS
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data():
    print("Loading test datasets...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load test datasets
    mnist_test = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    fashion_test = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
    kmnist_test = datasets.KMNIST(root="./data", train=False, download=True, transform=transform)
    
    # We create dataloaders with shuffle=False to make it completely deterministic and reproducible
    mnist_loader = DataLoader(mnist_test, batch_size=BATCH_SIZE, shuffle=False)
    fashion_loader = DataLoader(fashion_test, batch_size=BATCH_SIZE, shuffle=False)
    kmnist_loader = DataLoader(kmnist_test, batch_size=BATCH_SIZE, shuffle=False)
    
    return mnist_loader, fashion_loader, kmnist_loader

def construct_streams(mnist_loader, fashion_loader, kmnist_loader):
    print("Constructing test streams...")
    # Convert loaders to iterators to draw batches
    mnist_iter = iter(mnist_loader)
    fashion_iter = iter(fashion_loader)
    kmnist_iter = iter(kmnist_loader)
    
    # Fetch all needed batches
    mnist_batches = [next(mnist_iter) for _ in range(NUM_BATCHS_PER_TASK)]
    fashion_batches = [next(fashion_iter) for _ in range(NUM_BATCHS_PER_TASK)]
    kmnist_batches = [next(kmnist_iter) for _ in range(NUM_BATCHS_PER_TASK)]
    
    # 1. Sequential Stream
    # 50 batches MNIST, 50 batches FashionMNIST, 50 batches KMNIST
    sequential_stream = []
    for b in mnist_batches:
        sequential_stream.append((b[0], b[1], 0))  # (inputs, targets, task_idx)
    for b in fashion_batches:
        sequential_stream.append((b[0], b[1], 1))
    for b in kmnist_batches:
        sequential_stream.append((b[0], b[1], 2))
        
    # 2. Alternating Stream
    # Cycle through tasks: MNIST, Fashion, KMNIST, MNIST, Fashion, KMNIST, ...
    alternating_stream = []
    for i in range(NUM_BATCHS_PER_TASK):
        alternating_stream.append((mnist_batches[i][0], mnist_batches[i][1], 0))
        alternating_stream.append((fashion_batches[i][0], fashion_batches[i][1], 1))
        alternating_stream.append((kmnist_batches[i][0], kmnist_batches[i][1], 2))
        
    return sequential_stream, alternating_stream

def run_ttmm_adaptation(stream, algorithm, base_state_dict, expert_state_dicts, fisher_dicts, stream_name):
    print(f"\n--- Running {algorithm} on {stream_name} ---")
    
    # Load a clean base model
    model = get_resnet18_model(num_classes=NUM_CLASSES, pretrained=False)
    model.load_state_dict(base_state_dict)
    model = model.to(DEVICE)
    model.eval()
    
    # Extract floating-point trainable parameters (exclude non-trainable BN buffers)
    trainable_names = {name for name, _ in model.named_parameters()}
    fp_keys = [k for k in base_state_dict.keys() if k in trainable_names and base_state_dict[k].is_floating_point()]
    
    # Pre-calculate Fisher sensitivity prior for each named parameter
    G = {}
    epsilon_scale = 1e-6
    alpha_damping = 1.0
    for name in fp_keys:
        # Calculate joint sensitivity: mean over experts
        mean_fisher_mnist = fisher_dicts[0][name].mean().item()
        mean_fisher_fashion = fisher_dicts[1][name].mean().item()
        mean_fisher_kmnist = fisher_dicts[2][name].mean().item()
        joint_fisher_mean = (mean_fisher_mnist + mean_fisher_fashion + mean_fisher_kmnist) / 3.0
        # G_w metric
        G[name] = (joint_fisher_mean + epsilon_scale) ** alpha_damping

    # Initialize raw coefficients c for each floating-point parameter
    # Initializing to [0.0, 0.0, 0.0] ensures softmax is uniform [1/3, 1/3, 1/3]
    c = {name: torch.zeros(3, device=DEVICE, requires_grad=True) for name in fp_keys}
    
    # Initialize momentum buffer for each parameter
    v = {name: torch.zeros(3, device=DEVICE) for name in fp_keys}
    
    beta_momentum = 0.9
    lr_global = 1e-3
    
    # History for tracking
    accuracies = []
    coef_history = []  # List of dict of parameter-average coefficients
    
    # Task accuracies trackers
    task_correct = {0: 0, 1: 0, 2: 0}
    task_total = {0: 0, 1: 0, 2: 0}
    
    for step, (inputs, targets, task_idx) in enumerate(stream):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        
        # 1. Update coefficients before predicting, if adaptation is active
        if algorithm != "Uniform":
            # For each adaptation step, we perform self-supervised entropy minimization
            # Standard TTMM performs 1 gradient step per test batch
            
            # Since we need to compute gradients of c, we require grad
            for name in c:
                if c[name].grad is not None:
                    c[name].grad.zero_()
                    
            # Set up the differentiable merged state dict
            merged_params = {}
            lambdas = {}
            for name in base_state_dict.keys():
                if name in fp_keys:
                    # Differentiable softmax to ensure lambdas sum to 1 and are positive
                    l = torch.softmax(c[name], dim=0)
                    lambdas[name] = l
                    # Merge: base + sum(lambda_k * (expert_k - base))
                    merged_params[name] = base_state_dict[name].to(DEVICE) + \
                                          l[0] * (expert_state_dicts[0][name].to(DEVICE) - base_state_dict[name].to(DEVICE)) + \
                                          l[1] * (expert_state_dicts[1][name].to(DEVICE) - base_state_dict[name].to(DEVICE)) + \
                                          l[2] * (expert_state_dicts[2][name].to(DEVICE) - base_state_dict[name].to(DEVICE))
                else:
                    merged_params[name] = base_state_dict[name].to(DEVICE)
                    
            # Forward pass
            outputs = functional_call(model, merged_params, inputs)
            p = torch.softmax(outputs, dim=1)
            entropy = -torch.sum(p * torch.log(p + 1e-8), dim=1).mean()
            
            # Backward to get gradients w.r.t c
            entropy.backward(retain_graph=True)
            
            # Collect gradients
            grads = {name: c[name].grad.clone() for name in fp_keys if c[name].grad is not None}
            
            # Spatial surgery (IGGS-Merge)
            if "IGGS" in algorithm:
                # Obtain predicted pseudo-labels
                with torch.no_grad():
                    preds = outputs.argmax(dim=1)
                
                unique_classes = preds.unique()
                class_grads = {}
                
                # Compute class-specific entropy gradients
                for cls in unique_classes:
                    mask = (preds == cls)
                    if mask.sum() == 0:
                        continue
                    cls_inputs = inputs[mask]
                    
                    # Forward for this class subset
                    cls_outputs = functional_call(model, merged_params, cls_inputs)
                    cls_p = torch.softmax(cls_outputs, dim=1)
                    cls_entropy = -torch.sum(cls_p * torch.log(cls_p + 1e-8), dim=1).mean()
                    
                    # Backward to get class-specific gradients
                    model.zero_grad()
                    for name in c:
                        if c[name].grad is not None:
                            c[name].grad.zero_()
                    cls_entropy.backward(retain_graph=True)
                    
                    class_grads[cls.item()] = {name: c[name].grad.clone() for name in fp_keys if c[name].grad is not None}
                
                # Pairwise projection of class gradients in Riemannian space
                classes = list(class_grads.keys())
                projected_grads = {cls: {name: g.clone() for name, g in class_grads[cls].items()} for cls in class_grads}
                
                for i in range(len(classes)):
                    for j in range(len(classes)):
                        if i != j:
                            ca, cb = classes[i], classes[j]
                            # Riemannian inner product: \sum_w G_w * (g_a * g_b)
                            inner_prod = 0.0
                            norm_b = 0.0
                            for name in fp_keys:
                                if name in projected_grads[ca] and name in projected_grads[cb]:
                                    g_aw = projected_grads[ca][name]
                                    g_bw = projected_grads[cb][name]
                                    # G[name] is scalar metric weight
                                    term = G[name] * torch.dot(g_aw, g_bw)
                                    inner_prod += term.item()
                                    norm_b += (G[name] * torch.dot(g_bw, g_bw)).item()
                                    
                            if inner_prod < 0:
                                # Project ga onto normal plane of gb
                                for name in fp_keys:
                                    if name in projected_grads[ca] and name in projected_grads[cb]:
                                        projected_grads[ca][name] -= (inner_prod / (norm_b + 1e-8)) * projected_grads[cb][name]
                                        
                # Sum class gradients to obtain final spatial-conflict-free gradient
                g_final = {name: torch.zeros_like(c[name]) for name in fp_keys}
                for name in fp_keys:
                    for cls in classes:
                        if name in projected_grads[cls]:
                            g_final[name] += projected_grads[cls][name]
            else:
                # Standard gradient (no spatial surgery)
                g_final = grads
                
            # Temporal Momentum Surgery (TMS)
            if "TMS" in algorithm:
                # Compute Riemannian inner product between historical momentum v and current gradient g_final
                inner_prod_v_g = 0.0
                norm_g = 0.0
                for name in fp_keys:
                    if name in v and name in g_final:
                        term = G[name] * torch.dot(v[name], g_final[name])
                        inner_prod_v_g += term.item()
                        norm_g += (G[name] * torch.dot(g_final[name], g_final[name])).item()
                
                # If temporal conflict exists (inner product < 0), project momentum onto normal plane of gradient
                if inner_prod_v_g < 0:
                    for name in fp_keys:
                        if name in v and name in g_final:
                            v[name] = v[name] - (inner_prod_v_g / (norm_g + 1e-8)) * g_final[name]
                            
            # Update momentum and parameters
            with torch.no_grad():
                for name in fp_keys:
                    if name in g_final:
                        # Update momentum buffer: v = beta * v_projected + g_final
                        v[name] = beta_momentum * v[name] + g_final[name]
                        
                        # Set parameter learning rate. 
                        # Standard AdaMerging: uniform learning rate lr_global
                        # FP-CA/IGGS: preconditioned learning rate lr_w = lr_global * (G_w)^(-1)
                        if algorithm == "AdaMerging":
                            lr_w = lr_global
                        else:
                            # Preconditioned update
                            lr_w = lr_global / G[name]
                            
                        # Update c: c = c - lr_w * v
                        c[name].copy_(c[name] - lr_w * v[name])
                        
        # 2. Evaluate performance on the current batch using updated coefficients
        with torch.no_grad():
            # Apply final merged parameters for evaluation
            eval_params = {}
            step_lambdas_avg = torch.zeros(3, device=DEVICE)
            for name in base_state_dict.keys():
                if name in fp_keys:
                    l = torch.softmax(c[name], dim=0)
                    step_lambdas_avg += l
                    eval_params[name] = base_state_dict[name].to(DEVICE) + \
                                        l[0] * (expert_state_dicts[0][name].to(DEVICE) - base_state_dict[name].to(DEVICE)) + \
                                        l[1] * (expert_state_dicts[1][name].to(DEVICE) - base_state_dict[name].to(DEVICE)) + \
                                        l[2] * (expert_state_dicts[2][name].to(DEVICE) - base_state_dict[name].to(DEVICE))
                else:
                    eval_params[name] = base_state_dict[name].to(DEVICE)
            
            # Parameter-average coefficients for logging
            step_lambdas_avg /= len(fp_keys)
            coef_history.append(step_lambdas_avg.cpu().numpy().tolist())
            
            # Forward pass for eval
            eval_outputs = functional_call(model, eval_params, inputs)
            _, predicted = eval_outputs.max(1)
            correct_cnt = predicted.eq(targets).sum().item()
            total_cnt = targets.size(0)
            batch_acc = correct_cnt / total_cnt * 100
            accuracies.append(batch_acc)
            
            # Track by task
            task_correct[task_idx] += correct_cnt
            task_total[task_idx] += total_cnt
            
            if (step + 1) % 10 == 0 or step == 0:
                print(f"Step {step+1:03d}/{TOTAL_BATCHES} | Batch Acc: {batch_acc:.2f}% | "
                      f"Lambdas: [{step_lambdas_avg[0]:.3f}, {step_lambdas_avg[1]:.3f}, {step_lambdas_avg[2]:.3f}]")
                
    # Final metrics
    avg_acc = np.mean(accuracies)
    task_accs = {k: (task_correct[k]/task_total[k]*100 if task_total[k] > 0 else 0) for k in task_correct}
    print(f"[{algorithm} - {stream_name}] Finished. Average Acc: {avg_acc:.2f}%")
    for k, acc in task_accs.items():
        print(f"  Task {k} Acc: {acc:.2f}%")
        
    return {
        "avg_acc": avg_acc,
        "task_accs": task_accs,
        "accuracies": accuracies,
        "coef_history": coef_history
    }

def main():
    os.makedirs("results", exist_ok=True)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = False
        print("Disabled cuDNN to avoid initialization errors.")
    
    # Check if checkpoints exist, if not, raise error (we must train them first)
    required_checkpoints = [
        "checkpoints/base_model.pt",
        "checkpoints/mnist_expert.pt",
        "checkpoints/fashion_expert.pt",
        "checkpoints/kmnist_expert.pt",
        "checkpoints/mnist_fisher.pt",
        "checkpoints/fashion_fisher.pt",
        "checkpoints/kmnist_fisher.pt"
    ]
    for cp in required_checkpoints:
        if not os.path.exists(cp):
            raise FileNotFoundError(f"Required checkpoint {cp} not found. Please run pre-training and Fisher computation first!")

    # Load state dicts
    print("Loading checkpoints...")
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
    
    # Load data and construct streams
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
    
    results = {}
    
    for stream_name, stream in streams.items():
        results[stream_name] = {}
        for algo in algorithms:
            res = run_ttmm_adaptation(stream, algo, base_state_dict, expert_state_dicts, fisher_dicts, stream_name)
            results[stream_name][algo] = res
            
    # Save results as JSON
    with open("results/ttmm_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved all results to results/ttmm_results.json")
    
    # Print comparison table
    print("\n" + "="*80)
    print(f"{'Algorithm':<25} | {'Sequential Acc':<18} | {'Alternating Acc':<18}")
    print("="*80)
    for algo in algorithms:
        seq_acc = results["Sequential"][algo]["avg_acc"]
        alt_acc = results["Alternating"][algo]["avg_acc"]
        print(f"{algo:<25} | {seq_acc:<18.2f} | {alt_acc:<18.2f}")
    print("="*80)
    
    # Plot curves and save
    print("Generating plots...")
    for stream_name in streams:
        plt.figure(figsize=(10, 6))
        for algo in algorithms:
            accs = results[stream_name][algo]["accuracies"]
            # Cumulative average accuracy
            cum_accs = np.cumsum(accs) / (np.arange(len(accs)) + 1)
            plt.plot(cum_accs, label=algo, linewidth=2)
            
        plt.title(f"Cumulative Average Accuracy on {stream_name} Stream", fontsize=14)
        plt.xlabel("Test Step (Batch)", fontsize=12)
        plt.ylabel("Cumulative Accuracy (%)", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(fontsize=10)
        plt.savefig(f"results/ttmm_{stream_name.lower()}_accuracy.png", dpi=300, bbox_inches="tight")
        plt.close()
        
    print("Plots saved in results/")

if __name__ == "__main__":
    main()
