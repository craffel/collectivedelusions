import argparse
import os
import sys
import json
import random
import math
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import timm

from optimizers import SAM, SABCD
from merging import merge_models

# Set cache directories to local folders

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_split_cifar100(task_idx, train=True, transform=None):
    # Load dataset
    dataset = datasets.CIFAR100(root='./data', train=train, download=False, transform=transform)
    # Filter classes for task_idx (0 to 4)
    # Task 0: 0-19, Task 1: 20-39, Task 2: 40-59, Task 3: 60-79, Task 4: 80-99
    start_class = task_idx * 20
    end_class = (task_idx + 1) * 20
    indices = [i for i, label in enumerate(dataset.targets) if start_class <= label < end_class]
    return Subset(dataset, indices)

def main():
    parser = argparse.ArgumentParser(description="Continual Learning Model Merging Demystification Suite")
    parser.add_argument("--model", type=str, default="vit_base_patch16_224", help="ViT model from timm")
    parser.add_argument("--optimizer", type=str, default="sabcd_adam_gt", 
                        choices=["adamw", "sam", "sabcd_literal", "sabcd_standard_adam", "sabcd_adam_gt"])
    parser.add_argument("--merging", type=str, default="isotropic", 
                        choices=["isotropic", "spectral_dampening", "task_arithmetic", "norm_matching", "scale_calibrated", "ties_merging", "dare"])
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--rho", type=float, default=0.05, help="SAM/SA-BCD perturbation radius")
    parser.add_argument("--p_ratio", type=float, default=0.3, help="SA-BCD parameter selection ratio")
    parser.add_argument("--epochs", type=int, default=2, help="Epochs per task")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--num_tasks", type=int, default=5, help="Number of split tasks")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--alpha", type=float, default=1.0, help="Merging scale factor")
    parser.add_argument("--lambda_val", type=float, default=0.0, help="History balance coefficient")
    parser.add_argument("--output_file", type=str, default="results.json", help="Path to save final metrics JSON")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print("=================================================================")
    print(f"Starting Continual Merging Run with settings:")
    print(f"Model: {args.model}")
    print(f"Optimizer: {args.optimizer}")
    print(f"Merging: {args.merging}")
    print(f"LR: {args.lr}, WD: {args.weight_decay}, Epochs: {args.epochs}")
    print(f"Rho: {args.rho}, P_ratio: {args.p_ratio}, Alpha: {args.alpha}")
    print(f"Seed: {args.seed}, Device: {args.device}")
    print("=================================================================")

    set_seed(args.seed)

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load initial pre-trained model as theta_pre (theta_0)
    print(f"Loading pre-trained {args.model} backbone...")
    base_model = timm.create_model(args.model, pretrained=True)
    base_model.to(args.device)
    base_model.eval()
    
    # Freeze standard classifier head, we will use separate heads
    for param in base_model.parameters():
        param.requires_grad = True

    # Dictionary to keep track of task heads and historical performance
    task_heads = {}
    
    # Store state dict of backbone (theta_0)
    # In timm, 'head' is the classification layer, we exclude it from the backbone merging state dict
    def get_backbone_state(model):
        state = {}
        for k, v in model.state_dict().items():
            if not k.startswith("head."):
                state[k] = v.clone()
        return state

    theta_pre = get_backbone_state(base_model)
    theta_curr_merged = {k: v.clone() for k, v in theta_pre.items()}

    # Accuracy matrix: acc_matrix[t][i] is accuracy of model_t on task_i
    acc_matrix = []
    task_durations = []

    for t in range(args.num_tasks):
        print(f"\n--- Training Task {t+1}/{args.num_tasks} ---")
        start_time = time.time()

        # Load datasets
        train_set = get_split_cifar100(t, train=True, transform=train_transform)
        val_set = get_split_cifar100(t, train=False, transform=val_transform)
        
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

        # Re-construct model with backbone from current merged parameters theta_curr_merged
        model = timm.create_model(args.model, pretrained=False)
        # Restore backbone
        model_state = model.state_dict()
        for k, v in theta_curr_merged.items():
            model_state[k].copy_(v)
        model.load_state_dict(model_state)
        
        # Instantiate task head
        hidden_dim = model.num_features
        task_head = nn.Linear(hidden_dim, 20) # 20 classes per task
        nn.init.xavier_uniform_(task_head.weight)
        nn.init.zeros_(task_head.bias)
        
        model.head = task_head
        model.to(args.device)
        model.train()

        # Set up optimizer
        if args.optimizer == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == "sam":
            optimizer = SAM(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, rho=args.rho)
        elif args.optimizer.startswith("sabcd"):
            mode = args.optimizer.replace("sabcd_", "")
            optimizer = SABCD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, 
                              rho=args.rho, p_ratio=args.p_ratio, mode=mode)
        else:
            raise ValueError(f"Unknown optimizer: {args.optimizer}")

        criterion = nn.CrossEntropyLoss()

        # Fine-tuning loop
        for epoch in range(args.epochs):
            model.train()
            total_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(args.device), targets.to(args.device)
                
                # Shift labels from [20t, 20t + 19] to [0, 19]
                targets = targets - (t * 20)
                
                if args.optimizer == "adamw":
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                else:
                    # SAM & SABCD two-step optimization cycle
                    # First step (unperturbed loss gradient computation)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.first_step(zero_grad=True)
                    
                    # Second step (perturbed loss gradient computation + step update)
                    outputs_p = model(inputs)
                    loss_p = criterion(outputs_p, targets)
                    loss_p.backward()
                    optimizer.second_step(zero_grad=True)
                    
                    # Log unperturbed loss for progress
                    loss = loss.detach()

                total_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            epoch_loss = total_loss / len(train_set)
            epoch_acc = correct / total * 100
            print(f"Epoch {epoch+1}/{args.epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.2f}%")

        duration = time.time() - start_time
        task_durations.append(duration)
        print(f"Task {t+1} training completed in {duration:.1f} seconds.")

        # Save trained task specific head
        task_heads[t] = {k: v.cpu().clone() for k, v in model.head.state_dict().items()}

        # Perform Model Merging (Merge theta_curr_merged and task_expert to get new theta_curr_merged)
        theta_expert = get_backbone_state(model)
        
        print(f"Merging models using: {args.merging}...")
        theta_curr_merged = merge_models(
            theta_pre=theta_pre,
            theta_prev_merged=theta_curr_merged,
            theta_expert=theta_expert,
            task_idx=t + 1,
            method=args.merging,
            alpha=args.alpha,
            lambda_val=args.lambda_val
        )

        # Evaluation of current merged model on all tasks trained so far (1 to t+1)
        # Create evaluation model
        eval_model = timm.create_model(args.model, pretrained=False)
        eval_state = eval_model.state_dict()
        for k, v in theta_curr_merged.items():
            eval_state[k].copy_(v)
        eval_model.load_state_dict(eval_state)
        eval_model.to(args.device)
        eval_model.eval()

        task_accuracies = []
        for i in range(t + 1):
            eval_set = get_split_cifar100(i, train=False, transform=val_transform)
            eval_loader = DataLoader(eval_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
            
            # Attach task specific head
            head_state = task_heads[i]
            eval_head = nn.Linear(hidden_dim, 20)
            eval_head.load_state_dict(head_state)
            eval_model.head = eval_head
            eval_model.head.to(args.device)
            eval_model.eval()

            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, targets in eval_loader:
                    inputs, targets = inputs.to(args.device), targets.to(args.device)
                    targets = targets - (i * 20)
                    outputs = eval_model(inputs)
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                    
            acc = correct / total * 100
            task_accuracies.append(acc)
            print(f"Accuracy on Task {i+1}: {acc:.2f}%")

        # Fill with zeros for future tasks to keep matrix square/aligned
        row_accs = task_accuracies + [0.0] * (args.num_tasks - len(task_accuracies))
        acc_matrix.append(row_accs)

    # Compute final evaluation metrics (ACC, BWT)
    # ACC is mean accuracy over all tasks at final step
    final_accs = acc_matrix[-1]
    acc_metric = sum(final_accs) / args.num_tasks
    
    # BWT (Backward Transfer / forgetting)
    # BWT = 1/(T-1) * sum_{i=1}^{T-1} (acc_matrix[T-1][i] - acc_matrix[i][i])
    bwt_metric = 0.0
    if args.num_tasks > 1:
        forgetting_sum = 0.0
        for i in range(args.num_tasks - 1):
            forgetting_sum += (acc_matrix[-1][i] - acc_matrix[i][i])
        bwt_metric = forgetting_sum / (args.num_tasks - 1)

    print("\n================ FINAL METRICS ================")
    print(f"Average Accuracy (ACC): {acc_metric:.2f}%")
    print(f"Backward Transfer (BWT): {bwt_metric:.2f}%")
    print(f"Total training time: {sum(task_durations):.1f} seconds")
    print("===============================================")

    # Save results
    results = {
        "model": args.model,
        "optimizer": args.optimizer,
        "merging": args.merging,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "rho": args.rho,
        "p_ratio": args.p_ratio,
        "epochs": args.epochs,
        "alpha": args.alpha,
        "lambda_val": args.lambda_val,
        "seed": args.seed,
        "acc_matrix": acc_matrix,
        "acc": acc_metric,
        "bwt": bwt_metric,
        "durations": task_durations,
        "total_duration": sum(task_durations)
    }

    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {args.output_file}")

if __name__ == "__main__":
    main()
