import torch
import copy
from experiment import get_datasets, generate_test_stream, run_tta_evaluation, get_pretrained_base_encoder, get_task_vector, ResNet18Expert
from torch.func import functional_call
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset

tasks = ['mnist', 'fmnist', 'kmnist']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = get_datasets()
test_datasets = [data[tasks[i]][1] for i in range(3)]

expert_encoders = []
expert_heads_list = []
for task in tasks:
    resnet = ResNet18Expert()
    resnet.encoder.load_state_dict(torch.load(f"./checkpoints/{task}_encoder.pt", map_location=device))
    resnet.fc.load_state_dict(torch.load(f"./checkpoints/{task}_head.pt", map_location=device))
    expert_encoders.append(resnet.encoder.to(device))
    expert_heads_list.append(resnet.fc.to(device))

base_encoder = get_pretrained_base_encoder()
base_encoder.eval()
task_vectors = [get_task_vector(expert_encoders[i], base_encoder) for i in range(3)]

# Pre-compute diagonal Fisher for expert heads
fisher_diags = []
for i, task in enumerate(tasks):
    train_set, _ = data[task]
    subset_indices = list(range(200))
    subset_train_set = Subset(train_set, subset_indices)
    from experiment import compute_diagonal_fisher
    fim = compute_diagonal_fisher(expert_encoders[i], expert_heads_list[i], subset_train_set, num_samples=200)
    fisher_diags.append(fim)

# Generate test streams once and reuse them
test_datasets = [data[tasks[i]][1] for i in range(3)]
seq_batches = generate_test_stream(test_datasets, 'sequential', seed=42)
alt_batches = generate_test_stream(test_datasets, 'alternating', seed=42)

# Sweep configuration
lambda_lrs = [0.05, 0.1, 0.2, 0.5]
gammas = [10.0, 100.0]

print("Starting Systematic Sweep...")
print("Format: Lambda LR | Stream | Method | Accuracy")
print("-" * 60)

results = []

for lr in lambda_lrs:
    for stream_name, batches in [('alternating', alt_batches), ('sequential', seq_batches)]:
        # 1. Standard TTA
        std_acc = run_tta_evaluation(
            base_encoder=base_encoder,
            task_vectors=task_vectors,
            expert_encoders=expert_encoders,
            expert_heads=expert_heads_list,
            batches=batches,
            reg_type='none',
            gamma=0.0,
            tta_lr_lambdas=lr
        )
        print(f"{lr:9.2f} | {stream_name:11} | Standard TTA | {std_acc:6.2f}%")
        results.append((lr, stream_name, 'Standard TTA', std_acc))
        
        # 2. L2 TTA
        for gamma in gammas:
            l2_acc = run_tta_evaluation(
                base_encoder=base_encoder,
                task_vectors=task_vectors,
                expert_encoders=expert_encoders,
                expert_heads=expert_heads_list,
                batches=batches,
                reg_type='l2',
                gamma=gamma,
                tta_lr_lambdas=lr
            )
            print(f"{lr:9.2f} | {stream_name:11} | L2 (G={gamma}) | {l2_acc:6.2f}%")
            results.append((lr, stream_name, f'L2-TTA (G={gamma})', l2_acc))
            
        # 3. EWC TTA (Ours)
        for gamma in gammas:
            ewc_acc = run_tta_evaluation(
                base_encoder=base_encoder,
                task_vectors=task_vectors,
                expert_encoders=expert_encoders,
                expert_heads=expert_heads_list,
                batches=batches,
                fisher_diags=fisher_diags,
                reg_type='ewc',
                gamma=gamma,
                tta_lr_lambdas=lr
            )
            print(f"{lr:9.2f} | {stream_name:11} | EWC (G={gamma}) | {ewc_acc:6.2f}%")
            results.append((lr, stream_name, f'EWC-TTA (G={gamma})', ewc_acc))

print("\n--- SWEEP COMPLETELY FINISHED ---")

# Print clean structured summary markdown table for the paper
print("\n### Summary Table (Sequential Stream)")
print("| Lambda LR | Standard TTA | L2-TTA (G=10) | L2-TTA (G=100) | EWC-TTA (G=10) | EWC-TTA (G=100) |")
print("| :---: | :---: | :---: | :---: | :---: | :---: |")
for lr in lambda_lrs:
    row = f"| {lr:.2f} | "
    for method in ['Standard TTA', 'L2-TTA (G=10.0)', 'L2-TTA (G=100.0)', 'EWC-TTA (G=10.0)', 'EWC-TTA (G=100.0)']:
        val = next(r[3] for r in results if r[0] == lr and r[1] == 'sequential' and r[2] == method)
        row += f"{val:.2f}% | "
    print(row)

print("\n### Summary Table (Alternating Stream)")
print("| Lambda LR | Standard TTA | L2-TTA (G=10) | L2-TTA (G=100) | EWC-TTA (G=10) | EWC-TTA (G=100) |")
print("| :---: | :---: | :---: | :---: | :---: | :---: |")
for lr in lambda_lrs:
    row = f"| {lr:.2f} | "
    for method in ['Standard TTA', 'L2-TTA (G=10.0)', 'L2-TTA (G=100.0)', 'EWC-TTA (G=10.0)', 'EWC-TTA (G=100.0)']:
        val = next(r[3] for r in results if r[0] == lr and r[1] == 'alternating' and r[2] == method)
        row += f"{val:.2f}% | "
    print(row)
