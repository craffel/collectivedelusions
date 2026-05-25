import torch
from experiment import get_datasets, generate_test_stream, evaluate_static_merged, get_pretrained_base_encoder, get_task_vector, ResNet18Expert, reconstruct_merged_encoder

tasks = ['mnist', 'fmnist', 'kmnist']
import os
from torch.utils.data import Subset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = get_datasets()
test_datasets = [data[tasks[i]][1] for i in range(3)]

# Load experts
expert_encoders = []
expert_heads_list = []
for task in tasks:
    resnet = ResNet18Expert()
    resnet.encoder.load_state_dict(torch.load(f"./checkpoints/{task}_encoder.pt", map_location=device))
    resnet.fc.load_state_dict(torch.load(f"./checkpoints/{task}_head.pt", map_location=device))
    expert_encoders.append(resnet.encoder.to(device))
    expert_heads_list.append(resnet.fc.to(device))

base_encoder = get_pretrained_base_encoder()
task_vectors = [get_task_vector(expert_encoders[i], base_encoder) for i in range(3)]

print("Generating alternating stream...")
alt_batches = generate_test_stream(test_datasets, 'alternating', seed=42)

print("Generating sequential stream...")
seq_batches = generate_test_stream(test_datasets, 'sequential', seed=42)

# Verify the number of batches and items
print(f"Alternating batches: {len(alt_batches)}, Sequential batches: {len(seq_batches)}")

# Compare the images in the two streams
# We can sum all pixel values of all images in both streams and compare the totals!
alt_pixels_sum = 0.0
seq_pixels_sum = 0.0

for _, x, _ in alt_batches:
    alt_pixels_sum += x.sum().item()
    
for _, x, _ in seq_batches:
    seq_pixels_sum += x.sum().item()

print(f"Alternating total pixels sum: {alt_pixels_sum:.4f}")
print(f"Sequential total pixels sum: {seq_pixels_sum:.4f}")

# Evaluate static merged on both
alt_acc = evaluate_static_merged(base_encoder, task_vectors, expert_heads_list, alt_batches)
seq_acc = evaluate_static_merged(base_encoder, task_vectors, expert_heads_list, seq_batches)

print(f"Static Merged Acc on Alternating: {alt_acc:.2f}%")
print(f"Static Merged Acc on Sequential: {seq_acc:.2f}%")
