import torch
import copy
from experiment import get_datasets, generate_test_stream, run_tta_evaluation, get_pretrained_base_encoder, get_task_vector, ResNet18Expert

tasks = ['mnist', 'fmnist', 'kmnist']

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
base_encoder.eval()
task_vectors = [get_task_vector(expert_encoders[i], base_encoder) for i in range(3)]

# Pre-compute diagonal Fisher for expert heads
from torch.utils.data import Subset
fisher_diags = []
for i, task in enumerate(tasks):
    train_set, _ = data[task]
    subset_indices = list(range(200))
    subset_train_set = Subset(train_set, subset_indices)
    from experiment import compute_diagonal_fisher
    fim = compute_diagonal_fisher(expert_encoders[i], expert_heads_list[i], subset_train_set, num_samples=200)
    fisher_diags.append(fim)

# Clone original states
orig_base_state = {k: v.clone() for k, v in base_encoder.state_dict().items()}
orig_expert_heads = [{k: v.clone() for k, v in h.state_dict().items()} for h in expert_heads_list]

print("Generating alternating stream...")
batches = generate_test_stream(test_datasets, 'alternating', num_batches_per_task=1, seed=42)

print("Running run_tta_evaluation...")
run_tta_evaluation(
    base_encoder=base_encoder,
    task_vectors=task_vectors,
    expert_encoders=expert_encoders,
    expert_heads=expert_heads_list,
    batches=batches,
    fisher_diags=fisher_diags,
    reg_type='ewc',
    gamma=10.0
)

# Check base_encoder
base_changed = False
for k, v in base_encoder.state_dict().items():
    if not torch.allclose(v, orig_base_state[k]):
        print(f"base_encoder parameter changed: {k}")
        base_changed = True

# Check expert heads
heads_changed = False
for idx, h in enumerate(expert_heads_list):
    for k, v in h.state_dict().items():
        if not torch.allclose(v, orig_expert_heads[idx][k]):
            print(f"expert_head {idx} parameter changed: {k}")
            heads_changed = True

if not base_changed:
    print("base_encoder is NOT modified by run_tta_evaluation.")
if not heads_changed:
    print("expert_heads are NOT modified by run_tta_evaluation.")
