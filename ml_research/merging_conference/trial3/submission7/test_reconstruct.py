import torch
from experiment import get_pretrained_base_encoder, reconstruct_merged_encoder, ResNet18Expert, get_task_vector

tasks = ['mnist', 'fmnist', 'kmnist']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_encoder = get_pretrained_base_encoder()
expert_encoders = []
for task in tasks:
    resnet = ResNet18Expert()
    resnet.encoder.load_state_dict(torch.load(f"./checkpoints/{task}_encoder.pt", map_location=device))
    expert_encoders.append(resnet.encoder.to(device))

task_vectors = [get_task_vector(expert_encoders[i], base_encoder) for i in range(3)]

# Clone the original state dict of base_encoder
orig_state = {k: v.clone() for k, v in base_encoder.state_dict().items()}

print("Running reconstruct_merged_encoder...")
expert_encoder = reconstruct_merged_encoder(base_encoder, task_vectors, [1.0, 0.0, 0.0])

# Check if base_encoder state has changed
changed = False
for k, v in base_encoder.state_dict().items():
    if not torch.allclose(v, orig_state[k]):
        print(f"Parameter changed: {k}!")
        changed = True

if not changed:
    print("reconstruct_merged_encoder does NOT modify base_encoder.")
else:
    print("WARNING: reconstruct_merged_encoder MODIFIES base_encoder!")
