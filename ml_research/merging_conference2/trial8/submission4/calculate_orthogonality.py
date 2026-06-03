import torch
import torchvision.models as models
import numpy as np

def calculate_similarity():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Loading ImageNet pre-trained progenitor...")
    progenitor = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    progenitor_state = progenitor.state_dict()
    
    tasks = ['mnist', 'fmnist', 'cifar10']
    expert_states = {}
    
    for task in tasks:
        ckpt_path = f"checkpoints/resnet18_{task}.pt"
        print(f"Loading expert {task.upper()} from {ckpt_path}...")
        expert_data = torch.load(ckpt_path, map_location=device)
        expert_states[task] = expert_data['state_dict']
        
    # Get backbone parameters (excluding the classification head 'fc.')
    backbone_keys = [k for k in progenitor_state.keys() if not k.startswith('fc.')]
    
    # Compute global task vectors
    task_vectors = {}
    for task in tasks:
        flat_vector = []
        for key in backbone_keys:
            # We only consider parameters that have grads / are weights/biases
            # Skip non-float buffers if any (like num_batches_tracked)
            p_init = progenitor_state[key]
            p_expert = expert_states[task][key]
            if p_init.is_floating_point():
                diff = (p_expert - p_init).cpu().flatten()
                flat_vector.append(diff)
        task_vectors[task] = torch.cat(flat_vector)
        
    print("\nGlobal Task Vector Norms:")
    for task in tasks:
        print(f"  {task.upper()}: {torch.norm(task_vectors[task]).item():.4f}")
        
    print("\nPairwise Cosine Similarities (Global Backbone):")
    pairs = [('mnist', 'fmnist'), ('mnist', 'cifar10'), ('fmnist', 'cifar10')]
    for t1, t2 in pairs:
        v1 = task_vectors[t1]
        v2 = task_vectors[t2]
        cos_sim = torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2))
        print(f"  {t1.upper()} vs {t2.upper()}: {cos_sim.item():.6f}")
        
    # Layer-wise analysis of representative layers
    print("\nLayer-wise Cosine Similarities:")
    rep_layers = [
        'conv1.weight',
        'layer1.0.conv1.weight',
        'layer2.0.conv1.weight',
        'layer3.0.conv1.weight',
        'layer4.0.conv1.weight'
    ]
    for layer in rep_layers:
        print(f"  Layer: {layer}")
        for t1, t2 in pairs:
            diff1 = (expert_states[t1][layer] - progenitor_state[layer]).cpu().flatten()
            diff2 = (expert_states[t2][layer] - progenitor_state[layer]).cpu().flatten()
            norm1 = torch.norm(diff1)
            norm2 = torch.norm(diff2)
            if norm1 > 0 and norm2 > 0:
                layer_cos_sim = torch.dot(diff1, diff2) / (norm1 * norm2)
                print(f"    {t1.upper()} vs {t2.upper()}: {layer_cos_sim.item():.6f}")
            else:
                print(f"    {t1.upper()} vs {t2.upper()}: N/A (zero norm update)")

if __name__ == '__main__':
    calculate_similarity()
