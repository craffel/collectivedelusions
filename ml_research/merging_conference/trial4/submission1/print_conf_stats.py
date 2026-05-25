import torch
import torch.nn.functional as F
from torch.func import functional_call
from models import SharedEncoder, ClassificationHead
from eval_tta import get_test_streams, apply_corruption, set_seed
import numpy as np

def print_conf_stats(corruption):
    set_seed(42)
    device = torch.device("cpu")
    seq_stream, _ = get_test_streams(batch_size=64, num_batches_per_task=50)
    
    encoder_mnist = torch.load("encoder_mnist.pth", map_location=device, weights_only=True)
    encoder_fmnist = torch.load("encoder_fmnist.pth", map_location=device, weights_only=True)
    encoder_kmnist = torch.load("encoder_kmnist.pth", map_location=device, weights_only=True)
    expert_encoders = [encoder_mnist, encoder_fmnist, encoder_kmnist]
    param_names = list(encoder_mnist.keys())
    
    heads = {
        0: ClassificationHead().to(device),
        1: ClassificationHead().to(device),
        2: ClassificationHead().to(device)
    }
    heads[0].load_state_dict(torch.load("head_mnist.pth", map_location=device, weights_only=True))
    heads[1].load_state_dict(torch.load("head_fmnist.pth", map_location=device, weights_only=True))
    heads[2].load_state_dict(torch.load("head_kmnist.pth", map_location=device, weights_only=True))
    
    base_encoder = SharedEncoder().to(device)
    num_layers = len(param_names)
    
    merging_logits = torch.zeros((num_layers, 3), device=device)
    weights = torch.softmax(merging_logits, dim=1)
    merged_params = {}
    for l_idx, name in enumerate(param_names):
        merged_params[name] = (
            weights[l_idx, 0] * expert_encoders[0][name] +
            weights[l_idx, 1] * expert_encoders[1][name] +
            weights[l_idx, 2] * expert_encoders[2][name]
        )
        
    confs = []
    for step, (x, y, task_idx) in enumerate(seq_stream):
        x_corrupted = apply_corruption(x, corruption)
        with torch.no_grad():
            features = functional_call(base_encoder, merged_params, x_corrupted)
            outputs = heads[task_idx](features)
            probs = F.softmax(outputs, dim=-1)
            max_p, _ = torch.max(probs, dim=-1)
            confs.append(max_p.mean().item())
            
    print(f"Corruption: {corruption:8s} | Conf Mean: {np.mean(confs):.4f} | Min: {np.min(confs):.4f} | Max: {np.max(confs):.4f}")

for corr in ["clean", "noise", "blur", "contrast"]:
    print_conf_stats(corr)
