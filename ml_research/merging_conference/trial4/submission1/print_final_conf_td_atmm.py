import torch
import torch.nn.functional as F
from torch.func import functional_call
from models import SharedEncoder, ClassificationHead
from eval_tta import get_test_streams, apply_corruption, set_seed, augment_batch
import numpy as np

def run_td_atmm_and_print_conf(corruption):
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
    
    merging_logits_dict = {
        0: torch.zeros((num_layers, 3), device=device, requires_grad=True),
        1: torch.zeros((num_layers, 3), device=device, requires_grad=True),
        2: torch.zeros((num_layers, 3), device=device, requires_grad=True)
    }
    
    optimizers_dict = {
        0: torch.optim.Adam([merging_logits_dict[0]], lr=0.01),
        1: torch.optim.Adam([merging_logits_dict[1]], lr=0.01),
        2: torch.optim.Adam([merging_logits_dict[2]], lr=0.01)
    }
    
    confs = []
    weights_evolution = {0: [], 1: [], 2: []}
    
    for step, (x, y, task_idx) in enumerate(seq_stream):
        x_corrupted = apply_corruption(x, corruption)
        head = heads[task_idx]
        
        # Merging with current task logits
        weights = torch.softmax(merging_logits_dict[task_idx], dim=1)
        
        # Track average weight for active expert at layer 0 as a representative
        weights_evolution[task_idx].append(weights[0, task_idx].item())
        
        merged_params = {}
        for l_idx, name in enumerate(param_names):
            merged_params[name] = (
                weights[l_idx, 0] * expert_encoders[0][name] +
                weights[l_idx, 1] * expert_encoders[1][name] +
                weights[l_idx, 2] * expert_encoders[2][name]
            )
            
        # Evaluate
        with torch.no_grad():
            features = functional_call(base_encoder, merged_params, x_corrupted)
            outputs = head(features)
            probs = F.softmax(outputs, dim=-1)
            max_p, _ = torch.max(probs, dim=-1)
            confs.append(max_p.mean().item())
            
        # Adapt
        optimizers_dict[task_idx].zero_grad()
        features = functional_call(base_encoder, merged_params, x_corrupted)
        outputs = head(features)
        probs = F.softmax(outputs, dim=-1)
        ent_loss = -torch.mean(torch.sum(probs * torch.log(probs + 1e-12), dim=-1))
        
        x_aug = augment_batch(x_corrupted, task_idx)
        features_aug = functional_call(base_encoder, merged_params, x_aug)
        outputs_aug = head(features_aug)
        probs_aug = F.softmax(outputs_aug, dim=-1)
        kl_loss = F.kl_div(torch.log(probs_aug + 1e-12), probs.detach(), reduction="batchmean")
        
        loss = ent_loss + 1.0 * kl_loss + 0.05 * torch.sum(merging_logits_dict[task_idx] ** 2)
        loss.backward()
        optimizers_dict[task_idx].step()
        
    print(f"Corruption: {corruption:8s} | Conf Mean: {np.mean(confs):.4f} | Final Conf: {confs[-1]:.4f}")
    # Print weights for the tasks at the end of their respective segments
    # Each task gets 50 batches
    print(f"  MNIST final weight: {weights_evolution[0][-1]:.4f} (started at 0.3333)")
    print(f"  FMNIST final weight: {weights_evolution[1][-1]:.4f} (started at 0.3333)")
    print(f"  KMNIST final weight: {weights_evolution[2][-1]:.4f} (started at 0.3333)")

for corr in ["clean", "blur", "noise", "contrast"]:
    run_td_atmm_and_print_conf(corr)
