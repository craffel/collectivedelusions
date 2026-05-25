import torch
import torch.nn.functional as F
from torch.func import functional_call
from models import SharedEncoder, ClassificationHead
from eval_tta import get_test_streams, apply_corruption, set_seed

def test_adaptive_l2_task_specific(alpha_high, alpha_low, stream_name, corruption):
    set_seed(42)
    device = torch.device("cpu")
    seq_stream, alt_stream = get_test_streams(batch_size=64, num_batches_per_task=50)
    stream = seq_stream if stream_name == "sequential" else alt_stream
    
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
    
    optimizers = {
        0: torch.optim.Adam([merging_logits_dict[0]], lr=0.01),
        1: torch.optim.Adam([merging_logits_dict[1]], lr=0.01),
        2: torch.optim.Adam([merging_logits_dict[2]], lr=0.01)
    }
    
    total_correct = 0
    total_samples = 0
    
    for step, (x, y, task_idx) in enumerate(stream):
        x_corrupted = apply_corruption(x, corruption)
        
        logits = merging_logits_dict[task_idx]
        weights = torch.softmax(logits, dim=1)
        merged_params = {}
        for l_idx, name in enumerate(param_names):
            merged_params[name] = (
                weights[l_idx, 0] * expert_encoders[0][name] +
                weights[l_idx, 1] * expert_encoders[1][name] +
                weights[l_idx, 2] * expert_encoders[2][name]
            )
            
        with torch.no_grad():
            features = functional_call(base_encoder, merged_params, x_corrupted)
            outputs = heads[task_idx](features)
            _, predicted = outputs.max(1)
            total_correct += predicted.eq(y).sum().item()
            total_samples += y.size(0)
            
        opt = optimizers[task_idx]
        opt.zero_grad()
        features = functional_call(base_encoder, merged_params, x_corrupted)
        outputs = heads[task_idx](features)
        probs = F.softmax(outputs, dim=-1)
        ent_loss = -torch.mean(torch.sum(probs * torch.log(probs + 1e-12), dim=-1))
        
        from eval_tta import augment_batch
        x_aug = augment_batch(x_corrupted, task_idx)
        features_aug = functional_call(base_encoder, merged_params, x_aug)
        outputs_aug = heads[task_idx](features_aug)
        probs_aug = F.softmax(outputs_aug, dim=-1)
        kl_loss = F.kl_div(torch.log(probs_aug + 1e-12), probs.detach(), reduction="batchmean")
        
        # Calculate dynamic confidence
        import numpy as np
        H = ent_loss.item()
        gamma_k = max(0.0, 1.0 - H / np.log(10.0))
        
        # Linear interpolation for alpha
        alpha = alpha_high * gamma_k + alpha_low * (1.0 - gamma_k)
        
        # Loss with dynamic L2 penalty on logits
        loss = ent_loss + 1.0 * kl_loss + alpha * torch.sum(logits ** 2)
        loss.backward()
        opt.step()
        
    acc = 100.0 * total_correct / total_samples
    return acc

print("Evaluating Adaptive L2:")
for corr in ["clean", "noise", "blur", "contrast"]:
    acc_seq = test_adaptive_l2_task_specific(0.01, 0.5, "sequential", corr)
    acc_alt = test_adaptive_l2_task_specific(0.01, 0.5, "alternating", corr)
    print(f"  Corruption: {corr:8s} | Seq Acc: {acc_seq:.2f}% | Alt Acc: {acc_alt:.2f}%")
