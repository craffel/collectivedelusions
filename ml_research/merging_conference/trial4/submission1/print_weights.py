import torch
import torch.nn.functional as F
from torch.func import functional_call
from models import SharedEncoder, ClassificationHead
from eval_tta import get_test_streams, apply_corruption, set_seed

def inspect_s2c():
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
    merging_logits = torch.zeros((num_layers, 3), device=device, requires_grad=True)
    
    optimizer = torch.optim.Adam([merging_logits], lr=0.005)
    
    print(f"Step | Task | Clean/Noise Accuracy | Merging Weights (Mean across layers)")
    print("-" * 75)
    
    for step, (x, y, task_idx) in enumerate(seq_stream):
        x_corrupted = apply_corruption(x, "noise")
        
        # 1. Reconstruct merged encoder parameters WITH gradients enabled
        weights = torch.softmax(merging_logits, dim=1)
        merged_params = {}
        for l_idx, name in enumerate(param_names):
            merged_params[name] = (
                weights[l_idx, 0] * expert_encoders[0][name] +
                weights[l_idx, 1] * expert_encoders[1][name] +
                weights[l_idx, 2] * expert_encoders[2][name]
            )
            
        # 2. Compute accuracy BEFORE adaptation
        with torch.no_grad():
            features = functional_call(base_encoder, merged_params, x_corrupted)
            outputs = heads[task_idx](features)
            _, predicted = outputs.max(1)
            acc = 100.0 * predicted.eq(y).sum().item() / y.size(0)
            
            # Print mean weights
            mean_weights = weights.mean(dim=0).tolist()
            if step % 10 == 0 or step == len(seq_stream)-1:
                print(f"{step:4d} | {task_idx:4d} | Acc: {acc:6.2f}% | Weights: [{mean_weights[0]:.4f}, {mean_weights[1]:.4f}, {mean_weights[2]:.4f}]")
        
        # 3. Adaptation step
        optimizer.zero_grad()
        features = functional_call(base_encoder, merged_params, x_corrupted)
        outputs = heads[task_idx](features)
        probs = F.softmax(outputs, dim=-1)
        ent_loss = -torch.mean(torch.sum(probs * torch.log(probs + 1e-12), dim=-1))
        
        # consistency loss
        from eval_tta import augment_batch
        x_aug = augment_batch(x_corrupted, task_idx)
        features_aug = functional_call(base_encoder, merged_params, x_aug)
        outputs_aug = heads[task_idx](features_aug)
        probs_aug = F.softmax(outputs_aug, dim=-1)
        kl_loss = F.kl_div(torch.log(probs_aug + 1e-12), probs.detach(), reduction="batchmean")
        loss = ent_loss + 1.0 * kl_loss
        
        loss.backward()
        optimizer.step()

inspect_s2c()
