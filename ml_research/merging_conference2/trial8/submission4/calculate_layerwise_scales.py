import torch
import torchvision.models as models
import numpy as np

def calculate_scales():
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
        
    backbone_keys = [k for k in progenitor_state.keys() if not k.startswith('fc.')]
    
    print("\nLayer-wise U-IPR Scaling Factors (S^l):")
    scales = []
    keys_with_scales = []
    
    for key in backbone_keys:
        # Only float weights and biases, exclude buffers (like running_mean)
        if 'running_mean' in key or 'running_var' in key or 'num_batches_tracked' in key or 'bn' in key:
            continue
            
        p_init = progenitor_state[key].cpu().float()
        
        # Check if the parameter actually gets updated (non-zero norm of updates)
        T_experts = [(expert_states[t][key].cpu().float() - p_init) for t in tasks]
        norms = [torch.norm(T_exp).item() for T_exp in T_experts]
        
        if sum(norms) < 1e-6:
            # Skip un-updated parameters
            continue
            
        T_merged = torch.stack(T_experts).mean(dim=0)
        norm_merged = torch.norm(T_merged).item()
        
        if norm_merged < 1e-8:
            continue
            
        Sl = (sum(norms) / len(tasks)) / norm_merged
        scales.append(Sl)
        keys_with_scales.append(key)
        print(f"  {key:<30}: S^l = {Sl:.4f}  (Norms: {[f'{n:.3f}' for n in norms]} -> Merged: {norm_merged:.3f})")
        
    scales = np.array(scales)
    print("\nSummary Statistics of Layer-wise Scale Factors:")
    print(f"  Count: {len(scales)}")
    print(f"  Mean : {np.mean(scales):.4f}")
    print(f"  Std  : {np.std(scales):.4f}")
    print(f"  Min  : {np.min(scales):.4f}")
    print(f"  Max  : {np.max(scales):.4f}")
    print(f"  Theoretical Target (sqrt(3)): {np.sqrt(3):.4f}")

if __name__ == '__main__':
    calculate_scales()
