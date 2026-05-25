import torch
import torch.nn as nn
import torch.nn.functional as F
from experiment import get_resnet18_model, get_dataloaders, compute_fisher

def analyze_v_distribution():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load loaders and base model
    loader_train_A, _, _, _, _ = get_dataloaders()
    model = get_resnet18_model(device)
    model.eval()
    
    print("Computing Fisher Information for Task A...")
    fisher = compute_fisher(model, loader_train_A, device, num_samples=512)
    
    gammas = [0.1, 0.5, 1.0, 2.0]
    
    print("\nEmpirical Statistics of v_j for different layers:")
    for gamma in gammas:
        print(f"\n" + "="*50)
        print(f"TEMPERATURE gamma = {gamma}")
        print("="*50)
        print(f"{'Layer Name':<40} | {'Mean v':<8} | {'Std v':<8} | {'Min v':<8} | {'Max v':<8}")
        print("-" * 80)
        
        all_v = []
        for name, param in model.named_parameters():
            if 'weight' in name and len(param.shape) == 4 and name in fisher:
                C_out = param.shape[0]
                F_param = fisher[name].view(C_out, -1)
                F_row = torch.mean(F_param, dim=1) # shape (C_out,)
                F_row_mean = torch.mean(F_row) + 1e-8
                
                # Weight v = exp(-gamma * F_row / F_row_mean)
                v = torch.exp(-gamma * F_row / F_row_mean)
                
                mean_v = torch.mean(v).item()
                std_v = torch.std(v).item()
                min_v = torch.min(v).item()
                max_v = torch.max(v).item()
                
                print(f"{name:<40} | {mean_v:.4f} | {std_v:.4f} | {min_v:.4f} | {max_v:.4f}")
                all_v.extend(v.cpu().tolist())
                
        # Overall statistics
        all_v = torch.tensor(all_v)
        print("-" * 80)
        print(f"{'OVERALL NETWORK':<40} | {torch.mean(all_v).item():.4f} | {torch.std(all_v).item():.4f} | {torch.min(all_v).item():.4f} | {torch.max(all_v).item():.4f}")

if __name__ == "__main__":
    analyze_v_distribution()
