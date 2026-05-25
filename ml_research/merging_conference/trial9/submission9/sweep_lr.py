import torch
import numpy as np
from run_experiments import SimpleCNN, get_stream_batches, run_bk_comerge

def main():
    print("Loading experts for learning rate sweep...")
    expert_0 = SimpleCNN()
    expert_0.load_state_dict(torch.load("./checkpoints/expert_0.pt", map_location="cpu"))
    expert_0.eval()
    
    expert_1 = SimpleCNN()
    expert_1.load_state_dict(torch.load("./checkpoints/expert_1.pt", map_location="cpu"))
    expert_1.eval()
    
    print("Generating stream batches...")
    batches = get_stream_batches()
    
    lrs = [0.005, 0.01, 0.02, 0.04, 0.08]
    
    print("\n" + "="*80)
    print(f"{'Method / Learning Rate':<35} | {'Overall Accuracy (%)'}")
    print("-"*80)
    
    for lr in lrs:
        # Standard BK-CoMerge
        accs_standard = run_bk_comerge(
            expert_0, expert_1, batches, 
            lr=lr, steps=3, gamma_c=0.01, beta=0.1, s_scale=1.0,
            use_ts=False, use_ega=False
        )
        overall_std = np.mean(accs_standard)
        print(f"BK-CoMerge (lr={lr:.3f})            | {overall_std:21.2f}%")
        
        # EGA-BK-CoMerge
        accs_ega = run_bk_comerge(
            expert_0, expert_1, batches, 
            lr=lr, steps=3, gamma_c=0.01, beta=0.1, s_scale=1.0,
            use_ts=False, use_ega=True, tau_gate=0.6, alpha_gate=0.1
        )
        overall_ega = np.mean(accs_ega)
        print(f"EGA-BK-CoMerge (lr={lr:.3f})        | {overall_ega:21.2f}%")
        print("-"*80)

if __name__ == "__main__":
    main()
