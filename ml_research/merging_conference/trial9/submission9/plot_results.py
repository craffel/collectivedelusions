import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from run_experiments import (
    SimpleCNN,
    get_stream_batches,
    compute_offline_fisher,
    run_static_merging,
    run_fixed_tta,
    run_bk_comerge
)

def main():
    print("Loading experts for plotting...")
    expert_0 = SimpleCNN()
    expert_0.load_state_dict(torch.load("./checkpoints/expert_0.pt", map_location="cpu"))
    expert_0.eval()
    
    expert_1 = SimpleCNN()
    expert_1.load_state_dict(torch.load("./checkpoints/expert_1.pt", map_location="cpu"))
    expert_1.eval()
    
    print("Generating stream batches...")
    batches = get_stream_batches()
    
    print("Precomputing offline joint Fisher...")
    offline_fisher = compute_offline_fisher(expert_0, expert_1, num_samples=100)
    
    # Run key methods
    print("Running evaluations for plotting...")
    static_accs = run_static_merging(expert_0, expert_1, batches)
    tta_accs = run_fixed_tta(expert_0, expert_1, batches, lr=0.01, steps=3)
    bk_accs = run_bk_comerge(expert_0, expert_1, batches, lr=0.02, steps=3, gamma_c=0.01, beta=0.1, s_scale=1.0, use_ts=False, use_ega=False)
    ega_accs = run_bk_comerge(expert_0, expert_1, batches, lr=0.02, steps=3, gamma_c=0.01, beta=0.1, s_scale=1.0, use_ts=False, use_ega=True, tau_gate=0.6, alpha_gate=0.1)
    att_accs = run_bk_comerge(expert_0, expert_1, batches, lr=0.02, steps=3, gamma_c=0.01, beta=0.1, s_scale=1.0, use_ts=False, use_ega=False, use_att_bn=True, rho=0.2)
    
    # Set up plot
    plt.figure(figsize=(10, 5))
    
    # Smooth with rolling average (window size 3)
    def smooth(y, box_pts=3):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        # fix edges to avoid distortion
        y_smooth[0] = y[0]
        y_smooth[-1] = y[-1]
        return y_smooth

    x = np.arange(len(batches))
    
    # Define beautiful academic colors
    colors = {
        'static': '#7f7f7f', # Gray
        'tta': '#d62728',    # Red
        'bk': '#1f77b4',     # Blue
        'ega': '#2ca02c',    # Green
        'att': '#9467bd'     # Purple
    }
    
    plt.plot(x, smooth(static_accs), label='Static Merging', color=colors['static'], linewidth=2)
    plt.plot(x, smooth(tta_accs), label='Fixed TTA (TENT)', color=colors['tta'], linewidth=2)
    plt.plot(x, smooth(bk_accs), label='BK-CoMerge (Ours)', color=colors['bk'], linewidth=2)
    plt.plot(x, smooth(ega_accs), label='EGA-BK-CoMerge (Proposed)', color=colors['ega'], linewidth=2.5)
    plt.plot(x, smooth(att_accs), label='ATT-BK-CoMerge (Proposed)', color=colors['att'], linewidth=2)
    
    # Plot raw points with low opacity to show the variation
    plt.scatter(x, static_accs, color=colors['static'], alpha=0.3, s=15)
    plt.scatter(x, tta_accs, color=colors['tta'], alpha=0.3, s=15)
    plt.scatter(x, bk_accs, color=colors['bk'], alpha=0.3, s=15)
    plt.scatter(x, ega_accs, color=colors['ega'], alpha=0.3, s=15)
    plt.scatter(x, att_accs, color=colors['att'], alpha=0.3, s=15)
    
    # Plot segment boundary lines
    boundaries = [10, 20, 30, 40]
    for b in boundaries:
        plt.axvline(x=b-0.5, color='gray', linestyle='--', alpha=0.7)
        
    # Annotate segments at the top of the plot
    segment_names = [
        "Clean\nMNIST",
        "Noisy\nMNIST",
        "Clean\nFashion",
        "Noisy\nFashion",
        "Novel\nKMNIST"
    ]
    segment_centers = [4.5, 14.5, 24.5, 34.5, 44.5]
    for name, center in zip(segment_names, segment_centers):
        plt.text(center, 105, name, fontsize=10, fontweight='bold', ha='center', va='bottom')
        
    plt.xlim(-1, 50)
    plt.ylim(-5, 115)
    plt.xlabel("Streaming Step (Batch Index)", fontsize=12)
    plt.ylabel("Batch Classification Accuracy (%)", fontsize=12)
    plt.title("Test-Time Model Merging Performance on Non-Stationary Stream", fontsize=14, pad=25)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower left', fontsize=10, frameon=True, shadow=False)
    
    plt.tight_layout()
    plt.savefig("accuracies.pdf", bbox_inches='tight')
    plt.savefig("accuracies.png", dpi=300, bbox_inches='tight')
    print("Plot saved successfully as accuracies.pdf and accuracies.png!")

if __name__ == '__main__':
    main()
