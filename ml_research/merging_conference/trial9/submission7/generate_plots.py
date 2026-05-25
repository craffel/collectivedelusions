import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from eval_stream import generate_stream, Evaluators, SimpleCNN

def run_evaluation_with_curves(evaluate_fn, stream_batches, expert0, expert1, device="cpu"):
    # Reset method state before stream starts
    evaluate_fn(None, None, None, expert0, expert1, reset=True, device=device)
    
    accuracies = []
    for idx, (images, labels, segment_name) in enumerate(stream_batches):
        correct, total = evaluate_fn(images, labels, idx, expert0, expert1, reset=False, device=device)
        acc = 100.0 * correct / total
        accuracies.append(acc)
    return accuracies

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device for plot generation: {device}")
    
    stream_batches = generate_stream(device=device)
    
    # Load Experts
    expert0_std = SimpleCNN(use_cosface=False).to(device)
    expert1_std = SimpleCNN(use_cosface=False).to(device)
    expert0_std.load_state_dict(torch.load("models/mnist_standard.pt", map_location=device))
    expert1_std.load_state_dict(torch.load("models/fashionmnist_standard.pt", map_location=device))
    
    expert0_cos = SimpleCNN(use_cosface=True).to(device)
    expert1_cos = SimpleCNN(use_cosface=True).to(device)
    expert0_cos.load_state_dict(torch.load("models/mnist_cosface.pt", map_location=device))
    expert1_cos.load_state_dict(torch.load("models/fashionmnist_cosface.pt", map_location=device))
    
    methods_to_plot = [
        ("Static Merging", "static_merging", "gray", "--"),
        ("Fixed TTA", "fixed_tta", "blue", "-."),
        ("BK-CoMerge", "bk_co_merge", "orange", ":"),
        ("AdaSim-CoMerge (Ours)", "adasim_co_merge", "red", "-")
    ]
    
    # Evaluate Standard Experts
    print("Evaluating Standard Experts for plotting...")
    evals_std = Evaluators()
    std_curves = {}
    for name, fn_name, color, style in methods_to_plot:
        fn = getattr(evals_std, fn_name)
        std_curves[name] = run_evaluation_with_curves(fn, stream_batches, expert0_std, expert1_std, device=device)
        
    # Evaluate CosFace Experts
    print("Evaluating CosFace Experts for plotting...")
    evals_cos = Evaluators()
    cos_curves = {}
    for name, fn_name, color, style in methods_to_plot:
        fn = getattr(evals_cos, fn_name)
        cos_curves[name] = run_evaluation_with_curves(fn, stream_batches, expert0_cos, expert1_cos, device=device)
        
    # Generate Plots
    print("Generating plot...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Define vertical boundary lines and text positions
    segment_boundaries = [10, 20, 30, 40]
    segment_names = ["Clean MNIST", "Noisy MNIST", "Clean Fashion", "Noisy Fashion", "Novel KMNIST"]
    segment_centers = [5, 15, 25, 35, 45]
    
    # Standard Experts Plot
    for name, _, color, style in methods_to_plot:
        ax1.plot(std_curves[name], label=name, color=color, linestyle=style, linewidth=2)
    
    for b in segment_boundaries:
        ax1.axvline(x=b-0.5, color="gray", linestyle="--", alpha=0.5)
    
    for center, s_name in zip(segment_centers, segment_names):
        ax1.text(center, 95, s_name, ha="center", va="center", fontsize=9, fontweight="bold",
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
        
    ax1.set_title("Standard CNN Experts", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Batch Accuracy (%)", fontsize=10)
    ax1.set_ylim(0, 105)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower left", fontsize=9)
    
    # CosFace Experts Plot
    for name, _, color, style in methods_to_plot:
        ax2.plot(cos_curves[name], label=name, color=color, linestyle=style, linewidth=2)
        
    for b in segment_boundaries:
        ax2.axvline(x=b-0.5, color="gray", linestyle="--", alpha=0.5)
        
    for center, s_name in zip(segment_centers, segment_names):
        ax2.text(center, 95, s_name, ha="center", va="center", fontsize=9, fontweight="bold",
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
        
    ax2.set_title("CosFace Experts", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Batch Index in Stream", fontsize=10)
    ax2.set_ylabel("Batch Accuracy (%)", fontsize=10)
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower left", fontsize=9)
    
    plt.tight_layout()
    plt.savefig("stream_performance.png", dpi=300)
    print("Plot saved to stream_performance.png")

if __name__ == "__main__":
    main()
