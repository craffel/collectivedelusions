import os
import torch
import json
import numpy as np
from run_tta import (
    CNNEncoder, TaskExpert, DifferentiableMergedModel,
    build_test_stream, run_evaluation
)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running ablation study on epsilon on device: {device}")
    
    # Load trained expert models
    print("Loading experts...")
    experts = []
    tasks = ["MNIST", "FashionMNIST", "KMNIST"]
    for task in tasks:
        encoder = CNNEncoder()
        expert = TaskExpert(encoder).to(device)
        checkpoint_path = f"./checkpoints/expert_{task}.pt"
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Expert model not found at {checkpoint_path}. Please run train_experts.py first.")
        expert.load_state_dict(torch.load(checkpoint_path, map_location=device))
        expert.eval()
        experts.append(expert)
        
    # Load Fisher Information Matrices
    print("Loading FIMs...")
    fims = []
    for task in tasks:
        fim_path = f"./checkpoints/fim_{task}.pt"
        fim = torch.load(fim_path, map_location=device)
        fims.append(fim)
        
    # Define epsilon sweep values
    epsilon_values = [1e-8, 1e-6, 1e-4, 1e-2, 1.0]
    
    # We will run on Alternating and Sequential Streams
    # under Gaussian Noise and Contrast corruptions.
    streams = ["Alternating", "Sequential"]
    environments = ["Gaussian Noise", "Contrast"]
    loss_modes = ["teacher-supervised", "teacher-free"]
    
    ablation_results = {}
    
    for loss_mode in loss_modes:
        print(f"\n=================== LOSS MODE: {loss_mode.upper()} ===================")
        ablation_results[loss_mode] = {}
        for stream_type in streams:
            print(f"\n--- Stream Type: {stream_type} ---")
            ablation_results[loss_mode][stream_type] = {}
            
            # Build the stream
            stream = build_test_stream(stream_type, batch_size=64, num_batches_per_task=50)
            
            for env in environments:
                print(f"\nDomain Shift / Corruption: {env}")
                ablation_results[loss_mode][stream_type][env] = {}
                
                for eps in epsilon_values:
                    # Fix alpha = 0.1 as default
                    acc, _ = run_evaluation("FiT-Merge (Ours)", stream, env, experts, fims, device, loss_mode, alpha=0.1, epsilon=eps)
                    ablation_results[loss_mode][stream_type][env][str(eps)] = acc
                    print(f"  epsilon = {eps:<5}: {acc:.2f}%")
                    
    # Save the ablation results
    with open("ablation_epsilon_results.json", "w") as f:
        json.dump(ablation_results, f, indent=4)
    print("\nAblation study complete! Results saved to ablation_epsilon_results.json")
    
    # Generate LaTeX Table
    print("\nGenerating LaTeX Table for Ablation Study...")
    latex = """\\begin{table*}[t]
\\caption{Ablation study on the effect of the smoothing epsilon $\\epsilon$ for FiT-Merge (with $\\alpha = 0.1$). We report multi-task average accuracy (\\%) across Alternating and Sequential streams under Gaussian Noise and Contrast corruptions.}
\\label{tab:ablation_epsilon}
\\vskip 0.05in
\\begin{center}
\\begin{scriptsize}
\\begin{sc}
\\begin{tabular}{lcccccccc}
\\toprule
& \\multicolumn{4}{c}{\\textbf{Alternating Stream}} & \\multicolumn{4}{c}{\\textbf{Sequential Stream}} \\\\
\\cmidrule(r){2-5} \\cmidrule(l){6-9}
& \\multicolumn{2}{c}{\\textbf{Supervised}} & \\multicolumn{2}{c}{\\textbf{Teacher-Free}} & \\multicolumn{2}{c}{\\textbf{Supervised}} & \\multicolumn{2}{c}{\\textbf{Teacher-Free}} \\\\
\\cmidrule(r){2-3} \\cmidrule(l){4-5} \\cmidrule(r){6-7} \\cmidrule(l){8-9}
$\\epsilon$ & Noise & Contrast & Noise & Contrast & Noise & Contrast & Noise & Contrast \\\\
\\midrule
"""
    for eps in epsilon_values:
        val_alt_sup_noise = ablation_results["teacher-supervised"]["Alternating"]["Gaussian Noise"][str(eps)]
        val_alt_sup_contrast = ablation_results["teacher-supervised"]["Alternating"]["Contrast"][str(eps)]
        val_alt_free_noise = ablation_results["teacher-free"]["Alternating"]["Gaussian Noise"][str(eps)]
        val_alt_free_contrast = ablation_results["teacher-free"]["Alternating"]["Contrast"][str(eps)]
        
        val_seq_sup_noise = ablation_results["teacher-supervised"]["Sequential"]["Gaussian Noise"][str(eps)]
        val_seq_sup_contrast = ablation_results["teacher-supervised"]["Sequential"]["Contrast"][str(eps)]
        val_seq_free_noise = ablation_results["teacher-free"]["Sequential"]["Gaussian Noise"][str(eps)]
        val_seq_free_contrast = ablation_results["teacher-free"]["Sequential"]["Contrast"][str(eps)]
        
        latex += f"{str(eps):<8} & {val_alt_sup_noise:.2f} & {val_alt_sup_contrast:.2f} & {val_alt_free_noise:.2f} & {val_alt_free_contrast:.2f} & {val_seq_sup_noise:.2f} & {val_seq_sup_contrast:.2f} & {val_seq_free_noise:.2f} & {val_seq_free_contrast:.2f} \\\\\n"
        
    latex += """\\bottomrule
\\end{tabular}
\\end{sc}
\\end{scriptsize}
\\end{center}
\\vskip -0.15in
\\end{table*}
"""
    with open("table_ablation_epsilon.tex", "w") as f:
        f.write(latex)
    print("LaTeX table saved to table_ablation_epsilon.tex")

if __name__ == "__main__":
    main()
