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
    print(f"Running ablation study on device: {device}")
    
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
        
    # Define hyperparameter sweep values
    alpha_values = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    
    # We will run on the Alternating Stream and Sequential Stream
    # under Gaussian Noise and Contrast corruptions to get a comprehensive view.
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
                
                for alpha in alpha_values:
                    # Note: alpha=0.0 with use_fim is mathematically standard TTA (since FIM weights become 1.0)
                    acc, _ = run_evaluation("FiT-Merge (Ours)", stream, env, experts, fims, device, loss_mode, alpha=alpha)
                    ablation_results[loss_mode][stream_type][env][str(alpha)] = acc
                    print(f"  alpha = {alpha:<5}: {acc:.2f}%")
                    
    # Save the ablation results
    with open("ablation_alpha_results.json", "w") as f:
        json.dump(ablation_results, f, indent=4)
    print("\nAblation study complete! Results saved to ablation_alpha_results.json")
    
    # Generate LaTeX Table
    print("\nGenerating LaTeX Table for Ablation Study...")
    latex = """\\begin{table}[h]
\\caption{Ablation study on the effect of the Fisher modulation exponent $\\alpha$ under the Alternating Stream setting. We report multi-task average accuracy (\\%). Note that $\\alpha = 0.0$ corresponds to standard layer-wise TTA.}
\\label{tab:ablation_alpha}
\\vskip 0.05in
\\begin{center}
\\begin{scriptsize}
\\begin{sc}
\\begin{tabular}{lcccc}
\\toprule
& \\multicolumn{2}{c}{\\textbf{Teacher-Supervised}} & \\multicolumn{2}{c}{\\textbf{Teacher-Free}} \\\\
\\cmidrule(r){2-3} \\cmidrule(l){4-5}
$\\alpha$ & Noise & Contrast & Noise & Contrast \\\\
\\midrule
"""
    for alpha in alpha_values:
        val_sup_noise = ablation_results["teacher-supervised"]["Alternating"]["Gaussian Noise"][str(alpha)]
        val_sup_contrast = ablation_results["teacher-supervised"]["Alternating"]["Contrast"][str(alpha)]
        val_free_noise = ablation_results["teacher-free"]["Alternating"]["Gaussian Noise"][str(alpha)]
        val_free_contrast = ablation_results["teacher-free"]["Alternating"]["Contrast"][str(alpha)]
        
        # Bold the maximum value in each column
        # Quick helper to format with bold if it is the max
        latex += f"{alpha:<5} & {val_sup_noise:.2f} & {val_sup_contrast:.2f} & {val_free_noise:.2f} & {val_free_contrast:.2f} \\\\\n"
        
    latex += """\\bottomrule
\\end{tabular}
\\end{sc}
\\end{scriptsize}
\\end{center}
\\vskip -0.15in
\\end{table}
"""
    with open("table_ablation_alpha.tex", "w") as f:
        f.write(latex)
    print("LaTeX table saved to table_ablation_alpha.tex")

if __name__ == "__main__":
    main()
