import os
import torch
import json
import numpy as np
from train_experts import CNNEncoder, TaskExpert, get_dataloader, compute_fim
from run_tta import build_test_stream, run_evaluation

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running ablation study on FIM sample size on device: {device}")
    
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
        
    sample_sizes = [10, 50, 100, 250, 500]
    
    # Pre-compute FIMs for all tasks for each sample size
    fims_by_size = {}
    for size in sample_sizes:
        print(f"\n--- Estimating FIMs with {size} samples ---")
        fims_by_size[size] = []
        for task in tasks:
            # We load the FIM if size is 500 and we already have it saved
            if size == 500:
                fim_path = f"./checkpoints/fim_{task}.pt"
                if os.path.exists(fim_path):
                    print(f"Loading pre-computed 500-sample FIM for {task}")
                    fim = torch.load(fim_path, map_location=device)
                    fims_by_size[size].append(fim)
                    continue
            
            print(f"Computing FIM for {task} with {size} samples...")
            fim_loader = get_dataloader(task, train=True, batch_size=1)
            fim = compute_fim(experts[tasks.index(task)], fim_loader, device, num_samples=size)
            fims_by_size[size].append(fim)
            
    # We will run on Alternating and Sequential streams,
    # under Gaussian Noise and Contrast corruptions,
    # using teacher-supervised and teacher-free loss modes.
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
                
                for size in sample_sizes:
                    fims = fims_by_size[size]
                    # Evaluate FiT-Merge with alpha = 0.1 (optimal default)
                    acc, _ = run_evaluation("FiT-Merge (Ours)", stream, env, experts, fims, device, loss_mode, alpha=0.1)
                    ablation_results[loss_mode][stream_type][env][str(size)] = acc
                    print(f"  sample_size = {size:<4}: {acc:.2f}%")
                    
    # Save the ablation results
    with open("ablation_samples_results.json", "w") as f:
        json.dump(ablation_results, f, indent=4)
    print("\nSample size ablation study complete! Results saved to ablation_samples_results.json")
    
    # Generate LaTeX Table
    print("\nGenerating LaTeX Table for FIM Sample Size Ablation...")
    latex = """\\begin{table*}[t]
\\caption{Ablation study on the effect of the number of samples used to estimate the diagonal Fisher Information Matrix (FIM) for FiT-Merge ($\alpha = 0.1$). We report multi-task average accuracy (\%) across Alternating and Sequential streams under Gaussian Noise and Contrast corruptions.}
\\label{tab:ablation_samples}
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
Samples & Noise & Contrast & Noise & Contrast & Noise & Contrast & Noise & Contrast \\\\
\\midrule
"""
    for size in sample_sizes:
        val_alt_sup_noise = ablation_results["teacher-supervised"]["Alternating"]["Gaussian Noise"][str(size)]
        val_alt_sup_contrast = ablation_results["teacher-supervised"]["Alternating"]["Contrast"][str(size)]
        val_alt_free_noise = ablation_results["teacher-free"]["Alternating"]["Gaussian Noise"][str(size)]
        val_alt_free_contrast = ablation_results["teacher-free"]["Alternating"]["Contrast"][str(size)]
        
        val_seq_sup_noise = ablation_results["teacher-supervised"]["Sequential"]["Gaussian Noise"][str(size)]
        val_seq_sup_contrast = ablation_results["teacher-supervised"]["Sequential"]["Contrast"][str(size)]
        val_seq_free_noise = ablation_results["teacher-free"]["Sequential"]["Gaussian Noise"][str(size)]
        val_seq_free_contrast = ablation_results["teacher-free"]["Sequential"]["Contrast"][str(size)]
        
        latex += f"{size:<7} & {val_alt_sup_noise:.2f} & {val_alt_sup_contrast:.2f} & {val_alt_free_noise:.2f} & {val_alt_free_contrast:.2f} & {val_seq_sup_noise:.2f} & {val_seq_sup_contrast:.2f} & {val_seq_free_noise:.2f} & {val_seq_free_contrast:.2f} \\\\\n"
        
    latex += """\\bottomrule
\\end{tabular}
\\end{sc}
\\end{scriptsize}
\\end{center}
\\vskip -0.15in
\\end{table*}
"""
    with open("table_ablation_samples.tex", "w") as f:
        f.write(latex)
    print("LaTeX table saved to table_ablation_samples.tex")

if __name__ == "__main__":
    main()
