import torch
import os
import random
import numpy as np
from evaluation import load_test_data, build_stream, compute_joint_fisher, evaluate_method

def main():
    # Set random seed for deterministic evaluation
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running alpha sweep on device: {device}")
    
    # Disable cuDNN to bypass CUDNN_STATUS_NOT_INITIALIZED error
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = False
        
    # Paths to checkpoints
    encoder_paths = [
        "checkpoints/mnist_encoder.pth",
        "checkpoints/fmnist_encoder.pth",
        "checkpoints/kmnist_encoder.pth"
    ]
    head_paths = [
        "checkpoints/mnist_head.pth",
        "checkpoints/fmnist_head.pth",
        "checkpoints/kmnist_head.pth"
    ]
    base_path = "checkpoints/base_encoder.pth"
    
    # Check if files exist
    for p in encoder_paths + head_paths + [base_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required checkpoint not found: {p}")
            
    # Load checkpoints
    expert_encoders = []
    expert_heads = []
    experts_sds = []
    
    for k in range(3):
        enc_sd = torch.load(encoder_paths[k], map_location=device)
        hd_sd = torch.load(head_paths[k], map_location=device)
        expert_encoders.append(enc_sd)
        expert_heads.append(hd_sd)
        experts_sds.append(enc_sd)
        
    base_sd = torch.load(base_path, map_location=device)
    
    # Load/compute joint Fisher
    if os.path.exists("checkpoints/joint_fisher.pth"):
        joint_fisher = torch.load("checkpoints/joint_fisher.pth", map_location=device)
        print("Loaded joint Fisher from checkpoints/joint_fisher.pth")
    else:
        joint_fisher = compute_joint_fisher(encoder_paths, head_paths, device)
        torch.save(joint_fisher, "checkpoints/joint_fisher.pth")
    
    test_data = load_test_data()
    
    # Alpha values to sweep over
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
    domains = ["clean", "noise", "blur", "contrast"]
    
    results = {}
    
    for alpha in alphas:
        results[alpha] = {}
        print(f"\nEvaluating alpha = {alpha}")
        
        for domain in domains:
            # Build test batches for sequential stream
            batches = build_stream(test_data, stream_type="sequential", corruption_type=domain)
            
            acc = evaluate_method(
                method_name="Fisher-PC-Merge (Proposed)",
                test_batches=batches,
                experts_sds=experts_sds,
                base_sd=base_sd,
                expert_encoders=expert_encoders,
                expert_heads=expert_heads,
                joint_fisher=joint_fisher,
                device=device,
                lr=0.10,
                alpha_lfwa=alpha
            )
            results[alpha][domain] = acc
            print(f"  Domain: {domain:10s} Accuracy: {acc:.2f}%")
            
    print("\n" + "="*80)
    print("ALPHA SWEEP RESULTS FOR SEQUENTIAL STREAM")
    print("="*80)
    header = "| Alpha | Clean | Gaussian Noise | Gaussian Blur | Contrast | Average |"
    sep = "| :---: | :---: | :---: | :---: | :---: | :---: |"
    print(header)
    print(sep)
    for alpha in alphas:
        acc_clean = results[alpha]["clean"]
        acc_noise = results[alpha]["noise"]
        acc_blur = results[alpha]["blur"]
        acc_contrast = results[alpha]["contrast"]
        acc_avg = (acc_clean + acc_noise + acc_blur + acc_contrast) / 4.0
        row = f"| {alpha:.2f} | {acc_clean:.2f}% | {acc_noise:.2f}% | {acc_blur:.2f}% | {acc_contrast:.2f}% | **{acc_avg:.2f}%** |"
        print(row)
        
    # Save sweep results to file
    with open("checkpoints/results_alpha_sweep.txt", "w") as f:
        f.write("Alpha Sweep Results for Fisher-PC-Merge on Sequential Stream\n")
        f.write(header + "\n")
        f.write(sep + "\n")
        for alpha in alphas:
            acc_clean = results[alpha]["clean"]
            acc_noise = results[alpha]["noise"]
            acc_blur = results[alpha]["blur"]
            acc_contrast = results[alpha]["contrast"]
            acc_avg = (acc_clean + acc_noise + acc_blur + acc_contrast) / 4.0
            row = f"| {alpha:.2f} | {acc_clean:.2f}% | {acc_noise:.2f}% | {acc_blur:.2f}% | {acc_contrast:.2f}% | **{acc_avg:.2f}%** |"
            f.write(row + "\n")

if __name__ == "__main__":
    main()
