import torch
import os
import random
import numpy as np
from evaluation import load_test_data, build_stream, compute_joint_fisher, get_or_compute_prototypes, evaluate_method

def main():
    # Set random seed for deterministic evaluation
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on device: {device}")
    
    # Disable cuDNN to bypass CUDNN_STATUS_NOT_INITIALIZED error
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = False
        
    # 1. Paths to expert checkpoints
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
            
    # 2. Load checkpoints into memory
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
    
    # 3. Compute joint Fisher sensitivity
    joint_fisher = compute_joint_fisher(encoder_paths, head_paths, device)
    
    # Save the computed joint Fisher sensitivity for reference
    torch.save(joint_fisher, "checkpoints/joint_fisher.pth")
    print("Saved joint Fisher sensitivity prior to checkpoints/joint_fisher.pth")
    
    # 3.5. Compute or load class prototypes (CPA-Merge)
    prototypes = get_or_compute_prototypes(encoder_paths, device)
    
    # 4. Load test datasets
    test_data = load_test_data()
    
    # 5. Define Sweep Parameters
    streams = ["sequential", "alternating"]
    domains = ["clean", "noise", "blur", "contrast"]
    methods = [
        "static",
        "TTA (AdaMerging)",
        "LFWA (Fisher)",
        "CPA-Merge (PD-Routing + Alignment)",
        "PC-Merge (OPR + Projection)",
        "Fisher-PC-Merge (Proposed)",
        "Fisher-PC-Merge + AD-Merge (Proposed)"
    ]
    
    results = {}
    
    # Loop over all streams and domains
    for stream in streams:
        results[stream] = {}
        for domain in domains:
            results[stream][domain] = {}
            print(f"\nEvaluating stream: {stream.upper()}, domain: {domain.upper()}")
            
            # Build the test batches for this stream and domain
            batches = build_stream(test_data, stream_type=stream, corruption_type=domain)
            
            for method in methods:
                # Map method to evaluation behavior
                # LFWA and Fisher-PC-Merge need joint_fisher. Other methods do not (or ignore it).
                fish = joint_fisher if ("LFWA" in method or "Fisher" in method) else None
                
                # We use lr=0.10 for TTA, and alpha_lfwa=0.5
                acc = evaluate_method(
                    method_name=method,
                    test_batches=batches,
                    experts_sds=experts_sds,
                    base_sd=base_sd,
                    expert_encoders=expert_encoders,
                    expert_heads=expert_heads,
                    joint_fisher=fish,
                    device=device,
                    lr=0.10,
                    alpha_lfwa=0.5,
                    prototypes=prototypes
                )
                
                results[stream][domain][method] = acc
                print(f"  {method:30s} Accuracy: {acc:.2f}%")
                
    # 6. Format and Print Results in Markdown Table
    print("\n" + "="*80)
    print("FINAL RESULTS SWEEP")
    print("="*80)
    
    for stream in streams:
        print(f"\n### {stream.upper()} STREAM ACCURACY RESULTS")
        header = f"| Method | Clean | Gaussian Noise | Gaussian Blur | Contrast | Average |"
        sep = "| :--- | :---: | :---: | :---: | :---: | :---: |"
        print(header)
        print(sep)
        for method in methods:
            acc_clean = results[stream]["clean"][method]
            acc_noise = results[stream]["noise"][method]
            acc_blur = results[stream]["blur"][method]
            acc_contrast = results[stream]["contrast"][method]
            acc_avg = (acc_clean + acc_noise + acc_blur + acc_contrast) / 4.0
            row = f"| {method} | {acc_clean:.2f}% | {acc_noise:.2f}% | {acc_blur:.2f}% | {acc_contrast:.2f}% | **{acc_avg:.2f}%** |"
            print(row)
            
    # Also save results to a text file for easy extraction
    with open("checkpoints/results_sweep.txt", "w") as f:
        for stream in streams:
            f.write(f"\n--- {stream.upper()} STREAM ---\n")
            for method in methods:
                acc_clean = results[stream]["clean"][method]
                acc_noise = results[stream]["noise"][method]
                acc_blur = results[stream]["blur"][method]
                acc_contrast = results[stream]["contrast"][method]
                acc_avg = (acc_clean + acc_noise + acc_blur + acc_contrast) / 4.0
                f.write(f"{method:30s} | Clean: {acc_clean:.2f}% | Noise: {acc_noise:.2f}% | Blur: {acc_blur:.2f}% | Contrast: {acc_contrast:.2f}% | Avg: {acc_avg:.2f}%\n")

if __name__ == "__main__":
    main()
