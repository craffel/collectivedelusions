import subprocess
import json
import os
import argparse
import pandas as pd

def run_eval(arch, merge, calib, scale=1.0, reset_thresh=20, drop_rate=0.2, gamma=2.0, compensation="inverse", num_bits=None, quant_mode="per_tensor", corruption=None):
    cmd = [
        "python3", "-m", "src.eval",
        "--arch", arch,
        "--merge_method", merge,
        "--calib_method", calib,
        "--scale", str(scale),
        "--reset_thresh", str(reset_thresh),
        "--drop_rate", str(drop_rate),
        "--gamma", str(gamma),
        "--compensation", compensation,
        "--quant_mode", quant_mode
    ]
    if num_bits is not None:
        cmd += ["--num_bits", str(num_bits)]
    if corruption is not None:
        cmd += ["--corruption", corruption]
        
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running command: {' '.join(cmd)}")
        print(result.stderr)
        return None
        
    # Parse output to find the average accuracy
    lines = result.stdout.split("\n")
    avg_acc = None
    for line in lines:
        if "Average Multi-Task Accuracy:" in line:
            # Format is "Average Multi-Task Accuracy: XX.XX%"
            pct_str = line.split(":")[-1].strip().replace("%", "")
            avg_acc = float(pct_str)
            break
    return avg_acc

def main():
    parser = argparse.ArgumentParser(description="Run Merging Experiments")
    parser.add_argument("--arch", type=str, required=True, choices=["resnet18", "mlp"])
    args = parser.parse_args()
    
    print(f"==================================================")
    print(f"Running Experiments for Architecture: {args.arch.upper()}")
    print(f"==================================================")
    
    results = []
    
    # 1. Base Uncalibrated and Tuned TA Baselines
    # Weight Averaging (WA)
    print("\nEvaluating Weight Averaging (WA)...")
    acc_clean = run_eval(args.arch, "wa", "none")
    acc_int8_tensor = run_eval(args.arch, "wa", "none", num_bits=8, quant_mode="per_tensor")
    acc_int8_channel = run_eval(args.arch, "wa", "none", num_bits=8, quant_mode="per_channel")
    acc_noise = run_eval(args.arch, "wa", "none", corruption="noise")
    acc_blur = run_eval(args.arch, "wa", "none", corruption="blur")
    results.append({
        "Method": "Weight Averaging (WA)",
        "FP32 Clean": acc_clean,
        "INT8 Per-Tensor": acc_int8_tensor,
        "INT8 Per-Channel": acc_int8_channel,
        "OOD Noise (FP32)": acc_noise,
        "OOD Blur (FP32)": acc_blur
    })
    
    # Task Arithmetic (TA) - Tune global scale lambda
    print("\nTuning Task Arithmetic (TA)...")
    best_scale = 1.0
    best_acc = 0.0
    scales_to_sweep = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5] if args.arch == "mlp" else [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
    for sc in scales_to_sweep:
        acc = run_eval(args.arch, "ta", "none", scale=sc)
        if acc is not None and acc > best_acc:
            best_acc = acc
            best_scale = sc
    print(f"Best scale for TA: lambda = {best_scale} (Accuracy: {best_acc}%)")
    
    acc_clean = run_eval(args.arch, "ta", "none", scale=best_scale)
    acc_int8_tensor = run_eval(args.arch, "ta", "none", scale=best_scale, num_bits=8, quant_mode="per_tensor")
    acc_int8_channel = run_eval(args.arch, "ta", "none", scale=best_scale, num_bits=8, quant_mode="per_channel")
    acc_noise = run_eval(args.arch, "ta", "none", scale=best_scale, corruption="noise")
    acc_blur = run_eval(args.arch, "ta", "none", scale=best_scale, corruption="blur")
    results.append({
        "Method": f"Tuned TA (lambda = {best_scale:.2f})",
        "FP32 Clean": acc_clean,
        "INT8 Per-Tensor": acc_int8_tensor,
        "INT8 Per-Channel": acc_int8_channel,
        "OOD Noise (FP32)": acc_noise,
        "OOD Blur (FP32)": acc_blur
    })
    
    # 2. Advanced Merging without Calibration: TIES and DARE
    for merge in ["ties", "dare"]:
        print(f"\nEvaluating {merge.upper()} without calibration...")
        acc_clean = run_eval(args.arch, merge, "none")
        acc_int8_tensor = run_eval(args.arch, merge, "none", num_bits=8, quant_mode="per_tensor")
        acc_int8_channel = run_eval(args.arch, merge, "none", num_bits=8, quant_mode="per_channel")
        acc_noise = run_eval(args.arch, merge, "none", corruption="noise")
        acc_blur = run_eval(args.arch, merge, "none", corruption="blur")
        results.append({
            "Method": f"{merge.upper()} (Uncalibrated)",
            "FP32 Clean": acc_clean,
            "INT8 Per-Tensor": acc_int8_tensor,
            "INT8 Per-Channel": acc_int8_channel,
            "OOD Noise (FP32)": acc_noise,
            "OOD Blur (FP32)": acc_blur
        })
        
    # 3. Dense Calibration Methods on top of average updates (TA with scale = 1.0)
    for calib in ["u_ipr", "hns", "qr_ipr", "wcpr", "qr_sc_wcpr"]:
        print(f"\nEvaluating TA + {calib.upper()} Calibration...")
        acc_clean = run_eval(args.arch, "ta", calib, compensation="none")
        acc_int8_tensor = run_eval(args.arch, "ta", calib, compensation="none", num_bits=8, quant_mode="per_tensor")
        acc_int8_channel = run_eval(args.arch, "ta", calib, compensation="none", num_bits=8, quant_mode="per_channel")
        acc_noise = run_eval(args.arch, "ta", calib, compensation="none", corruption="noise")
        acc_blur = run_eval(args.arch, "ta", calib, compensation="none", corruption="blur")
        
        results.append({
            "Method": f"TA + {calib.upper()}",
            "FP32 Clean": acc_clean,
            "INT8 Per-Tensor": acc_int8_tensor,
            "INT8 Per-Channel": acc_int8_channel,
            "OOD Noise (FP32)": acc_noise,
            "OOD Blur (FP32)": acc_blur
        })
        
    # 4. Sparse Merging (TIES and DARE) with Wasserstein and Our QR-SC-WCPR
    # This evaluates our hypothesis about Sparsity-Calibration Mismatch
    for merge in ["ties", "dare"]:
        # Standard WCPR (which collapses sparsity/collapses under TIES/DARE)
        print(f"\nEvaluating {merge.upper()} + WCPR Calibration...")
        acc_clean = run_eval(args.arch, merge, "wcpr")
        acc_int8_tensor = run_eval(args.arch, merge, "wcpr", num_bits=8, quant_mode="per_tensor")
        acc_int8_channel = run_eval(args.arch, merge, "wcpr", num_bits=8, quant_mode="per_channel")
        acc_noise = run_eval(args.arch, merge, "wcpr", corruption="noise")
        acc_blur = run_eval(args.arch, merge, "wcpr", corruption="blur")
        results.append({
            "Method": f"{merge.upper()} + Standard WCPR",
            "FP32 Clean": acc_clean,
            "INT8 Per-Tensor": acc_int8_tensor,
            "INT8 Per-Channel": acc_int8_channel,
            "OOD Noise (FP32)": acc_noise,
            "OOD Blur (FP32)": acc_blur
        })
        
        # Our QR-SC-WCPR (proposed)
        print(f"\nEvaluating {merge.upper()} + QR-SC-WCPR (Ours)...")
        acc_clean = run_eval(args.arch, merge, "qr_sc_wcpr", compensation="inverse", gamma=2.0)
        acc_int8_tensor = run_eval(args.arch, merge, "qr_sc_wcpr", compensation="inverse", gamma=2.0, num_bits=8, quant_mode="per_tensor")
        acc_int8_channel = run_eval(args.arch, merge, "qr_sc_wcpr", compensation="inverse", gamma=2.0, num_bits=8, quant_mode="per_channel")
        acc_noise = run_eval(args.arch, merge, "qr_sc_wcpr", compensation="inverse", gamma=2.0, corruption="noise")
        acc_blur = run_eval(args.arch, merge, "qr_sc_wcpr", compensation="inverse", gamma=2.0, corruption="blur")
        results.append({
            "Method": f"{merge.upper()} + QR-SC-WCPR (Ours)",
            "FP32 Clean": acc_clean,
            "INT8 Per-Tensor": acc_int8_tensor,
            "INT8 Per-Channel": acc_int8_channel,
            "OOD Noise (FP32)": acc_noise,
            "OOD Blur (FP32)": acc_blur
        })
        
    # Print beautiful results table
    df = pd.DataFrame(results)
    print(f"\n\nResults for {args.arch.upper()}:")
    print(df.to_markdown(index=False))
    
    # Save results to a file
    os.makedirs("results", exist_ok=True)
    df.to_json(f"results/{args.arch}_results.json", orient="records", indent=4)
    df.to_markdown(open(f"results/{args.arch}_results.md", "w"), index=False)
    print(f"Results saved to results/{args.arch}_results.json and results/{args.arch}_results.md")

if __name__ == "__main__":
    main()
