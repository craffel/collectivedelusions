import subprocess
import json
import os
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
        "--quant_mode", quant_mode,
        "--device", "cpu"
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
            pct_str = line.split(":")[-1].strip().replace("%", "")
            avg_acc = float(pct_str)
            break
    return avg_acc

def evaluate_setting(arch, compensation, gamma):
    print(f"Running: Arch={arch.upper()}, Compensation={compensation}, Gamma={gamma}")
    acc_clean = run_eval(arch, "ties", "qr_sc_wcpr", compensation=compensation, gamma=gamma)
    acc_int8_tensor = run_eval(arch, "ties", "qr_sc_wcpr", compensation=compensation, gamma=gamma, num_bits=8, quant_mode="per_tensor")
    acc_int8_channel = run_eval(arch, "ties", "qr_sc_wcpr", compensation=compensation, gamma=gamma, num_bits=8, quant_mode="per_channel")
    acc_noise = run_eval(arch, "ties", "qr_sc_wcpr", compensation=compensation, gamma=gamma, corruption="noise")
    acc_blur = run_eval(arch, "ties", "qr_sc_wcpr", compensation=compensation, gamma=gamma, corruption="blur")
    
    return {
        "FP32 Clean": acc_clean,
        "INT8 Per-Tensor": acc_int8_tensor,
        "INT8 Per-Channel": acc_int8_channel,
        "OOD Noise": acc_noise,
        "OOD Blur": acc_blur
    }

def main():
    print("==================================================")
    print("Starting Comprehensive Ablation Study for QR-SC-WCPR")
    print("==================================================")
    
    results = []
    
    for arch in ["mlp", "resnet18"]:
        print(f"\n--- Sweeping Sparsity Compensation Methods ({arch.upper()}) ---")
        # 1. Compensation sweep (with default Gamma=2.0)
        for comp in ["none", "sqrt", "inverse"]:
            metrics = evaluate_setting(arch, comp, 2.0)
            results.append({
                "Architecture": arch.upper(),
                "Ablation Dimension": "Sparsity Compensation",
                "Setting": f"Comp: {comp}",
                "Compensation": comp,
                "Gamma": 2.0,
                **metrics
            })
            
        print(f"\n--- Sweeping Outlier Rejection Threshold Gamma ({arch.upper()}) ---")
        # 2. Gamma sweep (with default Compensation=inverse)
        for gamma in [1.0, 1.5, 2.0, 3.0, 4.0, 100.0]:
            # Skip repeating Gamma=2.0 since it was already run above
            existing = [r for r in results if r["Architecture"] == arch.upper() and r["Ablation Dimension"] == "Sparsity Compensation" and r["Compensation"] == "inverse" and r["Gamma"] == 2.0]
            if existing:
                metrics = {
                    "FP32 Clean": existing[0]["FP32 Clean"],
                    "INT8 Per-Tensor": existing[0]["INT8 Per-Tensor"],
                    "INT8 Per-Channel": existing[0]["INT8 Per-Channel"],
                    "OOD Noise": existing[0]["OOD Noise"],
                    "OOD Blur": existing[0]["OOD Blur"]
                }
            else:
                metrics = evaluate_setting(arch, "inverse", gamma)
                
            gamma_label = "No Clamping (gamma=100)" if gamma == 100.0 else f"gamma={gamma}"
            results.append({
                "Architecture": arch.upper(),
                "Ablation Dimension": "Outlier Clamping",
                "Setting": gamma_label,
                "Compensation": "inverse",
                "Gamma": gamma,
                **metrics
            })
            
    df = pd.DataFrame(results)
    print("\n\nAblation Study Results:")
    print(df.to_markdown(index=False))
    
    # Save results
    os.makedirs("results", exist_ok=True)
    df.to_json("results/ablation_results.json", orient="records", indent=4)
    
    # Format and save beautiful markdown table for report/appendix
    with open("results/ablation_results.md", "w") as f:
        f.write("# QR-SC-WCPR Ablation Study Results\n\n")
        
        for arch in ["MLP", "RESNET18"]:
            f.write(f"## {arch} Architecture\n\n")
            
            f.write("### Sparsity Compensation Method\n")
            f.write("(Evaluated at $\\gamma=2.0$)\n\n")
            sub_df1 = df[(df["Architecture"] == arch) & (df["Ablation Dimension"] == "Sparsity Compensation")]
            f.write(sub_df1[["Setting", "FP32 Clean", "INT8 Per-Tensor", "INT8 Per-Channel", "OOD Noise", "OOD Blur"]].to_markdown(index=False))
            f.write("\n\n")
            
            f.write("### Outlier Rejection Threshold $\\gamma$\n")
            f.write("(Evaluated with `inverse` compensation)\n\n")
            sub_df2 = df[(df["Architecture"] == arch) & (df["Ablation Dimension"] == "Outlier Clamping")]
            f.write(sub_df2[["Setting", "FP32 Clean", "INT8 Per-Tensor", "INT8 Per-Channel", "OOD Noise", "OOD Blur"]].to_markdown(index=False))
            f.write("\n\n")
            
    print("Ablation study completed and saved successfully!")

if __name__ == "__main__":
    main()
