import os
import glob

def parse_csv_from_file(filepath):
    lines = []
    started = False
    with open(filepath, "r") as f:
        for line in f:
            if "RESULTS_CSV_START" in line:
                started = True
                continue
            if "RESULTS_CSV_END" in line:
                started = False
                break
            if started:
                lines.append(line.strip())
    return lines

def main():
    out_files = sorted(glob.glob("sweep_hp_*.out"))
    print(f"Found {len(out_files)} hyperparameter log files.")
    
    lr_results = []
    epoch_results = []
    
    for filepath in out_files:
        csv_lines = parse_csv_from_file(filepath)
        if not csv_lines:
            continue
            
        # Parse configs
        config_line = ""
        with open(filepath, "r") as f:
            for line in f:
                if "Configurations:" in line:
                    config_line = line.strip()
                    break
                    
        # Extract LR and Epoch values
        # Configurations: Seed=42, CalSize=128, HeadLR=0.001, HeadEpochs=15, Merging=WA, Lambda=0.4
        parts = config_line.split(",")
        lr_val = 0.0
        epochs_val = 0
        for p in parts:
            if "HeadLR=" in p:
                lr_val = float(p.split("=")[1])
            if "HeadEpochs=" in p:
                epochs_val = int(p.split("=")[1])
                
        # Extract SPJA-SFT and SPJA-TTA
        sft_line = ""
        tta_line = ""
        for line in csv_lines:
            if "SPJA-SFT" in line:
                sft_line = line
            if "SPJA-TTA" in line:
                tta_line = line
                
        sft_avg = float(sft_line.split(",")[4])
        tta_avg = float(tta_line.split(",")[4])
        
        is_lr_sweep = "lr_" in filepath
        
        if is_lr_sweep:
            lr_results.append((lr_val, sft_avg, tta_avg, filepath))
        else:
            epoch_results.append((epochs_val, sft_avg, tta_avg, filepath))
            
    # Print tables
    print("\n" + "="*60)
    print("LEARNING RATE SWEEP RESULTS (HeadEpochs=15)")
    print("="*60)
    print(f"{'Learning Rate':<15} | {'SPJA-SFT (Avg)':<15} | {'SPJA-TTA (Avg)':<15}")
    print("-" * 60)
    for lr, sft, tta, fn in sorted(lr_results):
        print(f"{lr:<15g} | {sft:<15.2f} | {tta:<15.2f}")
    print("-" * 60)
    
    print("\n" + "="*60)
    print("EPOCH ABLATION SWEEP RESULTS (HeadLR=0.001)")
    print("="*60)
    print(f"{'Epochs':<15} | {'SPJA-SFT (Avg)':<15} | {'SPJA-TTA (Avg)':<15}")
    print("-" * 60)
    for ep, sft, tta, fn in sorted(epoch_results):
        print(f"{ep:<15} | {sft:<15.2f} | {tta:<15.2f}")
    print("-" * 60)

if __name__ == "__main__":
    main()
