import os
import json
import numpy as np

def main():
    if not os.path.exists('results'):
        print("Results folder not found.")
        return
        
    print("\n=========================================")
    # 1. Main Benchmark Accuracies
    print("MAIN BENCHMARK ACCURACIES Preview:")
    print("=========================================")
    print(f"{'Merged Configuration':<20} | {'Uncalibrated':<12} | {'TCAC':<10} | {'TAAC':<10} | {'LSC':<10} | {'SP-TAAC (Ours)':<15}")
    print("-"*90)
    
    settings = ["WA", "TA (λ=0.1)", "TA (λ=0.2)", "TA (λ=0.3)", "TA (λ=0.4)"]
    files = {
        "WA": "results/main_wa.json",
        "TA (λ=0.1)": "results/main_ta_lam0.1.json",
        "TA (λ=0.2)": "results/main_ta_lam0.2.json",
        "TA (λ=0.3)": "results/main_ta_lam0.3.json",
        "TA (λ=0.4)": "results/main_ta_lam0.4.json"
    }
    
    for s in settings:
        f_path = files[s]
        if os.path.exists(f_path):
            try:
                with open(f_path, 'r') as f:
                    res = json.load(f)
                none_avg = res.get('none', {}).get('avg', 0.0)
                tcac_avg = res.get('tcac', {}).get('avg', 0.0)
                taac_avg = res.get('taac', {}).get('avg', 0.0)
                lsc_avg = res.get('lsc', {}).get('avg', 0.0)
                sp_taac_avg = res.get('sp_taac', {}).get('avg', 0.0)
                print(f"{s:<20} | {none_avg:<12.2f} | {tcac_avg:<10.2f} | {taac_avg:<10.2f} | {lsc_avg:<10.2f} | {sp_taac_avg:<15.2f}")
            except Exception as e:
                print(f"{s:<20} | Error loading file: {e}")
        else:
            print(f"{s:<20} | [Waiting...]")
    print("-"*90)
    
    # 2. Sample Efficiency Sweep
    print("\n=========================================")
    print("SAMPLE EFFICIENCY N-SWEEP (Weight Averaging) Preview:")
    print("=========================================")
    print(f"{'N Samples/Task':<15} | {'Uncalibrated':<12} | {'TCAC':<10} | {'TAAC':<10} | {'LSC':<10} | {'SP-TAAC (Ours)':<15}")
    print("-"*80)
    for N in [4, 8, 16, 32, 64, 128, 256]:
        f_path = f"results/sample_efficiency/wa_N{N}.json"
        if os.path.exists(f_path):
            try:
                with open(f_path, 'r') as f:
                    res = json.load(f)
                none_avg = res.get('none', {}).get('avg', 0.0)
                tcac_avg = res.get('tcac', {}).get('avg', 0.0)
                taac_avg = res.get('taac', {}).get('avg', 0.0)
                lsc_avg = res.get('lsc', {}).get('avg', 0.0)
                sp_taac_avg = res.get('sp_taac', {}).get('avg', 0.0)
                print(f"N = {N:<11} | {none_avg:<12.2f} | {tcac_avg:<10.2f} | {taac_avg:<10.2f} | {lsc_avg:<10.2f} | {sp_taac_avg:<15.2f}")
            except Exception:
                pass
        else:
            print(f"N = {N:<11} | [Waiting...]")
    print("-"*80)

if __name__ == '__main__':
    main()
