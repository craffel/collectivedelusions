import numpy as np

def main():
    try:
        data = np.load('evaluation_results.npz', allow_pickle=True)
        print("=== EVALUATION RESULTS SUMMARY ===")
        print("Oracle Accuracies:")
        for k, v in data['oracle_accs'].item().items():
            print(f"  {k}: {v:.2f}%")
            
        print("\nSweep 1: Outlier Corruption p Sweep (Weight Averaging)")
        p_levels = data['p_levels']
        sweep1 = data['sweep1_results'].item()
        for method, accs in sweep1.items():
            print(f"  Method: {method.upper()}")
            for p, acc in zip(p_levels, accs):
                print(f"    p={p:.1f}: {acc:.2f}%")
                
        print("\nSweep 2: Sample Size Budget N Sweep (Clean WA)")
        n_budgets = data['n_budgets']
        sweep2 = data['sweep2_results'].item()
        for method, accs in sweep2.items():
            print(f"  Method: {method.upper()}")
            for n, acc in zip(n_budgets, accs):
                print(f"    N={n}: {acc:.2f}%")
                
        print("\nSweep 3: Ablation of QRC Components (p=0.2, N=128)")
        for k, v in data['ablation_results'].item().items():
            print(f"  {k}: {v:.2f}%")
            
        print("\nSweep 4: Task Arithmetic Merging (p=0.2, N=128)")
        for k, v in data['ta_results'].item().items():
            print(f"  {k}: {v:.2f}%")
            
    except FileNotFoundError:
        print("evaluation_results.npz not found.")

if __name__ == '__main__':
    main()
