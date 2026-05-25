import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from train import get_base_model, get_dataloaders, merge_models, evaluate, set_seed

def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Dataloaders
    print("Loading dataloaders...")
    dataloaders = get_dataloaders(batch_size=256) # Use larger batch size for faster evaluation
    test_a = dataloaders['test_a']
    test_b = dataloaders['test_b']
    test_full = dataloaders['test_full']
    
    # Checkpoints
    checkpoint_pairs = {
        'AdamW': ('expert_a_adamw.pt', 'expert_b_adamw.pt'),
        'SAM': ('expert_a_sam.pt', 'expert_b_sam.pt'),
        'TAA-SR (Ours)': ('expert_a_taa_sr.pt', 'expert_b_taa_sr.pt')
    }
    
    # Merging configurations
    merging_modes = ['TA', 'C-Ortho', 'OrthoMerge']
    lambda_vals = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    # Verify checkpoints exist
    for opt_name, (chk_a, chk_b) in checkpoint_pairs.items():
        if not os.path.exists(chk_a) or not os.path.exists(chk_b):
            print(f"Error: Checkpoints for {opt_name} are missing! ({chk_a}, {chk_b})")
            return
            
    # Load Base Model
    base_model = get_base_model().to(device)
    base_model.eval()
    
    results = []
    
    for opt_name, (chk_a, chk_b) in checkpoint_pairs.items():
        print(f"\nEvaluating Optimizer: {opt_name}")
        # Load Expert A and Expert B
        model_a = get_base_model().to(device)
        model_a.load_state_dict(torch.load(chk_a, map_location=device))
        model_a.eval()
        
        model_b = get_base_model().to(device)
        model_b.load_state_dict(torch.load(chk_b, map_location=device))
        model_b.eval()
        
        # Evaluate individual experts first (as reference)
        acc_a_on_a = evaluate(model_a, test_a, device, 'A')
        acc_a_on_b = evaluate(model_a, test_b, device, 'B')
        acc_b_on_a = evaluate(model_b, test_a, device, 'A')
        acc_b_on_b = evaluate(model_b, test_b, device, 'B')
        print(f"Expert A standalone - Acc A: {acc_a_on_a:.2f}%, Acc B: {acc_a_on_b:.2f}%")
        print(f"Expert B standalone - Acc A: {acc_b_on_a:.2f}%, Acc B: {acc_b_on_b:.2f}%")
        
        for mode in merging_modes:
            print(f"  Merging Mode: {mode}")
            for l_val in lambda_vals:
                merged_model = merge_models(model_a, model_b, base_model, mode, l_val)
                merged_model.to(device)
                merged_model.eval()
                
                acc_a = evaluate(merged_model, test_a, device, 'A')
                acc_b = evaluate(merged_model, test_b, device, 'B')
                acc_full = evaluate(merged_model, test_full, device, 'full')
                
                results.append({
                    'Optimizer': opt_name,
                    'Merging_Mode': mode,
                    'Lambda': l_val,
                    'Task_A_Acc': acc_a,
                    'Task_B_Acc': acc_b,
                    'Full_Acc': acc_full
                })
                print(f"    Lambda {l_val:.1f} -> Task A: {acc_a:.2f}%, Task B: {acc_b:.2f}%, Full: {acc_full:.2f}%")
                
    df = pd.DataFrame(results)
    df.to_csv('merging_results.csv', index=False)
    print("\nSaved all results to merging_results.csv")
    
    # Print a summary of best results for each combination
    print("\n=== SUMMARY OF BEST RESULTS (Max Full Accuracy) ===")
    summary_rows = []
    for opt_name in checkpoint_pairs.keys():
        for mode in merging_modes:
            sub_df = df[(df['Optimizer'] == opt_name) & (df['Merging_Mode'] == mode)]
            best_idx = sub_df['Full_Acc'].idxmax()
            best_row = sub_df.loc[best_idx]
            summary_rows.append(best_row)
            print(f"{opt_name} + {mode}: Best Lambda = {best_row['Lambda']:.1f} | Task A: {best_row['Task_A_Acc']:.2f}% | Task B: {best_row['Task_B_Acc']:.2f}% | Full: {best_row['Full_Acc']:.2f}%")
            
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv('best_merging_results.csv', index=False)
    
    # Plot curves
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for idx, mode in enumerate(merging_modes):
        ax = axes[idx]
        for opt_name in checkpoint_pairs.keys():
            sub_df = df[(df['Optimizer'] == opt_name) & (df['Merging_Mode'] == mode)]
            ax.plot(sub_df['Lambda'], sub_df['Full_Acc'], marker='o', label=opt_name)
        ax.set_title(f"Merging Method: {mode}")
        ax.set_xlabel("Merging Coefficient (Lambda)")
        if idx == 0:
            ax.set_ylabel("Full CIFAR-10 Accuracy (%)")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        
    plt.tight_layout()
    plt.savefig('merging_curves.png', dpi=300)
    print("Saved plot to merging_curves.png")

if __name__ == '__main__':
    main()
