import os
import torch

lr_lambdas = [0.1, 0.2, 0.5]
lr_heads = [1e-4, 5e-4, 1e-3]
gamma_regs = [1.0, 10.0, 100.0]

SAVE_DIR = "/fsx/craffel/collectivedelusions/ml_research/merging_conference/trial4/submission3"

best_seq_acc = 0.0
best_seq_cfg = None

best_alt_acc = 0.0
best_alt_cfg = None

results_list = []

for lr_l in lr_lambdas:
    for lr_h in lr_heads:
        for gr in gamma_regs:
            suffix = f"l{lr_l}_h{lr_h}_g{gr}".replace(".", "_")
            filename = f"evaluation_results_{suffix}.pt"
            path = os.path.join(SAVE_DIR, filename)
            
            if os.path.exists(path):
                try:
                    res = torch.load(path, map_location="cpu")
                    mc_vti_seq = res["sequential"]["mc_vti"]["overall_accuracy"]
                    mc_vti_alt = res["alternating"]["mc_vti"]["overall_accuracy"]
                    
                    results_list.append((lr_l, lr_h, gr, mc_vti_seq, mc_vti_alt))
                    
                    if mc_vti_seq > best_seq_acc:
                        best_seq_acc = mc_vti_seq
                        best_seq_cfg = (lr_l, lr_h, gr)
                        
                    if mc_vti_alt > best_alt_acc:
                        best_alt_acc = mc_vti_alt
                        best_alt_cfg = (lr_l, lr_h, gr)
                except Exception as e:
                    pass

print(f"Total results collected: {len(results_list)} / 27")
if results_list:
    print("\nTop 5 MC-VTI configurations by Sequential Accuracy:")
    sorted_by_seq = sorted(results_list, key=lambda x: x[3], reverse=True)
    for i, (l, h, g, seq, alt) in enumerate(sorted_by_seq[:5]):
        print(f"[{i+1}] lr_lambda={l}, lr_head={h}, gamma_reg={g} -> Sequential: {seq:.2f}%, Alternating: {alt:.2f}%")
        
    print("\nBest Sequential Configuration:")
    print(f"lr_lambda={best_seq_cfg[0]}, lr_head={best_seq_cfg[1]}, gamma_reg={best_seq_cfg[2]} -> Accuracy: {best_seq_acc:.2f}%")
    
    print("\nBest Alternating Configuration:")
    print(f"lr_lambda={best_alt_cfg[0]}, lr_head={best_alt_cfg[1]}, gamma_reg={best_alt_cfg[2]} -> Accuracy: {best_alt_acc:.2f}%")
else:
    print("No sweep results found yet. Some jobs might still be running or queued.")
