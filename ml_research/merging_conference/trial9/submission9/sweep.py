import torch
import numpy as np
from run_experiments import SimpleCNN, get_stream_batches, run_bk_comerge

def main():
    print("Loading experts for sweep...")
    expert_0 = SimpleCNN()
    expert_0.load_state_dict(torch.load("./checkpoints/expert_0.pt", map_location="cpu"))
    expert_0.eval()
    
    expert_1 = SimpleCNN()
    expert_1.load_state_dict(torch.load("./checkpoints/expert_1.pt", map_location="cpu"))
    expert_1.eval()
    
    print("Generating stream batches...")
    batches = get_stream_batches()
    
    best_acc = 0.0
    best_params = {}
    
    print("Sweeping EGA-BK-CoMerge...")
    tau_gates = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
    alpha_gates = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 1.5, 2.0]
    
    for tg in tau_gates:
        for ag in alpha_gates:
            accs = run_bk_comerge(
                expert_0, expert_1, batches, 
                lr=0.02, steps=3, gamma_c=0.01, beta=0.1, s_scale=1.0,
                use_ts=False, use_ega=True, tau_gate=tg, alpha_gate=ag
            )
            overall = np.mean(accs)
            print(f"tau_gate={tg:.2f}, alpha_gate={ag:.2f} | Overall: {overall:.4f}%")
            if overall > best_acc:
                best_acc = overall
                best_params = {'tau_gate': tg, 'alpha_gate': ag}
                
    print(f"\nBest EGA-BK-CoMerge: {best_acc:.4f}% with {best_params}")

    best_ts_acc = 0.0
    best_ts_params = {}
    
    print("\nSweeping TS-EGA-BK-CoMerge...")
    for tg in tau_gates:
        for ag in alpha_gates:
            accs = run_bk_comerge(
                expert_0, expert_1, batches, 
                lr=0.02, steps=3, gamma_c=0.01, beta=0.1, s_scale=1.0,
                use_ts=True, ema_factor=0.9, use_ega=True, tau_gate=tg, alpha_gate=ag
            )
            overall = np.mean(accs)
            print(f"TS-EGA: tau_gate={tg:.2f}, alpha_gate={ag:.2f} | Overall: {overall:.4f}%")
            if overall > best_ts_acc:
                best_ts_acc = overall
                best_ts_params = {'tau_gate': tg, 'alpha_gate': ag}
                
    print(f"\nBest TS-EGA-BK-CoMerge: {best_ts_acc:.4f}% with {best_ts_params}")

if __name__ == "__main__":
    main()
