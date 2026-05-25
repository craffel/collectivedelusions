import torch
import numpy as np
from run_stream import load_experts, prepare_test_stream, evaluate_method

def main():
    torch.manual_seed(42)
    experts = load_experts()
    stream_batches = prepare_test_stream()
    
    # Define hyperparameter search space around defaults
    momentums = [0.70, 0.80, 0.85]
    etas = [0.03, 0.05, 0.07]
    betas = [1.0, 1.5, 2.0]
    gamma_cs = [0.01, 0.02, 0.03, 0.04]
    
    best_overall = 0.0
    best_params = {}
    best_results = {}
    
    print("Starting hyperparameter sweep on Method F...")
    
    count = 0
    for s_momentum in momentums:
        for eta in etas:
            for beta in betas:
                for gamma_c in gamma_cs:
                    count += 1
                    print(f"\n[{count}] Testing: mom={s_momentum}, eta={eta}, beta={beta}, gamma_c={gamma_c}")
                    
                    r = evaluate_method(
                        'Method F (SMT-LDAC, Ours)', 
                        experts, 
                        stream_batches, 
                        s_momentum=s_momentum, 
                        depth_coherence=True,
                        eta=eta,
                        beta=beta,
                        gamma_c=gamma_c
                    )
                    
                    if r['Overall'] > best_overall:
                        best_overall = r['Overall']
                        best_params = {
                            's_momentum': s_momentum,
                            'eta': eta,
                            'beta': beta,
                            'gamma_c': gamma_c
                        }
                        best_results = r
                        print(f"--> NEW BEST OVERALL: {best_overall:.2f}% with {best_params}")
                        
    print("\n====================================")
    print("Sweep complete!")
    print(f"Best parameters: {best_params}")
    print("Best results:")
    for dom in best_results:
        print(f"  {dom}: {best_results[dom]:.2f}%")
    print("====================================")

if __name__ == "__main__":
    main()
