import time
import torch
import json
from eval_stream import generate_stream, Evaluators, SimpleCNN

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Measuring runtimes on device: {device}")
    
    stream_batches = generate_stream(device=device)
    
    # Load Experts
    expert0 = SimpleCNN(use_cosface=False).to(device)
    expert1 = SimpleCNN(use_cosface=False).to(device)
    expert0.load_state_dict(torch.load("models/mnist_standard.pt", map_location=device))
    expert1.load_state_dict(torch.load("models/fashionmnist_standard.pt", map_location=device))
    
    evals = Evaluators()
    
    methods = [
        ("Static Merging", evals.static_merging),
        ("Fixed TTA", evals.fixed_tta),
        ("CLW-Fisher", evals.clw_fisher),
        ("KT-Fisher", evals.kt_fisher),
        ("DF-Bayes-TTMM", evals.df_bayes_ttmm),
        ("BK-CoMerge", evals.bk_co_merge),
        ("AdaSim-CoMerge (Ours)", evals.adasim_co_merge)
    ]
    
    runtimes = {}
    
    # Warmup with proper reset!
    print("Warming up...")
    images, labels, _ = stream_batches[0]
    
    evals.static_merging(None, None, None, expert0, expert1, reset=True, device=device)
    evals.static_merging(images, labels, 0, expert0, expert1, reset=False, device=device)
    
    evals.adasim_co_merge(None, None, None, expert0, expert1, reset=True, device=device)
    evals.adasim_co_merge(images, labels, 0, expert0, expert1, reset=False, device=device)
        
    for name, fn in methods:
        print(f"Measuring {name}...")
        # Reset state
        fn(None, None, None, expert0, expert1, reset=True, device=device)
        
        start_time = time.perf_counter()
        # Measure over 20 batches
        num_batches = 20
        for idx in range(num_batches):
            images, labels, _ = stream_batches[idx]
            fn(images, labels, idx, expert0, expert1, reset=False, device=device)
            
        end_time = time.perf_counter()
        avg_time = (end_time - start_time) / num_batches * 1000.0 # in ms
        runtimes[name] = avg_time
        print(f"{name}: {avg_time:.2f} ms per batch")
        
    with open("runtimes.json", "w") as f:
        json.dump(runtimes, f, indent=4)
    print("Runtimes saved to runtimes.json")

if __name__ == "__main__":
    main()
