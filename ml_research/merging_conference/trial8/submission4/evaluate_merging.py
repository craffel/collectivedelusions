import torch
import os
import numpy as np
from src.data import download_datasets, get_expert_loaders, build_streams
from src.merging import run_static_merging, run_adamerging, run_df_bayes_ttmm, run_frtr_ttmm

def main():
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")
    
    # Load expert state dicts
    print("Loading experts...")
    mnist_expert = torch.load("checkpoints/mnist_expert.pt", map_location=device)
    kmnist_expert = torch.load("checkpoints/kmnist_expert.pt", map_location=device)
    experts = [mnist_expert, kmnist_expert]
    
    # Load datasets and build streams
    print("Loading datasets and building streams...")
    datasets_dict = download_datasets()
    expert_loaders = get_expert_loaders(datasets_dict)
    streams = build_streams(expert_loaders)
    
    # Methods to evaluate
    methods = {
        "Static Merging": lambda s: run_static_merging(experts, s, device=device),
        "AdaMerging": lambda s: run_adamerging(experts, s, device=device),
        "DF-Bayes-TTMM": lambda s: run_df_bayes_ttmm(experts, s, device=device),
        "FRTR-TTMM (Ours)": lambda s: run_frtr_ttmm(experts, s, device=device)
    }
    
    for stream_name, stream in streams.items():
        print(f"\n================ Stream: {stream_name} ================")
        for method_name, run_fn in methods.items():
            try:
                acc, batch_accs = run_fn(stream)
                print(f"{method_name:20s} | Accuracy: {acc:.4f}")
            except Exception as e:
                print(f"Error running {method_name}: {e}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    main()
