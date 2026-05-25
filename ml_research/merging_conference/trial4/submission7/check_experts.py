import torch
import os

def check_checkpoints():
    tasks = ["MNIST", "FashionMNIST", "KMNIST"]
    print("Checking expert checkpoints and FIMs...")
    
    for task in tasks:
        ckpt_path = f"./checkpoints/expert_{task}.pt"
        fim_path = f"./checkpoints/fim_{task}.pt"
        
        print(f"\n--- {task} ---")
        if os.path.exists(ckpt_path):
            state_dict = torch.load(ckpt_path, map_location="cpu")
            print(f"  Checkpoint: FOUND ({len(state_dict)} keys)")
            # Check for NaN values in parameters
            has_nan = False
            for k, v in state_dict.items():
                if torch.isnan(v).any():
                    print(f"    WARNING: NaN values found in parameter: {k}")
                    has_nan = True
            if not has_nan:
                print("    All parameters are valid (no NaN).")
        else:
            print("  Checkpoint: NOT FOUND")
            
        if os.path.exists(fim_path):
            fim = torch.load(fim_path, map_location="cpu")
            print(f"  FIM: FOUND ({len(fim)} keys)")
            # Check for NaN or negative values in FIM
            has_nan = False
            has_negative = False
            for k, v in fim.items():
                if torch.isnan(v).any():
                    print(f"    WARNING: NaN values found in FIM: {k}")
                    has_nan = True
                if (v < 0).any():
                    print(f"    WARNING: Negative values found in FIM: {k}")
                    has_negative = True
            if not has_nan and not has_negative:
                print("    All FIM values are valid (non-negative, no NaN).")
        else:
            print("  FIM: NOT FOUND")

if __name__ == "__main__":
    check_checkpoints()
