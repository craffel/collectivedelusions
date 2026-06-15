import os
import sys
import subprocess
import json

def main():
    os.makedirs("results", exist_ok=True)
    
    runs = [
        # AdamW base optimizer
        {"optimizer": "adamw", "merging": "task_arithmetic"},
        {"optimizer": "adamw", "merging": "isotropic"},
        {"optimizer": "adamw", "merging": "spectral_dampening"},
        
        # Standard SAM
        {"optimizer": "sam", "merging": "task_arithmetic"},
        {"optimizer": "sam", "merging": "isotropic"},
        {"optimizer": "sam", "merging": "spectral_dampening"},
        
        # SABCD Literal (the exact formula from paper)
        {"optimizer": "sabcd_literal", "merging": "task_arithmetic"},
        {"optimizer": "sabcd_literal", "merging": "isotropic"},
        {"optimizer": "sabcd_literal", "merging": "spectral_dampening"},
        
        # SABCD Standard Adam (standard Adam on perturbed grad, restricted to Omega)
        {"optimizer": "sabcd_standard_adam", "merging": "task_arithmetic"},
        {"optimizer": "sabcd_standard_adam", "merging": "isotropic"},
        {"optimizer": "sabcd_standard_adam", "merging": "spectral_dampening"},
        
        # SABCD Adam GT (Adam on unperturbed grad, restricted to Omega)
        {"optimizer": "sabcd_adam_gt", "merging": "task_arithmetic"},
        {"optimizer": "sabcd_adam_gt", "merging": "isotropic"},
        {"optimizer": "sabcd_adam_gt", "merging": "spectral_dampening"},
    ]
    
    print(f"Total configurations to run: {len(runs)}")
    
    for i, run in enumerate(runs):
        opt = run["optimizer"]
        merg = run["merging"]
        out_file = f"results/{opt}_{merg}.json"
        
        print(f"\n==================================================")
        print(f"Executing Sweep {i+1}/{len(runs)}")
        print(f"Optimizer: {opt} | Merging: {merg}")
        print(f"==================================================")
        
        cmd = [
            sys.executable, "train.py",
            "--model", "vit_tiny_patch16_224",
            "--optimizer", opt,
            "--merging", merg,
            "--epochs", "3",
            "--batch_size", "128",
            "--output_file", out_file
        ]
        
        # Execute training command
        try:
            result = subprocess.run(cmd, check=True)
            print(f"Completed run: {opt}_{merg}")
        except subprocess.CalledProcessError as e:
            print(f"Error during run: {opt}_{merg}: {e}")
            
    print("\nSweep completed! All results are saved in the results/ directory.")

if __name__ == "__main__":
    main()
