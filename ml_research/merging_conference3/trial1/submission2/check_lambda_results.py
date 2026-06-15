import os
import json

def main():
    files = [
        ("AdamW + Task Arithmetic (lambda=0.2)", "results/adamw_task_arithmetic_lambda_02.json"),
        ("AdamW + Isotropic Merging (lambda=0.2)", "results/adamw_isotropic_lambda_02.json"),
        ("SAM + Task Arithmetic (lambda=0.2)", "results/sam_task_arithmetic_lambda_02.json"),
        ("SAM + Isotropic Merging (lambda=0.2)", "results/sam_isotropic_lambda_02.json"),
    ]
    
    print("=== Checking Lambda = 0.2 Results ===")
    for name, path in files:
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
            acc = data.get("acc", 0.0)
            bwt = data.get("bwt", 0.0)
            print(f"{name:<45} | Accuracy (ACC): {acc:.2f}% | Forgetting (BWT): {bwt:.2f}%")
        else:
            print(f"{name:<45} | File not found yet.")

if __name__ == "__main__":
    main()
