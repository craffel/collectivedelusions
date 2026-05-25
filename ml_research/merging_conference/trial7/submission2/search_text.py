import re

files = ["submission3.txt", "submission6.txt", "submission10.txt"]

for f in files:
    print("="*60)
    print(f"File: {f}")
    print("="*60)
    with open(f, "r") as file_in:
        content = file_in.read()
    
    # Find sentences containing "expert" or "MNIST" or "K=2"
    for line in content.split("\n"):
        if any(w in line for w in ["expert", "K=2", "K = 2", "K=3", "FashionMNIST", "library"]):
            if len(line.strip()) > 30:
                print(line.strip()[:150])
