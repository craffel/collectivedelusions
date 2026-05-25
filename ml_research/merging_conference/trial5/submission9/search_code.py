import re

files = ["submission6.txt", "submission8.txt", "submission9.txt"]

for fpath in files:
    print(f"=== Embedded code in {fpath} ===")
    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Let's search for code blocks, or lines starting with import torch, etc.
    # We can also search for code-like patterns
    python_blocks = re.findall(r"(\bimport\s+torch\b.*?)(?=\n\n|\n[A-Z]|$)", content, re.DOTALL)
    for i, block in enumerate(python_blocks):
        print(f"Code block {i+1}:\n{block}\n")
    
    # Let's also search for 'class ' definition
    class_blocks = re.findall(r"(\bclass\s+\w+\b.*?)(?=\n\n|\n[A-Z]|$)", content, re.DOTALL)
    for i, block in enumerate(class_blocks):
        print(f"Class block {i+1}:\n{block}\n")
