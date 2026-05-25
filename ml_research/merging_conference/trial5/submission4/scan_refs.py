import re

keywords = ["merge", "merging", "adaptation", "tta", "ttmm", "entropy", "adapter", "test-time"]

def scan_references(filepath):
    print(f"\n===== References in {filepath} =====")
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Try to find the References or Bibliography section
    ref_match = re.search(r"(References|Bibliography)\b", content, re.IGNORECASE)
    if not ref_match:
        print("References section not found. Printing lines containing keywords:")
        lines = content.split("\n")
        for idx, line in enumerate(lines):
            if any(kw in line.lower() for kw in keywords):
                print(f"L{idx}: {line[:120]}")
        return
        
    ref_start = ref_match.start()
    ref_text = content[ref_start:]
    
    # Print lines in references section that match keywords
    lines = ref_text.split("\n")
    for idx, line in enumerate(lines):
        if any(kw in line.lower() for kw in keywords):
            print(f"L{idx}: {line[:120]}")

scan_references("papers/submission6.txt")
scan_references("papers/submission8.txt")
scan_references("papers/submission9.txt")
