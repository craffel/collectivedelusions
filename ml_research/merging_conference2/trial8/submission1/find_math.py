def find_sections_and_math(filepath):
    print(f"\n==================== Analyzing {filepath} ====================")
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Let's find some key terms and print lines around them
    lines = content.split("\n")
    keywords = ["U-IPR", "HNS", "Isotropic Parameter Resonance", "Holographic Norm Scaling", "Equation", "Frob", "norm", "scaling factor", "alpha", "beta", "gamma"]
    
    # We want to find sections like "4. " or "3. "
    for i, line in enumerate(lines):
        # check if it starts with section header or contains important formulas
        if any(kw.lower() in line.lower() for kw in keywords) or "equation" in line.lower() or "formula" in line.lower():
            # Print a window around it
            start = max(0, i - 2)
            end = min(len(lines), i + 4)
            print(f"--- Lines {start+1}-{end} (Match on: '{line.strip()}') ---")
            for j in range(start, end):
                print(f"{j+1}: {lines[j]}")
            print("-" * 40)

find_sections_and_math("papers/submission7.txt")
find_sections_and_math("papers/submission9.txt")
