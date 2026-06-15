import os
import re

base_dir = "/fsx/craffel/collectivedelusions/ml_research/merging_conference3/trial5/metareview"

def parse_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    score_info = []
    # Find any line that has Recommendation, Score, or Rating
    for idx, line in enumerate(lines):
        l_lower = line.lower()
        if "overall recommendation" in l_lower or "recommendation:" in l_lower or "score:" in l_lower or "rating:" in l_lower or "rating" in l_lower or "score" in l_lower:
            # Let's clean and save the line and the next 2 lines
            segment = []
            for offset in range(-1, 4):
                if 0 <= idx + offset < len(lines):
                    segment.append(f"{idx+offset+1}: {lines[idx+offset].strip()}")
            score_info.append("\n".join(segment))
            
    return score_info

for n in range(1, 11):
    print(f"\n==================== SUBMISSION {n} ====================")
    for r in range(1, 4):
        filepath = os.path.join(base_dir, f"submission{n}", f"reviewer{r}", "review.md")
        if os.path.exists(filepath):
            print(f"--- Reviewer {r} ---")
            info = parse_file(filepath)
            # Filter and print the most relevant one
            printed = False
            for chunk in info:
                # If it looks like it contains a numeric rating 1-6 or Accept/Reject
                if any(x in chunk.lower() for x in ["accept", "reject", "score", "rating", "recommendation"]):
                    if any(str(digit) in chunk for digit in range(1, 7)):
                        print(chunk)
                        print("-" * 20)
                        printed = True
            if not printed:
                # Just print the last chunk or all chunks
                for chunk in info[-2:]:
                    print(chunk)
                    print("-" * 20)
