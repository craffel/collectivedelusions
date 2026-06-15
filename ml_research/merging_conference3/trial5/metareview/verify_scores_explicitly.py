import os

base_dir = "/fsx/craffel/collectivedelusions/ml_research/merging_conference3/trial5/metareview"

def find_recommendation_section(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Let's search for "overall recommendation" or "recommendation" section headers
    # Standard Markdown headers
    lines = content.split('\n')
    rec_lines = []
    found = False
    for i, line in enumerate(lines):
        if line.strip().startswith("##") and any(x in line.lower() for x in ["recommendation", "recommendations"]):
            found = True
            # Print next 10 lines
            rec_lines = lines[i:i+10]
            break
            
    if not found:
        # Search for lines containing "Score:" or "Rating:" or "Recommendation:" near the end of the file
        for i, line in enumerate(lines[-30:]):
            idx = len(lines) - 30 + i
            if any(term in line.lower() for term in ["score:", "rating:", "recommendation:", "overall recommendation"]):
                rec_lines = lines[idx:idx+10]
                break
                
    return "\n".join(rec_lines)

for n in range(1, 11):
    print(f"\n==================== SUBMISSION {n} ====================")
    for r in range(1, 4):
        filepath = os.path.join(base_dir, f"submission{n}", f"reviewer{r}", "review.md")
        if os.path.exists(filepath):
            print(f"--- Reviewer {r} ---")
            print(find_recommendation_section(filepath))
            print("-" * 30)
