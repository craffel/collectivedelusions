import os

base_dir = "/fsx/craffel/collectivedelusions/ml_research/merging_conference3/trial5/metareview"

for n in range(1, 11):
    filepath = os.path.join(base_dir, f"submission{n}", "reviewer1", "review.md")
    if os.path.exists(filepath):
        print(f"\n==================== SUBMISSION {n} ====================")
        with open(filepath, 'r', encoding='utf-8') as f:
            for _ in range(5):
                line = f.readline()
                if not line:
                    break
                print(line.strip())
