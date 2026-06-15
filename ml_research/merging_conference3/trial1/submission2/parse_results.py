import os
import json
import pandas as pd

def main():
    results_dir = "results"
    if not os.path.exists(results_dir):
        print("No results directory found.")
        return
        
    files = [f for f in os.listdir(results_dir) if f.endswith(".json")]
    if not files:
        print("No JSON results found yet.")
        return
        
    data = []
    for file in files:
        path = os.path.join(results_dir, file)
        try:
            with open(path, "r") as f:
                res = json.load(f)
                
            opt = res.get("optimizer", "unknown")
            merg = res.get("merging", "unknown")
            acc = res.get("acc", 0.0)
            bwt = res.get("bwt", 0.0)
            dur = res.get("total_duration", 0.0)
            
            data.append({
                "Optimizer": opt,
                "Merging Strategy": merg,
                "Accuracy (ACC) %": f"{acc:.2f}%",
                "Forgetting (BWT) %": f"{bwt:.2f}%",
                "Duration (s)": f"{dur:.1f}s"
            })
        except Exception as e:
            print(f"Error reading {file}: {e}")
            
    df = pd.DataFrame(data)
    # Sort by Optimizer and Merging
    df = df.sort_values(by=["Optimizer", "Merging Strategy"]).reset_index(drop=True)
    
    # Generate Markdown Table
    md_table = df.to_markdown(index=False)
    print("\n" + "="*30 + " CURRENT SCOREBOARD " + "="*30)
    print(md_table)
    print("="*80 + "\n")
    
    # Save to a temporary file
    with open("current_scoreboard.md", "w") as f:
        f.write("# Current Experimental Results Scoreboard\n\n")
        f.write(md_table)
        f.write("\n")

if __name__ == "__main__":
    main()
