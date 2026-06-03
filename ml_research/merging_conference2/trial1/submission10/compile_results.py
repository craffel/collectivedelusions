import os
import glob
import json

def main():
    print("Compiling all partial sweep results...")
    
    all_results = []
    
    # Locate all JSON files in sweep_results/
    json_files = glob.glob("sweep_results/*.json")
    for f_path in json_files:
        try:
            with open(f_path, 'r') as f:
                data = json.load(f)
                all_results.extend(data)
        except Exception as e:
            print(f"Error reading {f_path}: {e}")
            
    # Save to global sweep_results.json
    with open("sweep_results.json", "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"Saved aggregated results ({len(all_results)} entries) to sweep_results.json")
    
    # Generate markdown summary
    markdown_lines = [
        "# Model Merging Hyperparameter Sweep Results",
        "",
        "| Architecture | Method | Alpha | Gamma | CIFAR-10 Acc | SVHN Acc | FMNIST Acc | Average Acc |",
        "|---|---|---|---|---|---|---|---|",
    ]
    
    for res in sorted(all_results, key=lambda x: (x.get('arch', ''), x.get('method', ''), x.get('Average', 0.0)), reverse=True):
        arch = res.get('arch', 'resnet18')
        m = res.get('method', 'N/A')
        a = res.get('alpha', 'N/A')
        g = res.get('gamma', 'N/A')
        c10 = res.get('cifar10', 0.0)
        svhn = res.get('svhn', 0.0)
        fmn = res.get('fmnist', 0.0)
        avg = res.get('Average', 0.0)
        
        # Format values
        a_str = f"{a:.1f}" if isinstance(a, float) else str(a)
        g_str = f"{g:.1f}" if isinstance(g, float) else str(g)
        markdown_lines.append(f"| {arch} | {m} | {a_str} | {g_str} | {c10:.2f}% | {svhn:.2f}% | {fmn:.2f}% | {avg:.2f}% |")
        
    with open("sweep_summary.md", "w") as f:
        f.write("\n".join(markdown_lines))
    print("Saved sweep summary to sweep_summary.md")
    
    # Trigger paper generation
    if os.path.exists("generate_paper.py"):
        print("Regenerating LaTeX paper with new best results...")
        import subprocess
        subprocess.run("python generate_paper.py", shell=True)
        
if __name__ == "__main__":
    main()
