import glob
import re

runs = {
    'AdaMerging': 'adamerging-8tasks_*.out',
    'Surgery': 'surgery-8tasks_*.out',
    'SyMerge': 'symerge-8tasks_*.out',
    'FoldMerge (Ours)': 'foldmerge-8tasks_*.out'
}

datasets = ['SUN397', 'Cars', 'RESISC45', 'EuroSAT', 'SVHN', 'GTSRB', 'MNIST', 'DTD']

results = {}

for name, pattern in runs.items():
    files = glob.glob(pattern)
    if not files:
        print(f"No files found for {name} with pattern {pattern}")
        continue
    # Get the latest file
    latest_file = sorted(files)[-1]
    print(f"Parsing {latest_file} for {name}...")
    
    with open(latest_file, 'r') as f:
        content = f.read()
        
    results[name] = {}
    
    # Parse individual dataset accuracies
    for ds in datasets:
        # Match "Done evaluating on SUN397. Accuracy: 74.48%"
        match = re.search(fr"Done evaluating on {ds}\. Accuracy: ([\d\.]+)%", content)
        if match:
            results[name][ds] = float(match.group(1))
        else:
            # Match "Final Eval: dataset: SUN397 ACC: 0.7448"
            match2 = re.search(fr"Final Eval: dataset: {ds} ACC: ([\d\.]+)", content)
            if match2:
                val = float(match2.group(1))
                if val <= 1.0:
                    val *= 100
                results[name][ds] = val
            else:
                results[name][ds] = 0.0
                
    # Parse average accuracy
    avg_vals = [results[name][ds] for ds in datasets if results[name][ds] > 0.0]
    if len(avg_vals) == len(datasets):
        results[name]['Avg'] = sum(avg_vals) / len(datasets)
    else:
        results[name]['Avg'] = 0.0

# Print as Markdown Table
print("\n### Multi-Task Model Merging Comparison (8-Tasks ViT-B-32)\n")
header = "| Method | " + " | ".join(datasets) + " | Avg ACC |"
separator = "| :--- | " + " | ".join([":---:" for _ in datasets]) + " | :---: |"
print(header)
print(separator)

for name in runs.keys():
    if name not in results:
        continue
    row = f"| **{name}** | "
    row += " | ".join([f"{results[name].get(ds, 0.0):.2f}%" for ds in datasets])
    row += f" | **{results[name].get('Avg', 0.0):.2f}%** |"
    print(row)
