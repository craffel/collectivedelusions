import glob
import re
import numpy as np

runs = {
    'SyMerge-8Tasks': ['symerge-8tasks_*.out', 'SyMerge/logs/ViT-B-32/*_SyMerge.txt'],
    'FoldMerge-8Tasks': ['foldmerge-8tasks_*.out', 'SyMerge/logs/ViT-B-32/*_FoldMerge.txt'],
    'FoldMerge-Barycentric': ['foldmerge-barycentric_*.out', 'SyMerge/logs/ViT-B-32/*_Barycentric.txt'],
    'FoldMerge-Warping': ['foldmerge-warping_*.out', 'SyMerge/logs/ViT-B-32/*_Warping.txt'],
    'SyMerge-Frozen': ['symerge-frozen_*.out', 'SyMerge/logs/ViT-B-32/*_Frozen.txt'],
    'FoldMerge-Frozen': ['foldmerge-frozen_*.out', 'SyMerge/logs/ViT-B-32/*_FoldMerge_Frozen.txt']
}

datasets = ['SUN397', 'Cars', 'RESISC45', 'EuroSAT', 'SVHN', 'GTSRB', 'MNIST', 'DTD']

for name, patterns in runs.items():
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    print(f"\n--- {name} ({len(files)} files found) ---")
    
    all_runs_avg = []
    dataset_runs = {ds: [] for ds in datasets}
    
    for file in sorted(files):
        try:
            with open(file, 'r') as f:
                content = f.read()
        except Exception:
            continue
            
        # Extract seed
        seed_match = re.search(r"seed:\s*(\d+)", content)
        seed = seed_match.group(1) if seed_match else "unknown"
        
        # Check if the run completed
        completed = "Done evaluating on DTD" in content or "Final Eval:" in content or "DTD" in content
        if not completed:
            continue
            
        run_accs = {}
        for ds in datasets:
            match = re.search(fr"Done evaluating on {ds}\. Accuracy: ([\d\.]+)%", content)
            if match:
                run_accs[ds] = float(match.group(1))
            else:
                match2 = re.search(fr"Final Eval: dataset: {ds} ACC: ([\d\.]+)", content)
                if match2:
                    val = float(match2.group(1))
                    if val <= 1.0:
                        val *= 100
                    run_accs[ds] = val
        
        if len(run_accs) == len(datasets):
            avg = sum(run_accs.values()) / len(datasets)
            print(f"File: {file} | Seed: {seed} | Avg ACC: {avg:.4f}%")
            all_runs_avg.append(avg)
            for ds in datasets:
                dataset_runs[ds].append(run_accs[ds])
        else:
            # Maybe it is printed on separate lines? Let's check if at least some are present
            pass
            
    if all_runs_avg:
        print(f"\nSummary for {name}:")
        print(f"Number of completed runs: {len(all_runs_avg)}")
        print(f"Overall Average ACC: {np.mean(all_runs_avg):.4f}% +/- {np.std(all_runs_avg):.4f}%")
        for ds in datasets:
            accs = dataset_runs[ds]
            print(f"  {ds}: {np.mean(accs):.2f}% +/- {np.std(accs):.2f}%")


