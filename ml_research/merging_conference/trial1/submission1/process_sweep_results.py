import os

if not os.path.exists("sweep_results.txt"):
    print("sweep_results.txt does not exist yet. Please wait for the sweep to complete.")
else:
    results = []
    with open("sweep_results.txt", "r") as f:
        # Skip header
        header = f.readline().strip().split(",")
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 8:
                continue
            sam_type = parts[0]
            rho = float(parts[1])
            clean = float(parts[2])
            noise = float(parts[3])
            blur = float(parts[4])
            contrast = float(parts[5])
            rotation = float(parts[6])
            ood_avg = float(parts[7])
            results.append({
                'sam_type': sam_type,
                'rho': rho,
                'clean': clean,
                'noise': noise,
                'blur': blur,
                'contrast': contrast,
                'rotation': rotation,
                'ood_avg': ood_avg
            })
            
    # Sort by clean avg
    results_sorted_clean = sorted(results, key=lambda x: x['clean'], reverse=True)
    print("\n=== SWEEP RESULTS SORTED BY CLEAN ACCURACY ===")
    for i, r in enumerate(results_sorted_clean[:10]):
        print(f"{i+1}. sam_type={r['sam_type']}, rho={r['rho']:.4f} | Clean: {r['clean']:.2f}% | OOD Avg: {r['ood_avg']:.2f}%")
        
    # Sort by OOD avg
    results_sorted_ood = sorted(results, key=lambda x: x['ood_avg'], reverse=True)
    print("\n=== SWEEP RESULTS SORTED BY OOD AVERAGE ACCURACY ===")
    for i, r in enumerate(results_sorted_ood[:10]):
        print(f"{i+1}. sam_type={r['sam_type']}, rho={r['rho']:.4f} | Clean: {r['clean']:.2f}% | OOD Avg: {r['ood_avg']:.2f}%")
        
    print("\nBest config overall (Clean):", results_sorted_clean[0])
    print("Best config overall (OOD):", results_sorted_ood[0])
