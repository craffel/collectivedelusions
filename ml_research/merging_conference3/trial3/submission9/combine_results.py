import json
from run_flatq_merge import generate_plots

def main():
    print("Combining results...")
    seeds = [42, 100, 2026]
    
    all_results = {}
    all_curvature = {}
    
    for seed in seeds:
        filename = f"flatq_merge_results_seed{seed}.json"
        with open(filename, "r") as f:
            data = json.load(f)
            all_results[str(seed)] = data['results'][str(seed)]
            all_curvature[str(seed)] = data['curvature'][str(seed)]
            
    # Save combined results
    combined_filename = "flatq_merge_results.json"
    with open(combined_filename, "w") as f:
        json.dump({
            'results': all_results,
            'curvature': all_curvature
        }, f, indent=2)
    print(f"Saved combined results to {combined_filename}")
    
    # Generate Plots
    print("Generating combined plots...")
    generate_plots(all_results, all_curvature)
    print("Plots successfully generated.")

if __name__ == '__main__':
    main()
