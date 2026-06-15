import json
import numpy as np
from run_real_world_experiment import run_real_world_experiment

if __name__ == "__main__":
    models = {
        "BERT-Tiny": "prajjwal1/bert-tiny",
        "BERT-Mini": "prajjwal1/bert-mini",
        "BERT-Medium": "prajjwal1/bert-medium",
        "BERT-Base": "bert-base-uncased"
    }
    
    seeds = [42, 43, 44, 45, 46]
    all_results = {}
    
    for friendly_name, model_path in models.items():
        print(f"\n==================================================")
        print(f"Evaluating {friendly_name} ({model_path})")
        print(f"==================================================")
        
        runs = []
        for s in seeds:
            try:
                run_res = run_real_world_experiment(seed=s, model_name=model_path)
                runs.append(run_res)
            except Exception as e:
                print(f"Error during run with seed {s} on model {friendly_name}: {e}")
                
        if runs:
            keys = runs[0].keys()
            aggregated = {}
            for k in keys:
                vals = [run[k] for run in runs]
                mean = np.mean(vals) * 100.0
                std = np.std(vals) * 100.0
                aggregated[k] = {"mean": mean, "std": std}
                
            all_results[friendly_name] = aggregated
            
            print(f"\nAggregated Results for {friendly_name} (Mean ± SD %):")
            for k in keys:
                print(f"  {k:25s}: {aggregated[k]['mean']:6.2f}% ± {aggregated[k]['std']:4.2f}%")
        else:
            print(f"No successful runs completed for {friendly_name}.")
            
    # Save the consolidated results to a JSON file
    with open("all_real_world_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
        
    print("\nConsolidated scale-up evaluation complete. Results saved to all_real_world_results.json.")
