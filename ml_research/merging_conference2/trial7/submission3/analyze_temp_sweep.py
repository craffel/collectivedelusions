import json
import numpy as np

def main():
    json_path = 'results/temperature_sweep.json'
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {json_path} not found.")
        return

    print("=========================================================================")
    print("                     TEMPERATURE SWEEP ANALYSIS                          ")
    print("=========================================================================")

    # Group by strategy, temperature, batch_size, alpha
    results = {}
    for entry in data:
        strat = entry['strategy']
        temp = entry['temperature']
        bs = entry['test_batch_size']
        alpha = entry['alpha']
        avg_acc = entry['avg_accuracy']
        
        if strat not in results:
            results[strat] = {}
        if temp not in results[strat]:
            results[strat][temp] = {}
        if bs not in results[strat][temp]:
            results[strat][temp][bs] = {}
        results[strat][temp][bs][alpha] = avg_acc

    # List of unique keys
    strategies = sorted(list(results.keys()))
    temperatures = sorted(list(results[strategies[0]].keys()))
    batch_sizes = sorted(list(results[strategies[0]][temperatures[0]].keys()))
    alphas = sorted(list(results[strategies[0]][temperatures[0]][batch_sizes[0]].keys()))

    print(f"Strategies: {strategies}")
    print(f"Temperatures: {temperatures}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Alphas: {alphas}")
    print("-------------------------------------------------------------------------")

    for strat in strategies:
        print(f"\nStrategy: {strat.upper()}")
        print("-" * 80)
        # Table 1: Static evaluation (alpha = 0.0)
        print("Table 1: Static Evaluation (alpha = 0.0)")
        print(f"{'Temp':<6} | {'B=1':<8} | {'B=4':<8} | {'B=16':<8} | {'B=64':<8} | {'B=256':<8}")
        print("-" * 65)
        for temp in temperatures:
            vals = []
            for bs in [1, 4, 16, 64, 256]:
                acc = results[strat][temp][bs].get(0.0, -1)
                vals.append(f"{acc:6.2f}%" if acc >= 0 else "N/A")
            print(f"{temp:<6.2f} | {vals[0]:<8} | {vals[1]:<8} | {vals[2]:<8} | {vals[3]:<8} | {vals[4]:<8}")
        print("-" * 65)

        # Table 2: Blended evaluation (alpha = 0.8)
        print("\nTable 2: Blended Evaluation (alpha = 0.8)")
        print(f"{'Temp':<6} | {'B=1':<8} | {'B=4':<8} | {'B=16':<8} | {'B=64':<8} | {'B=256':<8}")
        print("-" * 65)
        for temp in temperatures:
            vals = []
            for bs in [1, 4, 16, 64, 256]:
                acc = results[strat][temp][bs].get(0.8, -1)
                vals.append(f"{acc:6.2f}%" if acc >= 0 else "N/A")
            print(f"{temp:<6.2f} | {vals[0]:<8} | {vals[1]:<8} | {vals[2]:<8} | {vals[3]:<8} | {vals[4]:<8}")
        print("-" * 65)

if __name__ == '__main__':
    main()
