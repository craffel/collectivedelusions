import pandas as pd
import numpy as np

def main():
    try:
        df = pd.read_csv("experiment_results.csv")
    except FileNotFoundError:
        print("experiment_results.csv not found yet!")
        return
        
    print("Loaded results from experiment_results.csv.")
    
    # Let's find the best hyperparameters for each method on average across all 8 environments
    # Each environment is defined by (corruption, stream)
    methods = df["method"].unique()
    print(f"Available methods: {methods}")
    
    # We want to identify the best hyperparameter setting per method.
    # Hyperparameters are: (sparsity_p, eta, alpha, opt)
    best_configs = {}
    
    for method in methods:
        method_df = df[df["method"] == method]
        # Group by hyperparams and calculate mean accuracy
        hyperparams = ["sparsity_p", "eta", "alpha", "opt"]
        grouped = method_df.groupby(hyperparams)["accuracy"].mean().reset_index()
        # Find the row with maximum accuracy
        best_row = grouped.loc[grouped["accuracy"].idxmax()]
        best_configs[method] = {
            "sparsity_p": best_row["sparsity_p"],
            "eta": best_row["eta"],
            "alpha": best_row["alpha"],
            "opt": best_row["opt"],
            "avg_accuracy": best_row["accuracy"]
        }
        print(f"\nBest Config for {method}:")
        print(f"  Sparsity P%: {best_row['sparsity_p']}%")
        print(f"  Learning Rate (eta): {best_row['eta']}")
        print(f"  Sensitivity Power (alpha): {best_row['alpha']}")
        print(f"  Optimizer: {best_row['opt']}")
        print(f"  Average Accuracy across environments: {best_row['accuracy']:.2f}%")

    # Now let's generate a beautiful markdown/LaTeX table for the paper
    # Tectonic supports Markdown and LaTex.
    # Let's create a dataframe where index is method, and columns are (corruption, stream) combinations
    environments = [
        ("clean", "alternating"), ("clean", "sequential"),
        ("noise", "alternating"), ("noise", "sequential"),
        ("blur", "alternating"), ("blur", "sequential"),
        ("contrast", "alternating"), ("contrast", "sequential")
    ]
    
    table_rows = []
    
    for method in sorted(best_configs.keys()):
        cfg = best_configs[method]
        # Extract the row for this method and its best hyperparams
        method_df = df[
            (df["method"] == method) &
            (df["sparsity_p"] == cfg["sparsity_p"]) &
            (df["eta"] == cfg["eta"]) &
            (df["alpha"] == cfg["alpha"]) &
            (df["opt"] == cfg["opt"])
        ]
        
        row_dict = {"Method": f"{method} (P={cfg['sparsity_p']}%, lr={cfg['eta']}, a={cfg['alpha']})"}
        for corr, stream in environments:
            sub_df = method_df[(method_df["corruption"] == corr) & (method_df["stream"] == stream)]
            if len(sub_df) > 0:
                row_dict[f"{corr}_{stream}"] = f"{sub_df['accuracy'].values[0]:.2f}%"
            else:
                row_dict[f"{corr}_{stream}"] = "N/A"
        row_dict["Average"] = f"{cfg['avg_accuracy']:.2f}%"
        table_rows.append(row_dict)
        
    table_df = pd.DataFrame(table_rows)
    print("\n--- Summary Table ---")
    print(table_df.to_markdown(index=False))
    
    # Also save to markdown file
    with open("results_summary.md", "w") as f:
        f.write("# Experimental Results Summary\n\n")
        f.write(table_df.to_markdown(index=False))
        f.write("\n")

if __name__ == "__main__":
    main()
