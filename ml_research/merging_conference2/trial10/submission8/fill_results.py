import json
import os

def load_results():
    with open("results/results.json", "r") as f:
        return json.load(f)

def format_pct(val, decimals=2):
    return f"{val * 100:.{decimals}f}\\%"

def main():
    if not os.path.exists("results/results.json"):
        print("results/results.json not found! Please wait for the benchmark to finish.")
        return
        
    res = load_results()
    
    # Extract oracle
    mnist_oracle = format_pct(res["oracle"]["mnist"])
    fmnist_oracle = format_pct(res["oracle"]["fmnist"])
    cifar_oracle = format_pct(res["oracle"]["cifar10"])
    avg_oracle = format_pct(res["oracle_avg"])
    
    # Create a mapping for sweep results
    # Key: (merging, precision, calibration) -> (mean, std)
    sweep_map = {}
    for entry in res["sweeps"]:
        key = (entry["merging"], entry["precision"], entry["calibration"])
        sweep_map[key] = (entry["mean_acc"], entry["std_acc"])
        
    # Read submission_template.tex if it exists, otherwise we'll write a default template
    # Let's write the complete submission_template.tex directly in python if not found
    template_path = "submission_template.tex"
    if not os.path.exists(template_path):
        print("submission_template.tex not found, creating standard template...")
        write_default_template()
        
    with open(template_path, "r") as f:
        tex = f.read()
        
    # Replace oracle placeholders
    tex = tex.replace("__ORACLE_MNIST__", mnist_oracle)
    tex = tex.replace("__ORACLE_FMNIST__", fmnist_oracle)
    tex = tex.replace("__ORACLE_CIFAR10__", cifar_oracle)
    tex = tex.replace("__ORACLE_AVG__", avg_oracle)
    
    # Replace sweep placeholders in the format __MERGE_PRECISION_CALIB__
    # e.g., __WA_FP32_No-Cal__ or __TA_INT4-Channel_DEM-BC-32x8-b0.1__
    # We will find all placeholders in the text and replace them if we have matches
    placeholders = []
    import re
    matches = re.findall(r"__[A-Za-z0-9_\-\.\+]+__", tex)
    for m in matches:
        if m.startswith("__") and m.endswith("__"):
            content = m[2:-2] # Safely strip __ prefix and suffix
            parts = content.split("_")
            if len(parts) == 3:
                merging, precision, calib = parts
                # Map precision names back if needed
                key = (merging, precision, calib)
                if key in sweep_map:
                    mean, std = sweep_map[key]
                    if calib == "No-Cal":
                        replacement = format_pct(mean)
                    else:
                        replacement = f"${format_pct(mean)} \\pm {format_pct(std, 2)}$"
                    tex = tex.replace(m, replacement)
                    print(f"Replaced {m} with {replacement}")
                else:
                    print(f"Warning: {key} not found in sweep map!")
                
    # Write to submission.tex
    with open("submission.tex", "w") as f:
        f.write(tex)
    print("Wrote updated LaTeX to submission.tex")
    
    # Try to compile with tectonic
    print("Compiling submission.tex with Tectonic...")
    os.system("/fsx/craffel/miniconda3/bin/tectonic submission.tex")
    if os.path.exists("submission.pdf"):
        print("COMPILATION SUCCESSFUL! Saved as submission.pdf")
    else:
        print("Compilation failed or submission.pdf not generated.")

def write_default_template():
    # We will write the template directly via our main code to keep things structured.
    pass

if __name__ == "__main__":
    main()
