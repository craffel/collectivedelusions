import json
import os
import subprocess
import time

def build_and_compile():
    results_path = "results.json"
    if not os.path.exists(results_path):
        print("Waiting for results.json...")
        return False
        
    with open(results_path, "r") as f:
        results = json.load(f)
        
    print("Found results.json. Reading values:")
    print(json.dumps(results, indent=2))
    
    # Read the original submission.tex
    with open("submission.tex", "r") as f:
        tex_content = f.read()
        
    # Replacements mapping
    replacements = {
        "[STATIC_MNIST]": f"{results['Static']['mnist']:.2f}",
        "[STATIC_KMNIST]": f"{results['Static']['kmnist']:.2f}",
        "[STATIC_FASHION]": f"{results['Static']['fashion']:.2f}",
        "[STATIC_OVERALL]": f"{results['Static']['overall']:.2f}",
        
        "[PROTO_MNIST]": f"{results['PROTO-TTMM']['mnist']:.2f}",
        "[PROTO_KMNIST]": f"{results['PROTO-TTMM']['kmnist']:.2f}",
        "[PROTO_FASHION]": f"{results['PROTO-TTMM']['fashion']:.2f}",
        "[PROTO_OVERALL]": f"{results['PROTO-TTMM']['overall']:.2f}",
        
        "[KT_MNIST]": f"{results['KT-Fisher']['mnist']:.2f}",
        "[KT_KMNIST]": f"{results['KT-Fisher']['kmnist']:.2f}",
        "[KT_FASHION]": f"{results['KT-Fisher']['fashion']:.2f}",
        "[KT_OVERALL]": f"{results['KT-Fisher']['overall']:.2f}",
        
        "[FDF_MNIST]": f"{results['FDF-DPA']['mnist']:.2f}",
        "[FDF_KMNIST]": f"{results['FDF-DPA']['kmnist']:.2f}",
        "[FDF_FASHION]": f"{results['FDF-DPA']['fashion']:.2f}",
        "[FDF_OVERALL]": f"{results['FDF-DPA']['overall']:.2f}",
    }
    
    # Perform replacements
    for placeholder, val in replacements.items():
        tex_content = tex_content.replace(placeholder, val)
        
    # Write updated LaTeX file
    with open("submission_filled.tex", "w") as f:
        f.write(tex_content)
    print("Wrote updated LaTeX code to submission_filled.tex")
    
    # Overwrite submission.tex so that tectonic compiles it directly
    with open("submission.tex", "w") as f:
        f.write(tex_content)
        
    # Run plot generation
    print("Generating plot results_plot.png...")
    try:
        from plot_results import generate_plot
        generate_plot()
    except Exception as e:
        print(f"Plot generation failed: {e}")
        
    # Run tectonic compilation
    print("Compiling submission.tex to PDF using tectonic...")
    t0 = time.time()
    try:
        # tectonic will automatically download any packages if needed (requires internet, which is available!)
        cmd = ["./myenv/bin/tectonic", "submission.tex"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        print("Tectonic stdout:")
        print(res.stdout)
        if res.returncode != 0:
            print("Tectonic stderr:")
            print(res.stderr)
            raise RuntimeError(f"Tectonic failed with return code {res.returncode}")
        print(f"Tectonic compilation successful in {time.time() - t0:.2f}s!")
    except Exception as e:
        print(f"Compilation failed: {e}")
        return False
        
    return True

if __name__ == "__main__":
    build_and_compile()
