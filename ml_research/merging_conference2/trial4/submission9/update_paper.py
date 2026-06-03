import subprocess

def main():
    with open("submission.tex", "r") as f:
        text = f.read()
        
    # Replacements dictionary
    reps = {
        "[ORACLE_MNIST]": "96.12",
        "[ORACLE_FMNIST]": "85.11",
        "[ORACLE_CIFAR]": "69.62",
        "[ORACLE_AVG]": "83.62",
        
        "[WA_MNIST]": "63.27",
        "[WA_FMNIST]": "29.56",
        "[WA_CIFAR]": "27.10",
        "[WA_AVG]": "39.98",
        
        "[TCAC_MNIST]": "9.95",
        "[TCAC_FMNIST]": "7.97",
        "[TCAC_CIFAR]": "10.01",
        "[TCAC_AVG]": "9.31",
        
        "[SP_MNIST]": "20.48",
        "[SP_FMNIST]": "17.55",
        "[SP_CIFAR]": "20.68",
        "[SP_AVG]": "19.57",
        
        "[N_MNIST]": "89.97",
        "[N_FMNIST]": "56.22",
        "[N_CIFAR]": "21.61",
        "[N_AVG]": "55.93",
        
        "[SRAC_MNIST]": "20.80",
        "[SRAC_FMNIST]": "23.60",
        "[SRAC_CIFAR]": "22.80",
        "[SRAC_AVG]": "22.36",
        
        "[N_SRAC_MNIST]": "83.81",
        "[N_SRAC_FMNIST]": "52.99",
        "[N_SRAC_CIFAR]": "20.22",
        "[N_SRAC_AVG]": "52.34",
    }
    
    # Table 2 Ablation rows replacement
    ablation_rows = """Base Merged & Layer-wise & $\\beta=5.0$ & 21.25\\% \\\\
Base Merged & Layer-wise & $\\beta=15.0$ & 22.17\\% \\\\
Base Merged & Layer-wise & $\\beta=30.0$ & 22.36\\% \\\\
Base Merged & Safe Channel (Clamp 0.2) & $\\beta=5.0$ & 31.40\\% \\\\
Base Merged & Safe Channel (Clamp 0.2) & $\\beta=15.0$ & 29.67\\% \\\\
Base Merged & Safe Channel (Clamp 0.2) & $\\beta=30.0$ & 27.23\\% \\\\
\\midrule
N-TAAC Base & Layer-wise & $\\beta=5.0$ & 47.07\\% \\\\
N-TAAC Base & Layer-wise & $\\beta=15.0$ & 44.85\\% \\\\
N-TAAC Base & Layer-wise & $\\beta=30.0$ & 42.93\\% \\\\
N-TAAC Base & Safe Channel (Clamp 0.2) & $\\beta=5.0$ & 33.30\\% \\\\
N-TAAC Base & Safe Channel (Clamp 0.2) & $\\beta=15.0$ & 31.24\\% \\\\
N-TAAC Base & Safe Channel (Clamp 0.2) & $\\beta=30.0$ & 25.36\\% \\\\
\\midrule
N-TAAC Base & Head Routing Only & $\\beta=5.0$ & 48.43\\% \\\\
N-TAAC Base & Head Routing Only & $\\beta=15.0$ & 52.03\\% \\\\
N-TAAC Base & Head Routing Only & $\\beta=30.0$ & \\textbf{52.34}\\% \\\\"""

    for k, v in reps.items():
        text = text.replace(k, v)
        
    text = text.replace("[ABLATION_ROWS]", ablation_rows)
    
    with open("submission.tex", "w") as f:
        f.write(text)
        
    print("Successfully updated submission.tex with results!")
    
    # Run tectonic to compile
    print("Compiling submission.tex to PDF using tectonic...")
    cmd = ["/fsx/craffel/miniconda3/bin/tectonic", "submission.tex"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0:
        print("Compilation succeeded! submission.pdf has been generated.")
    else:
        print("Compilation failed!")
        print("Stdout:", res.stdout)
        print("Stderr:", res.stderr)

if __name__ == "__main__":
    main()
