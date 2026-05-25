import subprocess
import re

# Sweep configurations
coherence_weights = [0.005, 0.01, 0.02, 0.05, 0.1]
variance_weights = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]

results = []

print(f"{'Method':<12} | {'Coherence W':<12} | {'Variance W':<12} | {'Clean MNIST':<12} | {'Noisy MNIST':<12} | {'Clean Fashion':<13} | {'Noisy Fashion':<13} | {'Novel KMNIST':<12} | {'Overall':<10}")
print("-" * 115)

# 1. Run STATIC Baseline
cmd = "python run_ttmm.py --method static"
res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
stdout = res.stdout

# Parse results
mnist = float(re.search(r"Phase Clean MNIST\s+:\s+([\d.]+)%", stdout).group(1))
noisy_mnist = float(re.search(r"Phase Noisy MNIST\s+:\s+([\d.]+)%", stdout).group(1))
fashion = float(re.search(r"Phase Clean Fashion\s+:\s+([\d.]+)%", stdout).group(1))
noisy_fashion = float(re.search(r"Phase Noisy Fashion\s+:\s+([\d.]+)%", stdout).group(1))
kmnist = float(re.search(r"Phase Novel KMNIST\s+:\s+([\d.]+)%", stdout).group(1))
overall = float(re.search(r"Overall Accuracy\s+:\s+([\d.]+)%", stdout).group(1))

print(f"{'STATIC':<12} | {'-':<12} | {'-':<12} | {mnist:<12.2f} | {noisy_mnist:<12.2f} | {fashion:<13.2f} | {noisy_fashion:<13.2f} | {kmnist:<12.2f} | {overall:<10.2f}")

# 2. Run Sweeps for BK-CoMerge (variance_weight = 0.0) and VAKP-BC
for cw in coherence_weights:
    for vw in variance_weights:
        method = "bk_comerge" if vw == 0.0 else "vakp_bc"
        cmd = f"python run_ttmm.py --method {method} --coherence_weight {cw} --variance_weight {vw}"
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        stdout = res.stdout
        
        try:
            mnist = float(re.search(r"Phase Clean MNIST\s+:\s+([\d.]+)%", stdout).group(1))
            noisy_mnist = float(re.search(r"Phase Noisy MNIST\s+:\s+([\d.]+)%", stdout).group(1))
            fashion = float(re.search(r"Phase Clean Fashion\s+:\s+([\d.]+)%", stdout).group(1))
            noisy_fashion = float(re.search(r"Phase Noisy Fashion\s+:\s+([\d.]+)%", stdout).group(1))
            kmnist = float(re.search(r"Phase Novel KMNIST\s+:\s+([\d.]+)%", stdout).group(1))
            overall = float(re.search(r"Overall Accuracy\s+:\s+([\d.]+)%", stdout).group(1))
            
            print(f"{method.upper():<12} | {cw:<12.4f} | {vw:<12.4f} | {mnist:<12.2f} | {noisy_mnist:<12.2f} | {fashion:<13.2f} | {noisy_fashion:<13.2f} | {kmnist:<12.2f} | {overall:<10.2f}")
            results.append((method, cw, vw, mnist, noisy_mnist, fashion, noisy_fashion, kmnist, overall))
        except Exception as e:
            print(f"Error parsing {method} with cw={cw}, vw={vw}: {e}")

# Save results to a report file
with open("sweep_results.txt", "w") as f:
    f.write("Method,Coherence_Weight,Variance_Weight,Clean_MNIST,Noisy_MNIST,Clean_Fashion,Noisy_Fashion,Novel_KMNIST,Overall\n")
    for r in results:
        f.write(",".join(map(str, r)) + "\n")
