import os
import time
import subprocess
import re

def parse_txt_results():
    results_dir = "results"
    methods = ["Static", "PROTO-TTMM", "IGGS-OW", "Rob-OW"]
    corruptions = ["clean", "corrupted"]
    
    # Check if all files exist
    all_files = []
    for method in methods:
        for corr in corruptions:
            all_files.append(f"result_{method}_{corr}_seed42.txt")
            
    print("Checking for results files...")
    while True:
        missing = [f for f in all_files if not os.path.exists(os.path.join(results_dir, f))]
        if not missing:
            print("All results files found!")
            break
        print(f"Still waiting for {len(missing)} files: {missing}")
        time.sleep(15)
        
    # Parse files
    data = {}
    for method in methods:
        data[method] = {}
        for corr in corruptions:
            filepath = os.path.join(results_dir, f"result_{method}_{corr}_seed42.txt")
            data[method][corr] = {}
            with open(filepath, "r") as f:
                content = f.read()
            for line in content.splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    k = k.strip()
                    v = v.strip()
                    if k in ["MNIST_Acc", "KMNIST_Acc", "FashionMNIST_Acc", "Overall_Acc", "NDR", "FPR"]:
                        data[method][corr][k] = float(v)
                        
    return data

def update_latex(data):
    print("Updating LaTeX template with actual results...")
    with open("submission.tex", "r") as f:
        latex_content = f.read()
        
    # Replacements map
    replacements = {}
    
    # Mapping table from placeholders to values
    # Clean stream placeholders
    replacements["[M_S_C]"] = f"{data['Static']['clean']['MNIST_Acc']:.2f}"
    replacements["[K_S_C]"] = f"{data['Static']['clean']['K_Acc'] if 'K_Acc' in data['Static']['clean'] else data['Static']['clean'].get('KMNIST_Acc', 0.0):.2f}"
    replacements["[F_S_C]"] = f"{data['Static']['clean']['FashionMNIST_Acc']:.2f}"
    replacements["[O_S_C]"] = f"{data['Static']['clean']['Overall_Acc']:.2f}"
    replacements["[N_S_C]"] = f"{data['Static']['clean']['NDR']:.2f}"
    replacements["[P_S_C]"] = f"{data['Static']['clean']['FPR']:.2f}"
    
    replacements["[M_P_C]"] = f"{data['PROTO-TTMM']['clean']['MNIST_Acc']:.2f}"
    replacements["[K_P_C]"] = f"{data['PROTO-TTMM']['clean']['KMNIST_Acc']:.2f}"
    replacements["[F_P_C]"] = f"{data['PROTO-TTMM']['clean']['FashionMNIST_Acc']:.2f}"
    replacements["[O_P_C]"] = f"{data['PROTO-TTMM']['clean']['Overall_Acc']:.2f}"
    replacements["[N_P_C]"] = f"{data['PROTO-TTMM']['clean']['NDR']:.2f}"
    replacements["[P_P_C]"] = f"{data['PROTO-TTMM']['clean']['FPR']:.2f}"
    
    replacements["[M_I_C]"] = f"{data['IGGS-OW']['clean']['MNIST_Acc']:.2f}"
    replacements["[K_I_C]"] = f"{data['IGGS-OW']['clean']['KMNIST_Acc']:.2f}"
    replacements["[F_I_C]"] = f"{data['IGGS-OW']['clean']['FashionMNIST_Acc']:.2f}"
    replacements["[O_I_C]"] = f"{data['IGGS-OW']['clean']['Overall_Acc']:.2f}"
    replacements["[N_I_C]"] = f"{data['IGGS-OW']['clean']['NDR']:.2f}"
    replacements["[P_I_C]"] = f"{data['IGGS-OW']['clean']['FPR']:.2f}"
    
    replacements["[M_R_C]"] = f"{data['Rob-OW']['clean']['MNIST_Acc']:.2f}"
    replacements["[K_R_C]"] = f"{data['Rob-OW']['clean']['KMNIST_Acc']:.2f}"
    replacements["[F_R_C]"] = f"{data['Rob-OW']['clean']['FashionMNIST_Acc']:.2f}"
    replacements["[O_R_C]"] = f"{data['Rob-OW']['clean']['Overall_Acc']:.2f}"
    replacements["[N_R_C]"] = f"{data['Rob-OW']['clean']['NDR']:.2f}"
    replacements["[P_R_C]"] = f"{data['Rob-OW']['clean']['FPR']:.2f}"
    
    # Corrupted stream placeholders
    replacements["[M_S_N]"] = f"{data['Static']['corrupted']['MNIST_Acc']:.2f}"
    replacements["[K_S_N]"] = f"{data['Static']['corrupted']['KMNIST_Acc']:.2f}"
    replacements["[F_S_N]"] = f"{data['Static']['corrupted']['FashionMNIST_Acc']:.2f}"
    replacements["[O_S_N]"] = f"{data['Static']['corrupted']['Overall_Acc']:.2f}"
    replacements["[N_S_N]"] = f"{data['Static']['corrupted']['NDR']:.2f}"
    replacements["[P_S_N]"] = f"{data['Static']['corrupted']['FPR']:.2f}"
    
    replacements["[M_P_N]"] = f"{data['PROTO-TTMM']['corrupted']['MNIST_Acc']:.2f}"
    replacements["[K_P_N]"] = f"{data['PROTO-TTMM']['corrupted']['KMNIST_Acc']:.2f}"
    replacements["[F_P_N]"] = f"{data['PROTO-TTMM']['corrupted']['FashionMNIST_Acc']:.2f}"
    replacements["[O_P_N]"] = f"{data['PROTO-TTMM']['corrupted']['Overall_Acc']:.2f}"
    replacements["[N_P_N]"] = f"{data['PROTO-TTMM']['corrupted']['NDR']:.2f}"
    replacements["[P_P_N]"] = f"{data['PROTO-TTMM']['corrupted']['FPR']:.2f}"
    
    replacements["[M_I_N]"] = f"{data['IGGS-OW']['corrupted']['MNIST_Acc']:.2f}"
    replacements["[K_I_N]"] = f"{data['IGGS-OW']['corrupted']['KMNIST_Acc']:.2f}"
    replacements["[F_I_N]"] = f"{data['IGGS-OW']['corrupted']['FashionMNIST_Acc']:.2f}"
    replacements["[O_I_N]"] = f"{data['IGGS-OW']['corrupted']['Overall_Acc']:.2f}"
    replacements["[N_I_N]"] = f"{data['IGGS-OW']['corrupted']['NDR']:.2f}"
    replacements["[P_I_N]"] = f"{data['IGGS-OW']['corrupted']['FPR']:.2f}"
    
    replacements["[M_R_N]"] = f"{data['Rob-OW']['corrupted']['MNIST_Acc']:.2f}"
    replacements["[K_R_N]"] = f"{data['Rob-OW']['corrupted']['KMNIST_Acc']:.2f}"
    replacements["[F_R_N]"] = f"{data['Rob-OW']['corrupted']['FashionMNIST_Acc']:.2f}"
    replacements["[O_R_N]"] = f"{data['Rob-OW']['corrupted']['Overall_Acc']:.2f}"
    replacements["[N_R_N]"] = f"{data['Rob-OW']['corrupted']['NDR']:.2f}"
    replacements["[P_R_N]"] = f"{data['Rob-OW']['corrupted']['FPR']:.2f}"
    
    # Differences
    diff_c = data['Rob-OW']['clean']['FashionMNIST_Acc'] - data['PROTO-TTMM']['clean']['FashionMNIST_Acc']
    diff_n = data['Rob-OW']['corrupted']['FashionMNIST_Acc'] - data['PROTO-TTMM']['corrupted']['FashionMNIST_Acc']
    
    replacements["[DIFF_C]"] = f"{diff_c:.2f}"
    replacements["[DIFF_N]"] = f"{diff_n:.2f}"
    
    for placeholder, val in replacements.items():
        latex_content = latex_content.replace(placeholder, val)
        
    with open("submission.tex", "w") as f:
        f.write(latex_content)
        
    print("LaTeX file updated successfully!")

def compile_pdf():
    print("Compiling LaTeX to PDF using tectonic...")
    if os.path.exists("submission.pdf"):
        os.remove("submission.pdf")
            
    # Tectonic compiles everything in a single, robust step (including bibtex)
    cmd = "./my_env/bin/tectonic submission.tex"
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if os.path.exists("submission.pdf"):
        print("Success! submission.pdf compiled successfully!")
        return True
    else:
        print("Error: submission.pdf failed to compile.")
        print("--- tectonic output ---")
        print(res.stdout[:1000])
        print(res.stderr[:1000])
        return False


def main():
    data = parse_txt_results()
    update_latex(data)
    compile_pdf()

if __name__ == "__main__":
    main()
