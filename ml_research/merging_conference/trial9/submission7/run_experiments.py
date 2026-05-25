import json
import subprocess
import os

print("Starting stream evaluation experiments...")
# 1. Run evaluation
print("Executing eval_stream.py...")
subprocess.run(["python", "eval_stream.py"], check=True)

# 2. Parse results
if not os.path.exists("results.json"):
    print("Error: results.json not found!")
    exit(1)

with open("results.json", "r") as f:
    data = json.load(f)

def print_table(results_dict, title):
    print("\n" + "=" * 80)
    print(f" {title.upper()} EXPERTS RESULTS ")
    print("=" * 80)
    
    headers = ["Method", "Clean MNIST", "Noisy MNIST", "Clean Fashion", "Noisy Fashion", "Novel KMNIST", "Overall"]
    print(f"{headers[0]:<25} | {headers[1]:<12} | {headers[2]:<12} | {headers[3]:<13} | {headers[4]:<13} | {headers[5]:<12} | {headers[6]:<8}")
    print("-" * 110)
    
    for method, metrics in results_dict.items():
        print(f"{method:<25} | "
              f"{metrics['Clean MNIST']:<12.2f}% | "
              f"{metrics['Noisy MNIST']:<12.2f}% | "
              f"{metrics['Clean Fashion']:<13.2f}% | "
              f"{metrics['Noisy Fashion']:<13.2f}% | "
              f"{metrics['Novel KMNIST']:<12.2f}% | "
              f"{metrics['Overall']:<8.2f}%")
              
    # LaTeX formatting
    print("\nLaTeX Code for Paper:")
    print("-" * 50)
    print("\\begin{table*}[t]")
    print("\\centering")
    print("\\caption{Test-Time Model Merging accuracy (\\%) across non-stationary stream segments using " + title + " experts.}")
    print("\\label{tab:results_" + title.lower() + "}")
    print("\\begin{tabular}{lccccccc}")
    print("\\toprule")
    print("Method & C-MN & N-MN & C-FN & N-FN & Nov-K & Overall \\\\")
    print("\\midrule")
    for method, metrics in results_dict.items():
        m_name = method.replace("(Ours)", "\\textbf{(Ours)}")
        print(f"{m_name} & {metrics['Clean MNIST']:.2f}\\% & {metrics['Noisy MNIST']:.2f}\\% & {metrics['Clean Fashion']:.2f}\\% & {metrics['Noisy Fashion']:.2f}\\% & {metrics['Novel KMNIST']:.2f}\\% & {metrics['Overall']:.2f}\\% \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table*}")
    print("-" * 50)

print_table(data["standard"], "Standard")
print_table(data["cosface"], "CosFace")
