import os
import re
import glob

def parse_logs():
    # Find the newest merging-research_*.out file
    files = glob.glob('merging-research_*.out')
    if not files:
        print("No merging-research_*.out log files found!")
        return None
    
    # Sort files by modification time (newest first)
    files.sort(key=os.path.getmtime, reverse=True)
    newest_file = files[0]
    print(f"Parsing log file: {newest_file}")
    
    with open(newest_file, 'r') as f:
        content = f.read()
        
    results = {}
    
    # 1. Parse best lambdas
    ta_match = re.search(r"Best Task Arithmetic: Lambda=([\d\.]+)", content)
    if ta_match:
        results['BEST_TA_LAMBDA'] = ta_match.group(1)
    else:
        results['BEST_TA_LAMBDA'] = "0.5" # Default if not found
        
    tattbn_match = re.search(r"Best TA \+ TTBN: Lambda=([\d\.]+)", content)
    if tattbn_match:
        results['BEST_TATTBN_LAMBDA'] = tattbn_match.group(1)
    else:
        results['BEST_TATTBN_LAMBDA'] = "0.5" # Default if not found
        
    tsbm_match = re.search(r"Best TSBM: Lambda=([\d\.]+)", content)
    if tsbm_match:
        results['BEST_TSBM_LAMBDA'] = tsbm_match.group(1)
    else:
        results['BEST_TSBM_LAMBDA'] = "0.5" # Default if not found
        
    # 2. Parse results table lines
    # Example line: Individual Experts     | 99.30    | 91.38    | 76.20    | 88.96
    table_pattern = r"([\w\s\(\),\+\-]+)\s*\|\s*([\d\.]+)\s*\|\s*([\d\.]+)\s*\|\s*([\d\.]+)\s*\|\s*([\d\.]+)"
    matches = re.findall(table_pattern, content)
    
    for name, mnist, fashion, cifar, avg in matches:
        clean_name = name.strip()
        print(f"Matched row: {clean_name} -> {mnist}, {fashion}, {cifar}, {avg}")
        
        if "Individual Experts" in clean_name:
            results['EXPERT_MNIST'] = mnist
            results['EXPERT_FASHION'] = fashion
            results['EXPERT_CIFAR'] = cifar
            results['EXPERT_AVG'] = avg
        elif "Weight Averaging (WA)" in clean_name:
            results['WA_MNIST'] = mnist
            results['WA_FASHION'] = fashion
            results['WA_CIFAR'] = cifar
            results['WA_AVG'] = avg
        elif "WA + TTBN" in clean_name:
            results['WATTBN_MNIST'] = mnist
            results['WATTBN_FASHION'] = fashion
            results['WATTBN_CIFAR'] = cifar
            results['WATTBN_AVG'] = avg
        elif "Task Arithmetic (TA)" in clean_name:
            results['TA_MNIST'] = mnist
            results['TA_FASHION'] = fashion
            results['TA_CIFAR'] = cifar
            results['TA_AVG'] = avg
        elif "TA + TTBN" in clean_name:
            results['TATTBN_MNIST'] = mnist
            results['TATTBN_FASHION'] = fashion
            results['TATTBN_CIFAR'] = cifar
            results['TATTBN_AVG'] = avg
        elif "REPAIR" in clean_name:
            results['REPAIR_MNIST'] = mnist
            results['REPAIR_FASHION'] = fashion
            results['REPAIR_CIFAR'] = cifar
            results['REPAIR_AVG'] = avg
        elif "TCAC" in clean_name:
            results['TCAC_MNIST'] = mnist
            results['TCAC_FASHION'] = fashion
            results['TCAC_CIFAR'] = cifar
            results['TCAC_AVG'] = avg
        elif "TSBM" in clean_name:
            results['TSBM_MNIST'] = mnist
            results['TSBM_FASHION'] = fashion
            results['TSBM_CIFAR'] = cifar
            results['TSBM_AVG'] = avg
            
    return results

def main():
    results = parse_logs()
    if not results:
        print("Failed to parse logs.")
        return
        
    print("\nParsed Results:")
    for k, v in results.items():
        print(f"  {k}: {v}")
        
    # Read the template
    if not os.path.exists('submission_template.tex'):
        print("submission_template.tex not found!")
        return
        
    with open('submission_template.tex', 'r') as f:
        template = f.read()
        
    # Replace all placeholders
    filled = template
    for key, val in results.items():
        filled = filled.replace(f"{{{{{key}}}}}", str(val))
        
    # Check if there are any remaining placeholders
    remaining = re.findall(r"\{\{[\w_]+\}\}", filled)
    if remaining:
        print(f"\nWarning: Unfilled placeholders remaining: {remaining}")
        
    # Write the filled latex file
    with open('submission.tex', 'w') as f:
        f.write(filled)
    print("\nSuccessfully wrote to submission.tex")
    
    # Compile with Tectonic
    print("\nCompiling submission.tex using tectonic...")
    os.system('tectonic submission.tex')
    
    if os.path.exists('submission.pdf'):
        print("\nSUCCESS! submission.pdf has been generated successfully.")
    else:
        print("\nERROR: compilation failed or submission.pdf was not generated.")

if __name__ == '__main__':
    main()
