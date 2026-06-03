import os
import glob

def parse_csv_from_file(filepath):
    lines = []
    started = False
    with open(filepath, "r") as f:
        for line in f:
            if "RESULTS_CSV_START" in line:
                started = True
                continue
            if "RESULTS_CSV_END" in line:
                started = False
                break
            if started:
                lines.append(line.strip())
    return lines

def main():
    out_files = glob.glob("*.out")
    print(f"Found {len(out_files)} log files.")
    
    all_results = {}
    
    for filepath in out_files:
        # Check if this log file contains CSV results
        csv_lines = parse_csv_from_file(filepath)
        if not csv_lines:
            continue
            
        print(f"Parsing results from {filepath}...")
        # Header is the first line: Method,MNIST,Fashion,CIFAR10,Average
        header = csv_lines[0].split(",")
        
        # We can extract the run configurations from the file as well
        config_line = ""
        with open(filepath, "r") as f:
            for line in f:
                if "Configurations:" in line:
                    config_line = line.strip()
                    break
                    
        # Let's save the results
        all_results[filepath] = {
            "config": config_line,
            "data": csv_lines[1:]
        }
        
    print("\n" + "="*50)
    print("SUMMARY OF EXPERIMENTAL RESULTS")
    print("="*50)
    for filepath, res in all_results.items():
        print(f"\nLog File: {filepath}")
        print(f"Configs: {res['config']}")
        print("-" * 50)
        for line in res["data"]:
            print(line)
        print("-" * 50)

if __name__ == "__main__":
    main()
