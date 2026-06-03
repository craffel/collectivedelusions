import os
import re
import glob

def parse_sttbn_log():
    # Find the newest sttbn-hook_*.out file
    files = glob.glob('sttbn-hook_*.out')
    if not files:
        print("No sttbn-hook_*.out log files found!")
        return None
    
    # Sort files by modification time (newest first)
    files.sort(key=os.path.getmtime, reverse=True)
    newest_file = files[0]
    print(f"Parsing STTBN log file: {newest_file}")
    
    with open(newest_file, 'r') as f:
        content = f.read()
        
    sections = re.split(r"--- STTBN ", content)
    
    # First section has baseline standard TTBN
    baseline_sec = sections[0]
    baseline_table = {}
    table_pattern = r"(\d+)\s*\|\s*([\d\.]+)\s*\|\s*([\d\.]+)\s*\|\s*([\d\.]+)\s*\|\s*([\d\.]+)"
    matches = re.findall(table_pattern, baseline_sec)
    for bs, mnist, fashion, cifar, avg in matches:
        baseline_table[int(bs)] = float(avg)
        
    results = {
        'Baseline Standard TTBN': baseline_table
    }
    
    # Other sections correspond to STTBN variations
    for sec in sections[1:]:
        header_match = re.search(r"\(([\w\-,\s\=\.]+)\) ---", sec)
        if not header_match:
            continue
        header = header_match.group(1).strip()
        
        table = {}
        matches = re.findall(table_pattern, sec)
        for bs, mnist, fashion, cifar, avg in matches:
            table[int(bs)] = float(avg)
            
        results[header] = table
        
    return results

def main():
    results = parse_sttbn_log()
    if not results:
        return
        
    print("\nParsed STTBN Results:")
    for name, table in results.items():
        print(f"\nMethod: {name}")
        for bs, avg in sorted(table.items()):
            print(f"  Batch Size {bs:3d} -> Average Accuracy: {avg:.2f}%")

if __name__ == '__main__':
    main()
