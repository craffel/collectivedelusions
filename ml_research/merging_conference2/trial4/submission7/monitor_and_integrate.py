import time
import subprocess
import os
import re
import sys

JOB_ID = "22163860"
LOG_FILE = f"run-experiments_{JOB_ID}.out"
TEX_FILE = "submission.tex"

def check_job_active(job_id):
    try:
        res = subprocess.run(["/run/slurm-real/bin/squeue", "-j", job_id], capture_output=True, text=True)
        if job_id in res.stdout:
            return True
    except Exception as e:
        print(f"Error checking job queue: {e}")
    return False

def wait_for_job(job_id):
    print(f"Waiting for Slurm job {job_id} to complete...")
    start_time = time.time()
    while check_job_active(job_id):
        elapsed = time.time() - start_time
        print(f"Job is still active... Elapsed time: {elapsed/60:.1f} minutes. Checking again in 30 seconds.")
        time.sleep(30)
    print("Slurm job is no longer in the queue.")

def clean_val(val):
    val = val.strip().replace('%', '')
    if '±' in val:
        mean, std = val.split('±')
        mean = mean.strip()
        std = std.strip()
        if float(std) == 0.0:
            return f"{mean}"
        return f"{mean} \\pm {std}"
    return val

def parse_and_update():
    if not os.path.exists(LOG_FILE):
        print(f"Log file {LOG_FILE} not found!")
        return False
        
    with open(LOG_FILE, 'r') as f:
        content = f.read()
        
    # Verify job completion
    if "FINAL MULTI-SEED AGGREGATED RESULTS TABLE" not in content:
        print("Final aggregated results are not in the log file yet.")
        return False
        
    print("Parsing results from log file...")
    
    # 1. Parse Main Table
    main_pattern = r"FINAL MULTI-SEED AGGREGATED RESULTS TABLE\n=+\n.*?\n-+\n(.*?)\n=+"
    main_match = re.search(main_pattern, content, re.DOTALL)
    if not main_match:
        print("Could not parse main table.")
        return False
        
    main_rows_raw = main_match.group(1).strip().split('\n')
    main_rows = []
    for row in main_rows_raw:
        parts = [p.strip() for p in row.split('|')]
        if len(parts) == 7:
            n_val, method, mnist, fashion, cifar, mean, sft = parts
            n_val = n_val.strip()
            method_name = method.strip()
            if method_name == 'HSC':
                method_name = "\\textbf{HSC (Ours)}"
            elif method_name == 'uncalibrated':
                method_name = "Uncalibrated (WA)"
            
            mnist_fmt = clean_val(mnist)
            fashion_fmt = clean_val(fashion)
            cifar_fmt = clean_val(cifar)
            mean_fmt = clean_val(mean)
            sft_fmt = clean_val(sft)
            
            if 'HSC' in method_name:
                mean_fmt = f"\\textbf{{{mean_fmt}}}"
                sft_fmt = f"\\textbf{{{sft_fmt}}}"
                
            main_rows.append((n_val, method_name, mnist_fmt, fashion_fmt, cifar_fmt, mean_fmt, sft_fmt))

    # 2. Parse Split Sweep
    split_pattern = r"HSC SPLIT SWEEP \(N=128\)\n=+\n.*?\n-+\n(.*?)\n=+"
    split_match = re.search(split_pattern, content, re.DOTALL)
    split_rows = []
    if split_match:
        split_rows_raw = split_match.group(1).strip().split('\n')
        for row in split_rows_raw:
            parts = [p.strip() for p in row.split('|')]
            if len(parts) == 3:
                split_idx, mean, sft = parts
                split_idx_val = split_idx.strip()
                mean_fmt = clean_val(mean)
                sft_fmt = clean_val(sft)
                
                label = split_idx_val
                if split_idx_val == '0':
                    label = "0 (Pure R-TAAC)"
                elif split_idx_val == '15':
                    label = "\\textbf{15 (Layer 4 Block)}"
                    mean_fmt = f"\\textbf{{{mean_fmt}}}"
                    sft_fmt = f"\\textbf{{{sft_fmt}}}"
                    
                split_rows.append((label, mean_fmt, sft_fmt))

    # 3. Parse Alpha Sweep
    alpha_pattern = r"HSC ALPHA SWEEP \(N=128, Split=15\)\n=+\n.*?\n-+\n(.*?)\n=+"
    alpha_match = re.search(alpha_pattern, content, re.DOTALL)
    alpha_rows = []
    if alpha_match:
        alpha_rows_raw = alpha_match.group(1).strip().split('\n')
        for row in alpha_rows_raw:
            parts = [p.strip() for p in row.split('|')]
            if len(parts) == 3:
                alpha_val, mean, sft = parts
                alpha_val_str = alpha_val.strip()
                mean_fmt = clean_val(mean)
                sft_fmt = clean_val(sft)
                
                label = alpha_val_str
                if float(alpha_val_str) == 0.25:
                    label = "\\textbf{0.25}"
                    mean_fmt = f"\\textbf{{{mean_fmt}}}"
                    sft_fmt = f"\\textbf{{{sft_fmt}}}"
                    
                alpha_rows.append((label, mean_fmt, sft_fmt))

    # Load submission.tex
    with open(TEX_FILE, 'r') as f:
        tex = f.read()

    # Generate Main Table LaTeX block
    new_main_table = []
    new_main_table.append("\\begin{table*}[t]")
    new_main_table.append("\\caption{Multi-task model merging accuracies (\\%) on ResNet-18 across calibration sizes $N$ (aggregated over multiple seeds). Methods are evaluated both without head SFT (Representation only) and with head SFT (Mean+SFT).}")
    new_main_table.append("\\label{tab:main}")
    new_main_table.append("\\vskip 0.15in")
    new_main_table.append("\\begin{center}")
    new_main_table.append("\\begin{small}")
    new_main_table.append("\\begin{tabular}{llccccc}")
    new_main_table.append("\\toprule")
    new_main_table.append("N & Method & MNIST Acc & F-MNIST Acc & CIFAR-10 Acc & Mean Acc & Mean+SFT Acc \\\\")
    new_main_table.append("\\midrule")
    new_main_table.append("- & Expert Models (Upper Bound) & 98.45 & 85.40 & 67.38 & 83.74 & -- \\\\")
    new_main_table.append("\\midrule")
    
    prev_n = None
    for row in main_rows:
        n_val, method, mnist, fashion, cifar, mean, sft = row
        if prev_n is not None and prev_n != n_val:
            new_main_table.append("\\midrule")
        new_main_table.append(f"{n_val} & {method} & {mnist} & {fashion} & {cifar} & {mean} & {sft} \\\\")
        prev_n = n_val
        
    new_main_table.append("\\bottomrule")
    new_main_table.append("\\end{tabular}")
    new_main_table.append("\\end{small}")
    new_main_table.append("\\end{center}")
    new_main_table.append("\\vskip -0.15in")
    new_main_table.append("\\vspace{-3mm}")
    new_main_table.append("\\end{table*}")
    new_main_table_str = "\n".join(new_main_table)

    # Surgically replace main table
    tex_replaced = re.sub(
        r"\\begin\{table\*\}\[t\].*?\\label\{tab:main\}.*?\\end\{table\*\}",
        new_main_table_str.replace("\\", "\\\\"),  # escape backslashes for regex sub
        tex,
        flags=re.DOTALL
    )

    # Generate Split Sweep tabular block
    split_tabular = []
    split_tabular.append("\\begin{tabular}{ccc}")
    split_tabular.append("\\toprule")
    split_tabular.append("Split Index ($M$) & Mean Acc (\\%) & Mean+SFT Acc (\\%) \\\\")
    split_tabular.append("\\midrule")
    for row in split_rows:
        label, mean, sft = row
        split_tabular.append(f"{label} & {mean} & {sft} \\\\")
    split_tabular.append("\\bottomrule")
    split_tabular.append("\\end{tabular}")
    split_tabular_str = "\n".join(split_tabular)

    # Surgically replace split tabular inside split minipage
    # Locate the split minipage and replace the tabular within it
    # We find the minipage that contains label{tab:split}
    split_minipage_pattern = r"(\\begin\{minipage\}\{0\.48\\textwidth\}\s*\\centering\s*\\caption\{.*?split.*?\}.*?\\label\{tab:split\}.*?\\begin\{tabular\}.*?\\end\{tabular\}\s*\\end\{small\}\s*\\end\{minipage\})"
    split_match = re.search(split_minipage_pattern, tex_replaced, re.DOTALL)
    if split_match:
        old_split_mp = split_match.group(1)
        new_split_mp = re.sub(
            r"\\begin\{tabular\}.*?\\end\{tabular\}",
            split_tabular_str.replace("\\", "\\\\"),
            old_split_mp,
            flags=re.DOTALL
        )
        tex_replaced = tex_replaced.replace(old_split_mp, new_split_mp)

    # Generate Alpha Sweep tabular block
    alpha_tabular = []
    alpha_tabular.append("\\begin{tabular}{ccc}")
    alpha_tabular.append("\\toprule")
    alpha_tabular.append("Alpha ($\\alpha$) & Mean Acc (\\%) & Mean+SFT Acc (\\%) \\\\")
    alpha_tabular.append("\\midrule")
    for row in alpha_rows:
        label, mean, sft = row
        alpha_tabular.append(f"{label} & {mean} & {sft} \\\\")
    alpha_tabular.append("\\bottomrule")
    alpha_tabular.append("\\end{tabular}")
    alpha_tabular_str = "\n".join(alpha_tabular)

    # Surgically replace alpha tabular inside alpha minipage
    alpha_minipage_pattern = r"(\\begin\{minipage\}\{0\.48\\textwidth\}\s*\\centering\s*\\caption\{.*?alpha.*?\}.*?\\label\{tab:alpha\}.*?\\begin\{tabular\}.*?\\end\{tabular\}\s*\\end\{small\}\s*\\end\{minipage\})"
    alpha_match = re.search(alpha_minipage_pattern, tex_replaced, re.DOTALL)
    if alpha_match:
        old_alpha_mp = alpha_match.group(1)
        new_alpha_mp = re.sub(
            r"\\begin\{tabular\}.*?\\end\{tabular\}",
            alpha_tabular_str.replace("\\", "\\\\"),
            old_alpha_mp,
            flags=re.DOTALL
        )
        tex_replaced = tex_replaced.replace(old_alpha_mp, new_alpha_mp)

    # Save back to submission.tex
    with open(TEX_FILE, 'w') as f:
        f.write(tex_replaced)
        
    print(f"Successfully integrated all multi-seed results into {TEX_FILE}!")
    return True

def compile_and_verify():
    print("Compiling LaTeX manuscript...")
    # Run tectonic
    res = subprocess.run(["tectonic", TEX_FILE], capture_output=True, text=True)
    if res.returncode != 0:
        print("LaTeX compilation failed!")
        print(res.stderr)
        return False
        
    print("Compilation successful! Checking page count...")
    
    # Check page count
    try:
        import pypdf
        reader = pypdf.PdfReader('submission.pdf')
        total_pages = len(reader.pages)
        print(f"Total compiled PDF pages: {total_pages}")
        
        ref_page = -1
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if 'References' in text or 'REFERENCES' in text:
                ref_page = i + 1
                break
                
        print(f"References section starts on page: {ref_page}")
        
        if total_pages == 10 and ref_page == 9:
            print("PERFECT! Main text is exactly 8 pages, and references start on page 9.")
            return True
        elif total_pages > 10 or ref_page > 9:
            print("WARNING: Main text exceeds the 8-page limit! Refinement is needed.")
            return False
        else:
            print("Main text is shorter than 8 pages (completely fine).")
            return True
    except Exception as e:
        print(f"Error checking page count: {e}")
        
    return False

def main():
    wait_for_job(JOB_ID)
    success = parse_and_update()
    if success:
        compile_and_verify()

if __name__ == '__main__':
    main()
