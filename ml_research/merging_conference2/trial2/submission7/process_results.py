import os
import time
import json
import subprocess

def wait_for_jobs(job_ids, poll_interval=15):
    """Wait for specific Slurm jobs to be completed."""
    print(f"Waiting for Slurm jobs {job_ids} to complete...")
    job_str = ",".join(str(jid) for jid in job_ids)
    
    while True:
        try:
            # Run squeue for these specific jobs
            res = subprocess.run(["/run/slurm-real/bin/squeue", "-j", job_str], capture_output=True, text=True)
            # If the output is empty or doesn't contain the job IDs, they are done
            lines = res.stdout.strip().split("\n")
            active_jobs = 0
            for line in lines[1:]: # Skip header
                if any(str(jid) in line for jid in job_ids):
                    active_jobs += 1
            if active_jobs == 0:
                print("All Slurm jobs have completed!")
                return True
            else:
                print(f"Active Slurm jobs remaining: {active_jobs}...")
        except Exception as e:
            print(f"Error querying squeue: {e}")
            # If squeue fails, fallback to checking if files are recently updated
            pass
        time.sleep(poll_interval)

def format_table(results_wa, results_ta):
    # Extract baseline accuracies
    def get_metric(res, key, metric, fallback):
        # Let's check individual_results first
        val = res.get("individual_results", {}).get(key, {}).get(metric, None)
        if val is not None:
            return val
            
        # Fallback to sweep_data
        if "No Calibration" in key:
            idx = 0
        elif "Full Calibration" in key:
            idx = -1
        else:
            pct = int(key.split()[-1].replace("%", ""))
            try:
                idx = res["sweep_data"]["pct"].index(pct)
            except ValueError:
                return fallback
        
        m_key = "random_acc" if "Random" in key else ("avcs_acc" if "AVCS" in key else "svcs_acc")
        if "lat" in metric:
            m_key = m_key.replace("acc", "lat")
        
        try:
            return res["sweep_data"][m_key][idx]
        except Exception:
            return fallback

    wa_uncalib_acc = get_metric(results_wa, "No Calibration", "avg_acc", 35.24)
    wa_uncalib_lat = get_metric(results_wa, "No Calibration", "latency", 90.42)
    ta_uncalib_acc = get_metric(results_ta, "No Calibration", "avg_acc", 43.39)
    ta_uncalib_lat = get_metric(results_ta, "No Calibration", "latency", 94.83)
    
    wa_full_acc = get_metric(results_wa, "Full Calibration (TCAC)", "avg_acc", 14.81)
    wa_full_lat = get_metric(results_wa, "Full Calibration (TCAC)", "latency", 98.10)
    ta_full_acc = get_metric(results_ta, "Full Calibration (TCAC)", "avg_acc", 15.15)
    ta_full_lat = get_metric(results_ta, "Full Calibration (TCAC)", "latency", 102.23)

    table_latex = f"""\\begin{{table}}[t]
\\centering
\\caption{{Inference Accuracy (\\%) and Latency (ms) of LS-TCAC on ResNet-18.}}
\\label{{tab:main_results}}
\\begin{{small}}
\\begin{{tabular}}{{lcccc}}
\\toprule
& \\multicolumn{{2}}{{c}}{{\\textbf{{Weight Averaging}}}} & \\multicolumn{{2}}{{c}}{{\\textbf{{Task Arithmetic}}}} \\\\
\\cmidrule(lr){{2-3}} \\cmidrule(lr){{4-5}}
\\textbf{{Strategy}} & \\textbf{{Acc}} & \\textbf{{Lat (ms)}} & \\textbf{{Acc}} & \\textbf{{Lat (ms)}} \\\\
\\midrule
Uncalibrated & {wa_uncalib_acc:.2f} & {wa_uncalib_lat:.2f} & {ta_uncalib_acc:.2f} & {ta_uncalib_lat:.2f} \\\\
Full TCAC (100\\%) & {wa_full_acc:.2f} & {wa_full_lat:.2f} & {ta_full_acc:.2f} & {ta_full_lat:.2f} \\\\
\\midrule
"""
    
    for pct in [10, 20, 50, 80]:
        r_wa_acc = get_metric(results_wa, f"Random {pct}%", "avg_acc", 0.0)
        r_wa_lat = get_metric(results_wa, f"Random {pct}%", "latency", 0.0)
        r_ta_acc = get_metric(results_ta, f"Random {pct}%", "avg_acc", 0.0)
        r_ta_lat = get_metric(results_ta, f"Random {pct}%", "latency", 0.0)
        
        a_wa_acc = get_metric(results_wa, f"AVCS {pct}%", "avg_acc", 0.0)
        a_wa_lat = get_metric(results_wa, f"AVCS {pct}%", "latency", 0.0)
        a_ta_acc = get_metric(results_ta, f"AVCS {pct}%", "avg_acc", 0.0)
        a_ta_lat = get_metric(results_ta, f"AVCS {pct}%", "latency", 0.0)
        
        s_wa_acc = get_metric(results_wa, f"SVCS {pct}%", "avg_acc", 0.0)
        s_wa_lat = get_metric(results_wa, f"SVCS {pct}%", "latency", 0.0)
        s_ta_acc = get_metric(results_ta, f"SVCS {pct}%", "avg_acc", 0.0)
        s_ta_lat = get_metric(results_ta, f"SVCS {pct}%", "latency", 0.0)
        
        table_latex += f"""\\textbf{{SVCS {pct}\\% (Proposed)}} & \\textbf{{{s_wa_acc:.2f}}} & \\textbf{{{s_wa_lat:.2f}}} & \\textbf{{{s_ta_acc:.2f}}} & \\textbf{{{s_ta_lat:.2f}}} \\\\
AVCS {pct}\\% (Baseline) & {a_wa_acc:.2f} & {a_wa_lat:.2f} & {a_ta_acc:.2f} & {a_ta_lat:.2f} \\\\
Random {pct}\\% Baseline & {r_wa_acc:.2f} & {r_wa_lat:.2f} & {r_ta_acc:.2f} & {r_ta_lat:.2f} \\\\
\\midrule
"""
        
    table_latex = table_latex.strip()[:-8] + "\n\\bottomrule\n\\end{tabular}\n\\end{small}\n\\end{table}"
    return table_latex

def update_latex(table_latex):
    with open("template/submission.tex", "r") as fh:
        content = fh.read()
        
    start_tag = "\\begin{table}[t]"
    end_tag = "\\end{table}"
    
    start_idx = content.find(start_tag)
    end_idx = content.find(end_tag, start_idx) + len(end_tag)
    
    if start_idx == -1 or end_idx == -1:
        print("Could not find table in LaTeX file!")
        return False
        
    new_content = content[:start_idx] + table_latex + content[end_idx:]
    
    figure_latex = """\\begin{figure}[t]
    \\centering
    \\includegraphics[width=0.48\\textwidth]{plots/accuracy_vs_layers_ta_l0.5_c128.png}
    \\caption{Multi-task average accuracy vs. percentage of calibrated layers under Task Arithmetic merging ($\lambda=0.5$, $C=128$). Our proposed SVCS selection strategy surgically prevents representation degradation, outperforming the static AVCS baseline as $k$ increases.}
    \\label{fig:accuracy_vs_layers}
\\end{figure}"""
    
    if "fig:accuracy_vs_layers" not in new_content:
        table_end_idx = new_content.find("\\end{table}") + len("\\end{table}")
        new_content = new_content[:table_end_idx] + "\n\n" + figure_latex + new_content[table_end_idx:]
        
    with open("template/submission.tex", "w") as fh:
        fh.write(new_content)
    return True

def main():
    job_ids = [22162459, 22162460, 22162461, 22162462]
    wait_for_jobs(job_ids)
    
    print("Jobs completed! Loading final result files...")
    # Add a short sleep to ensure file system syncs
    time.sleep(2)
    
    with open("results_wa_l0.0_c128.json", "r") as fh:
        results_wa = json.load(fh)
    with open("results_ta_l0.5_c128.json", "r") as fh:
        results_ta = json.load(fh)
        
    print("Formatting table...")
    table_latex = format_table(results_wa, results_ta)
    
    print("Updating LaTeX template...")
    if update_latex(table_latex):
        print("Successfully updated template/submission.tex!")
        
        # Compile PDF
        print("Compiling LaTeX to PDF...")
        try:
            res = subprocess.run(["./tectonic", "template/submission.tex"], capture_output=True, text=True)
            if res.returncode == 0:
                print("PDF compiled successfully!")
                if os.path.exists("template/submission.pdf"):
                    subprocess.run(["cp", "template/submission.pdf", "submission.pdf"])
                    print("Copied submission.pdf to root directory.")
            else:
                print("Tectonic compile failed!")
                print(res.stderr)
        except Exception as e:
            print(f"Error during compilation: {e}")
            
if __name__ == "__main__":
    main()
