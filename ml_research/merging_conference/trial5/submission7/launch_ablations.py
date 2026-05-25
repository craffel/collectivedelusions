import os
import subprocess
import time
import json
import numpy as np

def run_cmd(cmd):
    result = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    return result.stdout.strip(), result.returncode

def submit_job(mode, stream, corruption, config_name=None, alpha=None):
    job_name = f"ab_{mode[:4]}_{stream[:3]}_{corruption[:3]}"
    if mode == "component":
        job_name += f"_{config_name[:4]}"
        export_vars = f"ALL,MODE=component,STREAM={stream},CORRUPTION={corruption},CONFIG_NAME={config_name}"
    else:
        job_name += f"_{str(alpha).replace('.', '')}"
        export_vars = f"ALL,MODE=alpha,STREAM={stream},CORRUPTION={corruption},ALPHA={alpha}"
        
    cmd = f"sbatch --job-name={job_name} --export={export_vars} ablate_runner.slurm"
    out, code = run_cmd(cmd)
    if code == 0:
        # Expected output: "Submitted batch job <job_id>"
        job_id = out.split()[-1]
        return job_id
    else:
        print(f"Error submitting job: {out}")
        return None

def main():
    streams = ['alternating', 'sequential']
    corruptions = ['noise', 'contrast']
    
    # 1. Component configs
    component_configs = ['Full_IGGS_Merge', 'No_OPR', 'Euclidean_Proj', 'No_Proj', 'No_Preconditioning']
    
    # 2. Alpha configs
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    job_ids = []
    
    print("Launching Component Ablation Jobs...")
    for stream in streams:
        for corr in corruptions:
            for cfg in component_configs:
                job_id = submit_job("component", stream, corr, config_name=cfg)
                if job_id:
                    job_ids.append(job_id)
                    print(f"Submitted Job {job_id} for component_{stream}_{corr}_{cfg}")
                    
    print("\nLaunching Alpha Sweep Jobs...")
    for stream in streams:
        for corr in corruptions:
            for a in alphas:
                job_id = submit_job("alpha", stream, corr, alpha=a)
                if job_id:
                    job_ids.append(job_id)
                    print(f"Submitted Job {job_id} for alpha_{stream}_{corr}_{a}")
                    
    print(f"\nTotal of {len(job_ids)} jobs submitted. Monitoring progress...")
    
    # Monitor jobs
    while True:
        # Check queue
        out, code = run_cmd("squeue")
        running_jobs = []
        for line in out.split("\n")[1:]: # skip header
            if line.strip():
                parts = line.split()
                if len(parts) > 0:
                    running_jobs.append(parts[0])
                    
        active_ours = [jid for jid in job_ids if jid in running_jobs]
        if not active_ours:
            print("\nAll submitted Slurm jobs have completed!")
            break
            
        print(f"Active/pending jobs: {len(active_ours)} / {len(job_ids)}. Sleeping 20s...")
        time.sleep(20)
        
    # Consolidate results
    print("\nConsolidating results from ablation_parts/ ...")
    parts_dir = "ablation_parts"
    
    ablation_results = {}
    alpha_results = {}
    
    # Initialize structures
    for stream in streams:
        ablation_results[stream] = {}
        alpha_results[stream] = {}
        for corr in corruptions:
            ablation_results[stream][corr] = {}
            alpha_results[stream][corr] = {}
            
    files = os.listdir(parts_dir)
    for file in files:
        if file.endswith(".json"):
            filepath = os.path.join(parts_dir, file)
            with open(filepath, "r") as f:
                data = json.load(f)
                
            stream = data["stream"]
            corr = data["corruption"]
            if data["mode"] == "component":
                cfg_name = data["config_name"]
                ablation_results[stream][corr][cfg_name] = data["accuracy"]
            elif data["mode"] == "alpha":
                a = data["alpha"]
                alpha_results[stream][corr][a] = data["accuracy"]
                
    # Save consolidate file
    np.savez('ablation_results.npz', ablation_results=ablation_results, alpha_results=alpha_results)
    print("Ablation results successfully consolidated and saved to ablation_results.npz!")

if __name__ == "__main__":
    main()
