import subprocess
import re

def submit_job(script_path, dependency=None):
    cmd = ["/opt/slurm/bin/sbatch"]
    if dependency:
        cmd.append(f"--dependency=afterok:{dependency}")
    cmd.append(script_path)
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print("STDOUT:", result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
        
    match = re.search(r"Submitted batch job (\d+)", result.stdout)
    if match:
        return match.group(1)
    return None

def main():
    # Cancel the old compile job
    scancel_cmd = ["/opt/slurm/bin/scancel", "22161859"]
    subprocess.run(scancel_cmd)
    
    sweep_job_ids = []
    
    # 1. Submit SAIM sweep
    saim_id = submit_job("slurm_scripts/sweep_vit_saim.slurm")
    if saim_id:
        sweep_job_ids.append(saim_id)
        
    # 2. Submit DOR-SAIM sweep
    dor_saim_id = submit_job("slurm_scripts/sweep_vit_dor_saim.slurm")
    if dor_saim_id:
        sweep_job_ids.append(dor_saim_id)
        
    if sweep_job_ids:
        # 3. Submit compile results job with dependency on both sweeps
        compilation_dependency = ":".join(sweep_job_ids)
        compile_script = "slurm_scripts/compile_all_results.slurm"
        compile_job_id = submit_job(compile_script, dependency=compilation_dependency)
        print(f"Submitted compile job with ID: {compile_job_id}")

if __name__ == "__main__":
    main()
