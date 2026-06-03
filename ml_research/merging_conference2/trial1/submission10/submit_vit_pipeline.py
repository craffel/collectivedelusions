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
    train_dependencies = "22161849:22161850"
    
    methods = ['arithmetic', 'ties', 'dare', 'orthomerge', 'saim', 'dor_saim']
    sweep_job_ids = []
    
    for method in methods:
        script = f"slurm_scripts/sweep_vit_{method}.slurm"
        job_id = submit_job(script, dependency=train_dependencies)
        if job_id:
            sweep_job_ids.append(job_id)
            
    if sweep_job_ids:
        compilation_dependency = ":".join(sweep_job_ids)
        compile_script = "slurm_scripts/compile_all_results.slurm"
        compile_job_id = submit_job(compile_script, dependency=compilation_dependency)
        print(f"Submitted compile job with ID: {compile_job_id}")

if __name__ == "__main__":
    main()
