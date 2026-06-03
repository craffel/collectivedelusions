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
    # Cancel previous single sweep and compile jobs
    subprocess.run(["/opt/slurm/bin/scancel", "22161913", "22161914"])
    
    sweep_job_ids = []
    
    # 1. Submit Part 1
    id1 = submit_job("slurm_scripts/sweep_vit_dor_saim_part1.slurm")
    if id1:
        sweep_job_ids.append(id1)
        
    # 2. Submit Part 2
    id2 = submit_job("slurm_scripts/sweep_vit_dor_saim_part2.slurm")
    if id2:
        sweep_job_ids.append(id2)
        
    # 3. Submit Part 3
    id3 = submit_job("slurm_scripts/sweep_vit_dor_saim_part3.slurm")
    if id3:
        sweep_job_ids.append(id3)
        
    if sweep_job_ids:
        # Submit compile results with dependency on all three parallel parts
        compilation_dependency = ":".join(sweep_job_ids)
        compile_script = "slurm_scripts/compile_all_results.slurm"
        compile_job_id = submit_job(compile_script, dependency=compilation_dependency)
        print(f"Submitted compile job with ID: {compile_job_id}")

if __name__ == "__main__":
    main()
