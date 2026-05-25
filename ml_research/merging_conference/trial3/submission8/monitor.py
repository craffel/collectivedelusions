import subprocess
import time

def get_active_jobs():
    try:
        out = subprocess.check_output(['/run/slurm-real/bin/squeue', '-h', '-u', 'craffel', '-o', '%j']).decode()
        return [line.strip() for line in out.split('\n') if line.strip() and line.strip().startswith('exp_')]
    except Exception as e:
        print(f"Error checking queue: {e}")
        return []

print("Starting monitoring...")
start_time = time.time()
while True:
    active_jobs = get_active_jobs()
    if not active_jobs:
        print("No active jobs left!")
        break
    print(f"Active jobs ({len(active_jobs)}): {active_jobs}")
    # Timeout after 8 minutes (480 seconds)
    if time.time() - start_time > 480:
        print("Timeout reached!")
        break
    time.sleep(20)
