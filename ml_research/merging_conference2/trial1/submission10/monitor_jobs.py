import time
import subprocess

def get_queue():
    result = subprocess.run(["squeue", "-h", "-u", "craffel"], capture_output=True, text=True)
    return result.stdout.strip()

def main():
    print("Starting job monitoring...")
    start_time = time.time()
    
    while True:
        queue_output = get_queue()
        if not queue_output:
            print("All Slurm jobs have completed!")
            break
            
        # Count remaining jobs
        lines = [line for line in queue_output.split("\n") if line.strip()]
        running = len([l for l in lines if " R " in l])
        pending = len([l for l in lines if " PD " in l])
        
        elapsed_min = (time.time() - start_time) / 60
        print(f"[{elapsed_min:.1f}m elapsed] Remaining jobs: {len(lines)} ({running} running, {pending} pending)...")
        
        # Print lines for visibility
        for line in lines[:5]:
            print("  ", line)
        if len(lines) > 5:
            print(f"   ... and {len(lines) - 5} more")
            
        time.sleep(30)

if __name__ == "__main__":
    main()
