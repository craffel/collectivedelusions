import json
import time
import subprocess
import os

def wait_for_jobs(job_ids, poll_interval=15):
    print(f"Waiting for Slurm jobs {job_ids} to complete...")
    job_str = ",".join(str(jid) for jid in job_ids)
    
    while True:
        try:
            res = subprocess.run(["/run/slurm-real/bin/squeue", "-j", job_str], capture_output=True, text=True)
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
            pass
        time.sleep(poll_interval)

def main():
    job_ids = [22162592, 22162593]
    wait_for_jobs(job_ids)
    
    # Wait extra time for files to sync/write
    time.sleep(5)
    
    # Load files
    fn_03 = "results_ta_l0.3_c128.json"
    fn_07 = "results_ta_l0.7_c128.json"
    
    if not os.path.exists(fn_03) or not os.path.exists(fn_07):
        print("Error: Result files do not exist yet!")
        return
        
    with open(fn_03, "r") as f:
        res_03 = json.load(f)
    with open(fn_07, "r") as f:
        res_07 = json.load(f)
        
    print("\n================== RESULTS FOR LAMBDA = 0.3 ==================")
    # Print No Calib
    # Let's check how the JSON stores No Calibration and Full Calibration
    # In results_ta_l0.5_c128.json, "No Calibration" was in sweep_data pct=0 index
    # and Full Calibration was at pct=100 index
    pcts = res_03["sweep_data"]["pct"]
    idx_0 = pcts.index(0)
    idx_100 = pcts.index(100)
    idx_10 = pcts.index(10)
    idx_20 = pcts.index(20)
    idx_50 = pcts.index(50)
    idx_80 = pcts.index(80)
    
    no_calib_03 = res_03["sweep_data"]["random_acc"][idx_0]
    full_calib_03 = res_03["sweep_data"]["random_acc"][idx_100]
    svcs_10_03 = res_03["sweep_data"]["svcs_acc"][idx_10]
    avcs_10_03 = res_03["sweep_data"]["avcs_acc"][idx_10]
    rand_10_03 = res_03["sweep_data"]["random_acc"][idx_10]
    
    svcs_20_03 = res_03["sweep_data"]["svcs_acc"][idx_20]
    avcs_20_03 = res_03["sweep_data"]["avcs_acc"][idx_20]
    rand_20_03 = res_03["sweep_data"]["random_acc"][idx_20]
    
    svcs_50_03 = res_03["sweep_data"]["svcs_acc"][idx_50]
    avcs_50_03 = res_03["sweep_data"]["avcs_acc"][idx_50]
    rand_50_03 = res_03["sweep_data"]["random_acc"][idx_50]
    
    svcs_80_03 = res_03["sweep_data"]["svcs_acc"][idx_80]
    avcs_80_03 = res_03["sweep_data"]["avcs_acc"][idx_80]
    rand_80_03 = res_03["sweep_data"]["random_acc"][idx_80]
    
    print(f"Uncalibrated Baseline (0%): {no_calib_03:.2f}%")
    print(f"Full Calibration (100%): {full_calib_03:.2f}%")
    print(f"SVCS 10%: {svcs_10_03:.2f}% | AVCS 10%: {avcs_10_03:.2f}% | Random 10%: {rand_10_03:.2f}%")
    print(f"SVCS 20%: {svcs_20_03:.2f}% | AVCS 20%: {avcs_20_03:.2f}% | Random 20%: {rand_20_03:.2f}%")
    print(f"SVCS 50%: {svcs_50_03:.2f}% | AVCS 50%: {avcs_50_03:.2f}% | Random 50%: {rand_50_03:.2f}%")
    print(f"SVCS 80%: {svcs_80_03:.2f}% | AVCS 80%: {avcs_80_03:.2f}% | Random 80%: {rand_80_03:.2f}%")
    
    print("\n================== RESULTS FOR LAMBDA = 0.7 ==================")
    no_calib_07 = res_07["sweep_data"]["random_acc"][idx_0]
    full_calib_07 = res_07["sweep_data"]["random_acc"][idx_100]
    svcs_10_07 = res_07["sweep_data"]["svcs_acc"][idx_10]
    avcs_10_07 = res_07["sweep_data"]["avcs_acc"][idx_10]
    rand_10_07 = res_07["sweep_data"]["random_acc"][idx_10]
    
    svcs_20_07 = res_07["sweep_data"]["svcs_acc"][idx_20]
    avcs_20_07 = res_07["sweep_data"]["avcs_acc"][idx_20]
    rand_20_07 = res_07["sweep_data"]["random_acc"][idx_20]
    
    svcs_50_07 = res_07["sweep_data"]["svcs_acc"][idx_50]
    avcs_50_07 = res_07["sweep_data"]["avcs_acc"][idx_50]
    rand_50_07 = res_07["sweep_data"]["random_acc"][idx_50]
    
    svcs_80_07 = res_07["sweep_data"]["svcs_acc"][idx_80]
    avcs_80_07 = res_07["sweep_data"]["avcs_acc"][idx_80]
    rand_80_07 = res_07["sweep_data"]["random_acc"][idx_80]
    
    print(f"Uncalibrated Baseline (0%): {no_calib_07:.2f}%")
    print(f"Full Calibration (100%): {full_calib_07:.2f}%")
    print(f"SVCS 10%: {svcs_10_07:.2f}% | AVCS 10%: {avcs_10_07:.2f}% | Random 10%: {rand_10_07:.2f}%")
    print(f"SVCS 20%: {svcs_20_07:.2f}% | AVCS 20%: {avcs_20_07:.2f}% | Random 20%: {rand_20_07:.2f}%")
    print(f"SVCS 50%: {svcs_50_07:.2f}% | AVCS 50%: {avcs_50_07:.2f}% | Random 50%: {rand_50_07:.2f}%")
    print(f"SVCS 80%: {svcs_80_07:.2f}% | AVCS 80%: {avcs_80_07:.2f}% | Random 80%: {rand_80_07:.2f}%")

if __name__ == "__main__":
    main()
