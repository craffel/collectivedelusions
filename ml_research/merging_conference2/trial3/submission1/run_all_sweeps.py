import os
import glob
import subprocess
import multiprocessing
import queue

def get_configurations():
    merge_modes = ["WA", "TA"]
    seeds = [42, 43, 44]
    cal_sizes = [4, 8, 16, 32, 64, 128, 256]
    ta_coeffs = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

    configs = []

    # 1. WA configs
    for seed in seeds:
        for cal_size in cal_sizes:
            filename = f"results_sweep_WA_0.3_N{cal_size}_S{seed}.json"
            if not os.path.exists(filename) or os.path.getsize(filename) < 10:
                configs.append(("WA", 0.3, cal_size, seed))

    # 2. TA configs
    for seed in seeds:
        for coeff in ta_coeffs:
            for cal_size in cal_sizes:
                filename = f"results_sweep_TA_{coeff}_N{cal_size}_S{seed}.json"
                if not os.path.exists(filename) or os.path.getsize(filename) < 10:
                    configs.append(("TA", coeff, cal_size, seed))

    return configs

def worker_proc(config_queue, gpu_id):
    # Set CUDA_VISIBLE_DEVICES if CUDA is available
    env = os.environ.copy()
    import torch
    if torch.cuda.is_available():
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        device_name = f"GPU {gpu_id}"
    else:
        device_name = f"CPU Worker {gpu_id}"

    while True:
        try:
            config = config_queue.get_nowait()
        except queue.Empty:
            break

        merge_mode, coeff, cal_size, seed = config
        print(f"[{device_name}] Starting sweep: mode={merge_mode}, coeff={coeff}, N={cal_size}, S={seed}")
        
        cmd = [
            "python", "experiment.py",
            "--mode", "sweep",
            "--merge_mode", merge_mode,
            "--coeff", str(coeff),
            "--cal_size", str(cal_size),
            "--seed", str(seed)
        ]
        
        # Run command
        res = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if res.returncode == 0:
            print(f"[{device_name}] Finished sweep: mode={merge_mode}, coeff={coeff}, N={cal_size}, S={seed}")
        else:
            print(f"[{device_name}] FAILED sweep: mode={merge_mode}, coeff={coeff}, N={cal_size}, S={seed}")
            print(res.stderr)

def main():
    configs = get_configurations()
    print(f"Total configurations to run: {len(configs)}")
    if not configs:
        print("All configurations already completed!")
        return

    # Queue to hold all remaining configurations
    config_queue = multiprocessing.Queue()
    for c in configs:
        config_queue.put(c)

    import torch
    if torch.cuda.is_available():
        num_workers = 8
    else:
        num_workers = 4 # Local CPU node has 4 CPUs
        
    processes = []
    
    for worker_id in range(num_workers):
        p = multiprocessing.Process(target=worker_proc, args=(config_queue, worker_id))
        processes.append(p)
        p.start()

    # Wait for all processes to finish
    for p in processes:
        p.join()

    print("All worker processes completed.")

if __name__ == "__main__":
    main()
