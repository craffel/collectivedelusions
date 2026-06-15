import run_experiments
import os

print("Running specific new experiments only...")
os.makedirs("results", exist_ok=True)
os.makedirs("submission", exist_ok=True)

# 1. Run GPU profiling benchmark
gpu_elati, gpu_pfsr, gpu_speedup = run_experiments.run_gpu_profiling_benchmark()

# 2. Run centroid adaptation experiment under domain drift
static_means_drift, adaptive_means_drift = run_experiments.run_centroid_adaptation_experiment()

print("Specific new experiments completed successfully!")
