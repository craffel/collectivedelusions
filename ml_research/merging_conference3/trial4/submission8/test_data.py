import sys
sys.path.insert(0, "./env_packages")
import torch
import torchvision
from experiments.run_merging import load_eval_data

print("Loading evaluation data...")
loaders, calib_batches = load_eval_data()
print("Data loaded successfully!")
