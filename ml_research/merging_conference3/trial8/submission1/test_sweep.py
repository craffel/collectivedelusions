import sys
print("Starting script...")
sys.stdout.flush()

import torch
print("Torch imported.")
sys.stdout.flush()

import numpy as np
print("Numpy imported.")
sys.stdout.flush()

print("Importing run_regime from run_experiments...")
sys.stdout.flush()
from run_experiments import run_regime
print("run_regime imported successfully!")
sys.stdout.flush()

subspaces_overlap = {
    0: (0, 96),
    1: (32, 128),
    2: (64, 160),
    3: (96, 192)
}

c_list = [0.01, 0.05, 0.1, 0.2, 0.5]

for c in c_list:
    print(f"Running sweep for c={c}...")
    sys.stdout.flush()
    res = run_regime(subspaces_overlap, c_hyperbolic=c)
    print(f"c={c:.2f} -> HyperMerge Acc: {res['hm_homo']*100:.2f}%, SABLE: {res['sable_e_homo']*100:.2f}%, SPS-ZCA: {res['sps_homo']*100:.2f}%")
    sys.stdout.flush()
