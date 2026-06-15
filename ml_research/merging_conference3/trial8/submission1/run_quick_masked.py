import torch
import numpy as np
import sys

# Reset seeds
torch.manual_seed(42)
np.random.seed(42)

from test_masked import run_regime

subspaces_overlap = {
    0: (0, 96),
    1: (32, 128),
    2: (64, 160),
    3: (96, 192)
}

print("Running quick masked HyperMerge test...")
sys.stdout.flush()

for c in [0.1, 0.2, 0.3, 0.5]:
    acc_sable, acc_sps, acc_hm = run_regime(subspaces_overlap, c_hyperbolic=c, tau=0.05, mask_layers=True)
    print(f"Overlap c={c:.2f} -> SABLE Early: {acc_sable*100:.2f}%, SPS-ZCA: {acc_sps*100:.2f}%, HyperMerge Masked: {acc_hm*100:.2f}%")
    sys.stdout.flush()
