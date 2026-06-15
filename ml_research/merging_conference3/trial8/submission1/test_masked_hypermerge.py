import sys
import torch
import numpy as np

# We will define a modified run_hypermerge with layer-wise top-M masking, and see what happens.
from run_experiments import run_regime

# Let's inspect the effect of layer-wise masking.
# We will copy the key parts from run_experiments.py to run a custom test.

def run_custom_test():
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Let's write a modified version of run_regime that supports masked hypermerge
    # We'll run it on subspaces_overlap with c=0.2, tau=0.05 or other parameters.
    pass

# Instead of rewriting the whole simulation, we can just run a python script that copies run_experiments.py
# and patches run_hypermerge inside it. Let's do a fast implementation of this.
