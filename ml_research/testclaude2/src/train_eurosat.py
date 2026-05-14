"""Patched EuroSAT trainer: disable SSL verification before torchvision dataset import."""
from __future__ import annotations
import os
import ssl
import sys

# Disable SSL verification globally for this process so torchvision can download.
ssl._create_default_https_context = ssl._create_unverified_context

# Now invoke train_expert with eurosat
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.train_expert import main

if __name__ == "__main__":
    sys.argv = ["train_eurosat",
                "--task", "eurosat",
                "--device", "cuda:5",
                "--epochs", "3",
                "--bs", "256",
                "--lr", "3e-5",
                "--max_train_samples", "20000",
                "--max_eval_samples", "4000",
                "--num_workers", "4",
                "--data_root", "./data",
                "--out", "./checkpoints"]
    main()
