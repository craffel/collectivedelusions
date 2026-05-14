"""Launch fine-tuning of all expert models in parallel on 8 GPUs."""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
import time


DEFAULT_TASKS = ["cifar10", "cifar100", "mnist", "svhn", "fashionmnist", "eurosat", "gtsrb", "dtd"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max_train_samples", type=int, default=20000)
    ap.add_argument("--max_eval_samples", type=int, default=4000)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--out", default="./checkpoints")
    ap.add_argument("--logs", default="./logs")
    ap.add_argument("--ngpus", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.logs, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)

    procs = []
    for i, task in enumerate(args.tasks):
        gpu = i % args.ngpus
        logf = open(os.path.join(args.logs, f"train_{task}.log"), "w")
        cmd = [
            sys.executable, "-m", "src.train_expert",
            "--task", task,
            "--device", f"cuda:{gpu}",
            "--epochs", str(args.epochs),
            "--bs", str(args.bs),
            "--lr", str(args.lr),
            "--max_train_samples", str(args.max_train_samples),
            "--max_eval_samples", str(args.max_eval_samples),
            "--num_workers", str(args.num_workers),
            "--data_root", args.data_root,
            "--out", args.out,
        ]
        print(f"[launch] gpu{gpu} -> {task}")
        p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
        procs.append((task, p, logf))
        time.sleep(2)  # stagger startup so datasets download serially

    failures = []
    for task, p, logf in procs:
        rc = p.wait()
        logf.close()
        status = "OK" if rc == 0 else f"FAIL rc={rc}"
        print(f"[done] {task}: {status}")
        if rc != 0:
            failures.append(task)
    print("FAILED:" if failures else "ALL OK", failures)
    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    main()
