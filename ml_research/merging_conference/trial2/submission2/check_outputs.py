import glob
import os

print("--- Slurm Output Files ---")
for f in sorted(glob.glob("cifar10-*.out")):
    print(f"\n[{f}]")
    with open(f, "r") as fh:
        lines = fh.readlines()
        print("".join(lines[-10:]))
        
print("--- Slurm Error Files ---")
for f in sorted(glob.glob("cifar10-*.err")):
    if os.path.getsize(f) > 0:
        print(f"\n[{f}]")
        with open(f, "r") as fh:
            lines = fh.readlines()
            print("".join(lines[-10:]))
