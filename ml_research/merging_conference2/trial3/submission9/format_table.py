import pandas as pd
import os

csv_file = "checkpoints/results.csv"
if not os.path.exists(csv_file):
    print(f"File {csv_file} not found yet. The Slurm job is probably still running.")
else:
    df = pd.read_csv(csv_file)
    print("--- TASK SPECIFIC HEADS ---")
    ts_df = df[df['head_setting'] == 'task_specific']
    for idx, row in ts_df.iterrows():
        print(f"{row['method']} & {row['mnist']:.2f} & {row['fashion']:.2f} & {row['cifar10']:.2f} & {row['average']:.2f} \\\\")
        
    print("\n--- SHARED HEAD ---")
    sh_df = df[df['head_setting'] == 'shared']
    for idx, row in sh_df.iterrows():
        print(f"{row['method']} & {row['mnist']:.2f} & {row['fashion']:.2f} & {row['cifar10']:.2f} & {row['average']:.2f} \\\\")
