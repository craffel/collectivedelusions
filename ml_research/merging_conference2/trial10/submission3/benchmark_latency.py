import torch
import torch.nn as nn
import time
import numpy as np
from torchvision.models import resnet18
from evaluate_merging_complete import ADSRModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Benchmarking on device: {device}")

# Load expert models
experts = []
for task in ["mnist", "fmnist", "cifar10"]:
    chk_path = f"checkpoints/{task}_expert.pt"
    model = resnet18()
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load(chk_path, map_location="cpu"))
    model = model.to(device)
    model.eval()
    experts.append(model)

# Instantiate models
wa_model = experts[0] # standard ResNet-18 for latency baseline

bw_adsr = ADSRModel(experts, temp=-5.0, channel_wise=False, adaptive_scale=False).to(device)
cw_adsr = ADSRModel(experts, temp_cw=-0.5, channel_wise=True, adaptive_scale=False).to(device)
as_adsr = ADSRModel(experts, temp=-5.0, temp_cw=-0.5, adaptive_scale=True).to(device)

bw_adsr.eval()
cw_adsr.eval()
as_adsr.eval()

models = {
    "Weight Averaging (Single Backbone)": wa_model,
    "BW-ADSR (Ours)": bw_adsr,
    "CW-ADSR (Ours)": cw_adsr,
    "AS-ADSR (Ours, Adaptive)": as_adsr
}

batch_sizes = [1, 32, 128]
num_warmup = 5
num_runs = 20

results = {}

for name, model in models.items():
    results[name] = {}
    print(f"\nBenchmarking {name}...")
    for bs in batch_sizes:
        # Create dummy input of shape [Batch, Channels, Height, Width]
        dummy_input = torch.randn(bs, 3, 32, 32).to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(dummy_input, task_id=0) if "ADSR" in name else model(dummy_input)
        
        # Timing
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(dummy_input, task_id=0) if "ADSR" in name else model(dummy_input)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        avg_latency_ms = (total_time / num_runs) * 1000.0
        throughput = (bs * num_runs) / total_time
        
        results[name][bs] = {
            "latency_ms": avg_latency_ms,
            "throughput": throughput
        }
        print(f"  Batch Size {bs:3d} | Latency: {avg_latency_ms:7.2f} ms | Throughput: {throughput:8.1f} img/sec")

# Print LaTeX table representation
print("\n--- LaTeX Table Output ---")
print("\\begin{table}[h]")
print("\\caption{Inference latency (ms) and throughput (images/sec) across different batch sizes $B$ on the CPU serving node.}")
print("\\label{tab:latency}")
print("\\vskip 0.15in")
print("\\begin{center}")
print("\\begin{small}")
print("\\begin{sc}")
print("\\setlength{\\tabcolsep}{4.0pt}")
print("\\begin{tabular}{lcccccc}")
print("\\toprule")
print("Method & \\multicolumn{3}{c}{Latency (ms)} & \\multicolumn{3}{c}{Throughput (img/sec)} \\\\")
print(" & $B=1$ & $B=32$ & $B=128$ & $B=1$ & $B=32$ & $B=128$ \\\\")
print("\\midrule")
for name in models.keys():
    lat_1 = results[name][1]["latency_ms"]
    lat_32 = results[name][32]["latency_ms"]
    lat_128 = results[name][128]["latency_ms"]
    tp_1 = results[name][1]["throughput"]
    tp_32 = results[name][32]["throughput"]
    tp_128 = results[name][128]["throughput"]
    
    clean_name = name.replace(" (Single Backbone)", "").replace(" (Ours)", "").replace(" (Ours, Adaptive)", "")
    if "ADSR" in name:
        print(f"\\textbf{{{clean_name}}} & \\textbf{{{lat_1:.2f}}} & \\textbf{{{lat_32:.2f}}} & \\textbf{{{lat_128:.2f}}} & \\textbf{{{tp_1:.1f}}} & \\textbf{{{tp_32:.1f}}} & \\textbf{{{tp_128:.1f}}} \\\\")
    else:
        print(f"{clean_name} & {lat_1:.2f} & {lat_32:.2f} & {lat_128:.2f} & {tp_1:.1f} & {tp_32:.1f} & {tp_128:.1f} \\\\")
print("\\bottomrule")
print("\\end{tabular}")
print("\\end{sc}")
print("\\end{small}")
print("\\end{center}")
print("\\vskip -0.1in")
print("\\end{table}")
