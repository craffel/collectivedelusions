import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# Create results directory if not exists
os.makedirs("results", exist_ok=True)

# Hardware Specifications: STM32H753XI (ARM Cortex-M7, 480 MHz, 1 MB SRAM, 2 MB Flash)
CLOCK_SPEED_MHZ = 480.0
CYCLE_TIME_NS = 1000.0 / CLOCK_SPEED_MHZ  # ~2.08 ns per cycle

# System parameters (matching Coordinate Sandbox)
D = 192
K = 4
r = 8
num_layers = 12
num_adapted_layers = 9  # Layers 4 to 12

# Weights sizes (in bytes)
# Base model weights: 12 layers of D x D.
# Plus 2 frozen base layers at layers 13-14 (since sandbox has 14 layers in total).
# Total base layers = 14 layers.
total_base_layers = 14
base_params = total_base_layers * D * D

# LoRA expert weights: 9 adapted layers. Each adapter has A (D x r) and B (r x D)
adapter_params_per_task = num_adapted_layers * (D * r + r * D)

# Size in KB
def size_kb(num_params, bytes_per_param):
    return (num_params * bytes_per_param) / 1024.0

# 1. Flash Storage Requirements (KB)
flash_fp16_ensemble = size_kb(base_params, 2) + K * size_kb(adapter_params_per_task, 2)
flash_pmq_4bit = size_kb(base_params, 0.5)  # 4-bit static merged
flash_sa_qab = size_kb(base_params, 0.5) + K * size_kb(adapter_params_per_task, 1) + size_kb(K * D, 1) # base INT4 + K INT8 adapters + centroids

# 2. Active SRAM Footprint (KB)
# Includes weights currently in SRAM + activations of a single layer (batch size 1 for edge)
activation_size_kb = size_kb(D, 4) # float activations in intermediate layers
sram_fp16_ensemble = size_kb(base_params, 2) + K * size_kb(adapter_params_per_task, 2) + activation_size_kb
sram_pmq_4bit = size_kb(base_params, 0.5) + activation_size_kb
# SA-QAB stores base INT4 and ALL adapters in SRAM for modular execution (or dynamically loads them, but let's assume they are stored in SRAM)
sram_sa_qab = size_kb(base_params, 0.5) + K * size_kb(adapter_params_per_task, 1) + activation_size_kb

# 3. MAC Operations & Estimated Latency (ms)
# PMQ FP16 Ensembling: runs entire base model + all K adapters in parallel (uniform blending)
macs_base = total_base_layers * D * D
macs_adapters_ensemble = K * num_adapted_layers * (D * r + r * D)
macs_fp16_ensemble = macs_base + macs_adapters_ensemble

# PMQ 4-bit: runs base model only (since adapters are static merged)
macs_pmq_4bit = macs_base

# SA-QAB: runs base model (INT4) + router (INT8, Block 3) + ONLY ONE adapter (INT8, Block 4-12)
macs_router = K * D # Cosine similarity of 192D feature against K centroids
macs_adapters_sa_qab = 1 * num_adapted_layers * (D * r + r * D) # execute exactly 1 adapter
macs_sa_qab = macs_base + macs_router + macs_adapters_sa_qab

# Cycle estimates per MAC on ARM Cortex-M7 (including memory access, unpacking overhead, instruction pipelining):
# - FP16 MAC takes ~1.5 cycles (utilizing single-cycle FPU, but with memory load overhead)
# - INT4 MAC takes ~0.75 cycles (using CMSIS-NN SIMD instructions, plus unpacking overhead)
# - INT8 MAC takes ~0.5 cycles (CMSIS-NN __SMLAD dual 16-bit / 8-bit MAC in 1 cycle)
CYCLES_PER_MAC_FP16 = 1.5
CYCLES_PER_MAC_INT4 = 0.75
CYCLES_PER_MAC_INT8 = 0.5

cycles_fp16_ensemble = macs_base * CYCLES_PER_MAC_FP16 + macs_adapters_ensemble * CYCLES_PER_MAC_FP16
cycles_pmq_4bit = macs_base * CYCLES_PER_MAC_INT4
# SA-QAB: base layers 1-14 are INT4, router is INT8, 1 active adapter is INT8
cycles_sa_qab = (macs_base * CYCLES_PER_MAC_INT4 + 
                  macs_router * CYCLES_PER_MAC_INT8 + 
                  macs_adapters_sa_qab * CYCLES_PER_MAC_INT8)

# Latency in ms: (Cycles * Cycle Time in ns) / 1,000,000
def latency_ms(cycles):
    return (cycles * CYCLE_TIME_NS) / 1e6

latency_fp16_ensemble = latency_ms(cycles_fp16_ensemble)
latency_pmq_4bit = latency_ms(cycles_pmq_4bit)
latency_sa_qab = latency_ms(cycles_sa_qab)

# 4. Energy Consumption (mJ)
# STM32H7 average power at 480 MHz is ~363 mW (110 mA @ 3.3V)
POWER_MW = 363.0
energy_fp16_ensemble = latency_fp16_ensemble * (POWER_MW / 1000.0)
energy_pmq_4bit = latency_pmq_4bit * (POWER_MW / 1000.0)
energy_sa_qab = latency_sa_qab * (POWER_MW / 1000.0)

# Print results
print("="*60)
print("STM32H7 MICROCONTROLLER PROFILING EMULATION RESULTS")
print("="*60)
print(f"PMQ FP16 Ensembling:")
print(f"  Flash Footprint:       {flash_fp16_ensemble:.2f} KB")
print(f"  Active SRAM Footprint: {sram_fp16_ensemble:.2f} KB")
print(f"  MAC Operations:        {macs_fp16_ensemble:,}")
print(f"  Estimated Latency:     {latency_fp16_ensemble:.3f} ms")
print(f"  Estimated Energy:      {energy_fp16_ensemble:.4f} mJ")
print("-"*60)
print(f"PMQ Static 4-bit (Collapsed):")
print(f"  Flash Footprint:       {flash_pmq_4bit:.2f} KB")
print(f"  Active SRAM Footprint: {sram_pmq_4bit:.2f} KB")
print(f"  MAC Operations:        {macs_pmq_4bit:,}")
print(f"  Estimated Latency:     {latency_pmq_4bit:.3f} ms")
print(f"  Estimated Energy:      {energy_pmq_4bit:.4f} mJ")
print("-"*60)
print(f"SA-QAB (Ours, INT4/INT8):")
print(f"  Flash Footprint:       {flash_sa_qab:.2f} KB")
print(f"  Active SRAM Footprint: {sram_sa_qab:.2f} KB")
print(f"  MAC Operations:        {macs_sa_qab:,}")
print(f"  Estimated Latency:     {latency_sa_qab:.3f} ms")
print(f"  Estimated Energy:      {energy_sa_qab:.4f} mJ")
print("="*60)

# Save to plot
methods = ['PMQ FP16\nEnsembling', 'PMQ Static\n4-bit (Collapsed)', 'SA-QAB\n(Ours)']
latencies = [latency_fp16_ensemble, latency_pmq_4bit, latency_sa_qab]
srams = [sram_fp16_ensemble, sram_pmq_4bit, sram_sa_qab]
energies = [energy_fp16_ensemble, energy_pmq_4bit, energy_sa_qab]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5))

colors = ['#C00000', '#7F7F7F', '#1F497D']

# Latency Plot
bars1 = ax1.bar(methods, latencies, color=colors, width=0.5, edgecolor='black', alpha=0.85)
ax1.set_ylabel('Inference Latency (ms)', fontsize=11, fontweight='bold')
ax1.set_title('Inference Latency', fontsize=12, fontweight='bold')
ax1.grid(True, linestyle='--', alpha=0.5)
for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05, f"{yval:.2f}ms", ha='center', va='bottom', fontweight='bold')

# SRAM Plot
bars2 = ax2.bar(methods, srams, color=colors, width=0.5, edgecolor='black', alpha=0.85)
ax2.set_ylabel('Active SRAM Footprint (KB)', fontsize=11, fontweight='bold')
ax2.set_title('SRAM Memory Footprint', fontsize=12, fontweight='bold')
ax2.grid(True, linestyle='--', alpha=0.5)
for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 10, f"{int(yval)}KB", ha='center', va='bottom', fontweight='bold')

# Energy Plot
bars3 = ax3.bar(methods, energies, color=colors, width=0.5, edgecolor='black', alpha=0.85)
ax3.set_ylabel('Inference Energy (mJ)', fontsize=11, fontweight='bold')
ax3.set_title('Energy Consumption', fontsize=12, fontweight='bold')
ax3.grid(True, linestyle='--', alpha=0.5)
for bar in bars3:
    yval = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f"{yval:.3f}mJ", ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig("results/hardware_profiling.png", dpi=150)
plt.close()
print("Saved profiling plot to results/hardware_profiling.png")
