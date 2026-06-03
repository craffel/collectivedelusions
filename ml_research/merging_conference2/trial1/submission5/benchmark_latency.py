import time
import torch
import torch.nn as nn
from experiment import get_resnet18_backbone_and_head, MultiTaskModel, device
import numpy as np
import os
import json

def benchmark_inference_latency():
    print(f"Starting latency benchmarking on device: {device}")
    
    # Generate dummy input batch
    batch_size = 128
    dummy_input = torch.randn(batch_size, 3, 32, 32, device=device)
    
    # 1. Load an expert model to serve as the baseline architecture
    backbone_expert, head_expert = get_resnet18_backbone_and_head()
    backbone_expert = backbone_expert.to(device)
    head_expert = head_expert.to(device)
    model_expert = MultiTaskModel(backbone_expert, head_expert).to(device)
    model_expert.eval()
    
    # Warmup
    print("Warming up standard model...")
    for _ in range(20):
        with torch.no_grad():
            _ = model_expert(dummy_input)
    if device.type == "cuda":
        torch.cuda.synchronize()
        
    # Measure standard model latency
    print("Measuring standard model latency...")
    start_time = time.time()
    num_runs = 200
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model_expert(dummy_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
    standard_latency = (time.time() - start_time) / num_runs * 1000.0  # ms
    print(f"Standard Model Latency: {standard_latency:.3f} ms")
    
    # 2. Measure TCAC Task-Switching overhead
    # In TCAC, we keep the shared backbone and load the task-specific BN statistics and weights/biases
    # Let's measure how long it takes to swap the state dict of the BN layers for a new task.
    print("\nMeasuring TCAC task-switching (buffer swapping) overhead...")
    
    # Let's create dummy BN states for 3 tasks
    tasks = ["mnist", "fashion", "cifar10"]
    bn_states = {}
    for task in tasks:
        state = {}
        for name, module in backbone_expert.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                state[name + ".running_mean"] = torch.randn_like(module.running_mean)
                state[name + ".running_var"] = torch.rand_like(module.running_var) + 0.1
                state[name + ".weight"] = torch.randn_like(module.weight)
                state[name + ".bias"] = torch.randn_like(module.bias)
        bn_states[task] = state
        
    # Swapping function: loads the pre-stored BN tensors into the active backbone
    def swap_task_bn(model_backbone, task_bn_state):
        with torch.no_grad():
            model_state = model_backbone.state_dict()
            for key, val in task_bn_state.items():
                model_state[key].copy_(val)
                
    # Measure swapping latency
    start_time = time.time()
    num_swaps = 500
    for i in range(num_swaps):
        task = tasks[i % len(tasks)]
        swap_task_bn(backbone_expert, bn_states[task])
    if device.type == "cuda":
        torch.cuda.synchronize()
    swapping_latency = (time.time() - start_time) / num_swaps * 1000.0  # ms
    print(f"TCAC Task-Switching (BN Swapping) Latency: {swapping_latency:.4f} ms")
    
    # 3. Measure TCAC Model Latency (Inference with swapped statistics)
    # Since swapping happens once before a task evaluation session, let's measure forward latency of the model
    # when TCAC is active (which is identical to standard model since it's native BatchNorm)
    print("\nMeasuring TCAC forward pass latency...")
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model_expert(dummy_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
    tcac_forward_latency = (time.time() - start_time) / num_runs * 1000.0  # ms
    print(f"TCAC Forward Pass Latency: {tcac_forward_latency:.3f} ms")
    
    results = {
        "standard_latency_ms": standard_latency,
        "switching_latency_ms": swapping_latency,
        "tcac_forward_latency_ms": tcac_forward_latency,
        "device": str(device)
    }
    
    os.makedirs("results", exist_ok=True)
    with open("results/latency_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nLatency results saved to results/latency_results.json!")

if __name__ == "__main__":
    benchmark_inference_latency()
