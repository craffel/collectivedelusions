import gc
import tracemalloc
import torch
import torch.nn as nn
from torchvision.models import resnet18

# Define standard ResNet-18 model (which is the architecture of ECC-Merge / Uncalibrated)
def get_standard_model():
    model = resnet18()
    model.fc = nn.Linear(512, 10)
    model.eval()
    model.requires_grad_(False)
    return model

def apply_sp_ttbc(model, alpha=0.9):
    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            original_forward = m.forward
            def make_custom_forward(bn_module, orig_forward):
                def custom_forward(x):
                    if x.size(0) > 1 and not bn_module.training:
                        batch_mean = x.mean(dim=(0, 2, 3))
                        batch_var = x.var(dim=(0, 2, 3), unbiased=False)
                        
                        blended_mean = alpha * bn_module.running_mean + (1 - alpha) * batch_mean
                        blended_var = alpha * bn_module.running_var + (1 - alpha) * batch_var
                        
                        return torch.nn.functional.batch_norm(
                            x, blended_mean, blended_var, bn_module.weight, bn_module.bias,
                            training=False, momentum=0.0, eps=bn_module.eps
                        )
                    else:
                        return orig_forward(x)
                return custom_forward
            m.forward = make_custom_forward(m, original_forward)

def profile_memory_usage(batch_size=64, num_runs=50):
    # Prepare dummy inputs
    x = torch.randn(batch_size, 3, 32, 32)
    
    # 1. Profile Standard Model (ECC-Merge)
    model_std = get_standard_model()
    gc.collect()
    tracemalloc.start()
    
    # Run warmups
    with torch.no_grad():
        for _ in range(5):
            _ = model_std(x)
        
    tracemalloc.reset_peak()
    
    # Run profiling
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model_std(x)
        
    current_std, peak_std = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # 2. Profile SP-TTBC Model
    model_sp = get_standard_model()
    apply_sp_ttbc(model_sp)
    
    gc.collect()
    tracemalloc.start()
    
    # Run warmups
    with torch.no_grad():
        for _ in range(5):
            _ = model_sp(x)
        
    tracemalloc.reset_peak()
    
    # Run profiling
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model_sp(x)
        
    current_sp, peak_sp = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"=== Memory Profile Results (Batch Size {batch_size}, {num_runs} Runs) ===")
    print(f"Standard Model (ECC-Merge) Peak Traced Memory: {peak_std / 1024:.2f} KB")
    print(f"SP-TTBC (Online) Peak Traced Memory:           {peak_sp / 1024:.2f} KB")
    increase_kb = (peak_sp - peak_std) / 1024
    increase_pct = (peak_sp - peak_std) / peak_std * 100 if peak_std > 0 else 0
    print(f"Online Memory Overhead:                        +{increase_kb:.2f} KB ({increase_pct:.2f}%)")
    
    # Save results to json
    import json
    results = {
        "batch_size": batch_size,
        "std_peak_kb": peak_std / 1024,
        "sp_peak_kb": peak_sp / 1024,
        "overhead_kb": increase_kb,
        "overhead_pct": increase_pct
    }
    with open("memory_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Saved memory profile results to memory_benchmark_results.json")

if __name__ == "__main__":
    profile_memory_usage()
