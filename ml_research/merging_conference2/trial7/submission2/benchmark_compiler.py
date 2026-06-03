import os
import time
import json
import torch
import torch.nn as nn
from torchvision.models import resnet18

def apply_sp_ttbc(model, alpha=0.9):
    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            original_forward = m.forward
            def make_custom_forward(bn_module, orig_forward):
                def custom_forward(x):
                    if x.size(0) > 1 and not bn_module.training:
                        # Compute batch statistics
                        batch_mean = x.mean(dim=(0, 2, 3))
                        batch_var = x.var(dim=(0, 2, 3), unbiased=False)
                        
                        # Blend statistics
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

def benchmark(model, x, num_iters=100):
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            _ = model(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        for _ in range(num_iters):
            _ = model(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
    return (end - start) / num_iters

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Standard Model (representing ECC-Merge / Uncalibrated)
    model_std = resnet18()
    model_std.eval()
    for p in model_std.parameters():
        p.requires_grad = False
    model_std = model_std.to(device)
    
    # SP-TTBC Model
    model_spttbc = resnet18()
    model_spttbc.eval()
    for p in model_spttbc.parameters():
        p.requires_grad = False
    model_spttbc = model_spttbc.to(device)
    apply_sp_ttbc(model_spttbc)
    
    # Test batch sizes
    batch_sizes = [1, 64]
    
    results = {}
    for bs in batch_sizes:
        x = torch.randn(bs, 3, 32, 32).to(device)
        print(f"\n--- Batch Size {bs} (Eager Mode) ---")
        
        t_std = benchmark(model_std, x)
        print(f"Standard Model (ECC-Merge): {t_std*1000:.4f} ms per pass")
        
        t_spttbc = benchmark(model_spttbc, x)
        print(f"SP-TTBC (Online): {t_spttbc*1000:.4f} ms per pass")
        
        results[f"eager_std_{bs}"] = t_std
        results[f"eager_spttbc_{bs}"] = t_spttbc
        
    print("\n--- Testing torch.compile Compatibility ---")
    
    # 1. Standard model with torch.compile
    print("Compiling Standard Model (ECC-Merge) with fullgraph=True...")
    try:
        compiled_std = torch.compile(model_std, fullgraph=True)
        # Warmup and compile trigger
        x_64 = torch.randn(64, 3, 32, 32).to(device)
        start_c = time.perf_counter()
        _ = compiled_std(x_64)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_c = time.perf_counter()
        print(f"Standard Model compiled successfully in {end_c - start_c:.2f} seconds.")
        
        t_compiled_std = benchmark(compiled_std, x_64)
        print(f"Compiled Standard Model (BS 64): {t_compiled_std*1000:.4f} ms")
        results["compile_std"] = True
        results["compile_std_time"] = end_c - start_c
        results["compile_std_latency"] = t_compiled_std
    except Exception as e:
        print(f"Standard Model compilation failed with fullgraph=True: {e}")
        results["compile_std"] = False
        
    # 2. SP-TTBC model with torch.compile
    print("\nCompiling SP-TTBC Model with fullgraph=True...")
    try:
        compiled_spttbc = torch.compile(model_spttbc, fullgraph=True)
        x_64 = torch.randn(64, 3, 32, 32).to(device)
        start_c = time.perf_counter()
        _ = compiled_spttbc(x_64)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_c = time.perf_counter()
        print(f"SP-TTBC Model compiled successfully with fullgraph=True in {end_c - start_c:.2f} seconds.")
        
        t_compiled_spttbc = benchmark(compiled_spttbc, x_64)
        print(f"Compiled SP-TTBC Model (BS 64): {t_compiled_spttbc*1000:.4f} ms")
        results["compile_spttbc"] = True
        results["compile_spttbc_time"] = end_c - start_c
        results["compile_spttbc_latency"] = t_compiled_spttbc
    except Exception as e:
        print(f"SP-TTBC Model compilation failed with fullgraph=True!")
        print(f"Error detail: {e}")
        results["compile_spttbc"] = False
        
        # Try compiling with fullgraph=False to see if graph breaks occur
        print("\nAttempting SP-TTBC compilation with fullgraph=False...")
        try:
            compiled_spttbc_fg_false = torch.compile(model_spttbc, fullgraph=False)
            x_64 = torch.randn(64, 3, 32, 32).to(device)
            start_c = time.perf_counter()
            _ = compiled_spttbc_fg_false(x_64)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_c = time.perf_counter()
            print(f"Compiled with fullgraph=False (fallback/graph breaks allowed) in {end_c - start_c:.2f} seconds.")
            t_compiled_spttbc_fg_false = benchmark(compiled_spttbc_fg_false, x_64)
            print(f"Compiled SP-TTBC (fullgraph=False, BS 64): {t_compiled_spttbc_fg_false*1000:.4f} ms")
            results["compile_spttbc_fg_false"] = True
            results["compile_spttbc_fg_false_time"] = end_c - start_c
            results["compile_spttbc_fg_false_latency"] = t_compiled_spttbc_fg_false
        except Exception as ex:
            print(f"SP-TTBC compilation failed completely: {ex}")
            results["compile_spttbc_fg_false"] = False
            
    # Save results to a json
    with open("compiler_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
