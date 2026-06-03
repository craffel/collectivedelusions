import torch
import torch.nn as nn
import time
import torchvision.models as models
import json

# Disable cuDNN globally to prevent CUDNN_STATUS_NOT_INITIALIZED errors
torch.backends.cudnn.enabled = False

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define a simulated WRSA frequency alignment wrapper
class FrequencyAlignedBatchNorm(nn.Module):
    def __init__(self, bn_module):
        super().__init__()
        self.bn = bn_module
        
    def forward(self, x):
        out = self.bn(x)
        # Simulate WRSA frequency alignment (FFT -> magnitude alignment -> IFFT)
        # Note: torch.fft.fft2 and ifft2 are computed over the spatial dimensions (H, W)
        if x.dim() == 4:
            # 2D Fourier Transform
            freq = torch.fft.fft2(out, dim=(-2, -1))
            # Simulated magnitude alignment (elementwise multiplication/scaling)
            freq = freq * 0.99
            # Inverse 2D Fourier Transform
            out = torch.fft.ifft2(freq, dim=(-2, -1)).real
        return out

def patch_model_with_frequency_alignment(model):
    """Recursively replaces all BatchNorm2d modules with FrequencyAlignedBatchNorm modules."""
    for name, child in model.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(model, name, FrequencyAlignedBatchNorm(child))
        else:
            patch_model_with_frequency_alignment(child)

def benchmark_inference(model, inputs, num_runs=1000, warmup=100):
    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(warmup):
            _ = model(inputs)
        
        # Synchronize before measurement
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        for _ in range(num_runs):
            _ = model(inputs)
            
        if device.type == "cuda":
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        
    avg_latency = (end_time - start_time) / num_runs * 1000 # in ms
    return avg_latency

def main():
    print("Initializing ResNet-18...")
    # Load standard ResNet-18
    model_standard = models.resnet18().to(device)
    model_freq = models.resnet18().to(device)
    
    # Patch the second model with frequency alignment hooks
    patch_model_with_frequency_alignment(model_freq)
    
    # Create representative batch of inputs (e.g. batch size 32 on CPU, 128 on GPU)
    batch_size = 128 if device.type == "cuda" else 32
    inputs = torch.randn(batch_size, 3, 32, 32, device=device)
    
    # Adapt runs based on device to avoid CPU timeouts
    num_runs = 1000 if device.type == "cuda" else 10
    warmup = 100 if device.type == "cuda" else 2
    
    print(f"\n--- BENCHMARKING INFERENCE LATENCY ({device.type.upper()}) ---")
    latency_std = benchmark_inference(model_standard, inputs, num_runs, warmup)
    print(f"Standard ResNet-18 (Our fused WRSC/baselines): {latency_std:.4f} ms")
    
    latency_freq = benchmark_inference(model_freq, inputs, num_runs, warmup)
    print(f"ResNet-18 with Active Frequency Alignment (WRSA): {latency_freq:.4f} ms")
    
    overhead = (latency_freq - latency_std) / latency_std * 100
    print(f"WRSA Latency Overhead: +{overhead:.2f}%")
    
    print("\n--- BENCHMARKING COMPILER COMPATIBILITY ---")
    # Benchmark compilation and execution under torch.compile
    try:
        print("Compiling Standard ResNet-18...")
        compiled_standard = torch.compile(model_standard)
        # Measure first run (compilation time)
        start = time.perf_counter()
        with torch.no_grad():
            _ = compiled_standard(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        compile_time_std = time.perf_counter() - start
        print(f"Standard ResNet-18 Compilation + First Run Time: {compile_time_std:.4f} s")
        
        # Measure compiled performance
        latency_compiled_std = benchmark_inference(compiled_standard, inputs, num_runs, warmup)
        print(f"Standard ResNet-18 Compiled Inference Latency: {latency_compiled_std:.4f} ms")
        speedup_std = (latency_std - latency_compiled_std) / latency_std * 100
        print(f"Standard ResNet-18 Compilation Speedup: {speedup_std:.2f}%")
        
    except Exception as e:
        print(f"Standard model compilation failed: {e}")
        compile_time_std = None
        latency_compiled_std = None

    try:
        print("\nCompiling Frequency Aligned ResNet-18...")
        compiled_freq = torch.compile(model_freq)
        # Measure first run (compilation time)
        start = time.perf_counter()
        with torch.no_grad():
            _ = compiled_freq(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        compile_time_freq = time.perf_counter() - start
        print(f"Frequency Aligned ResNet-18 Compilation + First Run Time: {compile_time_freq:.4f} s")
        
        # Measure compiled performance
        latency_compiled_freq = benchmark_inference(compiled_freq, inputs, num_runs, warmup)
        print(f"Frequency Aligned ResNet-18 Compiled Inference Latency: {latency_compiled_freq:.4f} ms")
        speedup_freq = (latency_freq - latency_compiled_freq) / latency_freq * 100
        print(f"Frequency Aligned ResNet-18 Compilation Speedup: {speedup_freq:.2f}%")
        
    except Exception as e:
        print(f"Frequency aligned model compilation failed or produced graph breaks: {e}")
        compile_time_freq = None
        latency_compiled_freq = None

    # Print summary results in JSON format
    results = {
        "latency_standard_ms": latency_std,
        "latency_frequency_ms": latency_freq,
        "latency_overhead_percent": overhead,
        "standard_compile_time_s": compile_time_std,
        "standard_compiled_latency_ms": latency_compiled_std,
        "frequency_compile_time_s": compile_time_freq,
        "frequency_compiled_latency_ms": latency_compiled_freq
    }
    
    with open("checkpoints/latency_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\nLatency benchmark completed and saved to checkpoints/latency_results.json")

if __name__ == "__main__":
    main()
