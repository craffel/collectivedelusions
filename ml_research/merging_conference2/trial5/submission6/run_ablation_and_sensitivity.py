import torch
import torch.nn as nn
import torchvision.models as models
import time
import os
import matplotlib.pyplot as plt
from datasets_utils import get_dataloaders
from calibration_methods import merge_models, calibrate_sequential
from merge_and_evaluate import evaluate, run_evaluation_for_all

def run_latency_benchmark(model, device, num_runs=100, batch_size=128):
    print("Running latency benchmark...")
    # Generate dummy input
    dummy_input = torch.randn(batch_size, 3, 32, 32).to(device)
    model = model.to(device)
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
            
    # Measure time
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
    end_time = time.time()
    
    avg_latency = (end_time - start_time) / num_runs * 1000.0 # in ms
    throughput = (batch_size * num_runs) / (end_time - start_time) # images / sec
    return avg_latency, throughput

def run_compile_benchmark(model, device):
    print("Testing torch.compile compatibility...")
    model = model.to(device)
    model.eval()
    
    try:
        compiled_model = torch.compile(model)
        dummy_input = torch.randn(2, 3, 32, 32).to(device)
        with torch.no_grad():
            _ = compiled_model(dummy_input)
        print("Compilation SUCCESSFUL!")
        return "PASS"
    except Exception as e:
        print(f"Compilation FAILED: {str(e)}")
        return f"FAIL: {str(e)[:50]}"

def main():
    torch.backends.cudnn.enabled = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    loaders = get_dataloaders(batch_size=128)
    
    # Load expert heads for evaluation
    mnist_expert = models.resnet18(weights=None)
    mnist_expert.fc = nn.Linear(512, 10)
    mnist_expert.load_state_dict(torch.load('mnist_expert.pt', map_location='cpu'))
    mnist_expert = mnist_expert.to(device)
    
    fashion_expert = models.resnet18(weights=None)
    fashion_expert.fc = nn.Linear(512, 10)
    fashion_expert.load_state_dict(torch.load('fashion_expert.pt', map_location='cpu'))
    fashion_expert = fashion_expert.to(device)
    
    cifar_expert = models.resnet18(weights=None)
    cifar_expert.fc = nn.Linear(512, 10)
    cifar_expert.load_state_dict(torch.load('cifar_expert.pt', map_location='cpu'))
    cifar_expert = cifar_expert.to(device)
    
    experts_list = [mnist_expert, fashion_expert, cifar_expert]
    experts_dict = {
        'mnist': mnist_expert,
        'fashion': fashion_expert,
        'cifar': cifar_expert
    }
    
    print("\n" + "="*50)
    print("EXPERIMENT 1: Ablation Study (SBR vs PBR)")
    print("="*50)
    
    ablation_results = {}
    for mode in ['wa', 'ta']:
        ablation_results[mode] = {}
        for method in ['sbr', 'pbr']:
            print(f"Running: Mode={mode.upper()}, Calibration={method.upper()}...")
            model = merge_models(merge_mode=mode, lambda_val=0.3)
            model = model.to(device)
            model, hooks = calibrate_sequential(model, experts_list, loaders, method, device, cal_size=128)
            res = run_evaluation_for_all(model, experts_dict, loaders, device)
            for h in hooks:
                h.remove()
            ablation_results[mode][method] = res
            print(f"  {method.upper()} -> Avg Acc: {res['avg']:.2f}% (MNIST: {res['mnist']:.2f}%, F-MNIST: {res['fashion']:.2f}%, CIFAR: {res['cifar']:.2f}%)")
            
    print("\n" + "="*50)
    print("EXPERIMENT 2: Data-Efficiency Sensitivity Study (N vs SBR Accuracy)")
    print("="*50)
    
    sensitivity_results = {}
    cal_sizes = [1, 4, 16, 64, 128]
    for n in cal_sizes:
        print(f"Running: SBR (WA) with cal_size N={n}...")
        model = merge_models(merge_mode='wa', lambda_val=0.3)
        model = model.to(device)
        model, hooks = calibrate_sequential(model, experts_list, loaders, 'sbr', device, cal_size=n)
        res = run_evaluation_for_all(model, experts_dict, loaders, device)
        for h in hooks:
            h.remove()
        sensitivity_results[n] = res
        print(f"  N={n} -> Avg Acc: {res['avg']:.2f}% (MNIST: {res['mnist']:.2f}%, F-MNIST: {res['fashion']:.2f}%, CIFAR: {res['cifar']:.2f}%)")
        
    print("\n" + "="*50)
    print("EXPERIMENT 3: Runtime Latency & Compile Benchmark")
    print("="*50)
    
    latency_results = {}
    
    # 1. Uncalibrated / SBR (Zero Overhead)
    model_sbr = merge_models(merge_mode='wa', lambda_val=0.3)
    model_sbr = model_sbr.to(device)
    model_sbr, hooks = calibrate_sequential(model_sbr, experts_list, loaders, 'sbr', device, cal_size=128)
    for h in hooks:
        h.remove()
    lat_sbr, tp_sbr = run_latency_benchmark(model_sbr, device)
    comp_sbr = run_compile_benchmark(model_sbr, device)
    latency_results['SBR (Ours)'] = {'latency': lat_sbr, 'throughput': tp_sbr, 'compile': comp_sbr}
    print(f"  SBR (Ours) -> Latency: {lat_sbr:.2f} ms | Throughput: {tp_sbr:.1f} imgs/sec | Compile: {comp_sbr}")
    
    # 2. C-FDSA (Fourier Hooks)
    model_fdsa = merge_models(merge_mode='wa', lambda_val=0.3)
    model_fdsa = model_fdsa.to(device)
    model_fdsa, hooks_fdsa = calibrate_sequential(model_fdsa, experts_list, loaders, 'c_fdsa', device, cal_size=128)
    lat_fdsa, tp_fdsa = run_latency_benchmark(model_fdsa, device)
    comp_fdsa = run_compile_benchmark(model_fdsa, device)
    for h in hooks_fdsa:
        h.remove()
    latency_results['C-FDSA'] = {'latency': lat_fdsa, 'throughput': tp_fdsa, 'compile': comp_fdsa}
    print(f"  C-FDSA -> Latency: {lat_fdsa:.2f} ms | Throughput: {tp_fdsa:.1f} imgs/sec | Compile: {comp_fdsa}")
    
    # Report summary
    print("\n" + "="*50)
    print("SUMMARY OF ALL NEW FINDINGS")
    print("="*50)
    print("\n1. SBR vs PBR Ablation:")
    for mode in ['wa', 'ta']:
        s_acc = ablation_results[mode]['sbr']['avg']
        p_acc = ablation_results[mode]['pbr']['avg']
        diff = s_acc - p_acc
        print(f"  {mode.upper()}: SBR = {s_acc:.2f}% | PBR = {p_acc:.2f}% | Delta = +{diff:.2f}%")
        
    print("\n2. Sensitivity to Calibration Size N:")
    for n in cal_sizes:
        print(f"  N = {n:<3} | Avg Accuracy = {sensitivity_results[n]['avg']:.2f}%")
        
    print("\n3. Latency & Compilability:")
    for method, info in latency_results.items():
        print(f"  {method:<12} | Latency: {info['latency']:.2f} ms | Throughput: {info['throughput']:.1f} imgs/sec | torch.compile: {info['compile']}")
        
    # Write summary to file for logging
    with open('ablation_sensitivity_summary.txt', 'w') as f:
        f.write("Ablation & Sensitivity Findings\n")
        f.write("="*30 + "\n\n")
        f.write("1. SBR vs PBR Ablation:\n")
        for mode in ['wa', 'ta']:
            s_acc = ablation_results[mode]['sbr']['avg']
            p_acc = ablation_results[mode]['pbr']['avg']
            f.write(f"  {mode.upper()}: SBR = {s_acc:.2f}% | PBR = {p_acc:.2f}% | Delta = +{s_acc - p_acc:.2f}%\n")
        f.write("\n2. Sensitivity to Calibration Size N:\n")
        for n in cal_sizes:
            f.write(f"  N = {n:<3} | Avg Accuracy = {sensitivity_results[n]['avg']:.2f}%\n")
        f.write("\n3. Latency & Compilability:\n")
        for method, info in latency_results.items():
            f.write(f"  {method:<12} | Latency: {info['latency']:.2f} ms | Throughput: {info['throughput']:.1f} imgs/sec | torch.compile: {info['compile']}\n")

if __name__ == '__main__':
    main()
