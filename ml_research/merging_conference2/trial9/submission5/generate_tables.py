import json
import os

def format_acc_std(mean, std=None):
    if mean is None:
        return "-"
    if std is None or std == 0.0:
        return f"{mean*100:.2f}"
    return f"{mean*100:.2f} \\pm {std*100:.2f}"

def main():
    if not os.path.exists("results.json"):
        print("results.json not found! Please run merge_and_evaluate.py first.")
        return
        
    with open("results.json", "r") as f:
        results = json.load(f)
        
    methods = ["WA", "TA (l=0.3)", "TA (l=0.5)", "QR-IPR", "TIES (l=0.5)", "DARE (l=0.5)", "WCPR", "QR-WCPR"]
    
    # 1. FP32 Clean Performance
    print("\n" + "="*20 + " TABLE 1: FP32 Clean (LaTeX Rows) " + "="*20)
    
    print("\nResNet-18 Clean (No Calib):")
    r18_clean = results.get("resnet18", {}).get("FP32 Clean", {})
    for m in methods:
        stats = r18_clean.get(m, {})
        avg = stats.get("average_acc")
        avg_std = stats.get("average_acc_std")
        mnist = stats.get("mnist", {}).get("acc")
        mnist_std = stats.get("mnist", {}).get("acc_std")
        fmnist = stats.get("fmnist", {}).get("acc")
        fmnist_std = stats.get("fmnist", {}).get("acc_std")
        cifar = stats.get("cifar10", {}).get("acc")
        cifar_std = stats.get("cifar10", {}).get("acc_std")
        print(f"ResNet-18 {m:12s} & {format_acc_std(mnist, mnist_std)} & {format_acc_std(fmnist, fmnist_std)} & {format_acc_std(cifar, cifar_std)} & {format_acc_std(avg, avg_std)} \\\\")
        
    print("\nResNet-18 Clean + DE-BN (16):")
    r18_clean_debn = results.get("resnet18", {}).get("FP32 Clean + DE-BN (16)", {})
    for m in methods:
        stats = r18_clean_debn.get(m, {})
        avg = stats.get("average_acc")
        avg_std = stats.get("average_acc_std")
        mnist = stats.get("mnist", {}).get("acc")
        mnist_std = stats.get("mnist", {}).get("acc_std")
        fmnist = stats.get("fmnist", {}).get("acc")
        fmnist_std = stats.get("fmnist", {}).get("acc_std")
        cifar = stats.get("cifar10", {}).get("acc")
        cifar_std = stats.get("cifar10", {}).get("acc_std")
        print(f"ResNet-18 {m:12s} + DE-BN (16) & {format_acc_std(mnist, mnist_std)} & {format_acc_std(fmnist, fmnist_std)} & {format_acc_std(cifar, cifar_std)} & {format_acc_std(avg, avg_std)} \\\\")
        
    print("\nMLP Clean:")
    mlp_clean = results.get("mlp", {}).get("FP32 Clean", {})
    for m in methods:
        stats = mlp_clean.get(m, {})
        avg = stats.get("average_acc")
        avg_std = stats.get("average_acc_std")
        mnist = stats.get("mnist", {}).get("acc")
        mnist_std = stats.get("mnist", {}).get("acc_std")
        fmnist = stats.get("fmnist", {}).get("acc")
        fmnist_std = stats.get("fmnist", {}).get("acc_std")
        cifar = stats.get("cifar10", {}).get("acc")
        cifar_std = stats.get("cifar10", {}).get("acc_std")
        print(f"MLP {m:12s} & {format_acc_std(mnist, mnist_std)} & {format_acc_std(fmnist, fmnist_std)} & {format_acc_std(cifar, cifar_std)} & {format_acc_std(avg, avg_std)} \\\\")

    # 2. Quantization (INT8)
    print("\n" + "="*20 + " TABLE 2: INT8 Quantization (LaTeX Rows) " + "="*20)
    r18_int8 = results.get("resnet18", {}).get("INT8 Quantized", {})
    r18_int8_debn = results.get("resnet18", {}).get("INT8 + DE-BN (16)", {})
    mlp_int8 = results.get("mlp", {}).get("INT8 Quantized", {})
    
    print("\nResNet-18 INT8:")
    for m in methods:
        s_no = r18_int8.get(m, {})
        avg_no = s_no.get("average_acc")
        std_no = s_no.get("average_acc_std")
        
        s_de = r18_int8_debn.get(m, {})
        avg_de = s_de.get("average_acc")
        std_de = s_de.get("average_acc_std")
        print(f"{m:12s} & {format_acc_std(avg_no, std_no)} & {format_acc_std(avg_de, std_de)} \\\\")
        
    print("\nMLP INT8:")
    for m in methods:
        s_no = mlp_int8.get(m, {})
        avg_no = s_no.get("average_acc")
        std_no = s_no.get("average_acc_std")
        print(f"{m:12s} & {format_acc_std(avg_no, std_no)} & - \\\\")

    # 3. Quantization (INT4)
    print("\n" + "="*20 + " TABLE 3: INT4 Quantization (LaTeX Rows) " + "="*20)
    r18_int4 = results.get("resnet18", {}).get("INT4 Quantized", {})
    r18_int4_debn = results.get("resnet18", {}).get("INT4 + DE-BN (16)", {})
    mlp_int4 = results.get("mlp", {}).get("INT4 Quantized", {})
    
    print("\nResNet-18 INT4:")
    for m in methods:
        s_no = r18_int4.get(m, {})
        avg_no = s_no.get("average_acc")
        std_no = s_no.get("average_acc_std")
        
        s_de = r18_int4_debn.get(m, {})
        avg_de = s_de.get("average_acc")
        std_de = s_de.get("average_acc_std")
        print(f"{m:12s} & {format_acc_std(avg_no, std_no)} & {format_acc_std(avg_de, std_de)} \\\\")
        
    print("\nMLP INT4:")
    for m in methods:
        s_no = mlp_int4.get(m, {})
        avg_no = s_no.get("average_acc")
        std_no = s_no.get("average_acc_std")
        print(f"{m:12s} & {format_acc_std(avg_no, std_no)} & - \\\\")

    # 4. Environmental Corruptions
    print("\n" + "="*20 + " TABLE 4: Environmental Corruptions (LaTeX Rows) " + "="*20)
    r18_noise = results.get("resnet18", {}).get("Noise (sev 1.5)", {})
    r18_noise_debn = results.get("resnet18", {}).get("Noise + DE-BN (16)", {})
    r18_blur = results.get("resnet18", {}).get("Blur (sev 1.5)", {})
    r18_blur_debn = results.get("resnet18", {}).get("Blur + DE-BN (16)", {})
    
    mlp_noise = results.get("mlp", {}).get("Noise (sev 1.5)", {})
    mlp_blur = results.get("mlp", {}).get("Blur (sev 1.5)", {})
    
    print("\nResNet-18 Robustness:")
    for m in methods:
        avg_n = r18_noise.get(m, {}).get("average_acc")
        std_n = r18_noise.get(m, {}).get("average_acc_std")
        
        avg_n_de = r18_noise_debn.get(m, {}).get("average_acc")
        std_n_de = r18_noise_debn.get(m, {}).get("average_acc_std")
        
        avg_b = r18_blur.get(m, {}).get("average_acc")
        std_b = r18_blur.get(m, {}).get("average_acc_std")
        
        avg_b_de = r18_blur_debn.get(m, {}).get("average_acc")
        std_b_de = r18_blur_debn.get(m, {}).get("average_acc_std")
        
        print(f"{m:12s} & {format_acc_std(avg_n, std_n)} & {format_acc_std(avg_n_de, std_n_de)} & {format_acc_std(avg_b, std_b)} & {format_acc_std(avg_b_de, std_b_de)} \\\\")
        
    print("\nMLP Robustness:")
    for m in methods:
        avg_n = mlp_noise.get(m, {}).get("average_acc")
        std_n = mlp_noise.get(m, {}).get("average_acc_std")
        avg_b = mlp_blur.get(m, {}).get("average_acc")
        std_b = mlp_blur.get(m, {}).get("average_acc_std")
        print(f"{m:12s} & {format_acc_std(avg_n, std_n)} & {format_acc_std(avg_b, std_b)} \\\\")

if __name__ == "__main__":
    main()
