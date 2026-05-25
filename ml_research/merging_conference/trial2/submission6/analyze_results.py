import os

methods = ["standard", "sam", "sosr", "sata"]
merge_methods = ["dpm", "svdm", "palm", "lrom"]

print("| Fine-tuning Method | Merging Method | Best Lambda | CIFAR-10 Acc (%) | SVHN Acc (%) | Multi-Task Avg (%) |")
print("|--------------------|----------------|-------------|------------------|--------------|--------------------|")

for method in methods:
    for merge_method in merge_methods:
        filepath = f"./results/results_{method}_{merge_method}.txt"
        if not os.path.exists(filepath):
            continue
            
        with open(filepath, "r") as f:
            lines = f.readlines()[1:] # skip header
            
        best_avg = -1.0
        best_lambda = 0.0
        best_cifar = 0.0
        best_svhn = 0.0
        
        for line in lines:
            parts = line.strip().split(",")
            l_val = float(parts[0])
            c_acc = float(parts[1])
            s_acc = float(parts[2])
            avg = float(parts[3])
            
            # Find best multi-task average (excluding endpoints lambda=0.0 and 1.0)
            # as intermediate lambdas measure joint performance on both tasks
            if l_val > 0.0 and l_val < 1.0:
                if avg > best_avg:
                    best_avg = avg
                    best_lambda = l_val
                    best_cifar = c_acc
                    best_svhn = s_acc
                    
        print(f"| {method.upper()} | {merge_method.upper()} | {best_lambda:.1f} | {best_cifar:.2f}% | {best_svhn:.2f}% | {best_avg:.2f}% |")
