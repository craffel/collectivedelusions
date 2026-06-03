import json
import numpy as np

with open("results_head_hyperparams.json", "r") as f:
    data = json.load(f)

# Seeds are 42, 43, 44
seeds = ["42", "43", "44"]
configs = list(data["42"].keys())
methods = list(data["42"][configs[0]].keys())

print(f"{'Config (LR & Epochs)':<25} | {'Method':<40} | {'MNIST Mean (±Std)':<20} | {'F-MNIST Mean (±Std)':<20} | {'CIFAR10 Mean (±Std)':<20} | {'Average Mean (±Std)':<20}")
print("-" * 155)

for config in configs:
    for method in methods:
        mnist_vals = []
        fmnist_vals = []
        cifar_vals = []
        avg_vals = []
        
        for seed in seeds:
            res = data[seed][config][method]
            mnist_vals.append(res["mnist"])
            fmnist_vals.append(res["fmnist"])
            cifar_vals.append(res["cifar10"])
            avg_vals.append(res["average"])
            
        print(f"{config:<25} | {method:<40} | "
              f"{np.mean(mnist_vals):.2f}% (±{np.std(mnist_vals):.2f}) | "
              f"{np.mean(fmnist_vals):.2f}% (±{np.std(fmnist_vals):.2f}) | "
              f"{np.mean(cifar_vals):.2f}% (±{np.std(cifar_vals):.2f}) | "
              f"{np.mean(avg_vals):.2f}% (±{np.std(avg_vals):.2f})")
    print("-" * 155)
