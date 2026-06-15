import torch
import torch.nn as nn
from run_experiments import RepresentationSandbox, evaluate_ceilings

seed = 42
sandbox = RepresentationSandbox(seed, rho=0.0)

experts = []
for k in range(sandbox.K):
    expert = nn.Linear(sandbox.D, sandbox.C, bias=True)
    with torch.no_grad():
        for c in range(sandbox.C):
            expert.weight[c] = sandbox.prototypes[k, c]
            expert.bias[c] = 0.0
    experts.append(expert)

X_test, Y_test, task_test = sandbox.generate_split(250)
ceilings = evaluate_ceilings(sandbox, experts, X_test, Y_test, task_test)
print("Ceiling accuracies without training:")
tasks = ['MNIST', 'FashionMNIST', 'CIFAR-10', 'SVHN']
for i, name in enumerate(tasks):
    print(f"  {name}: {ceilings[i]*100:.2f}%")
print(f"  Joint Mean: {sum(ceilings)/len(ceilings)*100:.2f}%")
