import torch
from test_high_accuracy import evaluate_model_high_accuracy

seeds = [10, 11, 12, 13, 14]
for s in seeds:
    res = evaluate_model_high_accuracy(s)
    print(f"Seed {s}:")
    for m in res:
        print(f"  {m:<15}: {['%.2f' % a for a in res[m]]}")
