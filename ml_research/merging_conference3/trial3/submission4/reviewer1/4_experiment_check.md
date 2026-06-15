# 4. Experiment Check

## Critical Evaluation of the Experimental Setup
The experimental setup is highly challenging, realistic, and rigorous:
* **Backbones:** The compact Vision Transformer (`vit_tiny_patch16_224`) with 5.7M parameters is a highly appropriate choice for representing tight edge-deployment constraints. ResNet-18 is evaluated to show generalizability across CNN families.
* **Tasks and Datasets:** The combination of MNIST, FashionMNIST, CIFAR-10, and SVHN represents a highly disparate, heterogeneous, and orthogonal task suite. It serves as an excellent extreme stress-test for linear weight-space operations.
* **Baselines:** The baselines are highly comprehensive and cover all natural sequencing variations: Uniform, AdaMerging (Dense), M-then-P, Ada-then-P, and P-then-M.

## Support of Claims
The empirical results fully and convincingly support all the authors' claims:
* **Representational Collapse:** Table 1 clearly shows that all merged configurations collapse to 10% to 14% accuracy, validating the catastrophic collapse hypothesis under extreme task shift.
* **P-then-M Superiority:** The unoptimized Prune-then-Merge (P-then-M) baseline consistently achieves the highest sparse joint mean (14.81% at 50% sparsity, 16.97% at 80% sparsity), significantly outperforming the complex ZipMerge-STE (11.23% and 11.32%). This supports the claim that pre-merging pruning acts as a spatial regularizer that removes conflicting noise.
* **Overfitting-Optimizer Paradox:** The authors provide quantitative evidence of the entropy dropping from 2.17 to 1.79 while test accuracy collapses, directly validating the transductive overfitting paradox.
* **Structured Pruning Latency:** Latency profiling on an ARM Cortex-A76 mobile CPU demonstrates that 50% structured block-pruning delivers a 1.89x physical speedup, validating the hardware utility of structured block designs.
* **PEFT and Procrustes Alignment:** The empirical LoRA merge (+29% improvement) and the subsequent Orthogonal Procrustes SVD alignment (+16.45% absolute accuracy gain over unaligned LoRA) provide strong empirical evidence that restricting updates to low-rank manifolds and rotating them into shared coordinates is the most effective way to close the performance gap.
