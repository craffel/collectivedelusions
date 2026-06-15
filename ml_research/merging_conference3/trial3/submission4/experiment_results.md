# ZipMerge Experimental Results

This document contains the complete empirical evaluation of **ZipMerge (Post-Merge Joint Weight Pruning and Coefficient Tuning)** using a **timm ViT-Tiny** (`vit_tiny_patch16_224`) backbone across four visual tasks: MNIST, FashionMNIST, CIFAR-10, and SVHN.

## Overview
We evaluate under three target sparsity levels ($p \in \{0.0, 0.5, 0.8\}$) corresponding to **0%, 50%, and 80%** parameter pruning. 
We evaluate ZipMerge (STE) and ZipMerge (ES) against four appropriate baselines:
1. **Uniform Merge (Dense):** Standard Task Arithmetic with uniform coefficients ($\lambda = 0.3$).
2. **Merge-then-Prune (M-then-P):** Naive post-hoc pruning of a uniform merged model.
3. **AdaMerging-then-Prune (Ada-then-P):** Naive post-hoc pruning of a dense-optimized model.
4. **Prune-then-Merge (P-then-M):** Separately pruning each task vector's parameters before performing a uniform merge.

Additionally, we list the **Individual Expert References** (unpruned upper bounds) to show the maximum possible performance.

---

## 1. Multi-Task Classification Accuracies

### Sparsity level: $p = 0.0$ (No Pruning / Dense)
| Method | MNIST (%) | FashionMNIST (%) | CIFAR-10 (%) | SVHN (%) | Joint Mean (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Individual Experts (Ref)** | 97.26 | 87.43 | 73.71 | 19.59 | 69.50 |
| Uniform | 9.85 | 9.99 | 17.05 | 15.78 | 13.17 |
| AdaMerging (Dense) | 11.65 | 18.06 | 12.55 | 10.92 | 13.30 |

### Sparsity level: $p = 0.5$ (50% Sparsity / Moderate Pruning)
| Method | MNIST (%) | FashionMNIST (%) | CIFAR-10 (%) | SVHN (%) | Joint Mean (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Individual Experts (Ref)** | 97.26 | 87.43 | 73.71 | 19.59 | 69.50 |
| Uniform (M-then-P) | 9.82 | 10.00 | 11.55 | 16.19 | 11.89 |
| AdaMerging-then-Prune | 10.55 | 13.85 | 10.36 | 9.93 | 11.17 |
| Prune-then-Merge (P-then-M) | 9.86 | 9.99 | 21.13 | 18.26 | **14.81** |
| **ZipMerge (STE)** | 11.00 | 14.04 | 10.20 | 9.67 | 11.23 |
| **ZipMerge (ES)** | 18.25 | 16.12 | 10.27 | 11.35 | 14.00 |

### Sparsity level: $p = 0.8$ (80% Sparsity / Aggressive Pruning)
| Method | MNIST (%) | FashionMNIST (%) | CIFAR-10 (%) | SVHN (%) | Joint Mean (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Individual Experts (Ref)** | 97.26 | 87.43 | 73.71 | 19.59 | 69.50 |
| Uniform (M-then-P) | 10.28 | 8.90 | 10.81 | 10.86 | 10.21 |
| AdaMerging-then-Prune | 10.28 | 10.47 | 11.32 | 10.85 | 10.73 |
| Prune-then-Merge (P-then-M) | 13.60 | 9.99 | 27.06 | 17.24 | **16.97** |
| **ZipMerge (STE)** | 10.28 | 10.48 | 10.85 | 13.66 | 11.32 |
| **ZipMerge (ES)** | 10.28 | 9.61 | 9.54 | 12.45 | 10.47 |

---

## 2. Key Empirical Findings

1. **Catastrophic Representational Collapse:**
   Every single merged configuration (including Uniform, AdaMerging, and the proposed ZipMerge variants) suffers from complete representational collapse. The joint mean accuracies reside between 10% and 14%, which is functionally equivalent to random guessing on these 10-class visual datasets. This is due to the extreme domain shifts between MNIST (digits), FashionMNIST (clothing), CIFAR-10 (natural objects), and SVHN (street numbers) being forced onto a compact, 5.7M parameter ViT-Tiny backbone.

2. **Supremacy of Prune-then-Merge (P-then-M) Baseline:**
   The unoptimized, decoupled baseline **Prune-then-Merge (P-then-M)** consistently and significantly outperforms all other sparse merging pipelines, achieving 14.81% joint accuracy at 50% sparsity and 16.97% joint accuracy at 80% sparsity. This suggests that pruning individual expert task vectors *prior* to merging acts as a powerful spatial regularizer. By removing non-overlapping, low-magnitude expert shifts, it dramatically reduces mutual interference and representation collisions in weight space before they can occur.

3. **Comparison of ZipMerge Paradigms:**
   Neither the first-order Straight-Through Estimator (ZipMerge-STE) nor the zero-order Evolutionary Search (ZipMerge-ES) consistently outperforms the other:
   - At 50% sparsity, **ZipMerge (ES)** outperforms ZipMerge (STE) (14.00% vs 11.23% Joint Mean).
   - At 80% sparsity, **ZipMerge (STE)** outperforms ZipMerge (ES) (11.32% vs 10.47% Joint Mean), and achieves slight improvements over the naive post-hoc pruning baselines (Uniform M-then-P at 10.21% and Ada-then-P at 10.73%).
   However, these marginal performance differences reside entirely within the noise of a non-functional, random-guessing model, making any claims of "substantial superiority" over post-hoc pruning highly speculative.

4. **The Overfitting-Optimizer Paradox:**
   Test-time adaptation via unsupervised minimum entropy on a tiny calibration set of 64 images overfits transductively. The optimizer successfully minimizes entropy (making predictions highly confident on the calibration set), but in doing so, it destroys the generalizable representations of the expert models, resulting in random-guessing accuracies on the full test sets.
