# 1. Summary of the Paper

## Main Topic and Real-World Context
The paper addresses a critical challenge in Edge AI and IoT deployment: the computational and storage bottleneck of deploying specialized, multi-task expert models on decentralized resource-constrained devices. While weight-space model merging via Task Arithmetic enables multi-task capabilities by storing task-specific delta vectors relative to a shared base model, storing several dense experts (e.g., visual encoders or large language models) remains prohibitively expensive (e.g., over 115 MB per CLIP ViT-B/32 expert). 

To solve this practical deployment bottleneck, the authors propose a post-hoc, training-free weight sparsification and merging framework called **Norm-Preserved Budgeted Task-Vector Pruning (NP-BTVP)**.

## Core Methodology
The framework compresses dense task vectors post-hoc down to a fraction of their original size (e.g., 90% to 95% sparsity), allowing them to be stored in highly efficient sparse formats (such as CSR or COO) on edge hardware. 

The paper introduces two deterministic, magnitude-based pruning strategies:
1. **Uniform Pruning (NP-BTVP-U):** Sparsifies each task vector globally to retain exactly the top $p\%$ absolute updates, then scales the surviving elements by a factor of $1/p$.
2. **Adaptive Saliency-Based Budget Allocation (NP-BTVP-S):** Measures average update intensity per parameter layer-wise (using an $L_1$-norm heuristic normalized by layer size) and solves for a layer-specific budget $p_l$ via binary search to strictly enforce the global budget $p$.

### Norm-Preserving Rescaling
Crucially, the paper introduces **norm-preserving rescaling** (scaling active updates by $1/p$ globally or $1/p_l$ layer-wise) as a signal-strength preservation heuristic. While magnitude-based pruning typically leads to update norm shrinkage, the authors show that scaling the largest (pruned) updates by $1/p$ mathematically boosts the expected $L_1$ norm of the update vectors (deriving a $3.30\times$ boost under Laplace and $2.58\times$ boost under Gaussian distributions for $p=0.10$). This "signal-strength boost" prevents specialized task vectors from being drowned out by the base pre-trained weights during multi-task fusion.

## Key Findings and Empirical Evidence
The authors evaluate the framework using a pre-trained CLIP ViT-B/32 backbone on 4 distinct classification datasets (MNIST, FashionMNIST, CIFAR-10, SVHN) across 3 independent random seeds, leading to several key findings:

1. **Pruning Resilience and Optimizer Independence:** Surprisingly, training-stage flatness (using Sharpness-Aware Minimization, or SAM) does not inherently provide an additional coordinate-aligned pruning buffer under well-converged regimes compared to standard AdamW. However, when paired with the norm-preserving rescaling, both AdamW and SAM experts exhibit extraordinary and nearly identical resilience to heavy sparsification.
2. **High Practical Performance:** At 90% sparsity ($p=0.10$), NP-BTVP-U achieves **90.34%** (AdamW) and **90.32%** (SAM) average accuracy, performing extremely close to the fully dense unpruned baselines (**90.94%** and **91.00%**).
3. **Competitive Edge over Baselines:** At 90% sparsity, NP-BTVP-U is competitive with the advanced stochastic DARE-Merging baseline (operating at 80% sparsity) and completely crushes TIES-Merging by **3.81%** average accuracy under SAM while using only half of TIES's parameter budget (10% vs 20%).
4. **The Saliency Double-Bind:** Saliency-Based Pruning (NP-BTVP-S) is slightly outperformed by global Uniform Pruning (NP-BTVP-U) because layer-specific budgets $p_l$ introduce scale instability under rescaling. Saliency-Global suffers from scale mismatch across layers, while Saliency-Layer suffers from extreme local variance and noise blowup.
5. **Ablation of Rescaling:** Without rescaling, Uniform Pruning at $p=0.10$ collapses to **80.94%** (AdamW) and **80.45%** (SAM). Rescaling restores performance completely, proving that update norm shrinkage is the primary failure mode in post-hoc pruning.
6. **INT8 Quantization Synergy:** Combining 90% sparsification with post-hoc 8-bit integer (INT8) quantization reduces on-disk storage per expert from 114.8 MB to **5.74 MB (a 40$\times$ compression)** with a negligible accuracy drop of only **0.12%** under SAM (90.20% average accuracy).

## Claimed Contributions and Verification
The explicitly claimed contributions are:
* **The NP-BTVP framework:** A training-free, post-hoc weight sparsification and merging method with norm-preserving rescaling. (Verified empirically with substantial performance gains).
* **Rigorous empirical evaluation:** Tested over 3 independent seeds on 4 datasets, establishing solid statistical significance (mean and standard deviation reported).
* **Geometric insight:** Demonstrating that SAM loss-landscape flatness does not provide coordinate-aligned pruning buffers under well-converged regimes, revealing a clear separation between dense weight-merging geometry and coordinate sparsification.
* **Analysis of the Saliency Double-Bind:** Identifying a fundamental trade-off in layer-wise adaptive budget allocation under rescaling, demonstrating the pragmatic superiority of simple uniform scaling.
* **Quantization Integration:** Demonstrating real-world deployment viability by achieving 40$\times$ storage reduction with negligible degradation.
