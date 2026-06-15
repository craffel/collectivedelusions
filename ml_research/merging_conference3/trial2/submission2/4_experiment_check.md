# Experimental Evaluation Check

## Evaluation of the Experimental Setup
The experimental setup is **highly standard and rigorously executed**:
- **Architecture:** The authors use `openai/clip-vit-base-patch32` (86M parameters), a standard and well-understood vision-language foundation model.
- **Protocol:** They perform full-backbone merging (all 86M parameters of the visual encoder), rather than simpler single-layer or subset-layer merging protocols. This is a strong, realistic test environment.
- **Datasets:** The four datasets used (MNIST, FashionMNIST, CIFAR-10, SVHN) are standard, diverse image classification tasks.
- **Evaluation Subset:** The authors evaluate on subsets of 1,000 test samples per dataset (total 4,000 samples). While using the full test set is the gold standard, a 1,000-sample subset per task is statistically significant and widely accepted in literature to accelerate extensive sweeps.

## Selection of Baselines
The selection of baselines is **excellent**:
- It includes **Zero-Shot CLIP** (lower bound), **Individual Experts** (empirical upper bound), **Task Arithmetic** (the baseline weight interpolation method), and the two most popular training-free coordinate-wise pruning methods, **TIES-Merging** and **DARE**.
- The exclusion of test-time optimization methods (like AdaMerging or FoldMerge) is explicitly and logically justified in Section 2 (Related Work) under data and compute constraints. Since offline model merging operates under strict zero-data and zero-training assumptions, comparing them against methods that require labeled validation sets or GPU-minutes of test-time backpropagation would be scientifically unfair.

## Critical Analysis of the Main Results (Table 1)
- **SVS vs. Task Arithmetic:** SVS with rank $k=128$ matches or slightly outperforms standard full-rank Task Arithmetic ($74.83\%$ vs $74.78\%$). This supports the core claim that downstream specialized updates reside on a low-rank manifold.
- **SVS vs. TIES & DARE:** SVS ($74.83\%$) is outperformed by DARE ($75.18\%$) and TIES-Merging ($77.98\%$).
- **Outstanding Honesty & Intellectual Integrity:** The authors do not try to hide the fact that TIES-Merging and DARE outperform SVS. Instead, they provide a **brilliant and highly insightful mathematical explanation** in Section 4.2 under the "Representation Gap" heading:
  - TIES and DARE operate in the spatial coordinate-basis of parameters, completely zeroing out overlapping coordinates to eliminate localized collisions.
  - SVS operates in the spectral domain, producing *dense* low-rank update matrices. While these filter out high-frequency noise, they still overlap entirely in the spatial coordinate-basis, creating cascading localized interference across multiple sequential Transformer layers.
  - This discussion represents an exceptionally honest, high-signal, and mature scientific contribution.

## Validation of BWN in Non-Normalized MLP Environments
To validate BWN's utility, the authors design a dedicated 3-layer MLP on MNIST and FashionMNIST *completely devoid of normalization layers*.
- The results (Figure 4) empirically confirm that:
  - Without BWN, low-scaling regimes ($\lambda = 0.1$) lead to activation and weight shrinkage (activation norm drops to $1.37$).
  - BWN analytically restores the scale to $1.62$ ($+17.8\%$) and recovers accuracy ($29.50\% \rightarrow 30.25\%$).
  - For SVS sweeps, BWN consistently stabilizes the activation scale.
- This is a well-designed, controlled experiment that successfully verifies the theoretical claims.

## Evaluation of Entropy-SVS
- **Pareto Efficiency:** Entropy-SVS traces a highly robust Pareto frontier. Under an entropy-scaling multiplier $m_{\text{entropy}}=1.0$, it achieves a $15.05\%$ compression of average rank (average rank 108.74) with virtually identical accuracy ($74.80\%$ vs. $74.83\%$).
- **Massive Compression Robustness:** At $m_{\text{entropy}}=0.4$, it prunes $65.70\%$ of the spectral directions (average rank of 43.90), while maintaining $74.55\%$ accuracy (only a $0.28\%$ drop). This is a remarkable empirical validation of the low-rank nature of downstream fine-tuning.

## Limitations of the Empirical Section
1. **Routing and Task Identity:** The multi-task evaluation relies on a multi-head setup where task identity is known at test-time to route activations to the correct task-specific linear head. While this is standard for model merging evaluations, it is a limitation compared to a fully unified single-head zero-shot classifier. The authors honestly acknowledge this in the limitations section.
2. **Evaluation Scale:** The experiments are restricted to a vision transformer of size 86M (CLIP-ViT-B/32). Evaluating SVS on larger, multi-billion parameter Large Language Models (LLMs) is absent, which is a major limitation for a general model merging method, though the authors discuss the challenges of gated SwiGLU MLPs and causal attention in Section 4.7.
