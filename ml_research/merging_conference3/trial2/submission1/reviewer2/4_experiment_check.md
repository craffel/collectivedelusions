# Experimental Evaluation Check

## Evaluation of Experimental Setup
- **Model Scale:** The paper uses **CLIP ViT-B/32** (86M parameters). While this is a standard academic benchmark, it is a relatively small model compared to modern, industrial-scale architectures (e.g., larger ViTs, CLIP ViT-L/14, or Large Language Models). 
- **Calibration Stream:** The calibration stream consists of exactly 1 unlabeled batch of size 16 per dataset ($N=64$ samples total). This mimics an ultra-low data environment, which is highly practical, but there is no validation showing how the system behaves under larger or variable calibration stream lengths (e.g., $N=128$, $N=512$, etc.).
- **Evaluation Splits:** Generalization is evaluated on 256 test samples per domain. This is a very small evaluation set, which can introduce high variance or sample-split noise. To ensure robust findings, a practitioner would want to see validation across full test sets (thousands of samples) rather than a small subset.

## Choice of Datasets (The Practitioner's Primary Concern)
- **The Toy Dataset Limitation:** The framework is evaluated exclusively on **MNIST, FashionMNIST, CIFAR-10, and SVHN**. These are small-scale, low-resolution (28x28 or 32x32), academic "toy" classification datasets.
- **Real-World Mismatch:** These datasets are not representative of modern, real-world, large-scale industrial computer vision tasks. They do not feature complex backgrounds, high-resolution objects, fine-grained categories, or severe domain shifts. To prove true real-world utility, the method must be evaluated on challenging, diverse vision benchmarks like **ImageNet-1K, VTAB, Stanford Cars, or Flowers102**, or in natural domain adaptation settings where model merging is actually used to solve industrial multi-task problems.

## Completeness of Baselines
- **Strengths:** The baselines are highly comprehensive and rigorously chosen:
  - Naive uniform Task Arithmetic (Method 1).
  - Unconstrained AdaMerging under both Adam GD and 1+1 ES (Methods 2 & 3).
  - Spatially Averaged AdaMerging (Method 4).
  - Shuffled diagnostics for both optimization paradigms (Methods 5 & 6).
  - A newly introduced **Calibrated Spatial Mean (Cal-Mean)** baseline (Method 9).
- **Adequacy:** The baseline comparison is mathematically fair and dense. Specifically, Cal-Mean isolates the contribution of SNEW and CCN on a single-scalar task model, enabling a direct, rigorous ablation of spatial layer-wise degrees of freedom.

## Support for Claims
- **Claim 1 (Overfitting-Optimizer Paradox): Supported.** The spatial shuffling treatments (Methods 5 & 6) show that randomizing optimized layer-wise coefficients across layers preserves almost all the accuracy gains. This empirically confirms that layer-wise optimizations do not learn genuine localized representational interactions but are primarily parameter-drift mechanisms fitting calibration noise.
- **Claim 2 (Sacrificial Task Bias): Supported.** The unconstrained evolutionary optimizer degrades SVHN performance (from 29.69% down to 28.26%). Our proposed unregularized calibration engine **CalMerge** (Method 8) restores and elevates SVHN accuracy to **32.03%** and Joint Mean to **61.82%**, confirming that scale-normalization successfully resolves task dominance.
- **Claim 3 (Value of Layer-wise Flexibility): Supported.** CalMerge (Method 8) outperforms Cal-Mean (Method 9) by 0.69% overall (61.82% vs 61.13%), and by much larger margins on CIFAR-10 (85.16% vs 78.65%) and SVHN (32.03% vs 30.08%). This supports the claim that layer-wise flexibility is valuable when properly calibrated.
- **Claim 4 (Generalization-Regularization Trade-off): Supported.** The 2D grid sweep ($\beta \times \gamma$) shows a smooth, monotonic decay in local adaptation performance as regularization weights increase. This validates that ESR acts as a stable, predictable structural stabilizer to control parameter drift.
- **Claim 5 (Heterogeneous Label Spaces): Supported.** The additional empirical simulation (Section 4.3.3) mathematically and empirically demonstrates the necessity of Class-Capacity Normalization (CCN) and SNEW under unequal class counts, showing a 1.0% improvement in Joint Mean accuracy for Cal-Mean over Task Arithmetic.
