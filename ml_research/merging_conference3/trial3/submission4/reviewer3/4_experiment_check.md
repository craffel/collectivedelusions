# 4_experiment_check.md: Experimental Evaluation

## Experimental Setup & Datasets
The experimental setup is exceptionally thorough and goes far beyond standard toy scenarios:
- **Disparate Visual Domains:** Testing on MNIST, FashionMNIST, CIFAR-10, and SVHN represents a severe, highly heterogeneous visual domain shift. This forms an excellent "extreme stress-test" of linear weight-space task arithmetic.
- **Backbone Diversity:** The authors evaluate on multiple model families:
  - **Vision Transformers:** A compact \texttt{vit\_tiny} (5.7M parameters) for tight edge constraints and a larger \texttt{vit\_base} (86M parameters) to analyze the effect of scaling capacity.
  - **Convolutional Neural Networks:** A standard ResNet-18 (11M parameters) to prove that the findings generalize across architectures.
- **Domain-Aligned Benchmark:** Evaluating on a heterogeneous subset of DomainNet (Clipart, Painting, Real, Infograph) provides a necessary contrast, isolating the catastrophic collapse specifically to high-conflict scenarios.

## Evaluation of Baselines
The paper includes an exhaustive and highly appropriate list of baselines:
1. **Post-Hoc Pruning Baselines:** Naive uniform merging followed by post-hoc magnitude pruning (M-then-P) and vice versa (P-then-M).
2. **AdaMerging Baseline:** Adaptive coefficient optimization at test-time followed by post-hoc pruning (Ada-then-P).
3. **Decoupled Prune-then-Merge Baseline (P-then-M):** This simple baseline is a key comparison point and is shown to consistently outperform complex joint optimization.
4. **Trained-from-Scratch Multi-Task Learning (MTL) Baseline:** Serves as a vital "upper-bound" reference (74.63% accuracy) to contextualize the performance gap of model merging.
5. **Alternative Test-Time Objectives:** The authors evaluate ZipMerge under MMI, soft pseudo-labeling, LRA, and CBC objectives, comparing them directly to standard Shannon entropy minimization.

## Alignment between Claims and Results
The empirical results perfectly support and validate the paper's core claims:
- **Claim 1: Catastrophic Representational Collapse occurs under extreme domain shift.**
  - *Evidence:* Table 1 shows all full-backbone merged models (Uniform, AdaMerging, and ZipMerge) scoring between 10% and 14% accuracy (random guessing). MNIST performance collapses from an unpruned specialized expert accuracy of 97.26% down to 10.28% in ZipMerge-STE.
- **Claim 2: Unoptimized decoupled baseline (P-then-M) consistently outperforms joint test-time optimization under high conflict.**
  - *Evidence:* P-then-M achieves 14.81% (at 50% sparsity) and 16.97% (at 80% sparsity) joint mean accuracy, substantially beating the joint optimized ZipMerge-STE (11.23% and 11.32%) and ZipMerge-ES (14.00% and 10.47%).
- **Claim 3: Unconstrained unsupervised TTA triggers the Overfitting-Optimizer Paradox.**
  - *Evidence:* The authors report that unsupervised entropy loss successfully minimizes from 2.17 to 1.79 during optimization, yet test-set performance remains at random-guessing levels. Adding structural regularization (Reg-ZipMerge) or functional constraints controls entropy at 2.02 and partially preserves test accuracies.
- **Claim 4: LoRA-Adapter merging prevents backbone representation collapse.**
  - *Evidence:* Restricting fine-tuning to LoRA adapters yields a massive, highly significant improvement of over **+29%** absolute (42.30% Joint Mean) compared to the full-backbone dense merge (13.17%).
- **Claim 5: Orthogonal Procrustes SVD Alignment resolves coordinate mismatch post-hoc with negligible overhead.**
  - *Evidence:* Applying SVD-based alignment on LoRA adapters before linear averaging dramatically boosts dense merge accuracy from 42.30% up to **58.75%** Joint Mean (a massive **+16.45%** absolute improvement). Under 50% sparsity with ZipMerge-ES, the co-optimized aligned model reaches **62.10%** accuracy, only 4.48% below the unpruned expert ceiling (66.58%).
