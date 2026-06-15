# Evaluation Phase 4: Experimental Evaluation and Claims Check

## 1. Experimental Setup and Baselines
The paper conducts a dual-evaluation using:
1. **A Synthetic 12-layer Sequential PyTorch Sandbox (192D):** Models MNIST, FashionMNIST, CIFAR-10, and SVHN using partitioned orthogonal subspaces (48D each) and 10 orthogonal class prototypes.
2. **A Real-World Feature Testbed (512D):** Extracts representations from an ImageNet pre-trained ResNet-18 model across the same four domains and trains specialized linear classification heads.

The paper evaluates a highly comprehensive and competitive suite of baselines:
- **Expert Ceiling:** Isolated execution with perfect routing (upper-bound).
- **Uniform Merging:** Static weight average.
- **PFSR:** Non-parametric classification-head dependent projection routing.
- **SPS-ZCA:** The supervised SOTA baseline (uses 64 labeled calibration samples).
- **Zero-Shot Cosine Routing (ZCR):** Weight-space average cosine similarity (calibration-free).
- **Streaming K-Means:** A purely unsupervised centroid-tracking baseline (with Hungarian matching).
- **Task Arithmetic:** Parameter-space merging with optimized scaling factor.
- **Traditional Test-Time Adaptation (TENT):** Gradient-based entropy minimization.

The inclusion of these baselines, particularly Streaming K-Means and ZCR, provides a thorough and rigorous standard for comparison.

## 2. Assessment of Claims and Findings
The empirical results are exceptionally detailed and refreshingly honest, and they generally support the paper's claims:

- **Verification of EER's Synthetic Success:** In the synthetic sandbox, EER achieves **71.38%** Joint Mean accuracy, outperforming the supervised SOTA SPS-ZCA (**66.76%**) by **+4.62%** and maintaining robustness to linear representation drift ($d=0.45$, **71.18%**).
- **Honesty in Reporting Real-World Collapse:** The authors deserve high praise for their transparency regarding the limitations of their zero-shot methods on real-world ResNet-18 features. They clearly document that:
  - **EER collapses to 35.38%** due to the *Entropy Calibration Discrepancy* (the MNIST expert being highly overconfident on OOD CIFAR-10 and SVHN inputs).
  - **EPL-OCA collapses to 27.45% (Hard) and 31.52% (Soft)** because overconfident pseudo-labels corrupt the running centroids in a *self-referential feedback loop*.
  - **UCG-EER collapses to 28.45%** when trying to make the gating unsupervised.
- **Validation of the Semi-Supervised Solution (CG-EER):** CG-EER resolves this overconfidence by gating routing using pre-computed offline centroids, achieving **61.50%** accuracy (outperforming supervised SPS-ZCA by **+0.70%**).
- **Ablation Studies:** The paper includes valuable ablations, showing that softening the ensembling temperature ($\tau = 0.5$) in EPL-OCA improves synthetic accuracy from **49.88%** to **61.62%** (acting as a spatial regularizer), and validating the scale-invariance of the proposed *Normalized Shannon Entropy*.

## 3. Methodological and Experimental Skepticism (Theorist View)
Despite the thoroughness, several aspects of the experimental evaluation warrant skepticism:

- **Idealized Sandbox vs. Real Embedding Gap:**
  The synthetic sandbox assumes **perfect subspace orthogonality** and **class orthogonality**. While this "provides algebraic tractability," it is a highly unrealistic representation of real deep-learning feature spaces. The fact that EER and EPL-OCA perform exceptionally well in the synthetic sandbox but **completely collapse on real ResNet-18 features** demonstrates that the synthetic environment is a poor proxy for real-world performance. This severe gap undercuts the value of the synthetic results and suggests that "completely calibration-free zero-shot ensembling" remains a theoretical ideal rather than a practical reality.
- **Overlapping Class Namespace Bias:**
  The authors evaluate all four tasks on 10 classes in the overlapping namespace $\{0, \dots, 9\}$. They admit that this introduces a background chance probability of $\approx 10\%$ for misrouted samples (e.g., routing a CIFAR-10 "cat" to MNIST "3", which is counted as correct if the ground truth index matches). While they acknowledge this, it introduces an optimistic evaluation bias across all models. A more rigorous evaluation would use disjoint class namespaces to measure pure, unbiased routing performance.
- **High SVHN Noise:**
  The SVHN noise scale is set extremely high (0.56), leading to an expert ceiling of only 39.44%. Although this is designed as an aggressive stress-test to show EER's resistance to cross-task noise infiltration, it heavily skews the overall Joint Mean accuracy downwards.
- **Scale of the Real-World Evaluation:**
  The real-world validation is restricted to 512-dimensional embeddings from a pre-trained ResNet-18 model on vision classification datasets (MNIST, FashionMNIST, CIFAR-10, SVHN). While this is a step in the right direction, modern model ensembling and model merging are typically applied to large-scale Vision Transformers (ViT) or Large Language Models (LLMs). Evaluating on these larger models and tasks with higher dimensionalities (e.g., 768 or 4096 dimensions) and more complex topological overlaps would be necessary to confirm the generalizability of their findings.
