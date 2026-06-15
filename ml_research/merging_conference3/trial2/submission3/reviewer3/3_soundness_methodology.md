# 3. Soundness and Methodology

## Clarity of the Description
The methodology of the paper is exceptionally well-described, mathematically rigorous, and highly transparent.
- The mathematical formulation in Section 3 is clear, precise, and easily readable. The transitions from task vectors to layer-wise coefficients, and finally to the polynomial subspace parameters, are logically step-by-step.
- The authors are highly commended for their complete transparency. In both the Introduction and Section 4, they explicitly and repeatedly clarify what portions of their experiments are executed within a *calibrated simulation environment* versus *physical PyTorch deep networks*. This level of honesty is rare and establishes excellent scientific integrity.
- Appendix A provides a detailed "Physical Validation Roadmap" with concrete PyTorch code snippets showing exactly how the polynomial synthesis and functional calls are implemented, making the implementation highly accessible.

## Appropriateness of Methods
- **Subspace Parameterization**: Parameterizing $\lambda_{k, l}$ as a continuous polynomial of normalized depth $\bar{l} = \frac{l}{L-1} \in [0, 1]$ is a mathematically sound way to enforce smoothness. Normalizing the depth is crucial for numerical stability and conditioning (preventing Vandermonde overflow), and the authors justify this well.
- **SplineMerge piecewise splines**: This extension is highly appropriate for addressing the realistic structural heterogeneity of deep architectures (e.g., the abrupt transition from attention blocks to feed-forward blocks). The inclusion of a dynamic programming recurrence relation to automate partition boundary selection is a very clever addition.
- **Dual Validation Regimes**: Validating the findings under both a stylized simulation environment (where factors can be perfectly controlled across 30 seeds and 700+ trajectories) and two distinct physical environments (a PyTorch Residual MLP and a pre-trained CLIP foundation model) is methodologically outstanding. This multi-axis validation ensures that the conclusions are not artifacts of a specific simulator.

## Potential Technical Flaws and Limitations

### 1. The Underfitting-Roughness Trade-off in Global Polynomials
As shown in the physical CLIP experiments (Table 4), while global PolyMerge ($d=2$ and $d=4$) successfully restricts spatial roughness ($28.1\times$ and $17.4\times$ reductions), it suffers from a notable **underfitting bottleneck**, dropping multi-task average accuracy on real CLIP weights to 89.00% and 90.00% respectively, which is below the static Task Arithmetic baseline of 94.00%. 
This indicates that the global polynomial constraint is **too restrictive** to capture the complex, non-monotonic layer sensitivity transitions in physical foundation models. While the authors successfully resolve this using SplineMerge (achieving 96.00%), this underfitting behavior represents a major practical limitation of the global PolyMerge formulation on real-world weights.

### 2. Transductive Overfitting of the Automated DP Boundary Discoverer
The authors implement a dynamic programming (DP) recurrence to automatically discover entropy-minimizing partition boundaries. However, their own empirical results (Table 3) show that DP-discovered partitioning actually yields *lower* generalization accuracy (86.12% vs. 86.80% for manual uniform partitioning). 
The authors' explanation is highly insightful: optimizing block boundaries at test-time on unlabeled target streams introduces another axis of transductive overfitting, allowing the partition boundaries themselves to fit local noise. While this explanation is intellectually satisfying, it reveals a fundamental flaw in the proposed automated boundary discovery method, making the manual uniform partition heuristic practically superior.

### 3. Simulative vs. Physical Landscapes
The primary quantitative sweeps in Section 4 are executed within a continuous weight-merging simulation. Although the simulator is calibrated on empirical ViT-B/32 statistics, it still incorporates highly simplified assumptions (e.g., additive linear merging, stylized convex quadratic loss, and simplified transductive noise). While the authors address this by adding physical validations and non-convex stress-tests (Model II) in the Appendix, the core of their statistical sweeps remains simulative.

## Reproducibility
The reproducibility of the paper is exceptionally high:
- The authors provide concrete PyTorch code integration patterns in the Appendix.
- They outline the exact hyperparameters used (Adam learning rate $10^{-2}$, ES mutation noise $\sigma=0.05$, batch size 64, steps 500, etc.).
- They specify the exact seeds (42 to 71 inclusive) used for all 30 random runs, which allows for exact statistical reproduction.
- The use of standard, publicly available checkpoints (such as `openai/clip-vit-base-patch32`) and datasets (CIFAR-10, GTSRB) ensures that any researcher can reproduce the physical CLIP validations.
- The CPU-only nature of the simulation and physical validation code enables democratized, low-resource validation.
