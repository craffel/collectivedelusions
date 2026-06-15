# Paper Summary: CR-PolySACM

**Title:** CR-PolySACM: Clipping-Regularized Sharpness-Aware Subspace Model Merging for Robust Post-Training Quantization
**Author:** Sophia Vance (Department of Mathematics, Harvard University)

---

## 1. Core Problem and Motivation
Post-hoc model merging enables separately fine-tuned task-specific expert models to be directly composed in weight space without retraining or accessing original datasets. To avoid task conflicts, recent frameworks employ test-time adaptation (TTA) to dynamically optimize layer-wise merging coefficients on small calibration streams (e.g., $N=64$ samples) by minimizing prediction entropy.

Despite their continuous-space (FP32) performance, the author identifies a critical and hitherto overlooked vulnerability in these adaptive paradigms: **Quantization-Operator Overfitting**:
- Unconstrained test-time coefficient optimization converges to extremely sharp local minima in continuous weight-space.
- When these models are deployed on edge devices under post-training quantization (PTQ) constraints (such as INT8 or INT4 formats) to meet strict memory and latency limits, the rounding noise triggers a catastrophic collapse in multi-task accuracy.
- This represents a fundamental bottleneck for deploying test-time adapted merged models on edge hardware.

---

## 2. Proposed Framework: CR-PolySACM
To resolve this dual bottleneck—overfitting to tiny calibration streams and extreme sensitivity to downstream quantization noise—the paper presents **CR-PolySACM** (Clipping-Regularized Sharpness-Aware Subspace Model Merging). This framework achieves a highly effective "division of labor" through two main pillars:

### A. Global Structural Regularization: Differentiable Polynomial Subspace (PolyMerge)
Instead of optimizing $L \times K$ independent layer-wise coefficients (e.g., $56$ parameters for a 14-layer ViT with 4 experts), PolyMerge restricts the layer-wise blending coefficients $\lambda_k^l$ to a low-degree polynomial of normalized network depth $d_l = (l-1)/(L-1) \in [0, 1]$:
$$
\lambda_k^l(\mathbf{p}_k) = \sigma\left(a_k + b_k \left(\frac{l-1}{L-1}\right) + c_k \left(\frac{l-1}{L-1}\right)^2\right)
$$
where $\mathbf{p}_k = [a_k, b_k, c_k]^T \in \mathbb{R}^3$ represents the polynomial coefficients for task $k$, and $\sigma$ is the logistic sigmoid function. This reduces the total optimization variables from $56$ to exactly $3 \times K = 12$ parameters. This global structural constraint prevents overfitting to the calibration stream, preserves global representation structures, and naturally shields the model against out-of-subspace noise.

### B. Local Flatness Optimization: Clipping-Regularized SACM (CR-SACM)
Within the stable, low-dimensional polynomial subspace, the author explicitly optimizes for local flatness. They analyze standard unnormalized sharpness-aware minimization (such as HessMerge or SACM) and identify a **task-vector norm scale pathology**:
- There is a massive, 50-fold discrepancy in layer-wise task-vector norms on a Vision Transformer backbone (e.g., intermediate blocks have norms of $0.40$--$0.68$, while the final layer normalization layer, Layer 13, has a norm of only $0.014$--$0.020$).
- Standard unnormalized sharpness optimizers are scale-blind, applying perturbations that are over $100\times$ smaller at Layer 13, rendering them blind to this highly sensitive layer norm.
- Conversely, unmitigated scale-invariant normalization scales Layer 13's perturbation by $>2500$ to $5000$ times, triggering immediate gradient explosion and representational collapse.

To resolve this, **CR-SACM** clips the task-vector norms to a robust minimum threshold ($\beta = 0.10$):
$$
V_{\text{clipped}, k}^l = \max\left( \|\tau_k^l\|_2, \beta \right)
$$
This balances scale sensitivity across layers and enables the optimizer to flatten low-norm layers without triggering gradient explosion.

---

## 3. Main Findings & Empirical Results
- **Experimental Setup:** Evaluated using a Vision Transformer backbone (`vit_tiny_patch16_224`) fine-tuned on four diverse classification datasets: MNIST, FashionMNIST, CIFAR-10, and SVHN.
- **SOTA in INT4:** Under aggressive 4-bit symmetric channel-wise quantization, CR-PolySACM achieves a joint mean accuracy of **19.07%**, outperforming standard PolyMerge (**18.10%**) by an absolute **+0.97%** (representing a relative improvement of over **5.3%**), demonstrating the critical synergy of global structural subspaces and local flatness regularization for robust edge deployment.
- **HessMerge Breakthrough:** Once scale-blindness is corrected by CR-SACM, the upgraded HessMerge baseline consistently and significantly outperforms AdaMerging across all six target schemas (+1.36% in FP32, rising from 49.12% to **50.48%**). This proves that unconstrained sharpness optimization is viable once scale-blindness is corrected.
- **Robustness and Sensitivity Ablations:**
  - Ablations on $\gamma$ show that unconstrained flatness optimization degrades performance due to the norm scale pathology.
  - Ablations on $\beta$ confirm a non-monotonic trend with two distinct failure modes: gradient explosion for small $\beta \le 0.01$ and scale-blindness for large $\beta \ge 0.25$.
- **Extended Appendix Analysis (Newly Added):**
  - **Calibration Stream Size ($N$):** Ablations on $N \in \{8, 16, 32, 64, 128\}$ show that CR-PolySACM remains stable down to $N=16$ ($B=4$ samples per task) due to its highly constrained 12-parameter search space.
  - **Overhead Analysis:** CR-PolySACM completes adaptation in just **1.56 seconds**, representing a negligible $+1.3\%$ overhead over standard AdaMerging ($1.54$ seconds) while delivering a massive **52.8$\times$ speedup** over exact Hessian trace optimization ($82.35$ seconds).
  - **Alternative Subspaces:** Low-degree polynomials outperform Random Projections (FP32 accuracy of $55.50\%$, a $-1.90\%$ drop) and match or outperform Discrete Cosine Transform subspaces ($56.82\%$).
  - **Scholarly Honesty:** The author transparently discusses the "expert-to-merge drop" (-31.27% gap in FP32) as an inherent domain disconnect challenge in multi-task merging, and honestly notes that absolute INT4 performance (19.07%) is still extremely low, establishing it as a scientific proof of concept rather than a production-ready solution.
  - **Sum-to-One Normalization & Parameter Scale Inflation (A.4):** Measures the average sum of coefficients at convergence ($\bar{\lambda} \approx 1.42$, far below the theoretical limit of 4.0), showing that entropy minimization combined with the structural polynomial depth constraint naturally regularizes scale inflation without explicit sum-to-one constraints.
  - **Robustness to Class Imbalance & Label Shift (A.5):** Evaluates skewed calibration streams, showing that CR-PolySACM is highly robust under extreme class imbalance (e.g., maintaining $18.81\%$ accuracy under 20\% class coverage), whereas unconstrained TTA (AdaMerging) collapses to random guessing ($10.15\%$).
  - **Convergence Curves & Trajectory Analysis (A.6):** Displays smooth, monotonic convergence of entropy loss and active minimization of local sharpness, showing that CR-PolySACM is highly stable over time.
