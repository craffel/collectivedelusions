# 4. Experimental Design and Validation Check

We rate the experimental design and empirical validation of this submission as **Excellent**. The experiments are comprehensive, well-structured, and include rigorous baselines, realistic stream configurations, and exceptional ablation studies.

## 1. Experimental Rigor and Protocol
- **Backbone Selection:** The paper uses a pre-trained **Vision Transformer (ViT-Tiny)** backbone (`vit_tiny_patch16_224`) consisting of 12 Transformer blocks. Fine-tuning the last two blocks (Blocks 10 and 11) is standard and computationally efficient.
- **Multi-Task Task Pool:** The choice of tasks (**MNIST, FashionMNIST, CIFAR-10, SVHN**) is diverse and represents a wide spectrum of difficulty.
- **Extremely Sparse Calibration Split:** To simulate severe data constraints, the authors use a tiny calibration set of $N=64$ samples (exactly 16 samples per task). This is a highly realistic setting for test-time routing.
- **Robust Optimization Protocol:** The routers are trained for 100 epochs using the AdamW optimizer, ensuring convergence.

## 2. Evaluation Stream Configurations
One of the major strengths of the paper is the division of the evaluation into three distinct stream protocols:
1. **Homogeneous Stream:** Single-task batches (idealized setting).
2. **Heterogeneous Stream (Sample-wise):** Mixed-task batches processed sample-by-sample (no hardware constraints, computationally expensive $O(B)$).
3. **Heterogeneous Stream (Collapsed):** Realistic edge deployment where mixed-task batches are processed, and the hardware averages the predicted coefficients ($\bar{\alpha} = \frac{1}{B} \sum_{b=1}^B \alpha_b$) to preserve single-model $O(1)$ efficiency, inducing **heterogeneity collapse**.
This classification provides a direct, practical bridge between machine learning theory and systems/hardware engineering.

## 3. Analysis of Main Results
- **Unconstrained Overfitting:** The unregularized Global Linear Router achieves 67.12% homogeneous accuracy, but collapses to 54.12% (**-13.00% drop**) under the heterogeneous collapsed stream. This empirically proves that unconstrained high-dimensional routing overfits severely to local calibration features and suffers mutual cancellation during batch averaging.
- **Quantum Failure:** SOTA quantum-inspired QWS-Merge drops by **-6.75%** under collapsed streams, falling to 60.12%.
- **Test-Time Adaptation Failure:** AdaMerging (which minimizes prediction entropy online) collapses completely to the uniform static baseline (54.88%) on heterogeneous streams due to *transductive overfitting* and *sacrificial task bias*, proving that online TTA is fragile in mixed-task environments.
- **R2D-Merge Excellence:** R2D-Merge with CFR achieves **65.62%** average accuracy across all three settings, demonstrating **absolute resilience (0.00% collapse drop)**. It outperforms unregularized baselines by **11.50%** and QWS-Merge by **5.50%** in the collapsed state.

## 4. Nuanced Comparison: CFR vs. Standard L2 Regularization
Comparing R2D-Merge (CFR) against the Standard L2-regularized L3-Router (L2 Reg) isolates the direct benefits of task-adaptive, covariance-weighted regularization:
- **On Complex Tasks:** CFR strictly dominates standard L2 decay:
  - **FashionMNIST:** R2D-Merge achieves **84.00%** accuracy compared to standard L2's 82.50% (**+1.50%** gain) in both configurations.
  - **CIFAR-10:** R2D-Merge achieves **85.00%** (homogeneous) and **84.50%** (collapsed) accuracy, outperforming standard L2's 83.50% and 82.00% respectively (**+1.50%** and **+2.50%** gains).
- **On Simple Tasks:** Standard L2 decay outperforms CFR on **MNIST** (73.50% collapsed vs. R2D-Merge's 68.50%). 
- **The Theoretical Rationale:** On simple canonical datasets like MNIST, uniform isotropic parameter shrinkage acts as an effective regularizer, pulling the weights closer to a uniform static configuration which performs well. However, this isotropic shrinkage excessively limits the router's representation capacity when confronted with more challenging distributions. In contrast, CFR's covariance-weighted regularization adaptively allocates parameter budget, releasing regularization constraints in directions of high task-specific sensitivity, yielding significantly higher performance on complex datasets.

## 5. Highly Detailed and Comprehensive Ablation Studies
The ablation studies are exceptionally thorough and substantiate every core theoretical claim:
- **Calibration Sample Size ($N$):** Ablating $N \in \{16, 32, 64, 128, 256\}$ reveals that under extreme sparsity ($N \le 32$), standard L2 decay is superior due to covariance estimation noise. However, as $N \ge 64$, CFR's covariance estimates stabilize, allowing R2D-Merge to strictly dominate L2 decay. This results in clear, actionable practitioner guidelines.
- **Latent Routing Dimension ($d$):** Ablating $d \in \{2, 4, 8, 16\}$ demonstrates that $d=4$ or $8$ is optimal, and $d=16$ leads to overfitting, confirming that low-dimensional feature projection is essential.
- **Feature Extraction Block:** Demonstrates that early layers (Block 0) capture dataset-invariant filters that generalize well, whereas deep blocks (Block 6 or 11) yield deep-layer activations highly biased toward specific tasks, inducing destructive feedback loops.
- **Regularization Strength Sweep (The Pareto Frontier):** This is one of the strongest empirical sections. Ablating $\lambda_{wd} \in \{0, 10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}\}$ elegantly maps the Pareto frontier of the Dynamic-Resilience Trade-off, showing how engineers can continuously trade off dynamic routing expressiveness (weight-to-bias ratio $\mathcal{M}_{\text{drift}}$) for hardware-level collapse resilience (retaining 65.12% collapsed accuracy with a mere -1.63% loss at $\lambda_{wd} = 10^{-3}$).

## Summary of Experimental Design and Validation:
The empirical validation of this paper is outstanding. The experimental protocols are clean, the baselines are representative, the evaluation streams capture real-world hardware realities, and the ablation studies are remarkably thorough, providing deep analytical insights rather than just binary success metrics.
