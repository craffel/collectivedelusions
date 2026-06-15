# Experiment Check: Löwdin-Orthogonalized Task-Space Projection

## 1. Overall Rating
**Excellent.**

The empirical section is exceptionally rigorous, statistically sound, and honest. Rather than resorting to selective reporting or weak baselines, the authors present exhaustive sweeps, realistic baselines, and real-world validations.

## 2. Baselines and Sandbox Realism
- **Comprehensive Baselines:** The paper compares PFSR/OTSP against multiple standard and SOTA baselines:
  - Static Uniform Merging (Task Arithmetic)
  - Naive Mean Centroid (Proving the necessity of SVD centroid extraction)
  - Parametric LinearRouter (Unregularized)
  - QWS-Merge (SOTA Parametric)
  - L3-Softmax (Unregularized, trainable Softmax)
  - L3-Softmax Well-Reg (Zero-initialized, trainable Softmax)
  All parametric routers were upgraded to train directly on raw $D$-dimensional representations with direct cross-entropy supervision, representing the strongest and most competitive parametric baseline configurations possible.
- **Calibrated Representation Sandbox:** To study coordinate-level routing dynamics in isolation, the authors construct a high-fidelity 192-dimensional representation sandbox simulating $K=4$ tasks (MNIST, FashionMNIST, CIFAR-10, SVHN) across Homogeneous, Heterogeneous ($B=256$), and Heterogeneous ($B=1$) streams. Results are averaged over 10 independent random seeds, ensuring statistical significance.
- **Orthogonal Masking Effect & Sandbox Realism:** The authors transparently address "Joint Metric Flatness" in disjoint orthogonal setups. Because out-of-subspace expert logits are exactly 0.0, any positive weight on the correct expert yields the ceiling accuracy of 74.46% (flat joint classification accuracy). The authors correctly identify **Routing Accuracy** as the only informative metric under this disjoint setup.

## 3. Key Quantitative Findings
- **Zero-Parameter Victory over Parametric Routers:** PFSR and OTSP achieve a perfect **100.00% ± 0.00%** routing accuracy on uncorrupted representations, whereas trainable Softmax routers only achieve 66.47% to 67.22% due to small-sample inductive overfitting on the tiny calibration split.
- **Vectorization Collapse Defense:** Under sample-wise vectorized ensembling ($B=1$), the unconstrained `LinearRouter` baseline's classification accuracy collapses to **55.57% ± 1.68%**. Both PFSR/OTSP and simplex-normalized routers are immune, maintaining stable ensembling weights bounded on the probability simplex.
- **Symmetric Overlap Results (Gating Temperature Sweeps):** The paper sweeps temperature $\tau \in [0.001, 2.0]$ in a symmetric overlap sandbox ($\rho = 0.33$). It deconstructs the **Hard Gating Penalty** under sharp gating ($\tau = 0.001$), where routing errors collapse classification to 71.71% (-4.33% relative to Uniform). It shows that softening the temperature to $\tau = 2.0$ completely recovers ensembling benefits, climbing accuracy to **75.81% ± 0.78%** (statistically identical to Uniform).
- **Asymmetric Sandbox Performance:** In an asymmetric sandbox with highly skewed noise (Task 3 SVHN noise $\sigma_3 = 1.95$), the authors show that under $\tau = 0.3$, PFSR/OTSP achieve outstanding routing accuracies of **69.74% ± 2.41%** and **70.76% ± 2.82%** respectively, far outperforming random (25.00%) and parametric routers (which fail to exceed 54.40%).
- **Löwdin Orthogonalization Limits:** Under asymmetric layout configurations, OTSP systematically underperforms unorthogonalized PFSR by 0.2% to 1.6% (Table 2). This is explained using multicollinearity theory, proving that raw unorthogonalized projection (PFSR) is not only computationally simpler but more robust.

## 4. Real-World Proof-of-Concept
To bridge the evaluation gap of the simulation sandbox, the authors conduct a proof-of-concept evaluation on a real pre-trained ResNet-18 manifold (ImageNet-1K). They define three semantic domains: Dogs (10 classes), Cats (5 classes), and Vehicles (10 classes), and evaluate on 1,250 real deep representation vectors under noise.
- Both PFSR and OTSP generalize exceptionally well, achieving outstanding routing accuracies of **92.00%** and **92.08%**.
- Because of the active positive semantic overlap between Dogs and Cats ($S_{01} = 0.1905$), Löwdin Orthogonalization (OTSP) successfully decouples the task-space coordinate axes, yielding a systematic routing accuracy improvement of **+0.08%** over raw PFSR, validating OTSP's utility on real-world manifolds.

## 5. Anisotropic Noise Verification
The authors simulate highly anisotropic noise on a 2-expert setup.
- Uncorrected anisotropic noise skews the coordinate projection and collapses OTSP routing accuracy to **77.10%**.
- Applying origin-centered second-moment covariance whitening ($\hat{\Sigma}^{-1/2}$) offline to both centroids and representations successfully spherizes the noise cloud, restoring OTSP's routing accuracy to **89.45%** (a massive **+12.35% absolute routing accuracy gain**). This empirically confirms the effectiveness of their covariance-whitening formulation.
