# Intermediate Review Evaluation: 4. Experimental Setup and Validation Check

## Evaluation of the Experimental Setup
The paper's experimental validation is **highly robust, comprehensive, and well-designed**. It employs a dual evaluation strategy:
1. **The Coordinate Sandbox (Synthetic representation space):**
   A 14-layer, 192-dimensional simulation environment specifically designed to model task-specific block subspaces under varying noise standard deviations (from 0.01 for MNIST-like to 1.35 for SVHN-like). This is a highly appropriate setup for isolation testing, allowing the authors to precisely control representation anisotropy, dimensionality, and heteroscedastic noise, which is crucial for proving the SVD overfitting post-mortem and the train-test feature scale mismatch.
2. **Real-World Vision Serving (ResNet-18 Backbone):**
   An actual vision ensembling task using three standard image benchmarks: MNIST, Fashion-MNIST, and CIFAR-10. It feeds images through a frozen ResNet-18 CPU backbone, places the router at `layer1` output (64-dim features), trains LoRA-like classification heads on `layer4` output (512-dim features), and executes single-pass activation blending on heterogeneous streams. This confirms that the theoretical insights generalize to real, non-orthogonal deep representations and physical image data.

The streaming batch sizes ($B=16$), calibration sample budgets ($N_c = 16$), and test sizes ($N_{\text{test}} = 300$) represent realistic edge-serving constraints.

---

## Datasets and Baselines
- **Datasets:** The choices of benchmarks—MNIST, Fashion-MNIST, CIFAR-10, and SVHN—are excellent. They represent a classic spectrum of difficulty, spatial variance, and noise complexity, making them perfect for testing robust ensembling under heteroscedasticity.
- **Baselines:** The paper evaluates against a comprehensive suite of 8-9 diverse baselines. These include:
  - Static weight averaging / model merging (Uniform weight-space average, PFSR CVPR'25).
  - Quantum Wavefunction Superposition (QWS-Merge).
  - Standard parameterized routing (Linear Router with L2 regularization).
  - Early-centroid cosine similarity heuristics (SABLE CVPR'26 on Raw coordinates and SEP features).
  - Standard unregularized Empirical Risk Minimization (Temp-Only ERM on Block, PCA, and UN-PCA features).
  - Oracle Expert Ceiling (to establish the absolute upper-bound performance).

This wide array of baselines represents the absolute state of the art and ensures that the claims are tested against all relevant serving configurations.

---

## Alignment of Claims and Empirical Support
The paper's empirical results exceptionally validate its core claims:

1. **Claim: PAC-ZCA resolves the "heterogeneity collapse" of static model merging.**
   - *Support:* Table 1 shows weight-space merging (PFSR) collapses to 40.56% $\pm$ 0.99% accuracy under mixed-task batch streams because static weights cannot process diverse sample paths simultaneously. PAC-ZCA (Block Ours) retains a robust **64.16% $\pm$ 2.23%** joint accuracy on both homogeneous and heterogeneous streams, proving complete immunity.
2. **Claim: Active temperature calibration outperforms static, uniform heuristics.**
   - *Support:* SABLE (Raw Coords) achieves only 40.46% $\pm$ 1.09%, whereas PAC-ZCA (Block Ours) achieves **64.16% $\pm$ 2.23%**, representing a massive **+23.70%** absolute improvement.
3. **Claim: PAC-Bayesian bound minimization reduces ensembling variance compared to ERM.**
   - *Support:* On synthetic Block features (Table 1), PAC-ZCA (Block) matches the mean performance of unregularized Temp-Only ERM while reducing standard deviation from 2.28% to 2.23%. On real images (Table 3), PAC-ZCA (Isotropic) achieves **70.87% $\pm$ 2.20%**, outperforming standard Temp-Only ERM (69.47% $\pm$ 2.21%) in both mean accuracy and ensembling stability. This validates the variance-reducing property of parameter-space KL complexity bounds.
4. **Claim: Unsupervised SVD on tiny calibration sets overfits to noise and causes expert task neglect.**
   - *Support:* Section 4.5 presents a quantitative post-mortem of the PCA-SEP collapse. It documents that during calibration, SVHN (Task 3) samples projected on their overfitted PCA basis retain a large norm of 17.29, but on unseen test-set noise, the projected norm collapses to 5.40 ($68.8\%$ drop). This causes the temperature-only policy (which scaled SVHN logits with a high temperature based on the huge calibration norm) to permanently neglect SVHN at test-time ($0.00\%$ routing accuracy).
5. **Claim: Unit-Norm PCA Projection (UN-PCA-SEP) resolves SVD overfitting.**
   - *Support:* Under UN-PCA-SEP, features are normalized to the unit sphere prior to projection, bounding coordinates on $[0, 1]$ and mathematically eliminating the noise scale mismatch. Under orthogonal manifolds, PAC-ZCA (UN-PCA) recovers joint classification accuracy to **44.36% $\pm$ 1.30%** (Table 1), completely recovering predictions for the high-noise expert.
6. **Claim: Calibration Sample Complexity scaling.**
   - *Support:* Table 4 systematically sweeps the calibration budget $N_c \in \{8, 16, 32, 64, 128\}$ per task. It shows that PAC-ZCA consistently maintains lower ensembling standard deviations than unregularized ERM across different sample budgets (e.g., $2.33\%$ vs. $2.43\%$ for $N_c=8$, and $1.48\%$ vs. $1.53\%$ for $N_c=32$), and both outperform uncalibrated SABLE by ~10% absolute.
