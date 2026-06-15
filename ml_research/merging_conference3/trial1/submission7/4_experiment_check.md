# Intermediate Review Report 4: Experimental Evaluation and Baseline Check

## 1. Evaluation of the Experimental Setup and Datasets
The authors evaluate their proposed diagnostic suite on a multi-task vision benchmark using:
*   **Backbone:** CLIP ViT-B/32 (12 transformer blocks + 1 projection layer, 13 parameter groups).
*   **Tasks/Datasets:** MNIST, FashionMNIST, CIFAR-10, SVHN.
*   **Evaluation Rigor:** 3 independent random trials (seeds 42, 100, 2026), reporting Mean $\pm$ Standard Deviation across all experiments. This represents an exceptional and rare standard of statistical rigor for model merging publications, where single-seed validation is unfortunately the norm.
*   **Calibration Set:** 256 images total (64 images per task) for test-time adaptation.

### Discussion of Datasets and Scope:
*   **Low-Resolution Vision Tasks:** The selected datasets are relatively low-resolution classification tasks.
*   **Excellent Limitations Discussion:** The authors demonstrate high academic integrity by explicitly acknowledging this limitation in Section 5. They clarify that because these tasks are saturated and task vectors reside close to their shared CLIP pre-trained initialization, a flat global average (Spatial Mean) naturally acts as a strong spatial regularizer.
*   **Generalization Acknowledged:** They explicitly discuss that in larger-scale networks (such as 7B+ modern autoregressive language models) or complex downstream tasks (such as instruction-tuning), layer-by-layer optimization may remain critical due to highly distinct representational hierarchies. This balanced discussion successfully mitigates the risk of overclaiming.

## 2. Evaluation of the Baselines
The paper evaluates and compares:
1.  **Task Arithmetic (Baseline, $\lambda=0.3$):** A fixed global scale.
2.  **Optimized AdaMerging (1+1 ES & Adam GD):** 13 optimized layer coefficients per task.
3.  **Spatially Averaged (Spatial Mean - 1+1 ES & Adam GD):** The layer-wise average of the optimized coefficients.
4.  **Intra-Task Layer Shuffling (1+1 ES & Adam GD):** The shuffled control treatment.

### Baseline Insights:
*   The inclusion of the properly selected Task Arithmetic baseline ($\lambda=0.3$) is excellent, showing that optimized Adam GD ($84.52\%$) barely outperforms the unoptimized baseline ($84.44\%$) while introducing 4x greater seed variance ($\pm 1.57\%$ vs. $\pm 0.37\%$).
*   This baseline comparison is critical to exposing the transductive overfitting behavior of Adam GD, proving that unconstrained optimization on a small calibration split is largely redundant and unstable for these datasets.

## 3. Support for Claims
The empirical results perfectly support the paper's central claims:
1.  **Overfitting-Optimizer Paradox:** Supported. Spatially averaging 1+1 ES parameters improves performance ($85.21 \pm 0.11\%$ vs. $85.07 \pm 0.47\%$), showing that zero-order variation is optimization noise. Under Adam GD, spatial averaging collapses CIFAR-10 performance (from 89.84% to 79.49%), proving the optimizer finds a highly precise, delicate configuration, yet this configuration fails to outperform the unoptimized baseline on unseen test data and increases seed variance by 4x, confirming transductive overfitting.
2.  **Extreme Landscape Flatness:** Supported. Both optimizers tolerate up to 50% relative Gaussian noise with negligible performance degradation, which is a highly robust and solid result.
3.  **CKA-Accuracy Decoupling:** Supported. Spatially averaged models exhibit marginally higher average CKA similarity, yet this activation alignment directly contradicts downstream classification accuracy (where CIFAR-10 accuracy collapses by 10.35% under Adam GD despite maintaining $>0.95$ CKA).
4.  **Proximity Regularization and Calibration sweeps (Appendix B & C):** Empirically supported by Figure 4 and Figure 5. Proximity regularization stabilizes Adam GD, showing peak accuracy of 86.57% at $\beta = 0.5$. Calibration sweeps clearly show the transductive overfitting threshold, with test accuracy stabilizing at $N_{\text{cal}} \ge 128$.
5.  **Coefficient Profiles (Appendix D):** Supported. Figure 6 visualizes the high-frequency optimization noise under 1+1 ES and the smooth but divergent patterns under Adam GD, confirming the joint entropy task-bias physically suppresses coefficients on the sacrificial SVHN task.

## 4. Minor Constructive Suggestions for Experiments

While the experimental evaluation is exceptionally complete, rigorous, and exhaustive, a few minor suggestions are offered for future research:

*   **Expand Dataset Scope:** Future iterations could evaluate these diagnostic treatments on larger-scale datasets like ImageNet or multi-modal benchmarks to confirm whether the transductive overfitting and spatial-averaging dynamics hold on high-resolution and more complex data distributions.
*   **Alternative Backbones:** Evaluating other vision backbones (e.g., ConvNeXt, Swin Transformer, or larger ViT variants like ViT-L/14) would help confirm that the observed optimization behaviors and landscape flatness are properties of model merging in general, rather than being specific to CLIP ViT-B/32.
*   **Evaluate More Seeds:** While 3 independent random trials is already an exceptionally rare and commendable standard for model merging papers, scaling the experiments to 5 or 10 seeds would provide even tighter confidence intervals and further solidify the statistical claims.
