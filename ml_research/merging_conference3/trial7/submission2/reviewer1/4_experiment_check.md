# Intermediate Review Evaluation: Experimental Evaluation (4_experiment_check.md)

This document provides a critical evaluation of the paper's experimental setup, simulated sandbox environments, baselines, real-world physical validations, and whether the empirical findings support the central claims.

---

## 1. Experimental Setup and Empirical Transparency
The authors deserve praise for their **explicit empirical transparency**. In Section 4.1, they state upfront that the primary quantitative results and stress tests are evaluated inside a simulated high-fidelity synthetic environment, termed the **Analytical Coordinate Sandbox**. 
* **The Sandbox Design:** The sandbox is a 192-dimensional representation space modeling 14 layers and 4 specialized expert task domains (Task 0: low-noise MNIST-equivalent, Task 1: moderate-noise FashionMNIST-equivalent, Task 2: high-noise CIFAR-10-equivalent, Task 3: extreme-noise SVHN-equivalent). 
* **Asymmetric Classes:** To test Class-Size Scaling Calibration (CSC), the task class vocabulary sizes are deliberately asymmetric ($C_{\text{tasks}} = [10, 10, 10, 4]$).
* **Statistical Rigor:** All primary results report means and standard deviations evaluated across 10 independent random seeds (seeds 42 to 51), establishing statistical significance.

---

## 2. Baselines
The baselines are comprehensive and cover the landscape of current ensembling methods:
1. **Static Uniform Merging:** Non-adaptive baseline (weights fixed at $0.25$).
2. **Linear Router (Unregularized):** Parametric baseline optimized on the small calibration split.
3. **QWS-Merge SOTA:** Quantum-inspired wavefunction superposition dynamic router, optimized at test-time.
4. **L3-Softmax (Well-Regularized):** Standard regularized layer-wise Softmax router, representing the state-of-the-art in parametric routing.
5. **PFSR + MBH:** Parameter-free baseline using unweighted cosine similarity. **Crucially, comparing against PFSR directly isolates the performance gain of our proposed Fisher Information metric.**

---

## 3. Analysis of Claims vs. Empirical Support

### Claim 1: Immunity to few-shot overfitting (Dynamic Routing Paradox)
* **Empirical Support: Strong.** In the homogeneous batch setting ($B=256$, Table 1), all parametric methods collapse to near-uniform accuracy ($\approx 36\% - 39\%$) due to overfitting on the microscopic 64-sample calibration split. Even the well-regularized L3-Softmax collapses to uniform averages. By contrast, the parameter-free methods completely bypass optimization, with FIOSR achieving a stellar joint accuracy of **76.86%** and PFSR achieving **68.30%**.

### Claim 2: Immunity to Vectorization Collapse under low-latency streaming
* **Empirical Support: Strong.** Under heterogeneous streams with single-sample sequential feeds ($B=1$, Table 2), parametric methods collapse or exhibit extreme volatility. In contrast, FIOSR (equipped with MBH) maintains perfectly flat performance of **76.83%** across all batch sizes ($B=1$ to $512$).

### Claim 3: Causal Necessity of Class-Size Scaling Calibration (CSC)
* **Empirical Support: Solid.** The ablation in Table 3 shows a statistically significant drop of **1.02% - 1.15%** absolute accuracy when CSC is removed across all 10 seeds, confirming that scaling by $\sqrt{2\log C_k / d}$ successfully corrects for maximum bias in asymmetric settings.

### Claim 4: Pooled Within-Class Variance vs. Task-Level Variance
* **Empirical Support: Solid.** Section 4.5 shows that using the pooled within-class covariance estimator to isolate pure coordinate-wise noise from discriminative centroid spread yields a statistically significant **+1.11%** absolute improvement in homogeneous joint accuracy ($76.88\%$ vs $75.77\%$), validating the theoretical choice.

### Claim 5: Scalability and Gating Safeguards
* **Empirical Support: Strong.** The ablation in Table 4 (Appendix) shows that Top-$M$ Expert Gating with $M=1$ (hard routing) completely eliminates sequential batch-partitioning overhead while achieving an outstanding joint accuracy of **76.87%**, representing an **8.84%** absolute improvement over flat Cosine. This supports the scalability claim, though it represents a conceptual compromise where weight ensembling is replaced by a hard task-routing selection.

### Claim 6: Robustness under Rotated, Correlated Noise
* **Empirical Support: Satisfactory.** Under rotated noise (Table 5), standard diagonal Fisher collapses below flat Cosine (67.38% vs 67.50%). However, the authors' training-free **FIOSR-Online** (utilizing an on-the-fly shrinkage covariance EVD alignment directly from the 16 calibration samples) achieves **67.68%**, successfully outclassing the flat baseline and unaligned diagonal Fisher. The Oracle K-FAC equivalent (**FIOSR-Rotated**) recovers **73.07%** ($+5.57\%$ gain), demonstrating the theoretical potential of block-diagonal/K-FAC formulations.

### Claim 7: Real-World Viability
* **Empirical Support: Solid but Modest.**
  * In the **LoRA Activation Space Simulation** (Table 6), FIOSR achieves **95.00%** routing accuracy and **77.00%** joint ensembling accuracy (a $+6.67\%$ absolute gain over flat Cosine), recovering **98.30%** of the oracle performance.
  * In the **End-to-End Physical ResNet-18 Validation** (Table 7) on real image datasets (MNIST, FashionMNIST, SVHN), FIOSR achieves a solid **59.00%** routing accuracy and **52.00%** joint accuracy, outperforming flat Cosine (**56.33% / 50.67%**).
  * *Critical Observation:* The performance gap in the end-to-end physical deployment is significantly narrower (**+1.33%** joint ensembling improvement) than in the simulated coordinate-aligned sandbox (**+8.56%** improvement). This suggests that while diagonal Fisher is exceptionally powerful in coordinate-aligned environments, actual physical activation spaces are highly complex with dense, non-axis-aligned covariance structures, where diagonal Fisher's coordinate-filtering effect is slightly dampened unless full block-diagonal (K-FAC) or shrinkage EVD alignment is employed. This is an insightful critique that highlights the external validity gap, although the authors deserve praise for being honest about physical validation and including global pre-calibration mean-centering and scale-regularization shrinkage ($\alpha=2.0$) to stabilize the FIM.
