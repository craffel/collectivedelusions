# Experimental Design and Evaluation Check

## 1. Quality of Empirical Evaluation and Baseline Selection

### Baseline Coverage
*   **Oracle Ceiling**: Included (78.82% / 78.98% joint accuracy), representing the maximum achievable accuracy given the noise level of individual tasks.
*   **Static weight-space averaging (Uniform)**: Included (25.00%), illustrating the baseline collapse under mixed streams.
*   **Advanced model merging baselines**: Includes **QWS-Merge** (40.16%) and **PFSR** (40.52%). Both are shown to collapse under mixed streams because weight-space merging cannot execute sample-specific expert paths inside a single parallel pass.
*   **Parameter-Free Subspace Routing (SABLE)**: Evaluated under both raw coordinates and Subspace Energy Projection (SEP) blocks/PCA.
*   **Empirical Risk Minimization (Temp-Only ERM)**: Included as a direct unregularized counterpart to PAC-ZCA, isolating the effect of the PAC-Bayesian KL complexity penalty.
*   **Assessment**: The baseline selection is highly comprehensive, covering static weight-space averaging, advanced dynamic weight routers, and uncalibrated activation-space blending routers.

---

## 2. Statistical Rigor

*   **Multi-Seed Evaluation**: Every single experiment in the Coordinate Sandbox and real-world Vision evaluation is run across **5 random seeds**, and the authors report both the mean and standard deviation.
*   **Statistical Significance**: A paired t-test was conducted to verify that PAC-ZCA's variance reduction and ensembling accuracy under orthogonal block-norms are statistically significant ($p < 0.05$) compared to unregularized ERM, demonstrating complete statistical confidence.
*   **Sensitivity Analysis (Table 2)**: The authors systematically sweep the prior variance hyperparameter $\sigma_0^2 \in \{0.1, 0.5, 1.0, 5.0, 10.0\}$ to study the regularization strength's effect, demonstrating that $\sigma_0^2 = 5.0$ lies in a highly stable, robust region.
*   **Sample Complexity Analysis (Table 4)**: They analyze how performance scales as a function of the total calibration budget $N_c \in \{8, 16, 32, 64, 128\}$ per task.
*   **Assessment**: The level of statistical rigor is outstanding. The inclusion of multi-seed runs with standard deviations, a paired t-test, hyperparameter sensitivity sweeps, and sample complexity analysis represents a gold standard for empirical validation in ML papers.

---

## 3. Real-World Serving Evaluation

*   To address the limitations of synthetic evaluation, the authors scale PAC-ZCA to real images (**MNIST, Fashion-MNIST, CIFAR-10**) using a frozen pre-trained **ResNet-18** feature extractor on CPU.
*   **Methodology**: Linear expert heads are trained on 1000 samples per task, and 16 calibration samples per task are partitioned into a Subspace Split (8 samples) and Optimization Split (8 samples). Testing is performed on a heterogeneous stream of 300 samples (100 per task).
*   **Results**: PAC-ZCA (Isotropic Ours) achieves **70.87% ± 2.20%** accuracy, outperforming SABLE (65.67%) and unregularized ERM (69.47% ± 2.21%) while successfully stabilizing ensembling variance.
*   **Assessment**: This real-world Vision experiment successfully bridges the gap between synthetic Coordinate Sandbox simulations and real-world deep feature manifolds.

---

## 4. Minor Empirical Weaknesses and Limitations

While the experimental design is excellent, several minor weaknesses are visible in the results:

1.  **Block-Feature Trade-off (SABLE vs. PAC-ZCA)**:
    In Table 1, standard SABLE (SEP-Block) with a static temperature of $\tau=0.05$ achieves **66.08% ± 0.78%** accuracy, which is slightly higher than PAC-ZCA (Block Ours) (**64.16% ± 2.23%**). The authors explain this as a "disjoint split penalty"—to satisfy McAllester's theorem, they must partition the 16 calibration samples, leaving only 8 samples for temperature optimization, which increases variance. While theoretically justified, this demonstrates a small empirical cost to maintaining absolute mathematical rigor.
2.  **Over-regularization Bottleneck under UN-PCA**:
    In Table 1, PAC-ZCA (UN-PCA Ours) slightly underperforms Temp-Only ERM (UN-PCA) (44.36% ± 1.30% vs. 44.58% ± 1.38% on orthogonal, and 45.86% ± 0.76% vs. 46.02% ± 0.93% on overlapping). Although the difference is small and within the standard deviation, it suggests that the fixed isotropic prior $\mathcal{N}(\mathbf{w}_0, 5.0 I)$ can be slightly too restrictive on normalized features.
3.  **Task Complexity in Real-World Evaluation**:
    While ResNet-18 is a real model, the tasks (MNIST, Fashion-MNIST, CIFAR-10) are relatively simple toy vision datasets. Validation on more complex visual benchmarks (e.g., VTAB-1k) or NLP benchmarks (e.g., GLUE) on modern large models (e.g., ViT-B/16 or Llama-3-8B) would strengthen the empirical claims. The authors address this by providing a highly detailed roadmap in Section 5.1, but the lack of actual empirical execution on these large benchmarks is a minor limitation.
4.  **Calibration Requirements**:
    The method depends on having 16 calibration samples of known identity offline per task. While standard for ZCA/SABLE, this requirement should be explicitly mentioned as a prerequisite.
