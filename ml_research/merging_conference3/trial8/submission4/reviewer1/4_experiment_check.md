# 4. Experimental Setup and Evaluation Check

This section provides a critical evaluation of the paper's experimental setup, baselines, and whether the results support the main claims.

## 1. Evaluation of the Experimental Setup

The paper evaluates the proposed approach on two distinct settings:
1. **The Coordinate Sandbox:** A custom, synthetic 14-layer, 192-dimensional analytical simulation environment designed to model 4 tasks of varying noise profiles.
2. **Real-Image Serving Benchmark:** A vision pipeline utilizing a pre-trained ResNet-18 frozen backbone on MNIST, Fashion-MNIST, and CIFAR-10.

### Criticisms of the Setup
* **Highly Artificial Toy Environments:** The primary evaluation environment is the **Coordinate Sandbox**, which is a highly simplified, synthetic Gaussian-noised toy setting. While it allows exact mathematical control over representation noise and anisotropy, success in an artificial simulation does not guarantee performance in real-world large-scale transformer serving systems (such as Llama-3-8B).
* **Extremely Small Calibration Budgets:** The paper utilizes ultra-low data regimes ($N_c=16$ total per task). While calibrating on tiny splits is interesting, it exacerbates ensembling variance, leading to large standard deviations in the reported results.
* **Low Task Cardinality ($K=4$ or $K=3$):** The experiments are limited to 3 or 4 tasks. Modern PEFT multi-tenant serving systems are designed to scale to dozens or hundreds of concurrent specialized adapters. The low-dimensional parameter space ($K \le 4$) makes it difficult to assess how the methods scale in high-dimensional serving registries.

---

## 2. Baselines Comparison

The authors compare their method against a wide variety of static and dynamic baselines, including Uniform Merging, QWS-Merge, Linear Router (Reg), PFSR, SABLE (Raw/Block/PCA), and Temp-Only ERM.
* **Appropriateness:** The selection of baselines is excellent. Including standard, unregularized **Empirical Risk Minimization (ERM)** is crucial, as it isolates the effect of the PAC-Bayesian KL complexity penalty on the log-temperature parameters.
* **The SABLE Comparison Disadvantage:** SABLE (SEP-Block), which is an uncalibrated baseline using a static, uniform hand-tuned temperature scale ($\tau=0.05$) and zero optimization, outperforms PAC-ZCA (Block Ours) in the Sandbox:
  - On Orthogonal Manifolds, SABLE (SEP-Block) achieves **66.08% $\pm$ 0.78%** joint accuracy, compared to **64.16% $\pm$ 2.23%** for PAC-ZCA (Ours). SABLE is **+1.92% better** in mean performance and has a much lower variance (0.78% standard deviation vs. 2.23%).
  - On Overlapping Manifolds, SABLE (SEP-Block) achieves **63.98% $\pm$ 0.66%**, compared to **63.38% $\pm$ 2.58%** for PAC-ZCA (Ours). SABLE is **+0.60% better** in mean performance and has a fraction of the variance (0.66% standard deviation vs. 2.58%).
  *Practically, this means that the simple, uncalibrated, zero-training-overhead baseline is superior to the highly complex, theoretically certified method proposed in this paper.*

---

## 3. Do the Results Actually Support the Claims?

The authors claim that the PAC-Bayesian bound minimization (1) successfully prevents overfitting, (2) reduces ensembling variance compared to standard unregularized ERM, and (3) stabilizes temperature optimization in ultra-low data regimes. 

A close analysis of the empirical results reveals that these claims are **minimally supported or statistically insignificant**:

### A. Lack of Empirical Benefit over Simple ERM (Table 1)
In Table 1, the performance of PAC-ZCA (Ours) and standard, unregularized Temp-Only ERM is practically identical:
* **Block Features (Orthogonal):** Both PAC-ZCA and Temp-Only ERM achieve the exact same joint accuracy (**64.16%**). The claimed variance reduction is a negligible **0.05%** standard deviation drop (from 2.28% to 2.23%), which is statistically completely insignificant.
* **Block Features (Overlapping):** PAC-ZCA achieves **63.38% $\pm$ 2.58%** while Temp-Only ERM achieves **63.06% $\pm$ 2.32%**. The 0.32% improvement in mean is well within the standard deviations of 2.58% and 2.32%, rendering the difference statistically meaningless.
* **UN-PCA Features (Orthogonal & Overlapping):** Standard unregularized ERM actually **outperforms** PAC-ZCA on both orthogonal features (44.58% vs. 44.36%) and overlapping features (46.02% vs. 45.86%).

### B. Lack of Statistical Significance on Real-Image Servings (Table 2)
In Table 2, the authors evaluate ResNet-18 serving performance:
* PAC-ZCA (Isotropic Ours) achieves **70.87% $\pm$ 2.20%** joint accuracy.
* Temp-Only ERM achieves **69.47% $\pm$ 2.21%** joint accuracy.
* The absolute improvement is only **1.40%**, which is smaller than the standard deviations of both runs ($\approx 2.2\%$). Without further seeds or a larger test set, this improvement cannot be claimed as statistically significant.

### C. Sample Complexity Convergence (Table 3)
Table 3 shows how performance scales with the calibration budget $N_c$ per task:
* For $N_c = 8$: Temp-Only ERM is **81.70% $\pm$ 2.43%**, while PAC-ZCA is **81.46% $\pm$ 2.33%** (ERM is better in mean).
* For $N_c = 16$: Temp-Only ERM is **80.50% $\pm$ 1.77%**, while PAC-ZCA is **80.46% $\pm$ 1.77%** (ERM is better in mean).
* For $N_c = 128$: Both methods achieve the exact same accuracy (**80.62%**).
* PAC-ZCA only outperforms ERM in mean for $N_c = 32$ (by a trivial 0.04% absolute) and $N_c = 64$ (by 0.08% absolute). These differences are negligible and statistically indistinguishable.

## Conclusion on Experimental Evaluation

The empirical results presented in the paper do not justify the introduction of the massive PAC-Bayesian mathematical machinery. In nearly all configurations, the proposed PAC-ZCA framework performs almost identically to standard unregularized Empirical Risk Minimization, and is frequently outperformed by the simple, uncalibrated SABLE baseline in terms of both mean accuracy and standard deviation. The authors' claims of "empirical superiority" and "stabilized ensembling variance" are not supported by the data.
