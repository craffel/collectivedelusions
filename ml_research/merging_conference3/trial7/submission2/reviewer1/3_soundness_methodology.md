# Intermediate Review Evaluation: Soundness and Methodology (3_soundness_methodology.md)

This document provides a critical evaluation of the clarity, appropriateness, reproducibility, and mathematical soundness of the methodology proposed in the paper, detailing key technical achievements as well as potential flaws.

---

## 1. Description Clarity and Appropriate Methods
* **Notation and Mathematical Presentation:** The methodology is written with exemplary clarity and mathematical rigor. The authors provide a complete notation table in Appendix Table 1 to guide the reader through the equations, which is highly appreciated.
* **Methodological Choices:** 
  * The transition from standard parametric routers to a parameter-free Riemannian formulation is well-motivated and logically structured.
  * Constructing the local metric tensor from diagonal empirical Fisher Information is highly appropriate. The derivation showing that dFIM represents the inverse coordinate noise variance ($F_j = 1/\sigma_j^2$) establishes a elegant, category-error-free information-geometric link.
  * The pooling of class-conditional variances ($\sigma_{k, j}^2$) to calculate dFIM is a crucial, high-impact choice. It successfully isolates pure coordinate noise from class centroid discriminative spread, preventing coordinates with high inter-class centroid spread from being artificially suppressed.
  * Micro-Batch Homogenization (MBH) and Class-Size Scaling Calibration (CSC) are highly appropriate and address the key vulnerabilities of batch-level non-stationarity and extreme value bias under asymmetric task vocabularies.

---

## 2. Mathematical Soundness and Potential Flaws

As a theory-minded reviewer, we carefully scrutinized the mathematical derivations in the main paper and appendix. We identified one definite mathematical error (a sign error) and several important conceptual assumptions.

### A. Definite Mathematical Sign Error in Appendix 1.2 (Rectified Activation FIM)
In Appendix Section 1.2, the authors derive the Fisher Information of a rectified Gaussian coordinate to prove the robustness of their diagonal Fisher formulation under ReLU/GELU sparsity. 
In Equation (36), the continuous component of the integral is evaluated using integration by parts:
$$\int_{-\mu_j/\sigma_j}^\infty t^2 \phi(t) dt = \left[ -t\phi(t) \right]_{-\mu_j/\sigma_j}^\infty + \int_{-\mu_j/\sigma_j}^\infty \phi(t) dt$$
Let us evaluate the boundary term $\left[ -t\phi(t) \right]_{-\mu_j/\sigma_j}^\infty$:
* The upper limit is $\lim_{t \to \infty} -t \phi(t) = 0$.
* The lower limit is evaluated at $t = -\mu_j/\sigma_j$.
* Therefore:
$$\left[ -t\phi(t) \right]_{-\mu_j/\sigma_j}^\infty = (0) - \left( - \left(-\frac{\mu_j}{\sigma_j}\right) \phi\left(-\frac{\mu_j}{\sigma_j}\right) \right) = -\frac{\mu_j}{\sigma_j} \phi\left(-\frac{\mu_j}{\sigma_j}\right)$$

However, in Equation (37), the authors write:
$$I_{\text{continuous}} = \frac{1}{\sigma_j^2} \left[ 1 - \Phi\left(-\frac{\mu_j}{\sigma_j}\right) + \frac{\mu_j}{\sigma_j} \phi\left(-\frac{\mu_j}{\sigma_j}\right) \right]$$
which features a **positive sign** on the boundary term instead of a negative sign. Consequently, the total Fisher Information expression in Equation (39) is also written with this incorrect sign:
$$F_{j} = \frac{1}{\sigma_j^2} \left[ 1 - \Phi\left(-\frac{\mu_j}{\sigma_j}\right) + \frac{\mu_j}{\sigma_j} \phi\left(-\frac{\mu_j}{\sigma_j}\right) + \frac{\phi(-\mu_j/\sigma_j)^2}{\Phi(-\mu_j/\sigma_j)} \right]$$

**Corrected Expression:**
The mathematically correct total Fisher Information for the rectified coordinate is:
$$F_{j} = \frac{1}{\sigma_j^2} \left[ 1 - \Phi\left(-\frac{\mu_j}{\sigma_j}\right) - \frac{\mu_j}{\sigma_j} \phi\left(-\frac{\mu_j}{\sigma_j}\right) + \frac{\phi(-\mu_j/\sigma_j)^2}{\Phi(-\mu_j/\sigma_j)} \right]$$

**Impact Assessment:**
This is a standard sign error in integration by parts. Crucially, the final asymptotic scaling result ($F_j \propto 1/\sigma_j^2$ as $\sigma_j \to \infty$) remains correct, because the term $\frac{\mu_j}{\sigma_j} \phi(-\mu_j/\sigma_j)$ still vanishes as $\sigma_j \to \infty$. Thus, the overall conceptual robustness of the diagonal Fisher coordinate filter holds. However, the exact mathematical expression of rectified Fisher Information contains an error that the authors must correct.

### B. The Zero Global Centroid Assumption and Translation Bias (Appendix 1.3)
The dual-space proof demonstrating that classification weights $W'_{k, c}$ align with the true activation means $\mu_{k, c}$ relies on the assumption of a zero global centroid ($\sum_{c'} \mu_{k, c'} = 0$).
* **Flaw/Limitation:** If the representation space features a large, non-zero global mean vector $M_k$, the classifier weights instead align with the mean-centered representation vector $\mu_{k, c} - M_k$. This introduces a **translation bias** which warps raw coordinate similarities.
* **Resolution:** The authors deserve significant credit for proactively addressing this. They explicitly integrate a global **pre-calibration mean-centering step** ($z' = z - \bar{z}_{\text{cal}}$) on both calibration and test activations, eliminating global mean shifts and preserving the geometric validity of the Riemannian inner product.

### C. Axis-Aligned Noise Assumption of Diagonal Fisher
By utilizing a *diagonal* Fisher Information Matrix, the framework implicitly assumes that coordinates are independent and that noise and task-irrelevant variations are aligned with the coordinate axes.
* **Flaw/Limitation:** In real-world networks, noise is rotated and highly correlated. Under rotated, non-axis-aligned noise, diagonal Fisher collapses (as shown in Table 5).
* **Resolution:** The authors provide a robust response in Section 4.6, evaluating an on-the-fly Covariance EVD shrinkage estimator (**FIOSR-Online**). This shrinkage estimator ($\alpha=0.2$) successfully stabilizes coordinates and outperforms the flat Cosine baseline under rotated noise, though full online EVD scales poorly to high-dimensional representation spaces ($d \ge 1024$).

### D. Extreme Value Calibration (CSC) Alternative-Hypothesis Penalty
The CSC divisor $\sqrt{2\log C_k / d}$ is derived under the null hypothesis (independent random noise variables).
* **Flaw/Limitation:** Under the alternative hypothesis (a true positive class prototype match), applying this divisor introduces an **asymmetrical penalty** on larger class vocabularies. If a 10-class task and a 4-class task both achieve an identical, genuine prototype match of similarity $0.8$, the 10-class task's score will be divided by a larger value, creating an artificial bias favoring smaller-vocabulary tasks when true matches occur.

---

## 3. Reproducibility
* **Code and Parameters:** The paper specifies all hyperparameters clearly ($\beta = 0.5$, $\gamma = 0.7$, $\tau=0.001$, calibration size $N_c=16$, dimension $D=192$, layers $L=14$, experts $K=4$, and asymmetric classes $C=[10, 10, 10, 4]$).
* **Statistical Significance:** The primary quantitative results and stress tests are evaluated across 10 independent random seeds (seeds 42 to 51) and report standard deviations, ensuring high reproducibility and statistical rigor.
* **Physical Validation:** The authors provide a detailed step-by-step end-to-end roadmap for physical deployment and successfully validate the framework end-to-end on a physical pre-trained ResNet-18 model on real images (MNIST, FashionMNIST, and SVHN), which greatly strengthens empirical reproducibility and validity.
