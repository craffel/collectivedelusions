# Intermediate Review Evaluation: 3. Soundness and Methodology Check

## Clarity of the Description
The mathematical exposition is **excellent**. The paper is exceptionally clear, well-structured, and rigorous. It defines the multi-tenant PEFT serving setting, details the "routing paradox" and why early-layer routing is required, and step-by-step builds the math for:
- Subspace Energy Projection (SEP) and its generalization to distributed non-orthogonal manifolds.
- The temperature-only Gibbs routing policy.
- Catoni's PAC-Bayesian bound optimization and its justification under unbounded Cross-Entropy loss.
- The localized Lipschitz bound on expected routing loss (Lemma 1, fully proved).
- The Lipschitz-Entropy Duality (Theorem 1, fully proved).
- The continuous-activation blending formulation and its theoretical discrepancy (the theory-practice gap, fully proved using Taylor's theorem).

The authors avoid hand-waving and instead provide rigorous proofs for every core theoretical claim.

---

## Appropriateness of Methods
1. **Strictly Temperature-Only Routing:** Training a complex high-dimensional routing network on a tiny calibration split ($N_c = 16$ per task) would cause immediate overfitting. Limiting the parameter space to $K$ temperature scales is highly appropriate and elegant.
2. **Active PAC-Bayesian Bound Minimization:** Direct optimization of Catoni's bound is theoretically justified for unbounded losses, replacing heuristic sweeps with a principled learning objective that bounds out-of-sample risk.
3. **Decoupled Calibration Splits:** SVD/PCA coordinate extraction on the same calibration set used to optimize temperatures would violate the i.i.d. assumption under McAllester's theorem. Partitioning the calibration set into disjoint splits ($\mathcal{C}^{\text{sub}}$ and $\mathcal{C}^{\text{opt}}$) is a highly appropriate and mathematically necessary step to preserve the integrity of the statistical learning guarantees.
4. **Unit-Norm PCA Subspace Projection (UN-PCA-SEP):** Standard unregularized PCA-SEP is highly vulnerable to noise alignment in small-sample regimes ($N_c \ll D$). Normalizing features to the unit $L_2$ sphere bounds coordinate values on $[0,1]$ and completely resolves the SVHN expert neglect collapse. This is an extremely clever, mathematically elegant solution.

---

## Potential Technical Flaws and Limitations
The paper is remarkably honest, rigorous, and mathematically transparent. It does not contain any obvious technical flaws, and it actively addresses its own limitations:

1. **Rigor-vs-Accuracy Trade-off under Decoupled Splits:**
   The paper transparently discusses that partitioning the tiny calibration set ($N_c=16$) into disjoint splits reduces the sample size for temperature optimization to $N_{\text{opt}}=8$ per task. This introduction of splitting variance is the "disjoint split penalty," which explain why the uncalibrated SABLE (SEP-Block) baseline—which avoids optimization and splitting variance entirely—can slightly outperform PAC-ZCA (SEP-Block) on raw block features (66.08% vs. 64.16%). This is a well-argued trade-off: in safety-critical edge serving, absolute learning-theoretic validity and certifiability are often more important than minor gains in empirical mean accuracy.
2. **Over-regularization Bottleneck:**
   PAC-ZCA (UN-PCA) slightly underperforms Temp-Only ERM (UN-PCA) by a tiny margin (e.g., 44.36% vs. 44.58% on orthogonal manifolds, and 45.86% vs. 46.02% on overlapping manifolds). The paper notes that this is due to an over-regularization bottleneck where the isotropic Gaussian prior centers the parameters near SABLE's uncalibrated physical scale. The authors propose several solid directions in future work (e.g., adaptive task-specific prior variances or optimizing the confidence parameter $\delta$) to address this.
3. **The Parameter $\beta$ in Catoni's Bound:**
   Catoni's bound introduces a positive real parameter $\beta > 0$ (set to $0.5$). While the bound is theoretically valid for any $\beta > 0$, the optimal choice of $\beta$ depends on the empirical risk and prior variance. The paper could discuss how sensitive the optimization is to the selection of $\beta$, or explore optimizing $\beta$ dynamically alongside the log-temperatures $\mathbf{w}$ as part of the training objective, which is a known technique in PAC-Bayesian literature.

---

## Reproducibility
The methodology is **highly reproducible**:
- The paper details all parameters of the 14-layer 192-dimensional analytical Coordinate Sandbox, including the exact task noise levels ($\sigma_0=0.01, \sigma_1=0.05, \sigma_2=0.28, \sigma_3=1.35$) and dimension allocations.
- It defines the exact architecture, routing layers, and expert training settings for the real-world ResNet-18 visual serving experiments.
- It reports mean and standard deviation over 5 random seeds for all configurations, ensuring that results are statistically significant and verifiable.
- It provides step-by-step algorithms and pseudocode-like explanations for the regularized subspace projection extensions and decoupled split protocols.
