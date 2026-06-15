# Peer Review Report: Dirichlet-PAC

## 1. Summary of the Paper
This paper introduces **Dirichlet-PAC**, a mathematically rigorous, simplex-constrained learning-theoretic framework for test-time multi-task model serving. The authors target the problem of dynamic activation blending of parameter-efficient expert adapters (e.g., LoRA) on a shared, frozen backbone. In real-world deployments, serve-time calibration streams are extremely data-scarce (often fewer than 64 samples per task), which causes standard unregularized Empirical Risk Minimization (ERM) to overfit to local transductive noise, leading to extreme, overconfident temperature routing.

To address this, Dirichlet-PAC stochastically models sample-specific ensembling weights directly as random vectors drawn from a Dirichlet posterior distribution over the probability simplex $\Delta^{K-1}$. Under McAllester's PAC-Bayesian theorem, the authors derive a closed-form generalization bound where the complexity penalty is the exact analytical Kullback-Leibler (KL) divergence between Dirichlet distributions on the simplex. For coordinate extraction, they propose **Subspace Energy Projection (SEP)**, which uses Singular Value Decomposition (SVD) on early-layer activations of calibration inputs to find task feature manifolds, projecting online queries onto these subspaces to extract coordinates normalized to the simplex. They propose a disjoint **Sample-Splitting** protocol to keep the prior data-independent, and establish uniform convergence over temperatures via discretization. 

Evaluation is conducted in a 14-layer simulated Analytical Coordinate Sandbox (ICS) and validated on pre-trained BERT backbones (\texttt{bert-tiny} to \texttt{bert-base-uncased}) using synthetic text classification tasks.

---

## 2. Strengths and Weaknesses

### Strengths
1. **Mathematical Rigor and Formal Foundations:** The paper is exceptionally well-grounded in statistical learning theory. Modeling ensembling uncertainty directly on the probability simplex via Dirichlet distributions is mathematically elegant and addresses a key limitation of prior unconstrained Gaussian approaches (such as PAC-ZCA).
2. **Watertight Prior Data-Independence:** The disjoint **Sample-Splitting** protocol ($\mathcal{S}_{\text{prior}}$ for SVD bases, $\mathcal{S}_{\text{opt}}$ for temperature optimization) is a highly sound design that strictly prevents prior-data contamination, resolving a common theoretical pitfall in test-time adaptation literature.
3. **Robust Feature Extraction Analysis:** Proposition 3.1 rigorously proves that Subspace Energy Projection (SEP) is mathematically scale-invariant and independent of any specific orthogonal basis representation. This provides a solid learning-theoretic justification for deploying the method on deeper and larger models.
4. **Strong Presentation and Clarity:** The manuscript is extremely clear, professional, and well-structured. The notation is consistent, and the step-by-step analytical derivation of the Dirichlet KL divergence in Appendix A is exceptionally thorough and correct.

### Weaknesses
1. **The Theoretical Surrogate Loss Gap:** The PAC-Bayesian bound is formulated and optimized using a linear proxy loss representing an expected expert-selection policy. However, the served model executes deep activation-space blending, which is passed through subsequent non-linear layers. This breaks the formal connection between the PAC-Bayesian generalization certificates and the physical served model.
2. **Underperforming Simple Static Baselines in the Real World:** On the larger, more realistic BERT backbones (BERT-Medium and BERT-Base), Dirichlet-PAC is consistently outperformed by a simple, static parameter average (Uniform Merging) by up to **6.67%**. This significantly undercuts the practical utility of the proposed optimization framework.
3. **Simulation-to-Reality Discrepancy:** In the custom 14-layer sandbox (ICS), Uniform Merging collapses to 46% accuracy, whereas in real-world BERT experiments, Uniform Merging achieves up to 99% accuracy. This indicates that the simulated environment is highly artificial and heavily biased against static merging, overstating the necessity of dynamic routing.
4. **Sensitivity to the Prior Temperature Hyperparameter ($\tau_0$):** In data-scarce settings, the KL penalty dominates the optimization, forcing learned temperatures to stick close to the prior $\tau_0$. Thus, the method's performance depends heavily on manual tuning of $\tau_0$, rather than truly "discovering" optimal task scales.

---

## 3. Ratings (Excellent, Good, Fair, Poor)

* **Soundness:** **Fair** (The mathematical derivations and SVD proofs are correct, but the theoretical "generalization guarantee" does not cover the actual non-linear activation-blended model that is served).
* **Presentation:** **Excellent** (The writing style, mathematical notation, and structuring of sections are superb; the step-by-step appendix proof is highly reproducible).
* **Significance:** **Fair** (While the theoretical contribution of simplex-constrained PAC-Bayes is valuable, the practical significance is limited because the method underperforms simple static baselines on real-world transformer backbones).
* **Originality:** **Good** (Treating ensembling weights as a Dirichlet distribution and using the closed-form Dirichlet KL penalty within PAC-Bayes is a novel and creative synthesis of existing techniques).

---

## 4. Overall Recommendation
* **Overall Rating:** **3: Weak Reject** (The paper has clear merits, particularly its elegant simplex-constrained formulation and rigorous SVD scale-invariance proof. However, the theoretical gap between the linear surrogate and non-linear blending, combined with the fact that the method underperforms simple static baselines on real-world BERT backbones, are significant weaknesses that require major revisions before publication).

---

## 5. Critical Flaws (Detailed Critique)

### Critical Flaw 1: The Theoretical Surrogate Loss Gap (Guarantees vs. Reality)
There is a fundamental mathematical gap between the theoretical guarantees provided by the PAC-Bayesian bound and the physical execution of the served model.
* **The Formulation:** The PAC-Bayesian generalization bound is formulated and optimized over a **linear calibration surrogate** (Equation 15):
  $$\widehat{\mathcal{L}}_{\text{cal}}(\boldsymbol{\tau}) = \frac{1}{N_{\text{opt}}} \sum_{b=1}^{N_{\text{opt}}} \left( 1 - \sum_{k=1}^K \alpha_{k, b} \cdot p_k(x_b) \right)$$
  which corresponds to the expected classification error of a stochastic *expert-selection* model.
* **The Reality:** The model actually served in production executes **activation-space blending** (Equation 4):
  $$h^{(l)} = h^{(l-1)} + \sum_{k=1}^K \alpha_{k, b} \cdot \Delta h_k^{(l)}$$
  which is propagated through subsequent non-linear blocks (such as attention layers, MLPs, layer norms) and final classification heads.
* **The Impact:** Because downstream layers are non-linear, the actual classification error of the blended activation network is *not* mathematically equal to the linear surrogate loss. Consequently, **the PAC-Bayesian bound does not formally guarantee the generalization of the physical activation-blending model.** It only guarantees the generalization of a linear expert-selection proxy. The authors gloss over this critical gap, which breaks the formal connection between their PAC-Bayesian certificates and the actual served network.

### Critical Flaw 2: Practical Significance Collapse on Real-World Backbones
The physical scale-up study on pre-trained BERT backbones reveals that **Dirichlet-PAC fails to outperform simple, zero-overhead static baselines on realistic models.**
* **The Performance Gap:** In Table 3, on `bert-medium`, standard `Uniform Merging` achieves **96.00% ± 3.89%** and `SABLE Norm` achieves **96.00% ± 3.89%**, while `Dirichlet-PAC (Ours)` underperforms them at **89.33% ± 3.89%** (lagging by **6.67%**). On `bert-base`, `Uniform Merging` achieves **95.33% ± 4.52%**, while `Dirichlet-PAC` lags at **92.00% ± 8.59%** (lagging by **3.33%**).
* **The Impact:** Uniform Merging is a zero-overhead baseline that requires *no* data splitting, *no* SVD subspace extraction, and *no* training/optimization epochs. In contrast, Dirichlet-PAC introduces substantial systems-level latency (splitting the scarce calibration set, computing SVDs, and running 100 Adam optimization epochs) only to lose in accuracy. This significantly undercuts the practical utility and justification of deploying Dirichlet-PAC.
* **Sandbox Bias:** There is a massive discrepancy between the simulated sandbox (ICS), where Uniform Merging collapses to **46.26%**, and the real-world BERT experiments, where it dominates at **94.00% to 99.33%**. This indicates that the simulated sandbox is highly artificial and heavily biased to exaggerate "representation interference" ($\eta = 0.05$) to overstate the necessity of dynamic routing.

### Critical Flaw 3: Extreme Sensitivity to the Manual Prior Temperature ($\tau_0$)
Dirichlet-PAC relies on an arbitrary prior temperature hyperparameter $\tau_0$ (set to $0.20$), which acts as an anchor for the temperature parameters.
* **The Mechanism:** In highly data-scarce settings (such as $N_{\text{opt}} = 8$ per task), the PAC-Bayesian complexity penalty ($D_{\text{KL}}$) completely dominates the optimization. Consequently, the learned temperatures $\boldsymbol{\tau}$ are heavily constrained and barely deviate from the prior (converging to $\approx 0.19$ as shown in the logs for Seed 42).
* **The Over-Regularization Trap:** This means that Dirichlet-PAC's superior performance over unregularized ERM (which suffers from "routing hijacking" and collapses to 67%) is not because it successfully "learns" optimal temperatures, but rather because **the KL penalty locks the temperatures to the manually-tuned prior temperature $\tau_0$.**
* **The Impact:** As shown in Ablation Study 3 (Table 4), if the prior temperature is poorly chosen (e.g., $\tau_0 = 0.05$), Dirichlet-PAC's accuracy collapses from $76.54\%$ down to $70.48\%$. Therefore, the model does not automatically resolve temperature calibration; it remains highly sensitive to the practitioner's manual choice of $\tau_0$.

---

## 6. Actionable Suggestions for Improvement
1. **Bridge the Theoretical Surrogate Gap:** The authors should either:
   * Perform Monte Carlo sampling of ensembling weights during optimization to evaluate the expected loss of the true non-linear blended network (though computationally heavier), OR
   * Explicitly bound the mathematical difference between the linear surrogate loss and the true blended non-linear loss, providing a formal correction term in the PAC-Bayesian bound.
2. **Evaluate on Harder, Real-World Benchmarks:** The BERT backbones are evaluated on highly simplistic, toy synthetic classification tasks where Uniform Merging trivially achieves 99% accuracy. The authors must validate their framework on standard, challenging multi-task benchmarks (such as GLUE/SuperGLUE tasks using LoRA adapters) where uniform parameter averaging actually suffers from task interference, which would demonstrate the true necessity of dynamic routing.
3. **Incorporate Covariance Regularization in SEP:** To prevent SVD underdetermined collapse as the expert registry $K$ scales under a fixed calibration budget, the authors should incorporate shrinkage estimators (such as Ledoit-Wolf covariance shrinkage) or randomized projection regularizers to stabilize the projection bases $V_k$.
4. **Compare against Continuous PAC-Bayes Bounds:** The authors should compare their discrete union-bound temperature optimization with continuous PAC-Bayesian bounds utilizing a continuous Gaussian hyper-prior over log-temperatures, which would bypass the coarse $|\Theta|$ discretization penalty and yield non-vacuous theoretical certificates.
