# Soundness & Methodology Critique: GranMerge

An in-depth analysis of the paper's methodology reveals excellent scientific rigor, exemplary transparency, and highly thorough reasoning.

## 1. Transparent Evaluation of the "Low-Fidelity Expert" Regime
The paper evaluates its framework in a resource-constrained scenario using a 12-layer Vision Transformer (\texttt{ViTTiny}) with model dimension $d_{\text{model}}=64$. The expert test accuracies are:
*   **MNIST Expert Accuracy:** 61.03%
*   **FashionMNIST Expert Accuracy:** 62.47%
*   **CIFAR-10 Expert Accuracy:** 24.93%
*   **SVHN Expert Accuracy:** 17.50%
*   **Overall Mean:** 41.48%

### Critique & Appreciation:
In standard model merging papers, expert models are fully converged (e.g., >95% accuracy). However, the authors are exceptionally transparent about this constraint, dedicating an entire **Limitations and Scope** section to it. They frame this setup as an **extreme, low-resource edge warm-start setting** designed to serve as an amplified "stress test" for deconstructing transductive overfitting dynamics. 
This is a highly commendable and scientifically honest choice. In poorly converged or low-fidelity experts, task vectors contain higher levels of high-frequency parameter noise, which amplifies vulnerability to transductive overfitting. This makes the experimental setup a rigorous sandbox for analyzing the limits of test-time adaptation, and the authors openly discuss how these boundaries might shift for high-fidelity models and large foundation models.

## 2. Rigorous and Nuanced Interpretation of 1+1 ES Dynamics
The paper demonstrates outstanding intellectual honesty in interpreting why zero-order 1+1 Evolution Strategies (ES) maintain higher test-set generalization than Adam under high-dimensional settings. Rather than presenting 1+1 ES as a "magic bullet," the authors propose and contrast two competing yet complementary lenses:
1.  **Isotropic Implicit Regularization:** Isotropic random mutations are naturally self-bounding, preventing the search from exploiting coordinate-aligned transductive noise.
2.  **Optimization Sluggishness (Curse of Dimensionality):** The authors present a highly compelling mathematical explanation: optimizing 288 parameters (Level 5) using 1+1 ES for only 100 steps is extremely inefficient. Because ES is sluggish and fails to optimize, it remains stuck near its initialization (the uniform blending scale, which is close to the static uniform baseline of 30.41%). 

This "sluggishness hypothesis" is beautifully supported by the empirical trends in Table 1, where ES performs worst at Level 1 (where the search space is small and it successfully optimizes/overfits, yielding 24.84%) and steadily rises back toward the uniform baseline as dimensionality increases. Presenting this alternative explanation is exceptionally rigorous and represents high-quality academic standards.

## 3. Clear and Reproducible Hyperparameter Disclosures
In Section 4.1, the authors explicitly disclose the exact joint regularization scale ($\beta = 1.0$) and depth balance ($\gamma = 0.2$), along with further breakdowns in Appendix A.3. This resolves any previous concerns regarding reproducibility, ensuring that an expert reader has all necessary parameters to replicate the studies.
