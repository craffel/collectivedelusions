# 5. Impact and Presentation Analysis

## Major Strengths
1. **Exceptional Mathematical Rigor:** The theoretical derivation of the subspace-projected full non-diagonal Hessian curvature, the Ridge/Lasso closed-form solutions, and the formal error bounds represent a masterclass in mathematical formulation.
2. **Innovative Gradient Subtraction:** The gradient subtraction scheme for finite-difference estimation is a highly clever, practical, and mathematically sound contribution that resolves the standard $1/\epsilon$ numerical instability under non-zero residual expert gradients.
3. **Intellectual Honesty and Transparency:** The authors are highly commendable for their deep, transparent, and rigorous discussion of limitations. They do not hide their negative results; instead, they mathematically formalize the local-global optimization gap (Appendix B.4), detail the failure mechanics of their block Gauss-Seidel coordination (Section 4.5), and admit the severe ill-conditioning risks of Layer 13.
4. **Writing and Presentation Quality:** The paper is exceptionally well-written, with high-quality tables, clear section transitions, and a highly polished narrative.

## Major Areas for Improvement
1. **Addressing the Practical Utility Gap:** The most critical weakness is that the proposed method fails to consistently outperform standard Task Arithmetic. The authors must bridge this local-global optimization gap—perhaps by incorporating higher-order tensor derivatives or dynamic zero-order scale corrections—before claiming ACM is a superior paradigm for physical model merging.
2. **Expanding Experimental Scale and Diversity:** The authors must evaluate ACM on realistic, large-scale architectures (such as ViT-Base/Large, ResNet-50, or RoBERTa/LLaMA backbones) and complex downstream datasets (e.g., ImageNet, GLUE) to prove that their theoretical scaling analysis holds empirically.
3. **Ensuring a Fair Baseline Comparison:** The Test-Time Adaptation baselines should be evaluated under their standard, recommended deployment settings (with larger target streams) to avoid a strawman setup where they are forced to collapse due to sample size constraints.

## Potential Impact and Significance
* **Theoretical Significance: High.** The paper provides a valuable, mathematically rigorous framework that formalizes the loss-landscape geometry of model merging. The derivations of the local-global gap bound and the cross-layer block coupling mismatch are landmark insights that will be highly useful for researchers seeking to bring mathematical guarantees to weight consolidation.
* **Practical Impact: Low.** Because the method is highly complex, requires calibration batching, and fails to consistently beat a simple uniform scaling baseline (Task Arithmetic), practitioners and engineers are highly unlikely to adopt ACM in its current form. It remains a beautiful mathematical theory that fails to translate into a superior physical solution.
