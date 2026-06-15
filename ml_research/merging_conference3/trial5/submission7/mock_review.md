# Mock Review

**Title:** Pruned Gradient Merging (PG-Merge): Deconstructing Complexity in Test-Time Model Fusion  
**Reviewer Recommendation:** Accept (Score: 5)  
**Soundness:** Excellent  
**Presentation:** Excellent  
**Significance:** Good  
**Originality:** Good  

---

## 1. Summary of the Paper
The paper addresses the **Overfitting-Optimizer Paradox** in active test-time model merging, where unconstrained on-the-fly optimization of layer-wise merging coefficients on small, unlabeled test streams leads to severe transductive overfitting and representational decay. Critiquing the rapid escalation of complexity in recent SOTA regularizers (such as the auxiliary distance penalties in RegCalMerge or the rigid geometric polynomial trajectories in PolyMerge), the authors propose **Pruned Gradient Merging (PG-Merge)**. Guided by the principle of Occam's razor, PG-Merge is a minimalist, training-free, and non-parametric approach that restricts the optimizer's active search space using a dynamic, binary sparse gradient mask. By sorting absolute gradient magnitudes and freezing the vast majority ($85\%$--$95\%$) of coefficients—updating only the top-$p\%$ most sensitive ones—the method acts as an online low-pass filter to prevent parameter drift and preserve the multi-task capabilities of merged networks.

---

## 2. Strengths
1. **High Conceptual Elegance and Simplicity:** Challenging the increasingly convoluted trajectory of model merging regularizers is a highly valuable and commendable direction. The proposed sparse gradient mask is intuitive, training-free, and mathematically clean, demonstrating that outstanding generalization can be achieved without introducing auxiliary loss terms, delicate penalty hyperparameters ($\lambda$), or complex geometric projections.
2. **Rigorous Empirical Foundation:** Unlike earlier toy setups, the evaluation uses a standard Vision Transformer (`vit_tiny`) across four diverse visual classification tasks (MNIST, FashionMNIST, CIFAR-10, SVHN) with properly converged, high-performing specialized expert models (Expert Ceiling of $78.08\%$). This lends high scientific validity to the results.
3. **Prevention of Momentum Leakage:** The identification of historical momentum leakage in advanced optimizers like Adam and the introduction of the strict post-update parameter projection (Equation 15) is a highly professional and technically rigorous solution that ensures masked coordinates remain mathematically frozen.
4. **Strong Performance Gains:** The results in Table 1 demonstrate that under the highly sparse setting of $p=0.05$ (updating only $\approx 3$ coefficients per step), PG-Merge achieves the highest Joint Mean Accuracy of **$62.70\%$**, outperforming unregularized AdaMerging ($61.08\%$), static Uniform Merging ($62.16\%$), and matching or exceeding the highly complex RegCalMerge ($62.35\%$).
5. **Outstanding Critique of Rigid Subspaces:** The paper provides a clear explanation and empirical verification of why rigid, pre-defined geometric subspaces (like PolyMerge) catastrophically fail ($46.97\%$) when merging fully converged experts with complex, non-linear layer-wise gradient relationships.

---

## 3. Weaknesses & Constructive Suggestions

### 1. Technical Nuance: Optimizer State Mismatch (Adam Momentum Buffer Decay)
While the post-update projection step (Equation 15) successfully overwrites updated coefficients to keep them frozen in weight space, it introduces an internal state mismatch within the Adam optimizer:
*   **The Issue:** The Adam optimizer's running moment vectors ($m$ and $v$) are still updated with zero gradients for the masked-out coordinates at each step.
*   **The Consequence:** Over multiple adaptation steps, the momentum of frozen parameters will progressively decay toward zero. When a parameter eventually becomes active again (due to a high gradient magnitude), its update step will be severely dampened because its momentum buffer has been depleted.
*   **Suggestion:** A more theoretically sound approach would be to freeze the optimizer state update entirely for the masked coordinates, rather than feeding zero gradients and overwriting weights post-update. Alternatively, the authors should ablate the choice of optimizer and try standard SGD without momentum. Standard SGD completely bypasses momentum leakage, eliminates the need for Equation 15, and would make the method even simpler and more elegant. (Note: The authors' inclusion of Section Appendix A discussing this nuance is highly appreciated and shows deep theoretical self-awareness.)

### 2. Hyperparameter Sensitivity: Selecting $p$ at Test-Time
The paper claims PG-Merge is "hyperparameter-lean" and "non-parametric," but the sparsity ratio $p$ is a critical hyperparameter that significantly impacts performance (average accuracy drops from $62.70\%$ at $p=0.05$ to $61.08\%$ at $p=1.0$).
*   **The Issue:** In a real-world online test-time adaptation scenario, practitioners do not have access to ground-truth validation labels to tune $p$ or select the optimal $5\%$ sweet spot.
*   **Suggestion:** The authors should discuss how a practitioner can select or adapt $p$ on-the-fly without labels. For example, can $p$ be chosen based on the distribution or variance of the raw gradients, or does a default value of $p \in [0.05, 0.15]$ generalize well across different backbones and tasks?

### 3. Performance Degradation on SVHN under High Sparsity
While PG-Merge ($p=0.05$) achieves peak performance on MNIST, FashionMNIST, and average scores, its accuracy on **SVHN** drops to **$32.03\%$**, which is worse than static Uniform Merging ($33.20\%$) and unconstrained AdaMerging ($35.16\%$).
*   In contrast, PolyMerge achieves its highest accuracy on SVHN ($40.43\%$) despite collapsing elsewhere.
*   **Suggestion:** This suggests that highly distinct or complex domains like SVHN may require more active coordinates to adapt effectively, or that their gradient signals are noisier. Some qualitative analysis or discussion of this SVHN anomaly would add significant depth to the paper.

### 4. Lack of Optimization/TTA Trajectory Curves
The paper would benefit greatly from plotting the prediction entropy (adaptation loss) and joint test accuracy over the 100 adaptation steps.
*   **Suggestion:** Including these plots in the appendix or main text would provide visual proof of the "Overfitting-Optimizer Paradox" in action, and illustrate how PG-Merge stabilizes the optimization path over time compared to the rapid representational collapse of unconstrained AdaMerging.

---

## 4. Questions for the Authors
1. **Standard SGD Ablation:** Have you experimented with using standard SGD without momentum for PG-Merge? If standard SGD achieves comparable performance, it would eliminate the need for the post-update projection step (Equation 15), making the method even simpler.
2. **Mask Stability:** How stable is the active set of coefficients selected by the dynamic mask $M_{k, l}$ across the 100 adaptation steps? Does the set of updated coordinates fluctuate wildly due to gradient noise on the tiny 64-sample calibration batch, or does it converge to a stable subset?
3. **SVHN Behavior:** Why does SVHN perform worse under high sparsity ($32.03\%$) than under unconstrained AdaMerging ($35.16\%$), unlike MNIST and FashionMNIST which peak at $p=0.05$?
