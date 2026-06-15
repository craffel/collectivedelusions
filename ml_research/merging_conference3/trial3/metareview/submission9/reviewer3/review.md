# Peer Review

## 1. Summary of the Paper
This paper introduces **FlatQ-Merge** (Flatness-Aware Quantization-Aware Model Merging), a comprehensive empirical study and mathematical framework investigating the role of expert loss landscape geometry (flatness) in low-precision model merging under post-training quantization (PTQ). 

To control expert flatness, the authors fine-tune task-specific experts using Sharpness-Aware Minimization (SAM) across multiple perturbation radii ($\rho$). They then merge these experts dynamically using layer-wise blending coefficients ($\Lambda$) and compress the model using per-channel symmetric uniform weight post-training quantization (PTQ) to 8-bit or extreme 4-bit precision. To adapt the merging coefficients without access to ground-truth labels at test-time, they optimize $\Lambda$ via the Straight-Through Estimator (STE) by minimizing joint prediction entropy on a small, unlabeled calibration dataset. 

Through extensive multi-axial sweeps on a Vision Transformer (\texttt{vit\_tiny}) backbone across four diverse tasks (MNIST, FashionMNIST, CIFAR-10, SVHN) across three random seeds, the authors discover a precision-dependent **Flatness-Robustness Synergy**: under moderate 8-bit noise, expert flatness yields negligible gains, but under extreme 4-bit rounding noise, pre-training with an optimal SAM radius ($\rho=0.05$) yields a substantial **+7.44% absolute multi-task accuracy improvement** over sharp SGD-trained experts. 

The authors also identify a critical non-linear **Over-Perturbation Threshold** at $\rho \ge 0.1$, which they geometrically profile and explain as "representation convergence"—where excessive SAM forces divergent tasks to converge to the same wide local minima of the pre-trained base, eroding specialized features and collapsing multi-task parameter fusion. Finally, they show that merging flat experts with static uniform weights ($\rho=0.05$, NaiveUniform) outperforms performing sophisticated test-time adaptation on sharp experts ($\rho=0.0$, FlatQ-Merge) by **+6.03%**, demonstrating that pre-merging landscape conditioning is far more critical than downstream adaptation algorithms.

---

## 2. Strengths (Soundness, Significance, and Originality)

### A. Exceptional Conceptual Leaps and Originality
The paper represents a **significant and highly original conceptual advance** in the fields of loss landscape geometry and model compression:
*   **Elegant Curvature Projection Bridge:** The authors establish a beautiful and mathematically rigorous link showing that the low-dimensional coefficient-space Hessian $H^l_{\Lambda}$ is exactly the projection of the weight-space Hessian $H^l_{\theta}$ onto the subspace spanned by the task vectors ($H^l_{\Lambda} = (T^l)^T H^l_{\theta} T^l$). They prove that minimizing weight-space curvature bounds and flattens both the trace and spectral norm of the coefficient-space Hessian ($\lambda_{\max}(H^l_{\Lambda}) \le \lambda_{\max}(H^l_{\theta}) \cdot \|T^l\|_2^2$). This formal connection is highly satisfying and conceptually elegant.
*   **Paradigm-Shifting Takeaway:** The discovery that pre-merging expert loss landscape conditioning (flatness) dominates downstream test-time adaptation sophistication (with NaiveUniform flat experts beating test-time optimized sharp experts by **+6.03%** in 4-bit) challenges the current heavy focus of the model merging community on designing increasingly complex post-merging blending algorithms. It shows that conditioning the expert geometry beforehand is a far more powerful lever.
*   **Geometric Discovery of Representation Convergence:** Explaining the non-linear over-perturbation threshold ($\rho \ge 0.1$) via task vector cosine similarity is highly original. Uncovering that excessively large perturbation radii force divergent task-specific experts to converge to the *same* wide local minima of the pre-trained base ("representation convergence") provides a highly valuable conceptual insight for researchers.

### B. Outstanding Depth of Empirical Validation and High-Signal Inquiries
The paper features an exceptionally thorough and creative suite of ablations that systematically investigate and validate every dimension of the framework:
*   **Adversarial Flatness vs. Average Flatness (SAM vs. SWA):** The authors show that while SWA (trajectory averaging) works well under moderate (8-bit) noise, it fails completely under extreme (4-bit) noise, proving that SAM's coordinate-wise adversarial formulation is uniquely necessary for low-bit robustness.
*   **Independent Clipping vs. Convex Softmax Combination:** Directly demonstrates that independent clipping bounds $[0, 1]$ dramatically outperform normalized convex combinations (by +8.20% in 8-bit and +3.03% in 4-bit) and provides a fine-grained layer-wise coefficient stability analysis showing that test-time adaptation operates on a stable, sub-pixel manifold without exploiting boundaries.
*   **Implicit Regularization vs. High-Dimensional TENT:** Proves that FlatQ-Merge's low-dimensional coefficient bottleneck (56 parameters) acts as a powerful implicit structural regularizer that prevents degenerate class collapse without requiring explicit diversity penalties (where TENT-style optimization of all 5.7M weights completely collapses to random guessing).
*   **Direct Weight-Space Curvature Measurement:** Actively measures Hessian trace proxy via random Gaussian parameter perturbations, demonstrating an **$8\times$ reduction in weight-space curvature** for SAM ($\rho=0.05$) experts. Cruvature metrics correlate perfectly with 4-bit merging resilience, beautifully closing the theoretical loop.
*   **DARE Compatibility:** Demonstrates that FlatQ-Merge is fully orthogonal to and compatible with advanced parameter-pruning and sign-conflict resolution techniques like DARE, yielding a +5.96% absolute gain when combined.

### C. Edge-Deployment Relevance
The paper demonstrates clear systems-level advantages. FlatQ-Merge keeps weights strictly in 4-bit compressed form throughout adaptation and inference, reducing peak adaptation memory by **$8\times$** ($2.85\text{MB}$ vs $22.8\text{MB}$ in FP32) compared to unquantized post-hoc adaptation, resolving a critical physical memory bottleneck on resource-constrained edge hardware.

---

## 3. Weaknesses and Areas for Improvement

While the paper is outstanding, addressing the following areas would further strengthen the work:
*   **Scale of Backbone and Evaluation:** The empirical evaluation is restricted to a relatively small backbone (\texttt{vit\_tiny}) and a restricted pre-training data budget (512 images per task). Although this scale is fully justified to make the extensive, multi-axial grid sweeps computationally tractable, demonstrating the flatness-robustness synergy on a larger model (e.g., ViT-Base or ResNet-50) would provide even greater empirical weight.
*   **Activation Quantization:** The framework focuses on weight-only post-training quantization (W8A32 and W4A32). In extreme edge environments, integer-only execution requires joint weight-activation quantization (e.g., W8A8 or W4A4). While the authors include an excellent theoretical discussion of how SAM-induced Lipschitz bounding naturally suppresses activation outliers and mitigates activation PTQ noise, including even a small empirical pilot under joint quantization would be a highly valuable addition.

These limitations are minor, and the authors are highly transparent and detailed in addressing them in their Limitations section, providing a clear blueprint for scaling and future work.

---

## 4. Evaluation Criteria Ratings

### Soundness: Excellent
The methodology is exceptionally sound, mathematically rigorous, and supported by flawless theoretical and empirical arguments. Every design choice (independent clipping, 56-parameter optimization, STE) is thoroughly justified and backed by comprehensive ablations. The reproducibility of the work is outstanding.

### Presentation: Excellent
The paper is exceptionally well-structured, clear, and professional. Complex mathematical and geometric ideas (Hessian projections, representation convergence, piecewise-constant loss landscapes, SWA vs. SAM coordinate-wise properties) are explained with exceptional clarity, precision, and intuition. Tables and equations are structured cleanly.

### Significance: Excellent
This work addresses a highly practical and relevant problem (low-precision model merging on resource-constrained hardware) and advances our understanding of loss landscape geometry in parameter-space fusion. By proving that pre-merging expert geometry dominates downstream test-time optimization, it will likely guide future researchers to focus more on pre-training geometries, representing a highly influential contribution.

### Originality: Excellent
The paper provides exceptional, deep insights and mathematical formalizations that are highly original and ambitious. Connecting weight-space curvature to coefficient-space curvature via projection, discovering "representation convergence," isolating why adversarial flatness is uniquely necessary for low-bit quantization, and direct weight-space curvature profiling are all highly original, high-signal contributions.

---

## 5. Overall Recommendation
**6: Strong Accept**

**Justification:** 
This is a technically flawless, exceptionally well-written, and deeply insightful paper that makes a major conceptual contribution to the machine learning community. It elegantly bridges loss landscape geometry, model compression, and parameter-space fusion with flawless mathematical proofs and overwhelming empirical support. The paper features an outstanding level of ablation depth and delivers profound, paradigm-shifting takeaways that challenge current research focus and open up exciting new avenues for deploying robust, parameter-fused, and highly compressed models. It fully deserves a Strong Accept at a top-tier machine learning conference.
