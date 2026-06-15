# Technical Soundness & Methodology Check: BC-Router

## 1. Soundness & Methodology Assessment
**Rating: Excellent**

The technical soundness of this paper is outstanding. Rather than merely presenting a new heuristic and showing a selective performance improvement, the authors adopt a highly rigorous, methodologically transparent approach. They design controlled classical router variants (BL-Router, GLS-Router, BSigmoid-Router) specifically to isolate and deconstruct key architectural and optimization confounders.

### A. Controlled Confounder-Isolation Design
The methodology is exceptionally robust because each router is designed to answer a specific scientific question:
1.  **Over-Scaling Confounder (BL-Router):** By capping the routing coefficients at $\lambda_{max} = 0.3$, the authors test whether classical routing failures are driven by weight over-scaling (which pushes weights out of local convergence basins in Task Arithmetic).
2.  **Layer-wise Specialization Confounder (GLS-Router):** By introducing trainable layer-wise scaling amplitudes to a shared global routing head, the authors control for the layer-wise capacity advantage of SOTA wave models.
3.  **Zero-Sum Competitive Bottleneck (BSigmoid-Router):** By replacing Softmax with independent Sigmoids, the authors eliminate the zero-sum competitive routing budget during mixed-batch multi-task calibration.

### B. Intellectual Honesty in Deconstructing Baselines
What makes this paper methodologically superb is its intellectual honesty. When empirical results challenge their initial hypotheses, the authors do not hide them. Instead, they mathematically deconstruct the failure modes of their own baselines:
*   **Deconstructing BL-Router's Under-Scaling:** The authors identify that the standard Softmax-based scale ceiling ($\alpha_{k} = \lambda_{\max} \times \text{Softmax}(o_k)$) introduces a severe under-scaling bottleneck (capping the sum of coefficients at 0.3, whereas Uniform Merge has a sum of 1.2). They show that under uniform uncertainty, this restricts each task to a scale of 0.075, causing collapse. They use this mathematical insight to motivate the transition to the Softmax-free **BSigmoid-Router**, which decouples activations and restores the scale capacity.
*   **Deconstructing GLS-Router's Overfitting:** The authors explain that the extreme optimization sensitivity of the unregularized GLS-Router across calibration seeds is driven by the 56 layer-wise scaling parameters ($R_k^{(l)}$) overfitting to the tiny 64-sample calibration set. They highlight this as a critical optimization lesson: weight decay must be applied directly to layer-wise amplitudes to stabilize layer-specific routing.

### C. Standardized and Reproducible Calibration Protocol
The calibration and training protocols are perfectly standardized to ensure fairness:
*   All specialized task experts are trained to high convergence (establishing strong ceiling bounds of 92.8% to 100.0% accuracy) using AdamW on GPU.
*   All trainable methods are optimized on an identical, tiny 64-sample offline calibration set for exactly 100 steps of Adam with a learning rate of $1\times 10^{-2}$.
*   Standard L2 regularization (weight decay $\gamma = 1\times 10^{-4}$) is systematically evaluated to control for routing-head overfitting.

---

## 2. Methodology Strengths and Weaknesses
*   **Strengths:**
    *   Highly systematic, confounder-driven experimental design.
    *   Superb mathematical precision and transparency throughout the formulation.
    *   Outstanding academic honesty, deconstructing and explaining the failure modes of intermediate baselines.
    *   Elegant transfer of Mixture-of-Experts (MoE) token gating insights to parameter-space model merging.
*   **Weaknesses:**
    *   The paper's deconstruction is primarily demonstrated on a compact Vision Transformer backbone (`vit_tiny_patch16_224`), and verifying if these scale-regularization and sigmoidal routing insights generalize to larger-scale backbones (e.g., ViT-Base/Large, Swin Transformers, or LLMs) remains a path for future validation.
