# Peer Review: CR-PolySACM

**Paper Title:** CR-PolySACM: Clipping-Regularized Sharpness-Aware Subspace Model Merging for Robust Post-Training Quantization

---

## 1. Summary of the Paper
This paper presents a mathematically rigorous and empirically grounded investigation into the post-training quantization (PTQ) robustness of test-time adaptive model merging (TTA). The author identifies and analyzes a critical, overlooked vulnerability in existing adaptive merging paradigms: **Quantization-Operator Overfitting**, where unconstrained test-time coefficient optimization converges to extremely sharp local minima in continuous weight-space. While these sharp minima yield high performance in FP32, they are exceptionally fragile under downstream post-training quantization (PTQ) rounding noise, resulting in catastrophic representational collapse.

To resolve this bottleneck, the paper proposes **CR-PolySACM** (Clipping-Regularized Sharpness-Aware Subspace Model Merging), a unified framework that combines global structural constraints with local landscape flatness optimization:
1. **Differentiable Polynomial Subspace Parameterization (PolyMerge):** Blending coefficients are restricted to a low-degree polynomial of network depth, compressing the optimization search space from $L \times K$ independent layer-wise parameters (e.g., $56$ parameters) to exactly $3 \times K = 12$ polynomial coefficients. This global structural constraint prevents overfitting to small calibration streams and naturally shields the model against out-of-subspace noise.
2. **Quantization Noise and Polynomial Subspace Curvature Decomposition:** The author derives a second-order Taylor expansion decomposition showing that the quantization-induced loss gap is governed by a division of labor: test-time adaptation can only minimize and flat-map the controllable second-order in-subspace error, whereas out-of-subspace noise dominates at ultra-low precisions.
3. **The Task-Vector Norm Scale Pathology:** Standard sharpness-aware optimization (e.g., HessMerge or SACM) applies uniform perturbations in coefficient space. However, the author identifies a massive, 50-fold discrepancy in layer-wise task-vector norms on a Vision Transformer backbone, rendering standard unnormalized flatness regularizers completely blind to low-norm layers (like final layer norm, Layer 13). Conversely, unmitigated scale-invariant normalization triggers gradient explosion due to a massive $>2500\times$ scale multiplier.
4. **Clipping-Regularized SACM (CR-SACM):** CR-SACM clips task-vector norms to a robust minimum floor ($\beta = 0.10$), balancing scale sensitivity across layers and enabling the optimizer to successfully flatten low-norm layers without triggering singular gradient explosion.

Under aggressive 4-bit quantization, CR-PolySACM achieves a joint mean accuracy of **19.07%** across four visual domains, setting a new state of the art and outperforming the previous PolyMerge baseline (**18.10%**) by an absolute **+0.97%** (representing a relative improvement of over **5.3%**). Furthermore, their upgraded HessMerge baseline (using CR-SACM) consistently and significantly outperforms AdaMerging across all target precisions (+1.36% in FP32, rising from 49.12% to **50.48%**).

---

## 2. Strengths

### A. Theoretical Soundness & Mathematical Rigor
- **Noise Decomposition and In-Subspace Bounds:** The second-order Taylor expansion of the multi-task loss gap:
  $$
  \Delta \mathcal{L} \approx \nabla_W \mathcal{L}^T \delta_{\perp} + \frac{1}{2} \boldsymbol{\epsilon}^T \mathcal{H}_{\mathbf{p}} \boldsymbol{\epsilon} + \frac{1}{2} \delta_{\perp}^T \mathcal{H}_W \delta_{\perp}
  $$
  is highly elegant and theoretically profound. It uncovers that test-time adaptation can only minimize and flat-map the second-order in-subspace error ($\frac{1}{2} \boldsymbol{\epsilon}^T \mathcal{H}_{\mathbf{p}} \boldsymbol{\epsilon}$) and has zero control over the out-of-subspace noise components $\delta_{\perp}$. This explains the fundamental physical limits of weight-space flatness optimization and why unconstrained TTA methods collapse under low-bit precisions.
- **Physical Justification of the Norm Scale Pathology:** Identifying the 50-fold discrepancy in layer-wise task-vector norms on a Vision Transformer backbone provides a direct, empirically grounded explanation for why standard unnormalized sharpness-aware minimization fails.
- **Elegant Scale-Invariant Mapping:** The mathematical derivation of CR-SACM's perturbation is flawless. By scaling the perturbation inversely with the square of the clipped task-vector norm $(V_{\text{clipped}, k}^l)^2$, CR-SACM ensures that the resulting weight-space perturbation $\Delta W^l \approx \sum_k \epsilon_k^l \tau_k^l$ has a uniform, balanced magnitude across all layers. The introduction of the clipping threshold $\beta = 0.10$ is a robust and elegant way to prevent division-by-zero or gradient explosion.

### B. Conceptual Framing & Originality
- **Bridging Geometry and Deployment:** Framing the vulnerability of test-time adaptive merging as quantization-operator overfitting bridges the gap between test-time adaptation geometry and edge deployment constraints.
- **The Integrated Subspace-Flatness Paradigm:** Combining global structural polynomial subspace constraints (which prevent calibration-stream overfitting) with local flatness optimization (which minimizes in-subspace sensitivity) represents a highly original and effective "division of labor".

### C. Empirical Rigor & Scholarly Honesty
- **Comprehensive Evaluation Sweep:** The evaluation across six diverse, hardware-relevant quantization schemas is highly complete.
- **Exceptional Transparency:** The author is highly commended for their scholarly honesty. They explicitly discuss the "expert-to-merge drop" (-31.27% gap in FP32) as an inherent domain disconnect challenge in multi-task merging, and honestly frame their absolute INT4 results (19.07%) as a valuable scientific proof of concept rather than overclaiming production readiness. This significantly enhances the paper's scientific credibility.
- **Exhaustive Appendix Additions (Addressing Previous Technical Queries):**
  - **Appendix A.1 (Calibration size $N$):** Validates the transductive gradient generalization gap across $N \in \{8, 16, 32, 64, 128\}$.
  - **Appendix A.1 (Percentile-Based Blueprint):** Successfully validates the automated percentile-based blueprint on the ViT backbone, where setting $\beta$ to the 10th percentile yields $\beta \approx 0.098$ and an accuracy of **19.05%**, matching the hand-tuned baseline. This provides a direct path for automated, parameter-free scaling to larger models.
  - **Appendix A.2 (Wall-clock overhead):** Shows that CR-PolySACM takes only **1.56 seconds** for 40 TTA steps, representing a negligible $+1.3\%$ overhead over standard AdaMerging while delivering a massive **52.8$\times$ speedup** over exact Hessian trace optimization. It also confirms long-term stability without boundary saturation (no parameter freezing).
  - **Appendix A.3 (Alternative subspaces):** Shows that depth-dependent polynomials outperform Random Projections and DCT subspaces, and evaluates mild domain shifts where the merge gap drops below $-4.50\%$.
  - **Appendix A.4 (Projected Noise Quantification):** Empirically measures and reports $\|J_{\mathbf{p}}\boldsymbol{\epsilon}\|_2$ and $\|\delta_{\perp}\|_2$ under INT8 and INT4 targets, providing strong empirical weight to the theoretical noise decomposition.
  - **Appendix A.4 (Sigmoid Saturation & Sum-to-One Normalization):** Discusses parameter scale inflation and demonstrates that prediction entropy minimization naturally acts as an implicit regularizer, preventing coefficient inflation and stabilizing converged sums at a safe $\bar{\lambda} \approx 1.42$.
  - **Appendix A.5 (Calibration Class Imbalance):** Shows that CR-PolySACM is highly robust under severe label shifts on the calibration stream (maintaining $18.81\%$ accuracy under 20\% class coverage), whereas unconstrained TTA collapses.
  - **Appendix A.6 (Convergence Curves):** Provides step-by-step convergence trajectories of the entropy loss and the local landscape sharpness, confirming excellent optimization stability.

---

## 3. Weaknesses & Areas for Improvement
The revised paper is exceptionally complete, technically sound, and highly polished. However, we identify a few subtle, advanced areas for improvement that would further strengthen the manuscript:

### A. Sensitivity and Robustness of the Percentile Choice in the Automated Blueprint
- **Critique:** The proposed percentile-based blueprint dynamically sets the clipping threshold $\beta$ as a lower percentile of the layer-wise task-vector norm distribution across the network. The authors show a highly successful proof of concept on the ViT backbone using the 10th percentile ($\beta \approx 0.098$ achieving 19.05% accuracy). However, the paper does not show the sensitivity of this blueprint to the choice of the percentile itself (e.g., how does performance behave if the user selects the 5th, 15th, or 25th percentile?). 
- **Actionable Suggestion:** It would be highly valuable to add a brief sentence or a small table in the appendix outlining the sensitivity of the automated blueprint to different percentile choices, ensuring practitioners have clear guidelines on how to choose this percentile for novel architectures.

### B. Impact of Backbone Scaling on the Task-Vector Norm Scale Pathology
- **Critique:** The empirical validation is primarily conducted on the relatively small Vision Transformer backbone (`vit_tiny_patch16_224` with 5.7M parameters). While the authors discuss scaling to LLMs/VLMs in their future work section, they do not show if the task-vector norm scale pathology becomes more or less pronounced as the backbone scales up (e.g., to `vit_small` or `vit_base`).
- **Actionable Suggestion:** The authors could discuss whether model scale (depth and width) theoretically intensifies the task-vector norm scale pathology (e.g., do deeper models exhibit even larger discrepancies in task-vector norms?). Explicitly addressing how deeper networks might exhibit wider norm spreads would further ground the scaling potential of CR-PolySACM.

### C. Adaptation Behavior of Task-Specific Head Parameters
- **Critique:** The paper evaluates model composition by keeping the task classification heads task-specific and swapping them dynamically during evaluation, while merging the shared backbone parameters. While this is standard practice in weight-space merging (especially across highly disparate classification domains), in some unified multi-task architectures (such as decoder-only language models), task heads are often unified.
- **Actionable Suggestion:** The authors should include a brief discussion on how CR-PolySACM translates to scenarios where task classification heads are also shared or when there is a unified head (e.g., a shared vocabulary head in generative language models), highlighting any potential challenges or straightforward extensions.

---

## 4. Questions for the Author
1. **Percentile Choice Sensitivity:** In your automated percentile-based blueprint, how sensitive is the performance to the choice of the percentile? Have you experimented with values other than the 10th percentile (e.g., 5th or 20th), and is there a stable range?
2. **Backbone Scale Progression:** Do you expect the task-vector norm scale discrepancy to expand as the model size increases? For example, would a ViT-Base or a 7B LLM exhibit an even larger gap than the 50-fold discrepancy observed in ViT-Tiny?
3. **Unified Heads:** How does CR-PolySACM adapt to architectures with a unified output head (e.g., decoder-only language models)? Would the final vocabulary projection layer represent another sensitive, low-norm layer similar to Layer 13 in the Vision Transformer?

---

## 5. Ratings & Decisions

### A. Soundness
- **Rating:** Excellent
- **Justification:** The mathematical derivations are exceptionally rigorous, correct, and based on reasonable assumptions. The quadratic noise decomposition, the task-vector norm scale pathology, and the clipping-regularized scale balancing are physically and mathematically sound. The newly added appendix sections exhaustively validate calibration size ($N$), class imbalance, scale inflation, and convergence trajectories, leaving no major technical questions unanswered.

### B. Presentation
- **Rating:** Excellent
- **Justification:** The paper is beautifully structured, highly readable, and exceptionally clear. It positions itself perfectly within prior literature and is highly transparent about its limitations (e.g., the domain disconnect gap and absolute low-precision accuracy). Figures and tables are of publication-grade quality.

### C. Significance
- **Rating:** Excellent
- **Justification:** The paper addresses an important and highly relevant problem: enabling robust post-training quantization for merged multi-task models on edge hardware. By linking test-time adaptation, subspace constraints, and flatness optimization, it advances the state-of-the-art and provides a practical framework (CR-PolySACM) that sets a new benchmark. It will influence future research on model merging and quantization-aware TTA.

### D. Originality
- **Rating:** Excellent
- **Justification:** The concept of Quantization-Operator Overfitting is novel. The discovery and theoretical analysis of the task-vector norm scale pathology are highly original, and the clipping-regularized scale-balancing solution (CR-SACM) represents a major methodological advancement.

### E. Overall Recommendation
- **Decision:** 5: Accept (Strongly Lean towards 6: Strong Accept)
- **Justification:** This is a technically solid, exceptionally well-written, and mathematically rigorous paper that makes significant contributions to post-hoc model merging and quantization-robust edge deployment. The theoretical insights into the limits of test-time flatness optimization and the task-vector norm scale pathology are outstanding. The empirical results across six diverse quantization schemas are highly convincing, and the newly added Appendix sections successfully and exhaustively resolve prior concerns regarding overhead, calibration stream size, alternative subspaces, scale inflation, and label shift, making the paper highly complete and ready for immediate publication.
