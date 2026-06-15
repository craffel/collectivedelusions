# Peer Review Report: Quantum Wavefunction Superposition Merging (QWS-Merge)

## Strengths and Weaknesses

### Major Strengths

1. **Outstanding Originality and Conceptual Ambition**: 
   The paper proposes a highly original, quantum-inspired paradigm (QWS-Merge) that represents a refreshing departure from incremental heuristics in parameter-space model merging. Shifting the foundational assumption of model merging from static, input-independent weight averaging to dynamic, wave-like parameter superposition is an ambitious conceptual leap. Treating task-specific experts as eigenstates $|\psi_k\rangle$ in a parameter Hilbert space is a highly creative and paradigm-shifting perspective.

2. **Wave-Like Subspace Regularization under Extreme Conflict**:
   The use of a non-monotonic cosine-based projection acting on normalized spherical phase states serves as a powerful regularization mechanism. This is empirically verified in a high-conflict domain (SVHN), where the classical, unconstrained Linear Router baseline collapses catastrophically to $15.30\%$ accuracy (near-random), while QWS-Merge maintains a robust $31.60\%$ ($91.5\%$ of the specialized expert capacity). This demonstrates that the wave-like phase formulation is not just a metaphor, but a highly effective regularized subspace for parameter routing.

3. **Exceptional Parameter and Calibration Efficiency**:
   By restricting the routing to layer-wise phase-basis vectors, QWS-Merge operates with an extremely compact parameter footprint (exactly $336$ trainable parameters for $L=14$ layer groups and $K=4$ tasks). This ultra-compact footprint successfully bypasses the Overfitting-Optimizer Paradox, allowing stable optimization using standard Adam on a tiny offline validation set of only 16 samples per task (64 total calibration samples).

4. **Rigorous and Transparent Evaluation**:
   Evaluating on a compact, resource-constrained Vision Transformer backbone ($\mathtt{vit\_tiny\_patch16\_224}$ with 5.7M parameters) is an excellent and challenging testbed. In compact model regimes, representational collapse is prominent, making this a much more rigorous benchmark than using highly over-parameterized models that naturally absorb parameter conflicts. Furthermore, the paper is intellectually honest and transparent in documenting and analyzing the "heterogeneity collapse" that occurs in mixed-task streams at larger batch sizes.

5. **Exceptional Presentation Quality**:
   The manuscript is exceptionally well-written, mathematically rigorous, and easy to follow. The mathematical notation is clean and precise, and the tables and figures are well-integrated and informative.

---

### Weaknesses and Areas for Improvement

1. **Practical Inference Bottleneck (Batch Dependency & I.I.D. Violation)**:
   A major theoretical and practical limitation is the batch-dependent nature of the "wavefunction collapse" (Equation 8). By averaging sample-level coefficients across the batch dimension to construct a batch-level weight representation, the prediction for any individual sample depends on the co-occurring samples in its batch. This violates the standard I.I.D. assumption of machine learning. Under mixed-task streams at larger batch sizes (e.g., $B=256$), this averaging causes the coefficients to collapse back to uniform compromises, resulting in "heterogeneity collapse" (degrading accuracy to $48.70\%$, which is below the static AdaMerging baseline of $57.20\%$).
   *Suggestion*: The paper would be significantly strengthened by exploring or discussing a lightweight engineering solution to decouple the batch dependency during deployment, such as employing an Exponential Moving Average (EMA) of routing coefficients or a small rolling queue of recent coefficients for single-sample inference streams ($B=1$).

2. **Demystifying the "Quantum" Metaphor**:
   The quantum mechanics analogy is conceptually rich and highly successful as a design inspiration. However, mathematically, QWS-Merge operates purely in a classical computing environment.
   *Suggestion*: For mathematical rigor, the authors should explicitly contrast the proposed mechanism with physical quantum mechanics (e.g., probability amplitudes here are real-valued and can be negative, and the "collapse" is a standard classical average over a batch). A brief discussion connecting this formulation to classical non-linear soft-routing layers with cosine activation would ground the method for readers from traditional machine learning backgrounds.

3. **Capacity-Regularization Trade-off on Low-Conflict Tasks**:
   Due to the heavily regularized nature of the wave phase-basis projections, QWS-Merge exhibits a slight performance penalty on simpler, low-conflict tasks (such as MNIST and FashionMNIST) compared to the unconstrained Linear Router ($77.60\%$ vs. $91.20\%$ on MNIST). This capacity-regularization trade-off is mathematically expected but should be recognized as a limitation of the heavy regularizing properties of the cosine projection.

---

## Soundness
**Rating**: **Good**

The proposed methodology is technically sound and mathematically rigorous. All equations are well-defined, and the dimensions are consistent. The empirical results on the multi-task benchmark provide strong, credible support for the core claims of resolving representational collapse and wave-like regularization. The authors are transparent and scientifically honest about the batch dependency and capacity-regularization trade-off, which reinforces the soundness of the paper's scientific contributions.

---

## Presentation
**Rating**: **Excellent**

The submission is exceptionally well-structured, clear, and highly professional. The mathematical formulation is presented with remarkable clarity. The narrative is highly engaging and effectively motivates the proposed quantum-inspired framework. The related work is properly contextualized, and the empirical results are presented with exceptional clarity.

---

## Significance
**Rating**: **Good**

The paper addresses an important and highly relevant problem in model merging (destructive parameter interference under low model capacity and high task conflict). By shifting the focus from static weight consensus to dynamic wave-inspired parameter-space routing, this paper introduces a non-incremental paradigm shift. It has high potential to influence future research in physics-inspired neural routing, dynamic multi-task learning, and edge deployment of foundation models where storage constraints prevent keeping multiple independent checkpoints.

---

## Originality
**Rating**: **Excellent**

The paper is exceptionally original. The conceptual leap of modeling fine-tuned expert weights as task eigenstates in a Hilbert space and using wave-like phase-coherence to dynamically assemble weights on-the-fly represents a major creative breakthrough. The design of layer-wise, low-dimensional phase-basis projections is an elegant and highly novel solution to the Overfitting-Optimizer Paradox that plagues existing test-time adaptation methods.

---

## Overall Recommendation

**Recommendation**: **5: Accept**

**Justification**: This is a highly creative, conceptually ambitious, and original paper. It challenges the traditional static-weight assumption of model merging and introduces a successful, physically-grounded wave-coherence formulation. QWS-Merge demonstrates exceptional regularization properties, preventing the catastrophic parameter-space collapse that plagues unconstrained classical routing baselines under high domain conflict. Although there is a practical bottleneck regarding batch-dependent inference, the authors are highly transparent and scientifically honest in documenting and analyzing this limitation. The sheer novelty, intellectual ambition, and strong empirical regularization of this work make it a highly valuable contribution to the conference.
