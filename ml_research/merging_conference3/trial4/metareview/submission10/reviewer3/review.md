# Peer Review of Conference Submission

## Summary of the Paper
This paper challenges a foundational assumption in the model-merging literature: that specialized expert models must be fused into a single, static set of parameters before deployment. Instead, drawing inspiration from physical wave mechanics, the authors introduce **Quantum Wavefunction Superposition Merging (QWS-Merge)**. Under this formulation, fine-tuned expert weights are treated as task eigenstates $|\psi_k\rangle$ in a parameter Hilbert space. During inference, these states are superposed and dynamically "collapsed" (via batch averaging) into a localized, classical weight configuration based on the wave-like phase-interference of the incoming features. 

To achieve this, the global representation of an input is projected onto a low-dimensional sphere, representing the input phase state. Layer-wise trainable phase-basis vectors, scaling amplitudes, and phase biases are then optimized on an extremely lightweight, 64-sample calibration set (16 samples per task). Empirical results on a challenging multi-task benchmark (MNIST, FashionMNIST, CIFAR-10, SVHN) using a compact Vision Transformer backbone (`vit_tiny_patch16_224`, 5.7M parameters) demonstrate that QWS-Merge completely resolves catastrophic representational collapse. It improves uniform merging accuracy from $49.35\%$ to \textbf{59.32\%} on homogeneous streams. Crucially, compared to a classical Linear Router baseline, QWS-Merge exhibits strong wave-inspired regularization: under extreme task conflict (SVHN), the Linear Router collapses to $15.30\%$, while QWS-Merge maintains a robust \textbf{31.60\%} (preserving $91.5\%$ of specialized expert capacity).

---

## Strengths and Weaknesses

### 1. Originality (Major Strength)
- **Conceptual Leap:** The core idea of modeling model weights as superposed eigenstates in a parameter Hilbert space that collapse dynamically via phase-coherence is a **bold, ambitious, and highly original conceptual contribution**. It represents a significant departure from standard incremental model-merging publications, which are often limited to minor heuristic or algebraic modifications of static parameter-combining.
- **Wave-Like Regularization:** Porting the physical principles of constructive and destructive wave-interference to regularize parameter space is exceptionally creative. The mathematical translation—constraining routing coefficients to spherical unit projections and non-monotonic cosine wave-functions—proves to be a highly functional regularizer rather than just a cosmetic metaphor.
- **Documenting Heterogeneity Collapse:** The paper is the first in the model-merging space to systematically analyze and formally document **"heterogeneity collapse"** under mixed-task streams and larger batch sizes. This provides a highly novel scientific insight that has been overlooked in typical homogeneous evaluation settings.

### 2. Significance (Strength)
- **High Potential Impact:** By shifting the focus of the community from static weight compromises to regularized, dynamic parameter routing, this paper opens up a promising new avenue of research. It demonstrates how physical wave principles can prevent parameter-space collapse in data-scarce, compact-capacity regimes. This can significantly influence future research in multi-task learning, parameter-efficient fine-tuning (PEFT), and dynamic foundation model serving.
- **Extreme Parameter and Sample Efficiency:** Optimizing only 336 parameters on a tiny 64-sample validation set in under 30 seconds completely bypasses the Overfitting-Optimizer Paradox, making the approach highly practical for resource-constrained edge deployments.

### 3. Soundness (Minor Weaknesses & Areas for Improvement)
- **Baseline Control Confounder (Weakness):** There is a notable architectural difference between QWS-Merge and the classical Linear Router baseline. The Linear Router maps the pooled representation directly to routing weights, which are then applied **globally** across all $L$ layers. In contrast, QWS-Merge utilizes **layer-wise** phase-basis vectors, allowing for localized functional specialization at each layer. This introduces a baseline confounder: is QWS-Merge's superiority under extreme conflict (SVHN) due to the wave-like cosine projection, or is it simply because it has layer-specific flexibility while the Linear Router is restricted globally? 
  - *Constructive Suggestion:* The authors should evaluate a **Layer-Wise Linear Router** baseline (with a comparable parameter footprint) to cleanly decouple the benefits of the cosine wave formulation from layer-wise routing flexibility.
- **Fidelity of the Metaphor (Minor Weakness):** Although the quantum mechanical framing is elegant and motivating, the system is entirely classical, real-valued, and deterministic. The paper would benefit from a brief, transparent statement explicitly acknowledging that the quantum framing acts as an elegant physical analogy rather than a strict quantum mechanical simulation.
- **Batch Dependency and I.I.D. Violation (Acknowledged Weakness):** The batch-averaging step used to collapse the wavefunction violates the standard independent-and-identically-distributed (I.I.D.) assumption during inference, making a sample's prediction dependent on its batch companions. While the authors transparently document this in their excellent "Limitations" section, it remains a challenge for online single-sample deployments.

### 4. Presentation (Strength)
- **Clarity and Flow:** The manuscript is exceptionally well-written, beautifully structured, and highly professional. The mathematical formulation is precise, and the narrative directly connects physical principles with architectural implementations.
- **Empirical Honesty:** The authors are highly commended for their scientific honesty and transparency regarding the "capacity-regularization trade-off" and "heterogeneity collapse." Instead of hiding the performance drops of dynamic methods at larger batch sizes, they analyze them thoroughly, establishing a clean, realistic benchmark for future research.

---

## Section Ratings

### Soundness: Good
The mathematical formulations are rigorous and correct, and the experimental setup (including a diverse multi-task benchmark and compact backbone) is highly appropriate. However, the rating is bounded at "Good" due to the global-vs-layer-wise confounder in the baseline comparisons.

### Presentation: Excellent
The writing is exceptionally clear, the figures are clean and highly informative, and the discussion of limitations and batch heterogeneity represents a model of scientific transparency and rigor.

### Significance: Excellent
The paper addresses an important problem in a highly compact parameter regime. It has high potential to influence future research in multi-task parameter interpolation and dynamic routing by introducing regularized parameter manifolds.

### Originality: Excellent
Porting wavefunction superposition, phase-coherent wave interference, and measurement collapse into the parameter-space of deep neural networks represents an incredibly creative and paradigm-shifting conceptual leap.

---

## Overall Recommendation

**Rating: 5 (Accept)**

### Justification:
This is a highly creative and intellectually refreshing paper that challenges the traditional static-weight assumption of model merging. By introducing a bold physical metaphor—Quantum Wavefunction Superposition Merging (QWS-Merge)—the authors successfully port wave-interference and spherical unit projection constraints to parameter space. This mathematical design provides exceptional regularizing properties, preventing unconstrained parameter-space collapse and outperforming classical routing under extreme task conflict (SVHN) by a massive $+16.30\%$ absolute margin. 

While there are minor areas for improvement—specifically the need to evaluate a Layer-Wise Linear Router baseline to isolate the exact source of performance gains and a minor clarification regarding the classical nature of the physical analogy—the conceptual novelty, scientific honesty, and impressive empirical robustness of the regularized subspace make this a highly valuable and non-incremental contribution to the conference.

---

## Questions and Constructive Feedback for the Authors

1. **Layer-wise Baseline:** To ensure a mathematically fair comparison, have you considered evaluating a **Layer-wise Linear Router** baseline? If each layer of a classical router had its own small linear projection layer (restricted to a comparable 336-parameter budget), how would it perform on SVHN compared to QWS-Merge? This would help decouple the regularization benefits of the wave-like cosine formulation from layer-specific flexibility.
2. **Frozen Random Projection Sensitivity:** The global features are projected into a low-dimensional $d$-dimensional space ($d = 4$) via a frozen random projection matrix $P$. Is the performance sensitive to the random initialization seed of $P$? Would using an orthogonal projection or a learned PCA projection improve feature-alignment?
3. **Single-Sample Deployment:** Since the batch-averaging step introduces transductive batch dependency and violates the I.I.D. assumption during inference, how do you envision deploying QWS-Merge in real-world single-sample streaming scenarios ($B=1$)? Have you experimented with maintaining an Exponential Moving Average (EMA) or a rolling buffer of coefficients during streaming?
4. **Clarification on Analogy:** Would you be open to explicitly adding a brief sentence in the Methodology section clarifying that while the mathematical design is inspired by quantum principles (e.g., eigenstates, wave superposition), the actual formulation is a classical, real-valued, and deterministic analogy? This would enhance the scientific precision of the paper.
