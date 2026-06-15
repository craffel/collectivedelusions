# 3. Soundness and Methodology

## Clarity of Description
The mathematical formulation of QWS-Merge is exceptionally clear, precise, and well-structured. The progression from input patch embeddings to the projected low-dimensional unit sphere phase state $\psi(x)_b$, the layer-wise phase alignment calculation, the wavefunction collapse via batch averaging, and the final parameter-space reconstruction is described with rigorous equations.

However, there are a few areas where the methodological description should be clarified or where the analogy warrants a brief caveat:
1. **Fidelity of the Physical Analogy:** The paper heavily uses quantum vocabulary (e.g., "wavefunction", "eigenstate", "Hilbert space", "measurement collapse"). In practice, the system is entirely classical, real-valued, and deterministic. No actual complex wavefunctions, density matrices, or quantum state operations are involved. While highly creative and motivating, the authors should transparently acknowledge that this formulation is an elegant classical physical analogy.
2. **Frozen Random Projection Sensitivity:** The global representation $z(x)_b$ is projected into a low-dimensional $d$-dimensional space ($d = K = 4$) via a frozen random projection matrix $P \in \mathbb{R}^{D \times d}$. Because the projection dimension $d$ is extremely low, the choice of random seed for $P$ could have a non-trivial impact on the feature-alignment and subsequent performance. The paper does not discuss this sensitivity or how $P$ is initialized (e.g., Gaussian, orthogonal).

## Appropriateness of Methods
The proposed highly constrained, low-dimensional phase-space (336 trainable parameters) is a highly appropriate and elegant solution to the data-scarce calibration regime. Standard fine-tuning or high-dimensional routing mechanisms are highly prone to the Overfitting-Optimizer Paradox when optimized on a tiny 64-sample set. Bounding the optimization to layer-wise phase angles on a unit sphere is a theoretically sound way to enforce extreme parameter-efficiency and strong generalization.

## Potential Technical Flaws & Baseline Confounders
A critical concern regarding the soundness of the empirical comparison is a **potential baseline confounder**:
- **Layer-Wise vs. Global Routing:** The paper notes that the classical *Linear Router* baseline maps the input's pooled backbone representations directly to routing weights via a standard linear layer, which are then applied **globally** across all $L$ layers. By contrast, QWS-Merge utilizes **layer-wise** phase-basis vectors to support fine-grained, localized routing.
- **The Confounder:** This design difference makes it difficult to isolate the exact source of QWS-Merge's performance gains. Is the superiority of QWS-Merge under extreme conflict (SVHN) truly due to the wave-like cosine formulation, or is it simply because QWS-Merge has layer-specific flexibility while the Linear Router is restricted to a single global set of routing coefficients?
- **The Fix:** To ensure a mathematically rigorous and scientifically fair comparison, the authors should evaluate a **Layer-wise Linear Router** baseline. In this baseline, each layer has its own small linear projection layer (with parameters comparable to QWS-Merge's layer-wise parameters), allowing for localized routing. Without this control baseline, the claim that the wave-like projection is the primary driver of regularization is not fully decoupled from layer-wise flexibility.

## Batch Dependency and I.I.D. Violation
The wavefunction collapse step performs a mean-measurement across the batch dimension:
$$ \bar{\alpha}_k(l) = \frac{1}{B} \sum_{b=1}^B \alpha_{k, b}(l) $$
This introduces a major batch dependency during inference, where the prediction for an individual image $x_b$ depends on the other samples present in the same batch. This represents a direct violation of the standard independent-and-identically-distributed (I.I.D.) assumption of inference. While the authors transparently acknowledge this in their excellent "Limitations" section, it remains a significant methodological limitation that restricts its immediate real-world deployability on online single-sample streams ($B=1$).
