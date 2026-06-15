# Novelty Check

**Originality Rating: Good**

The paper presents a highly original and creative combination of existing and novel concepts to solve a critical real-world problem: the vulnerability of test-time model merging to physical input noise.

### Key Aspects of Novelty:

1. **Dual-Regularization Concept:**
   While subspace-constrained merging coefficients (PolyMerge) and flatness-aware minimization (Sharpness-Aware Minimization, or SAM) exist as separate ideas, combining them into a unified test-time model merging framework represents a creative and effective synthesis.
   Specifically, the polynomial parameterization acts as a *spatial filter* to remove high-frequency noise across layers, while the flatness-aware objective acts as a *temporal/optimization regularizer* that prevents low-frequency transductive drift.

2. **Zeroth-Order Flatness Optimization for Edge Devices:**
   Applying flatness-aware minimization directly to the millions of network weights at test-time is computationally impossible for edge accelerators due to the double-backward pass and activation caching. FlatMerge's key novelty is applying flatness-aware optimization directly within the extremely compact ($K \times (d+1)$) coefficient space using a **Zeroth-Order (gradient-free) randomized smoothing approach**.
   This is a highly pragmatic and elegant design decision: it reformulates SAM into a forward-only perturbation scheme on the coefficients, requiring **zero weight backpropagation** and **zero activation memory caching**.

3. **Characterization of Noise-Entropy Collapse:**
   The paper provides a valuable, rigorous formalization and qualitative analysis of **Noise-Entropy Collapse** under test-time corruptions. It deepens the understanding of the *Overfitting-Optimizer Paradox* in TTA by showing how unconstrained first-order optimizers exploit corrupted test-time logits to minimize entropy at the cost of catastrophic generalization failures.

4. **Amortization via Asynchronous Adaptation:**
   To address the latency penalty inherent to zeroth-order randomized smoothing, the paper proposes a highly practical **Asynchronous, Periodic Adaptation** scheme. Since physical noise and environmental conditions change on a slow temporal scale, the blending coefficients are updated periodically on background threads and cached. This represents a highly pragmatic, system-level design contribution that makes ZO-FlatMerge feasible for real-world production.
