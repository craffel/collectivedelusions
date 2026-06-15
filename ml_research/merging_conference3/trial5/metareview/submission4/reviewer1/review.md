# Peer Review

## Summary of the Paper
This paper presents a critical, methodological deconstruction of Quantum Wavefunction Superposition Merging (QWS-Merge), a prominent state-of-the-art dynamic model-merging method. Utilizing a compact Vision Transformer (`vit_tiny_patch16_224`) backbone trained to high convergence across MNIST, FashionMNIST, CIFAR-10, and SVHN, the authors apply Occam's razor to dissect the necessity of QWS-Merge's complex "quantum eigenstate" wave equations. 

To isolate and control for various architectural and optimization confounders, the authors propose the **Bounded Classical Router (BC-Router)** framework, introducing three targeted variants:
1. **Bounded Linear Router (BL-Router):** Restricts Task Arithmetic coefficients to a static scale ceiling ($\lambda_{max} = 0.3$) to isolate the *Over-Scaling Confounder*.
2. **Global Router with Layer-wise Scaling (GLS-Router):** Employs global routing with layer-specific scaling amplitudes to isolate the *Layer-wise Specialization Confounder*.
3. **Bounded Sigmoidal Router (BSigmoid-Router):** Replaces Softmax with independent Sigmoids to resolve the *Softmax Zero-Sum Competitive Bottleneck* during mixed-batch calibration.

The paper yields four major scientific and methodological findings:
* **Paradigm Clarification:** Establishes the operational delta between Test-Time Adaptation (e.g., AdaMerging, which incurs heavy test-time backpropagation latency) and offline-calibrated dynamic routing (e.g., QWS-Merge or the proposed BC-Router, which run as lightweight forward passes with zero inference active optimization).
* **Deconstructing Classical Collapse via L2 Regularization:** Exposes that the unregularized classical Linear Router baseline collapses on SVHN ($74.00 \pm 16.14\%$) due to overfitting on the tiny calibration set. Applying basic L2 weight decay ($\gamma = 1 \times 10^{-4}$) completely rescues the classical baseline, boosting SVHN accuracy to **$91.73 \pm 3.71\%$** (outperforming SOTA QWS-Merge by **+12.00%**).
* **Decoupling Experts via independent Sigmoids:** Demonstrates that standard Softmax-based bounding forces an artificial under-scaling bottleneck (sum of coefficients capped at 0.3, restricting each task to 0.075 under uncertainty). Replacing Softmax with independent Sigmoids in **BSigmoid-Router** completely resolves this, achieving a stable joint homogeneous accuracy of $83.73 \pm 1.93\%$ and heterogeneous stream accuracy of **$83.96 \pm 2.27\%$** ($B=1$, outperforming QWS-Merge).
* **The Role of Metaphor as a Structural Regularizer:** Reveals that unregularized GLS-Router exhibits extreme sensitivity across seeds on SVHN (standard deviation of $24.30\%$) and overfits to the calibration set (FashionMNIST collapse). This proves that QWS-Merge's true value lies not in a physical "quantum eigenstate" property, but in its wave projection equations acting as a robust, stable structural regularizer.

---

## Strengths and Weaknesses

### Strengths
1. **Outstanding Conceptual Novelty (Occam's Razor):** The paper takes a courageous, intellectually refreshing, and highly original stance by applying Occam's razor to a prominent SOTA method. It exposes a critical trend in contemporary deep learning: introducing over-engineered, mathematically exotic metaphors (like quantum mechanics) to hide under-tuned, unregularized classical baselines. By showing that simple L2 regularization completely resolves classical "collapse," the paper shifts the paradigm of baseline design and baseline optimization in model merging.
2. **The "Macro-Level Mixture-of-Experts" Framing:** Reframing dynamic model merging under the lens of sparse Mixture-of-Experts (MoE) token gating is a highly elegant, original, and ambitious conceptual leap. By utilizing independent sigmoidal routing (**BSigmoid-Router**), the authors design a "macro-level MoE" that bypasses the severe memory copy and communication overhead of token-level gating by merging parameter-space vectors once per batch, preserving mathematical capacity and independent expert activation.
3. **Exceptional Scientific Honesty and Transparency:** The discussion on the "Generalist-Specialist Paradox" and the "Practical Utility Limits" of dynamic weight-routing is a major highlight. The authors candidly explain that dynamic routing does not create new capacity but merely reallocates existing parameter budgets. Consequently, specializing on one task (e.g., SVHN) inevitably degrades others, making simple static Uniform Merges superior for generalist applications. This level of transparency is rare and exceptionally valuable.
4. **Methodological and Empirical Rigor:** The experimental design is flawless. The authors use fully converged task experts to establish a proper ceiling, report standard deviations over multiple calibration seeds, evaluate robustness under interleaved heterogeneous stream noise across multiple batch sizes, and conduct detailed latency and PyTorch memory buffer profiling (Table 5).

### Weaknesses
1. **Localized Scaling of the Empirical Sandbox:** The empirical sandbox is conducted on a compact Vision Transformer backbone (`vit_tiny_patch16_224`) across four image datasets (MNIST, FashionMNIST, CIFAR-10, SVHN). While this capacity-constrained setup is standard and mathematically ideal to isolate and deconstruct model-merging routing mechanisms without confounding factors, verifying these findings on larger backbones (e.g., Swin, ViT-Base/Large) and larger-scale datasets (such as ImageNet classifications or DomainNet) remains a prioritized path for future empirical validation.
2. **Online Dynamics of Test-Time Adaptation:** In Table 2, the online Test-Time Adaptation baseline **AdaMerging** is modeled statically using its offline-calibrated homogeneous joint mean accuracy rather than executing active online entropy-minimization loops across the shuffled stream. While this is a highly practical simplification due to AdaMerging's massive latency, it fails to capture the true temporal order effects and parameter drift of online TTA. The authors openly acknowledge this limitation, which preserves high transparency.

---

## Dimension Ratings

* **Soundness: Excellent**
  The paper is exceptionally sound, rigorous, and carefully evaluated. The authors use converged experts, evaluate over multiple seeds to establish statistical significance, control for specific confounders using targeted baselines (BL-Router, GLS-Router, L2 weight decay), and mathematically and visually deconstruct the batch-averaging bottleneck.

* **Presentation: Excellent**
  The writing is clear, direct, and engaging. The mathematical formulations are highly precise and well-defined. The narrative hook using Occam's razor is compelling, and the comprehensive Appendix (including calibration specs, detailed latency benchmarks, and ablation studies on calibration size and regularization strength sensitivity) ensures the work is highly reproducible and complete.

* **Significance: Excellent**
  This work has the potential to profoundly influence future research in model merging, multi-task learning, and parameter-space Mixture-of-Experts. It serves as a vital call to action for the machine learning community to prioritize baseline optimization and proper regularization, and introduces a highly practical, parameter-efficient (772 parameters), low-latency sigmoidal routing paradigm suitable for edge deployments.

* **Originality: Excellent**
  The originality of this paper is highly significant. It lies in its bold conceptual deconstruction of exotic mathematical metaphors, the deep insight that such metaphors act as structural regularizers during optimization, and the ambitious parallel drawn between dynamic model merging and sparse Mixture-of-Experts.

---

## Overall Recommendation

**6: Strong Accept**

**Justification:**
This is an outstanding, conceptually bold, and scientifically rigorous paper. It applies Occam's razor to demystify a prominent state-of-the-art dynamic model-merging protocol and proves that simple, classical linear projection heads—when properly regularized with standard L2 weight decay or formulated with independent Sigmoids—match or exceed SOTA quantum wavefront performance. The mathematical deconstruction of the Softmax zero-sum under-scaling bottleneck and the original parallel drawn to sparse Mixture-of-Experts are highly significant conceptual leaps. Backed by exceptional writing quality, exhaustive latency profiling, and rare scientific honesty regarding the generalist-specialist tradeoff, this paper represents a flawless, high-impact contribution that should be accepted with the highest priority.

---

## Constructive Feedback for the Authors
1. **Explicit Regularization on Scaling Amplitudes:** The authors show that the layer-wise scaling parameters in GLS-Router overfit severely to the calibration set and collapse on FashionMNIST, concluding that QWS-Merge's wave equations act as a necessary structural regularizer. To further strengthen the classical baseline, it would be highly valuable to test whether applying explicit L2 regularization (weight decay) directly to the layer-wise scaling amplitudes $R_k^{(l)}$ (or applying a lower learning rate/gradient clipping on them) can stabilize the classical layer-wise scaling. If so, this would provide a simple, algorithmic alternative to wave-inspired regularization.
2. **Expanding Gating Topology for LLM Merging:** The discussion on the generalizability to Large Language Models in Section 4.3 is highly compelling. The authors are encouraged to expand on how the Softmax-free sigmoidal gating topology scales when the task suite grows to $K \ge 10$ experts, especially under sparse gating layouts (such as Top-1 or Top-2 gating), to further strengthen the connection to established sparse Mixture-of-Experts networks.
3. **Broader Generalization:** While the compact Vision Transformer backbone and vision datasets are ideal to isolate parameter-space conflicts, conducting a small, representative pilot on a larger model (e.g., ViT-Base or a multimodal CLIP-based model) would solidify the generalizability claims and make the deconstruction even more robust.
