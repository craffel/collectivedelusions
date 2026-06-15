# Peer Review

**Submission Title:** Quantization-Aware Model Merging (Q-Merge)

---

## 1. Summary of the Paper

This paper addresses a highly practical deployment challenge at the intersection of weight-space model merging and post-training network quantization (PTQ). While model merging (e.g., Task Arithmetic or AdaMerging) allows practitioners to fuse multiple task-specific expert models into a single checkpoint with zero retraining, subsequent post-training quantization to 8-bit (INT8) or 4-bit (INT4) formats degrades multi-task accuracy due to quantization noise. Merging already-quantized experts also fails due to the alignment loss of discrete integer grids.

To bridge this gap, the paper proposes **Quantization-Aware Model Merging (Q-Merge)**, a test-time adaptation framework that optimizes layer-wise merging coefficients directly under the non-differentiable quantization operator. The framework operates on an unlabeled calibration stream (64 images total) to minimize joint prediction entropy. It evaluates and compares two optimization paradigms: zero-order 1+1 Evolution Strategy (1+1 ES) and first-order Adam gradient descent with a Straight-Through Estimator (STE). 

Key empirical findings using a pre-trained ViT-Tiny backbone (5.7M parameters) across four classification tasks (MNIST, FashionMNIST, CIFAR-10, SVHN) demonstrate:
1.  Under 8-bit PTQ, Q-Merge (Adam GD with STE) achieves **74.30%** average accuracy, recovering the quantization loss and outperforming the unquantized uniform FP16 baseline (**71.88%**) and standard unquantized AdaMerging ES (**73.21%**).
2.  Propagating gradients through the rounding operator via STE is highly superior to zero-order mutations (1+1 ES), yielding higher performance, faster convergence, and over **2.7x lower seed-to-seed variance**.
3.  Under aggressive 4-bit PTQ, moving from per-tensor to standard per-channel weight quantization prevents catastrophic model collapse, and Q-Merge (STE) optimizes weight alignment further to achieve **63.36%** average accuracy, outperforming the naive post-merge baseline (**56.66%**) by **6.70%** absolute.
4.  Fully integer-quantized weight pipelines (W8A16/W4A16) are feasible via post-hoc 8-bit quantization of task heads with negligible performance loss ($<0.01\%$).

---

## 2. Strengths and Weaknesses

### Strengths:
*   **High Pragmatic Edge Utility:** Q-Merge is exceptionally lightweight, converging in seconds on a compact, unlabeled calibration stream. Once optimized, the continuous blending coefficients are discarded and weights are locked as static INT8 or INT4 integers, introducing **zero latency or parameter overhead during inference**.
*   **Exemplary Experimental Rigor & Controls:** The paper stands out for its meticulous experimental design. Most notably, the authors deconstruct the "optimizer confounding factor" (comparing AdaMerging and Q-Merge under both ES and Adam GD), which isolated the performance gains of the optimizer from the quantization operator. This is an outstanding scientific practice.
*   **Extensive Systems-Level Analyses:** The submission is remarkably thorough, addressing common edge-deployment and compiler-level questions via comprehensive ablations. These include fixed-point scale factor discretization sensitivity, dynamic activation quantization (W8A8/W4A4), peak memory consumption (activation caching in STE vs. 1+1 ES), dynamic stream task-balancing heuristics, and sequential integration with advanced PTQ frameworks (AdaRound).
*   **Excellent Performance:** The results demonstrate that 8-bit model merging is nearly lossless compared to unquantized ceilings, and 4-bit model merging is highly viable under per-channel representations.

### Weaknesses:
*   **Incremental Conceptual Novelty:** From a fundamental machine learning perspective, the paper represents an incremental engineering-driven synthesis of pre-existing blocks. The core components—AdaMerging (entropy-based test-time coefficient optimization), symmetric uniform quantization (PTQ), Straight-Through Estimator (STE), and per-channel quantization—are highly established. Differentiating coefficients through STE is a straightforward extension of AdaMerging to a quantized weight setting rather than a paradigm-shifting conceptual leap.
*   **Toy-Scale Experimental Generalizability:** The empirical evaluation relies exclusively on a pre-trained **ViT-Tiny backbone (5.7M parameters)** fine-tuned in a low-data regime (512 images per task) across a four-task classification benchmark. While this allows for rapid iterative research and multi-seed reporting, it represents a toy-scale configuration. In modern edge systems, model merging and PTQ are most commercially valuable on multi-billion parameter autoregressive Large Language Models (LLMs) or large-scale Vision-Language Models (VLMs). The generalizability of Q-Merge to these generative foundation models is not empirically demonstrated.
*   **Low Parameter Drift Regime:** Because experts are fine-tuned on only 512 images, they remain structurally and parameter-wise close to the pre-trained base model. In real-world enterprise applications, expert models are fully fine-tuned on massive datasets, resulting in extreme parameter drift where weights diverge significantly, severely challenging linear mode connectivity. Empirical validation under extreme parameter drift is missing.

---

## 3. Ratings on Dimensions

### Soundness: Excellent
The paper is technically flawless and highly rigorous. Claims are thoroughly supported by multi-seed evaluations (means and standard deviations reported). The mathematical derivations of dual-path gradient flow through both weights and dynamic per-channel scale factors are correct and elegant. The paper addresses all standard potential flaws (such as backpropagation memory, scale factor discretization, and non-stationary calibration streams) head-on with robust formulations and ablations.

### Presentation: Excellent
The paper is exceptionally well-written, structured, and easy to follow. Figure 1 and the tables are highly detailed and self-contained. The inclusion of the comprehensive appendix preempts and addresses almost every common reviewer query, making the work exceptionally polished.

### Significance: Good
The paper addresses an important, highly practical deployment bottleneck. If scaled to larger models, Q-Merge has high potential to become a standard tool in resource-constrained edge systems. However, its current significance is somewhat limited by the toy-scale experimental evaluation (ViT-Tiny).

### Originality: Fair
The originality of the work is limited. It represents a straightforward, logical engineering combination of AdaMerging, PTQ, and STE. The solution to "4-bit collapse" (per-channel quantization) is the standard industry-wide PTQ prescription and is not an algorithmic novelty of Q-Merge. The paper does not introduce a fundamentally new mathematical formulation or change how the community conceptualizes parameter fusion.

---

## 4. Overall Recommendation

**Score: 4 (Weak Accept)**

**Justification:** 
The paper is technically solid and exceptionally rigorous in its execution, evaluations, and systems-level analyses. It successfully solves a highly practical deployment problem, demonstrating that optimizing layer-wise merging coefficients directly under the quantization operator is nearly lossless at 8-bit and highly viable at 4-bit. However, the conceptual novelty is incremental (a straightforward extension of AdaMerging with STE and standard PTQ blocks), and the empirical evaluations are restricted to a toy-scale setting (ViT-Tiny on classification). While the paper represents a valuable engineering contribution that practitioners and researchers will build upon, these limitations keep it below the bar of a full "Accept" for a major machine learning conference.

---

## 5. Detailed Questions and Constructive Feedback for the Authors

1.  **Scale of Experiments:** While the ViT-Tiny (5.7M parameters) benchmark is excellent for demonstrating the algorithmic mechanics and running exhaustive ablations, it is too small to convince practitioners deploying large generative models. Can the authors provide preliminary or full results on a larger backbone, such as CLIP-ViT-B/32 or a small LLM (e.g., LLaMA-1B) on standard downstream tasks (e.g., text classification or reasoning)?
2.  **High Parameter Drift Validation:** Standard model merging often fails when experts diverge far from the base model due to extensive fine-tuning. Since your current experts are trained on only 512 images, they represent a low parameter drift regime. Can you evaluate your framework on high-capacity expert models that have been fully fine-tuned on massive datasets to demonstrate how Q-Merge scales when linear mode connectivity is more severely challenged?
3.  **SVHN Performance:** Across all configurations, the absolute performance on SVHN is low (e.g., $41.34\%$ for unmerged experts, $35.87\%$ for Q-Merge). Is this due to the low-data fine-tuning budget (512 images) or are there other factors (e.g., severe optimization conflict with the other tasks)?
