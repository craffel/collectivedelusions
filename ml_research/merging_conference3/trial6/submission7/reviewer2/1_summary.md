# Summary of the Paper

## 1. Main Topic
The paper addresses the problem of dynamic weight-space routing for merging multi-task neural network experts. Specifically, it focuses on identifying and mitigating two critical failure modes in existing dynamic model-merging routers:
1. **Optimization Bloat and Out-of-Distribution (OOD) Overfitting:** Previous parameterized dynamic routers (e.g., wave-inspired or multi-layer architectures) require training via iterative optimization on small calibration sets, leading to severe overfitting and catastrophic failure under out-of-distribution tasks.
2. **Heterogeneity Collapse:** Under mixed-task streaming deployment, dynamic routers that average sample-wise routing coefficients across the batch dimension to satisfy hardware accelerator constraints suffer from a flattening of coefficients, destroying task-specific expert specialization.

## 2. Approach
To resolve these limitations under a strict minimalist philosophy, the authors propose a co-designed algorithm-systems framework consisting of:
- **Parameter-Free Subspace Routing (PFSR):** A non-parametric, training-free routing method. It projects penultimate-layer feature representations onto the task coordinate subspace of frozen pre-trained expert classification heads using cosine similarity, deriving routing coefficients via a temperature-scaled Softmax.
- **Unit-Norm Calibration (UNC):** A training-free calibration step that applies unit-norm normalization to features and class prototype weights to resolve cross-expert representation and scale mismatches.
- **Class-Size Scaling Calibration:** A statistical normalization factor ($O(\sqrt{\log C_k / d})$) applied to raw cosine similarity scores to prevent statistical over-routing toward experts with larger vocabulary or label spaces.
- **Micro-Batch Homogenization (MBH):** A data-stream scheduling mechanism that dynamically partitions mixed-task input batches into homogeneous micro-batches based on dominant task projections, performs specialized merged-model inference on each, and re-assembles/re-sorts the outputs.
- **Bounded Top-$k$ Routing:** A mechanism to cap the number of active micro-batches $G \le k$ when scaling to large expert counts, maintaining constant-time inference scaling.
- **Sub-Vocabulary Prototype Selection:** A data-free heuristic selecting high-variance classification weights to prune the search space for large vocabulary tasks (e.g., LLMs), bypassing computational scaling bottlenecks.

## 3. Key Findings
- **OOD Collapse:** Parametric dynamic routing networks (like Quantum Wavefunction Superposition Merging - QWS-Merge) suffer from transductive overfitting on small calibration splits (64 samples), dropping to $10.00\%$ accuracy on SVHN. Classical $L_2$ regularization on standard linear routers easily replicates or outperforms these wave-inspired architectures.
- **Layer-Averaging Collapse:** Layer-wise routing networks are redundant and collinear because the contractive dynamics of deep-layer Jacobians project the gradient signals from a shared joint classification head onto a shared dominant task subspace. Hence, a global, single-layer Linear Router systematically outperforms multi-layer routers.
- **Batch Heterogeneity Degradation:** Shuffled, mixed-task batches force standard dynamic routers to average coefficients across different task requirements, resulting in a flat uniform average ($\approx 0.25$) and dropping accuracy (e.g., the Linear Router drops from $51.00\%$ to $43.40\%$).
- **Collapse Prevention:** Resolving mixed streams at the scheduling level via MBH completely shields weight-merging averages from cross-task interference. PFSR + MBH achieves a collapse-free $71.60\%$ Joint Mean accuracy on heterogeneous streams under a synthetic sandbox, matching sample-wise performance.
- **VRAM Viability via PEFT (LoRA):** Framing dynamic merging under the LoRA paradigm restricts the active memory footprint to $1.04\times$ base model size, allowing dynamic, on-device kernel-fused weight updates in milliseconds while bypassing PCIe transfer bounds.

## 4. Explicitly Claimed Contributions (with Evidence)
1. **Empirical Deconstruction of QWS-Merge:** Proves that wave-inspired routing is highly unstable and collapses on OOD tasks, showing that $L_2$ regularization on a basic linear router delivers superior stability. (Supported by Figure 1a, Table 1).
2. **Analytical and Empirical Proof of Layer-Averaging Collapse:** Proves mathematically and empirically that layer-specific routing coefficients are collinear and redundant under joint classification constraints. (Supported by Section 3.6, Figure 1b, Table 1).
3. **Introduction of PFSR + MBH + UNC:** Proposes a zero-shot, parameter-free dynamic merging framework. Under homogeneous streams, it achieves $75.00\%$ Joint Mean accuracy on a synthetic sandbox and maintains $71.60\%$ on heterogeneous streams. (Supported by Section 3, Table 1, Table 2).
4. **Diverse Real-World Evaluations:** Validates the framework on DomainNet using Vision Transformers (ViT-Base) and LLaMA-7B on NLP benchmarks (GSM8K, HumanEval, WMT, Alpaca), achieving $78.50\%$ and $79.12\%$ Mean accuracy respectively. (Supported by Table 5, Table 6).
5. **PEFT Integration for VRAM Viability:** Grounding dynamic weight merging using low-rank adapters to cap the memory footprint at $1.04\times$ base model size and analyzing systems trade-offs. (Supported by Section 4.5, Table 3, Table 4).
6. **Instantaneous Dynamic Task Adaptation:** Explains that because PFSR is training-free, practitioners can add/delete experts on the fly by updating the projection matrix, requiring zero retraining or joint calibration. (Supported by Section 4.6).
