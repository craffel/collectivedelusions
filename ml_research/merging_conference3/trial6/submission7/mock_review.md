# Peer Review: Parameter-Free Subspace Routing (PFSR) + Micro-Batch Homogenization (MBH)

**Review Title:** A Refreshing Minimalist Paradigm Shift in Dynamic Weight Merging with Rigorous Systems-ML Co-Design

---

## 1. Summary of the Submission
This paper presents a highly original and refreshing "minimalist" approach (applying Occam's razor) to dynamic weight-space routing for merging multi-task neural network experts under the Parameter-Efficient Fine-Tuning (PEFT/LoRA) paradigm. The authors identify and address two critical, overlooked failure modes in existing dynamic routing architectures: (1) **Optimization Bloat/OOD Overfitting** (e.g., in wave-inspired routers like QWS-Merge), and (2) **Heterogeneity Collapse** in mixed-task streams where batch-averaging task coefficients forces them toward a flat, uniform distribution, destroying task specificity.

To resolve these, the paper introduces:
1. **Parameter-Free Subspace Routing (PFSR):** A non-parametric, zero-shot framework that projects penultimate representations onto frozen expert classification heads via cosine similarity to derive routing coefficients, requiring zero trainable parameters, zero training, and zero calibration data.
2. **Micro-Batch Homogenization (MBH):** A stream-level batch partitioner that dynamically groups mixed-task streams into homogeneous micro-batches on the fly, performing specialized parameter merging and inference on each to completely bypass heterogeneity collapse.
3. **Statistical Class-Size Scaling Calibration:** Corrects statistical cosine similarity biases that occur in asymmetrical expert registries (e.g., LLM vs. classification experts).
4. **Systems-ML Co-designs:** Grounding the framework under LoRA to guarantee a strict $1.04\times$ memory footprint, implementing sequential weight materialization to cap RAM/VRAM on edge CPUs, and integrating parallel SGMV kernels to achieve true $O(1)$ constant-time inference on cloud GPUs with negligible overhead ($5.71\%$).

On a custom Isolating Coordinate Sandbox, Vision Transformers (DomainNet), and LLaMA-7B experts, the proposed PFSR + MBH + UNC framework systematically outperforms complex wave-inspired pipelines and strong static merging baselines, recovering up to $97.5\%$ and $96.8\%$ of the expert standalone ceilings under highly heterogeneous streams with zero optimization overhead.

---

## 2. Strengths and Weaknesses

### Strengths:
*   **Narrative and Conceptual Clarity (The Minimalist Philosophy):** The paper is exceptionally well-written, engaging, and adopts a powerful thematic persona ("The Minimalist") that runs consistently from the introduction's deconstruction of convoluted baselines to the method's elegant parameter-free formulation.
*   **Rigor of Empirical Deconstruction:** The systematic audit and exposure of wave-inspired QWS-Merge as unstable/overfitted, along with the demonstration that simple classical $L_2$ regularization matches or exceeds its performance, is a major, high-value service to the model-merging community.
*   **Rigorous Mathematical Soundness (Layer-Averaging Collapse Proof):** The paper provides a highly complete, elegant first-order Taylor approximation proving why layer-wise dynamic parameters collapse to collinear trajectories under single-head classification constraints, rendering multi-layer routing architectures mathematically redundant.
*   **Outstanding Systems-ML Integration:** The co-design with PEFT/LoRA, the deployment matrix (Table 11), the CPU sequential materialization strategy, and the parallel GPU SGMV benchmarks make this work exceptionally grounded in real-world systems engineering.
*   **Sub-Vocabulary Prototype Selection:** The data-free, parameter-centric pruning of massive vocabulary spaces ($C \ge 32,000$ to $C_{sub}=256$) relying purely on classification head weight variance is highly elegant, private, and computationally efficient.
*   **Plug-and-Play Adaptability:** Eliminating trainable routing parameters allows instantaneous, calibration-free addition or retirement of task experts in production model hubs.

### Weaknesses (Critical Constructive Feedback):
While the paper is of outstanding quality, we highlight up to **3 critical areas** that the authors should address or clarify:

1.  **Reliance on Simulated Penultimate Feature Representation Manifolds:**
    *   *Critique:* For the real-world Vision Transformer (DomainNet) and LLaMA-7B (NLP) benchmarks, the evaluations are *simulated* using representative feature embeddings and pre-calculated expert ceilings rather than running live, full-parameter active inference on raw splits. 
    *   *Impact:* While the authors are transparent about this resource-efficient protocol, simulated feature spaces are clean proxies that may hide practical, non-linear noise, representation distortions, or hardware caching bottlenecks that only manifest in full end-to-end forward passes on live weights. The paper would be significantly strengthened by validating at least one sub-scale sub-task on a live active model pipeline.
2.  **Infrastructure and Serving Complexity Trade-off:**
    *   *Critique:* PFSR + MBH aggressively prunes model-level parameters and calibration training but does so by shifting the engineering complexity to the underlying data-serving infrastructure. On-the-fly stream partitioning, dynamic weight-merging, sequential dispatching, and index-based scatter-gather output re-assembly require a highly sophisticated systems layer.
    *   *Impact:* Integrating advanced parallel GPU kernels (such as SGMV) introduces non-trivial CUDA compilation and software maintenance dependencies. For environments prioritizing pure infrastructure simplicity, this trade-off (model-level simplicity vs. system-level serving complexity) must be explicitly weighed.
3.  **Ambiguity in "Zero Calibration Split Data" Claim under OOD Rejection:**
    *   *Critique:* The paper’s central claim of "zero calibration split data" is slightly relaxed when integrating the high-yield Gaussian Mixture Model (GMM) Density Estimator for OOD rejection or offline $K$-means centroids for non-classification experts, as these require a small validation split of in-distribution samples to fit the coordinate density boundaries.
    *   *Impact:* While this low-dimensional GMM fitting (56 parameters) is highly sample-efficient and stable on 64-sample splits, the authors should harmonize their "completely training-free/zero-data" claims with this minor data dependency in the methodology text.

---

## 3. Detailed Evaluation

### Soundness: Good to Excellent
The mathematical formulations, coordinate projections, extreme-value statistical calibration (Eq. 2), and the first-order Layer-Averaging Collapse proof are exceptionally sound, complete, and mathematically rigorous. The systems-level co-designs (sequential materialization, SGMV parallelization, and UNC positive-definite ridge regularization) are methodologically robust. The main caveat preventing a perfect "excellent" is the reliance on simulated representation manifolds for the real-world evaluations.

### Presentation: Excellent
The writing is exceptionally clear, precise, and professional. The paper features outstanding visual layouts, with Figure 1 clearly illustrating the deconstructions and Algorithm 1 detailing the entire framework. Every table is accompanied by exhaustive, high-signal captions.

### Significance: Excellent
The paper addresses a highly important problem (heterogeneity collapse and over-parameterization in model merging). By introducing MBH, it shifts the robustness paradigm from complex weight-routing to clean data-stream engineering. Its actionable edge CPU guidelines and instantaneous dynamic task adaptation render it highly significant for real-world production registries and edge devices.

### Originality: Excellent
The conceptual shift of resolving stream heterogeneity at the data level (MBH), combined with non-parametric projection (PFSR) and extreme value statistical normalization for asymmetrical vocabularies (Eq. 2), is highly original and represents a major step forward for the field.

---

## 4. Overall Recommendation

**Rating:** **5: Accept**
*(An exceptionally solid, technically sound paper with high impact on weight-space model merging, excellent evaluation, outstanding systems co-design, and strong reproducibility.)*

**Justification:** 
The paper stands out as a highly valuable course correction for the model-merging community, elegantly deconstructing convoluted wave-inspired "quantum" routing architectures and replacing them with a completely non-parametric, zero-shot framework. By demonstrating that mixed-task streams can be resolved at the data level (MBH) under a strict PEFT/LoRA co-design, it achieves outstanding multi-task performance within tight VRAM limits. Despite minor limitations regarding simulated feature representation proxies and infrastructure complexity, the paper’s mathematical rigor, exhaustive ablations, and systems feasibility make it a clear Accept.

---

## 5. Actionable Suggestions for Revision
1.  **Delineate simulated vs. live benchmarks more prominently:** Ensure that the "simulated representations" protocol in Sec 4.4 and 4.5 is prominently highlighted in the main tables to prevent any reader misinterpretation.
2.  **Harmonize zero-data claims:** Soften the "zero calibration data" claim in the abstract/intro to reflect the minor, low-resource calibration split dependency introduced by GMM density estimation and non-classification centroid fitting.
3.  **Acknowledge Dynamic serving literature:** Connect MBH to high-performance serving and request batching literature (e.g., Orca, vLLM) in the Related Work to contextualize its dynamic partitioning approach within the broader systems-ML landscape.
