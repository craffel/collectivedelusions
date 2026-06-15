# Peer Review

## Summary of the Paper
This paper addresses the critical problem of dynamic weight-space routing in multi-task model merging. It identifies and deconstructs two major overlooked failure modes of existing parameterized routing models (e.g., QWS-Merge and layer-wise routers): (1) optimization bloat and out-of-distribution (OOD) overfitting due to redundant trainable parameters trained on tiny calibration splits, and (2) *heterogeneity collapse*, where mixed-task input streams force batch-wise coefficients to average out, destroying task specificity during dynamic weight blending.

To resolve these issues, the paper introduces **Parameter-Free Subspace Routing (PFSR)** combined with **Micro-Batch Homogenization (MBH)**—a training-free, zero-shot framework containing zero trainable parameters and requiring zero calibration data. PFSR projects penultimate representations onto a frozen task coordinate subspace using cosine similarity against pre-trained expert classification weights, deriving gating coefficients via a temperature-scaled Softmax. MBH shields weight merging from heterogeneity collapse by dynamically partitioning incoming mixed-task batches into homogeneous micro-batches on the fly, performing tailored inference on each, and re-sorting predictions back to the original batch sequence. 

Additionally, the paper presents a rigorous first-order Taylor expansion proof demonstrating **Layer-Averaging Collapse**, explaining why layer-wise multi-layer routers collapse to a redundant single-layer search space and are systematically outperformed by simple global, single-layer linear routers when merging weights under shared classification heads. The entire framework is evaluated on a synthetic Sandbox and a real-world Vision Transformer (ViT) benchmark on DomainNet, backed by exhaustive hardware and systems scaling audits.

---

## Strengths

1. **Elegance and Simplicity (Occam's Razor):** Stripping away $100\%$ of trainable routing parameters and eliminating low-resource calibration training entirely represents a highly elegant and refreshing research direction. Replacing a complex, wave-inspired optimization pipeline (which overfitting audits reveal to be unstable) with a parameter-free projection is structurally clean and highly effective.
2. **Data-Level Stream Partitioning:** Shifting the resolution of stream heterogeneity from complex model-level constraints (which suffer from the "Robustness-Accuracy Illusion") to data-stream partitioning (MBH) is a major, highly practical contribution. It achieves perfect task-specific specialization under heterogeneous streams while remaining completely transparent to downstream applications.
3. **Rigorous Theoretical Grounding:** The first-order Taylor expansion proof of Layer-Averaging Collapse is mathematically sound and provides deep, valuable insights into the optimization dynamics of weight merging. It demystifies the redundancy of multi-layer routers and aligns perfectly with the empirical findings.
4. **Exhaustive Systems-ML Co-design and Scaling Audits:** The paper goes far beyond typical accuracy tables by providing extensive systems-level evaluations. The authors address all standard critiques—measuring VRAM footprints under LoRA, vocabulary latency scaling in LLMs ($C=32,000$), expert scalability ($K=16$), and parallel multi-adapter GPU execution (using Punica-style SGMV kernels on an NVIDIA A100 GPU). This comprehensive systems validation makes the methodology highly credible.

---

## Weaknesses & Areas for Improvement

While the paper is outstanding, several technical areas require closer inspection and would benefit from constructive refinement:

1. **The Infrastructure Complexity Trade-off:** There is a subtle paradox in the paper's minimalist claims. While PFSR eliminates *routing parameters*, MBH shifts complexity from the model architecture to the **data-serving infrastructure**. On-the-fly stream partitioning, dynamic weight-merging, sequential inference passes, and index scatter re-assembly (or integrating parallel SGMV CUDA kernels) introduce a highly complex serving layer. The authors should explicitly discuss and acknowledge this trade-off between model simplicity and infrastructure-serving complexity.
2. **Dependence on Representational Compatibility under Drift:** PFSR and UNC rely herein on the assumption that the experts' penultimate representations remain semantically aligned. While this holds under PEFT/LoRA (as shown by their representation distance audit), they show that under full fine-tuning with large learning rates, representation manifolds undergo severe topological divergence. Although they propose mitigation strategies (MLP projections, contrastive objectives), these require training, which violates the core "zero trainable parameters" and "zero-shot" claims of their framework.
3. **Generalizability to Non-Classification and Regression Tasks:** The similarity projection in Eq. 1 relies on computing cosine similarities against class prototype rows. This limits its direct applicability to regression tasks and generative networks (like diffusion models) that lack classification heads. While they propose constructing task-alignment anchors using K-means clustering on a calibration split, this proposal remains purely theoretical. Including even a small proof-of-concept empirical validation for this non-classification extension would substantially strengthen this section.
4. **Performance Trade-offs in OOD Cosine Rejection:** Table 7 sweeps the OOD cosine rejection threshold $\gamma_{OOD}$. While setting $\gamma_{OOD} = 0.4$ successfully rejects $91.60\%$ of SVHN samples, it mis-rejects $23.73\%$ of in-distribution samples, causing overall Joint Mean accuracy to drop significantly from $71.50\%$ to $63.20\%$. Although they mention in text that a Gaussian Mixture Model (GMM) density estimator resolves this (rejecting 95.20% SVHN, 4.30% false-positive rate, 74.10% Joint Mean), these GMM results are not included in the main tables. Moving this GMM sweep into the main tables would make this critical OOD mitigation much more prominent and convincing.

---

## Questions for the Authors

1. Could you provide details on the latency of the indexing and scatter re-assembly step? In unaccelerated CPU environments, does the sequential dispatch of $G \le K$ active micro-batches introduce significant overhead, and how does this scale as the expert pool $K$ grows larger?
2. How sensitive is the GMM density estimator to the size of the calibration split? Does fitting a robust covariance matrix require significantly more than the 64-sample calibration split used in the baseline experiments?
3. In fully fine-tuned models where representational drift is severe, is it possible to perform PFSR on representations extracted from earlier, frozen layers of the backbone (e.g., before task-specific divergence becomes severe)? Have you evaluated this potential parameter-free mitigation strategy?

---

## Ratings

*   **Soundness:** Excellent (5/5) — The claims are fully supported by mathematically rigorous proofs, thorough ablations, systems audits, and strong empirical results on both synthetic and real-world benchmarks.
*   **Presentation:** Excellent (5/5) — The paper is exceptionally well-written, engaging, clearly structured, and provides complete details for reproducibility.
*   **Significance:** Good (4.5/5) — The paper addresses a major bottleneck in dynamic model merging and offers highly practical, hardware-efficient serving pipelines, serving as a valuable corrective to architectural over-engineering.
*   **Originality:** Excellent (5/5) — The combination of zero-shot prototype similarity projection (PFSR) and stream partitioning (MBH) is highly creative, and the Layer-Averaging Collapse proof is highly original.
*   **Overall Recommendation:** Accept (5/5) — A technically solid, beautifully written, and highly thorough paper that makes a strong contribution to the model-merging and systems-ML communities with high practical utility.
