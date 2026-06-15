# Intermediate Evaluation 5: Impact and Presentation

This document evaluates the overall strengths, areas for theoretical/conceptual improvement, writing quality, and potential significance/impact of the proposed paper.

## 1. Major Strengths
* **Rigorous Mathematical Grounding:** Unlike many model merging papers that rely heavily on coordinate-basis heuristics or randomized trials, this paper is grounded in fundamental numerical linear algebra (SVD, Eckart-Young-Mirsky Theorem) and information theory (Shannon entropy).
* **Elegant scale-invariance Proofs:** The derivation of scale-invariance under L2-normalization, LayerNorm, and RMSNorm is a beautiful contribution that exposes a major blind spot in model merging literature, explaining why complex scale-preservation algorithms are redundant in modern architectures.
* **Closed-Form, Dynamic Rank Allocation (Entropy-SVS):** Moving from uniform ranks to an information-theoretic, dynamic layer-wise capacity allocator represents a major step forward, achieving up to $65.7\%$ rank compression with near-zero performance loss.
* **Empirical Integrity:** The authors evaluate their methods on a complete 86M parameter backbone and conduct a controlled, un-normalized MLP validation to prove their boundary conditions. All claimed metrics are perfectly consistent with the raw experimental data.
* **Intellectually Honest Analysis:** The discussion of the "Representation Gap" between continuous spectral filtering (SVS) and discrete coordinate pruning (TIES/DARE) is highly insightful and provides a clear conceptual direction for future hybrid merging research.

---

## 2. Areas for Theoretical and Conceptual Improvement
To meet the absolute highest standards of mathematical and theoretical rigor, the authors must address the following limitations and gaps in their current draft:

### A. Address the Bias Term in Scale-Invariance Proofs (Section 3.4.1 & 3.4.2)
The proofs demonstrating exact cancellation of the global weight scaling factor $\alpha$ are technically only exact when the layer's bias vector is zero, or scaled proportionally. Because the authors merge biases via standard linear Task Arithmetic (omitting them from BWN), the scale-invariance is technically a high-precision approximation rather than an exact mathematical identity. The authors should explicitly qualify their proofs to account for non-zero biases.

### B. Correct the Residual Block Scaling Assumption (Section 3.4.4)
The statement that the parameterized block $\mathcal{F}$ scales linearly under global weight scaling ($\mathcal{F}_{\alpha}(\text{LN}(\mathbf{x})) = \alpha \mathcal{F}(\text{LN}(\mathbf{x}))$) is mathematically incorrect for standard MHA and MLP blocks:
* In **MHA**, scaling query and key projection weights scales the input to softmax quadratically ($\alpha^2$), modifying the attention distribution non-linearly (as a temperature scale).
* In **MLP**, modern activation functions (GELU, Swish/SiLU) are non-linear and not homogeneous of degree 1, meaning scaling their input by $\alpha$ does not scale their output by $\alpha$.

The authors must correct this explanation. The scale invariance fails inside residual blocks not only because the skip connection is unscaled, but also because MHA and MLP blocks are non-linear operators that do not exhibit linear homogeneity with respect to weight scaling.

### C. Detail Multi-Task Entropy Aggregation in Entropy-SVS (Section 3.5)
The authors should clarify how layer-wise ranks $k_l$ are determined when multiple task-specific update vectors $T_1, \dots, T_N$ exist at each layer. Are the Shannon spectral entropies of the task vectors averaged, or is a separate rank $k_{t, l}$ computed and applied to each task vector individually before they are combined?

---

## 3. Overall Presentation Quality
The presentation quality is **excellent**. The paper is beautifully structured, highly readable, and uses scholarly, precise language. The mathematical formulations are clean, and the notation is consistent. The figures and tables are informative and highly supportive of the core narrative. The limitations and future directions are discussed openly and comprehensively.

---

## 4. Potential Significance and Impact
The paper has the potential to make a **significant impact** on the model merging and parameter-efficient fine-tuning (PEFT) communities:
* It challenges the trend toward increasingly complex, overparameterized, and computationally expensive optimization-based merging schemes, demonstrating that competitive multi-task consolidation can be achieved through simple, closed-form linear algebra.
* It provides a clear, mathematically sound justification for why explicit weight-scale-preservation is redundant in modern Transformer architectures.
* It introduces an elegant, information-theoretic framework (singular value Shannon entropy) for analyzing and allocating layer-wise informational capacity, which could find applications in broader areas such as network pruning, quantization, and low-rank adaptation (LoRA).
* The concept of hybrid spectral-spatial merging (SVS + coordinate pruning) proposed in the limitations section could inspire a highly powerful new family of model merging algorithms.
