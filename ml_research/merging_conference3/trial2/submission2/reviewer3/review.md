# Peer Review of Conference Submission

**Title:** Spectral Model Merging via Singular Value Slicing (SVS)

---

## 1. Summary of the Paper
This paper investigates the complexity of multi-task model merging and proposes **Spectral Model Merging via Singular Value Slicing (SVS)** as a minimalist, training-free, and closed-form alternative to overparameterized or optimization-based merging frameworks. 

SVS operates under the hypothesis that downstream fine-tuning updates (task vectors) concentrate generalizable semantic knowledge on a low-dimensional spectral manifold, while high-frequency singular components correspond to fine-tuning noise that causes destructive coordinate-wise parameter interference when merged. To isolate this manifold, SVS performs a standard Singular Value Decomposition (SVD) on task vectors, retaining only the top $k$ principal singular components. 

To preserve the activation scales across deep, un-normalized layers, the authors propose **Barycentric Weight Normalization (BWN)**, an analytical scale-preservation operator. Furthermore, they prove mathematically that global weight scaling factors are neutralized under modern feature normalization operators (L2-normalization, LayerNorm, RMSNorm), rendering explicit scale-matching algorithms redundant in fully normalized architectures (such as CLIP). 

Lastly, they introduce an adaptive, information-theoretic rank allocation scheme (**Entropy-SVS**) that scales layer-wise slicing ranks $k_l$ using the Shannon spectral entropy of each layer's singular value distribution, allowing the network to dynamically allocate capacity where update trajectory complexity is high.

---

## 2. Strengths and Weaknesses

### Major Strengths
* **Rigorous Mathematical Grounding:** In contrast to many existing model merging techniques that rely on coordinate-basis heuristics, randomized dropout masks, or iterative optimization, this work is elegantly grounded in numerical linear algebra (SVD, Eckart-Young-Mirsky Theorem) and information theory (Shannon entropy).
* **Elegant scale-invariance Proofs (Section 3.4):** The mathematical proofs demonstrating how L2, LayerNorm, and RMSNorm cancel out positive global scaling factors provide a beautiful, unifying theoretical justification for why explicit scale-preservation algorithms are redundant in normalized networks. This reframes and simplifies a large body of literature on scale preservation.
* **Information-Theoretic Rank Allocation (Section 3.5):** Developing a dynamic layer-wise rank allocation framework using the Shannon spectral entropy of the singular value spectrum is a principled, highly adaptive conceptual contribution. This shifts the paradigm from uniform rank-picking to locally adaptive, representation-aware low-rank projection.
* **Exemplary Empirical Integrity & Validation:** The authors evaluate SVS across the entire 86M parameter visual backbone of CLIP-ViT-B/32. Crucially, they conduct a controlled, un-normalized MLP experiment to empirically validate the scale-preservation behavior of BWN when normalization is absent, demonstrating thoroughness and scientific rigour.
* **Intellectual Honesty in Analysis:** The paper provides a highly insightful discussion of the "Representation Gap" between continuous spectral filtering (SVS) and discrete coordinate pruning (TIES/DARE), pointing out why coordinate-basis pruning is more effective at preventing cross-layer representation interference in sequential transformer backbones.

### Major Weaknesses / Theoretical Areas for Improvement
* **Omission of the Bias Term in Scale-Invariance Proofs (Section 3.4.1 & 3.4.2):**
  In Case 1 (L2-norm) and Case 2 (LayerNorm), the cancellation of the global weight scaling factor $\alpha > 0$ relies on the assumption that the output of the layer scales exactly linearly with $\alpha$, i.e., $\mathbf{h}_{BWN} = \alpha \mathbf{h}$. However, the linear layer is defined as $\mathbf{h} = X W^T + \mathbf{b}$. Because bias vectors $\mathbf{b}$ are merged via standard linear Task Arithmetic and are **not scaled by $\alpha$** (as stated in Section 3.3), the scaled projected activations become:
  $$\mathbf{h}_{scaled} = \alpha X W^T + \mathbf{b} \ne \alpha (X W^T + \mathbf{b})$$
  Whenever a non-zero bias vector is present and not scaled proportionally, $\alpha$ does not factor out exactly, and thus does not mathematically cancel out. The scale-invariance is technically an approximation rather than an exact identity.
* **Inaccurate Linear-Scaling Assumption in Residual Blocks (Section 3.4.4):**
  The authors assert that under global scaling by the BWN factor $\alpha > 0$, the parameterized block $\mathcal{F}$'s output scales linearly: $\mathcal{F}_{\alpha}(\text{LN}(\mathbf{x})) = \alpha \mathcal{F}(\text{LN}(\mathbf{x}))$. This is mathematically incorrect for standard Transformer block components:
  1. *In the Multi-Head Attention (MHA) block:* Scaling query and key projection weights by $\alpha$ scales the input to the softmax quadratically ($\alpha^2$), altering the attention distribution non-linearly as a temperature scaling parameter:
     $$\text{Attention}_{scaled} = \text{softmax}\left(\alpha^2 \frac{Q K^T}{\sqrt{d_k}}\right) (\alpha V)$$
  2. *In the MLP block:* Modern activation functions (such as GELU, Swish/SiLU) are non-linear and not homogeneous of degree 1 (i.e., $\text{GELU}(\alpha z) \ne \alpha \text{GELU}(z)$). Thus, scaling weights by $\alpha$ does not scale the output of the MLP block linearly.
  
  The scale-invariance inside residual blocks fails not just because the skip connection is unscaled, but also because MHA and MLP blocks are non-linear operators that do not exhibit linear homogeneity with respect to weight scaling.

---

## 3. Soundness
* **Rating:** Good
* **Justification:** The core methodology (SVS, BWN, and Entropy-SVS) is mathematically elegant and empirically sound. SVD represents the optimal low-rank projection under the Eckart-Young-Mirsky Theorem. However, the rating is capped at "Good" due to the subtle theoretical gaps in Section 3.4's proofs (omission of the bias term in L2/LayerNorm cancellation, and the incorrect assumption of linear homogeneity in MHA/MLP blocks inside residual connections). Correcting these theoretical oversimplifications in the final version would easily elevate Soundness to "Excellent."

---

## 4. Presentation
* **Rating:** Excellent
* **Justification:** The paper is exceptionally well-written, structured, and easy to follow. The mathematical notation is clean and rigorous. The authors do an outstanding job of clearly conveying both their high-level intuition and low-level technical formulations. The empirical figures and tables are informative and highly supportive of the core narrative.

---

## 5. Significance
* **Rating:** Excellent
* **Justification:** This paper makes a highly significant contribution to the model merging literature. By demonstrating that competitive multi-task consolidation can be achieved through closed-form linear algebra, it challenges the need for complex, overparameterized, and computationally expensive optimization-based merging schemes. Furthermore, the Shannon spectral entropy rank allocation (Entropy-SVS) is a beautiful, highly generalizable contribution that could find applications in broader areas such as network compression and LoRA.

---

## 6. Originality
* **Rating:** Excellent
* **Justification:** While post-hoc SVD-based parameter compression has been explored concurrently, this work is highly distinguished by its original theoretical and algorithmic contributions. It provides the first formal proof of global scaling cancellation in normalized architectures, and introduces a completely novel, information-theoretic dynamic rank allocation scheme based on singular value Shannon entropy.

---

## 7. Overall Recommendation
* **Rating:** 5: Accept
* **Justification:** This is an outstanding, mathematically grounded, and scientifically rigorous paper. It successfully consolidates multiple experts into a single multi-task model with zero additional trainable parameters or test-time steps. SVS at rank $k=128$ (utilizing only $16.7\%$ of the rank space) matches or exceeds the performance of full-rank Task Arithmetic. Additionally, Entropy-SVS achieves up to $65.7\%$ average rank compression with virtually no loss in downstream accuracy. Despite the minor theoretical oversimplifications identified in Section 3.4's proofs, the paper's intellectual honesty, conceptual elegance, and strong empirical validation make it a highly valuable and complete contribution that is ready for acceptance.

---

## 8. Questions and Constructive Feedback for the Authors
1. **The Bias Term:** In Section 3.4.1 and 3.4.2, can you discuss the impact of non-zero bias vectors on the scale-invariance proofs? Since biases are merged via standard linear Task Arithmetic (and not scaled by the BWN coefficient $\alpha$), the cancellation of $\alpha$ is technically approximate rather than exact.
2. **Residual Block Scaling:** In Section 3.4.4, you state that the parameterized block $\mathcal{F}$ scales linearly under global weight scaling. However, because MHA involves a non-linear softmax (which undergoes a temperature shift of $\alpha^2$ when query/key projection weights are scaled) and MLP blocks use non-homogeneous activations (GELU/Swish), the output $\mathcal{F}_{\alpha}$ does not scale linearly as $\alpha \mathcal{F}$. I suggest updating this explanation to reflect that scale-invariance inside residual connections fails both because the identity path is unscaled and because MHA/MLP blocks are non-linear, non-homogeneous operators.
3. **Entropy-SVS Multi-Task Aggregation:** In Section 3.5, you define $k_l$ using the task-specific update matrix $T_t$. Since there are $N$ distinct task vectors at each layer, each with potentially different spectral distributions and entropies, how is the final layer-wise rank $k_l$ determined? Is $H(T_t)$ computed for each task vector individually and the resulting ranks averaged, or do you take the maximum? Clarifying this detail would improve reproducibility.
4. **Uniqueness:** In Section 3.2, under the Eckart-Young-Mirsky Theorem, the low-rank projection is unique if and only if the singular values satisfy $\sigma_k > \sigma_{k+1}$. I recommend adding this standard theoretical qualification for complete mathematical precision.
