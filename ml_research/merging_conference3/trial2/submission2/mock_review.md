# Peer Review

**Title:** Occam's Razor in Weight Space: Spectral Model Merging via Singular Value Slicing  
**Reviewer Recommendation:** Accept (Score: 5)  
**Soundness:** Excellent  
**Presentation:** Excellent  
**Significance:** Good  
**Originality:** Good  

---

## 1. Summary of the Paper
The paper addresses the challenge of multi-task model merging—the process of consolidating multiple task-specific expert neural networks into a single multi-task model without training or test-time adaptation. The authors challenge the growing complexity of contemporary merging frameworks (which rely on overparameterized normalizing flows, auxiliary parameters, or active test-time optimization streams) and advocate for a simpler, closed-form linear algebra approach.

The authors propose two primary operators:
1. **Spectral Model Merging via Singular Value Slicing (SVS):** A non-parametric, training-free, and closed-form operator that performs Singular Value Decomposition (SVD) on task-specific weight updates (task vectors) and retains only the top $k$ principal singular components. This serves as an analytical low-pass filter to remove fine-tuning noise and reduce parameter conflicts.
2. **Barycentric Weight Normalization (BWN):** An analytical scale-preservation operator that matches the merged weights' Frobenius norm to the weighted average of the original experts.

Crucially, the paper provides a mathematical proof demonstrating that in modern normalized architectures (such as those containing L2-normalization, LayerNorm, or RMSNorm), global weight scaling is factored out and completely canceled, rendering scale-preservation steps empirically redundant. The authors also propose **Entropy-SVS**, an information-theoretic scheme that uses Shannon spectral entropy of singular values to adaptively allocate slicing ranks across different layers of a deep network backbone completely offline. 

SVS is evaluated on the full 86M parameter visual backbone of CLIP-ViT-B/32 across four diverse image classification datasets (MNIST, FashionMNIST, CIFAR-10, SVHN).

---

## 2. Key Strengths

### 1. Philosophical Rigor and Elegant Simplicity (Occam's Razor)
This paper is a refreshing departure from the trend of designing increasingly bloated, highly parameterized, and fragile optimization-driven model merging frameworks. By showing that a simple, closed-form, training-free, and non-parametric linear algebra projection (SVS) can match or outperform standard full-rank Task Arithmetic, the paper demonstrates the power of elegant, minimalist design. If a simple method can match a complex one, the simpler one is strictly superior.

### 2. Outstanding Theoretical Contributions (Section 3.4)
The formal proof demonstrating why positive global weight scaling factors are completely neutralized in modern normalized architectures (LayerNorm, RMSNorm, L2-norm) is a **highly significant and elegant theoretical contribution**. This demystifies the properties of weight-space merging and exposes that many previous works spent considerable effort optimizing scaling parameters that are ultimately factored out and canceled by standard normalization layers during a forward pass. The discussion of the "Residual Block Boundary Condition" is also an excellent, honest addition that explains when scaling *does* matter (by regulating the relative ratio of the skip connection vs. the main path).

### 3. Mature Scientific Honesty and Intellectual Integrity
The paper does not employ standard academic obfuscation to hide when its method is outperformed. It openly reports that TIES-Merging and DARE outperform SVS in average multi-task accuracy. More importantly, it provides an **exceptionally deep, high-signal conceptual analysis of this representation gap** (Section 4.2). It explains that while spectral-domain filtering successfully isolates the core semantic directions of individual experts, it produces *dense* updates that still overlap in the spatial coordinate-basis, leading to cascading localized interference. This level of intellectual depth and theoretical clarity adds massive scientific value to the community, far exceeding a slightly higher number on a leaderboard.

### 4. Adaptive Information-Theoretic Rank Allocation (Entropy-SVS)
Entropy-SVS is a highly practical, training-free contribution. Using Shannon spectral entropy to dynamically allocate ranks across layers respects the architectural hierarchy of deep networks (where different layers encode updates at different complexities). The authors demonstrate a robust Pareto frontier: compressing $15\%$ of the rank space losslessly (retaining $74.80\%$ accuracy vs. SVS's $74.83\%$), and compressing up to $65.70\%$ of the rank space with a negligible drop in accuracy ($74.55\%$).

### 5. Excellent Literature Contextualization and Rigor
The authors provide outstanding contextualization against recent SVD-based merging literature, citing *Task Singular Vectors (TSV)* (Gargiulo et al., CVPR 2025), *Model Merging with SVD to Tie the Knots* (Stoica et al., ICLR 2025), and *Ortho-Merge* (2025). They cleanly position SVS as a purified, purest-form offline baseline, while clearly highlighting their own unique theoretical scale-invariance and adaptive rank allocation contributions.

### 6. Outstanding Presentation Quality & Scalability Analysis
The manuscript is beautifully written, highly structured, and mathematically precise. The introduction of Randomized SVD (Halko et al., 2011) successfully mitigates cubic SVD scalability concerns for multi-billion parameter LLMs. The paper is highly reproducible due to its closed-form, deterministic nature, and SVD Caching optimizes sweep times to under 1.2 seconds.

---

## 3. Weaknesses / Areas for Improvement (Constructive & Minor)

### 1. Discussion of Architectural Characteristics of LLMs
The evaluation is currently restricted to a Vision Transformer of size 86M (CLIP-ViT-B/32). While the authors successfully introduce Randomized SVD as an $\mathcal{O}(m n \log k)$ scalability solution, evaluating the method on multi-billion parameter LLMs (e.g., Llama-3, Mistral) represents an exciting next step.
- **Actionable Suggestion:** The authors could add a brief sentence discussing specific architectural characteristics of LLMs (such as gated SwiGLU MLP layers or causal attention blocks) that would require special considerations when applying SVS.

### 2. Deep Geometric Analysis of Flattening Axis Sensitivity
The authors flatten higher-dimensional tensors by grouping the output channel dimension and flattening other dimensions. While structurally sound and empirically supported by pilot evaluations, a more comprehensive exploration of alternative flattening axes would provide valuable geometric insights.
- **Actionable Suggestion:** Add a note in future directions suggesting a dedicated sensitivity study of tensor flattening choices.

---

## 4. Questions for the Authors
1. **Regarding the SVD flattening choice:** How sensitive is the singular value spectrum of SVS to the choice of the flattening axis? If you group the input channels instead of the output channels, does it significantly alter the multi-task accuracy or the Shannon spectral entropy of the layers?
2. **Regarding LLM structures:** How do you anticipate SVS performing on the gated SwiGLU projection layers of modern LLMs compared to standard multi-head attention weights?
3. **Regarding the hybrid merging direction:** You provide a highly valuable analysis of the representation gap between spectral-domain low-pass filtering (SVS) and spatial coordinate-basis pruning (TIES/DARE). Do you think a hybrid pipeline that applies SVS first as a continuous spectral filter, followed by TIES magnitude thresholding, could successfully combine the strengths of both paradigms?

---

## 5. Final Recommendation
This is a beautifully written, mathematically elegant, and philosophically refreshing paper. It successfully strips away unnecessary complexity to establish a clean, robust, and scale-invariant baseline for model merging. The mathematical proofs of scale-invariance represent an outstanding conceptual contribution that directly simplifies model merging theory. The authors have done an outstanding job addressing all previous concerns, and the submission is highly polished and ready for publication. I strongly recommend **Accept**.
