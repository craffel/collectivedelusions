# Presentation, Impact, and Significance Evaluation

## Strengths
1. **Outstanding Presentation and Readability:** The paper is exceptionally well-written, with a clear narrative, logical flow, and highly polished terminology. The formatting and structures are clean and professional, using standard academic writing conventions.
2. **Clear and Informative Illustrations:** Figure 1 (Heterogeneity Collapse) and Figure 2 (SLD-Merge Pipeline Overview) are of high quality and greatly assist the reader in quickly understanding the core motivation and the proposed pipeline.
3. **High Practical Significance:** For edge and on-device deployment (mobile, NPU, IoT), the proposed method offers a highly attractive trade-off:
   - **92.5% storage savings** for specialized task parameters.
   - **8.3% computational overhead** (FLOPs) during inference.
   - **Complete batch independence** which is essential for safe, predictable production deployments.
4. **Strong Theoretical/Empirical Generalization Insights:** The discovery that SVD low-rank truncation acts as a regularizer in low-resource regimes (outperforming full-rank baselines) is a significant academic finding that opens up interesting research directions on the implicit regularization of low-rank matrices in transfer learning.

## Areas for Improvement

### 1. Literary Contextualization & Academic Humility
From a Scholarly perspective, the paper would benefit from a more thorough and humble positioning within the historical academic literature:
- **Prototypical Representations:** The "Activation-Space Mean Initialization" is presented as a novel, custom design, but it is structurally identical to prototypical representation learning (Snell et al., 2017) and nearest-centroid classifiers. Citing this connection would strengthen the scholarly foundation of the paper.
- **SVD Task vector Factorization:** Taking the SVD of task vectors or weight matrices is an active area of research in model merging and parameter-efficient adapters (e.g., *AdaRank*, *FRISM*, *DiDi-Merging*). The paper should cite these concurrent works to accurately map the 2025/2026 merging landscape.

### 2. Clarity on Implementation vs. Efficiency Claims
The paper should clarify whether their vectorized parallel forward pass equation:
$$Y = X W_{\text{base}}^{(l)} + \sum_{k=1}^K \alpha_k \odot \left( (X A_k^{(l)}) B_k^{(l)} \right)$$
actually achieves $O(1)$ expert execution per sample in PyTorch, or if it naively evaluates all $K$ paths. If it evaluates all $K$ paths, the actual computational savings are only realized for $B=1$ (or if custom indexing/scatter-gather operations are used). A brief technical note on this implementation detail in the methodology or appendix would prevent misunderstandings.

### 3. Scaling Properties and Limitations
- **Model Size and Task Complexity:** The paper evaluates on `vit_tiny_patch16_224` (5.7M parameters) and very small subsets of public datasets (256 training samples). While this represents a challenging low-shot setting, it leaves open the question of how the method scales to standard-scale models (e.g., Llama-3, Mistral, ViT-Base) and full-scale datasets.
- **Limitations of Cosine Routing:** The cosine router normalizes activations, discarding magnitude information. The authors should briefly discuss how this might limit performance on fine-grained or hierarchical classification tasks, where relative activation magnitude may be critical for disambiguating closely related sub-categories.

## Potential Impact and Significance
The potential impact of this paper is **high**, particularly in the on-device and edge AI space. By demonstrating how to achieve stateless, batch-independent, and extremely storage-efficient dynamic model merging, this work provides a highly practical blueprint for deploying multi-task models on resource-constrained hardware. Furthermore, the conceptual bridge between weight merging and Mixture of Experts (MoE) via post-hoc SVD represents a valuable research direction that could influence future hybrid architectures.
