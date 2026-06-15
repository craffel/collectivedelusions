# 2. Novelty and Delta Analysis

## Key Novel Aspects
The proposed **CAM-Router** introduces the following novel aspects to the domain of weight-space dynamic model merging:
1. **Preservation of Spatial Token Sequences in Routing:** Instead of immediately flattening/pooling the hidden features into a single representation, CAM-Router retains the full spatial token dimension ($N$) from the early stages of a Vision Transformer backbone.
2. **Cross-Attention Multi-Expert Query Routing:** The use of learned query embeddings representing task domains ($Q \in \mathbb{R}^{K \times D}$) that attend to spatial patch tokens via Multi-Head Cross-Attention (MHCA). This enables spatially localized routing decisions that can focus on relevant feature regions.
3. **Decoupled Historical Gating (DHG):** Formulating an exponential moving average (EMA) smoothing of predicted routing coefficients across batches to mitigate batch-composition dependency and solve "heterogeneity collapse" in mixed-task batches.

---

## The "Delta" From Prior Work
The proposed method sits in close proximity to several prior works, and its incremental vs. significant contributions can be characterized as follows:

* **From MoE (Mixture of Experts):** MoE architectures dynamically route activations at the *token level* using specialized routing gates. The delta here is that CAM-Router merges entire model weights *on-the-fly* before the forward pass, meaning that during inference there is no extra routing computational overhead in layers 2 to $L$. This saves activation-routing memory and compute.
* **From Classical/Quantum Dynamic Merging (QWS-Merge, BSigmoid-Router):** Both QWS-Merge and BSigmoid-Router already introduced classical/quantum-inspired dynamic routing heads. Specifically, the independent bounded sigmoidal activation ($\alpha = \lambda_{max} \cdot \sigma(o)$ with $\lambda_{max} = 0.3$) was **already proposed** by BSigmoid-Router, as the authors acknowledge. The actual delta is purely in the **routing feature representation**: CAM-Router uses an MHCA mechanism on unpooled tokens rather than a simple linear layer on global-average-pooled tokens.
* **From Standard Batching/Temporal Tracking:** Decoupled Historical Gating (DHG) resolves the "heterogeneity collapse" of batched inference. However, applying an EMA to track stats across batches is an extremely standard architectural choice in machine learning (commonly used in Batch Normalization, training tracking, or temporal model ensembling).

---

## Characterization of Novelty
We characterize the novelty of this work as **Incremental to Moderate**:
* **Moderate Novelty in Router Feature Engineering:** Identifying that global average pooling is an empirical bottleneck for dynamic routers (due to spatial vulnerability and heterogeneity collapse) and replacing it with spatial cross-attention is a clever and highly practical design. The "First-Block Paradox Resolution" (using a pre-trained, static first block to extract stable token representations for routing before merging layers $2 \dots L$) is also a neat and practical workaround.
* **Incremental in Mathematical/Algorithmic Machinery:** The individual components (cross-attention, learned task queries, bounded sigmoid gating, and exponential moving averages) are standard deep learning building blocks. The mathematical formulation does not introduce fundamentally new theoretical paradigms, and the bounded sigmoidal activation is direct reuse of BSigmoid-Router.
* **Scope Limit:** While the spatial cross-attention makes sense for vision models, its applicability and novelty for other major domains of model merging—such as autoregressive Large Language Models (LLMs)—are not addressed. The paper presents this as a general merging framework, but the entire design (spatial tokens, patches, Gabor-like first-layer filters) is heavily vision-centric, limiting the claimed breadth of the contribution.
