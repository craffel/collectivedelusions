# Peer Review of Conference Submission

## Paper Summary
This paper introduces **Sparse Low-Rank Dynamic Merging (SLD-Merge)**, a novel, parameter-efficient, and completely batch-independent dynamic model-merging framework designed for robust multi-task deployment on edge devices. 

In weight-space model merging, existing dynamic merging methods (such as QWS-Merge and Linear Routers) reconstruct a single set of dense weights by averaging routing coefficients across the batch dimension. The authors argue that this introduces a critical deployment bottleneck: **batch-dependency** and **heterogeneity collapse**, where a sample's classification depends on other co-packaged samples in the batch, causing performance to degrade when processing mixed-task streams. 

To resolve this, SLD-Merge shifts dynamic routing from weight-reconstruction space to sample-wise activation-space routing. It performs offline Singular Value Decomposition (SVD) on expert task vectors to construct compact low-rank adapters ($B_k$ and $A_k$). At inference, a bounded cosine-similarity router extracts global spatial-pooled representation vectors from input activations and computes sample-wise alignment scores. It then applies Top-1 hard gating to activate only the most relevant low-rank expert adapter for each individual sample. Since this process is executed sample-by-sample, each sample is evaluated in complete isolation, achieving stateless, deterministic, batch-independent inference. 

Additionally, the authors introduce **Activation-Space Mean Initialization**—initializing the routing basis vectors to the mean activation of each task on a tiny unlabeled calibration set—to achieve high-quality zero-shot routing without backpropagation calibration. They also propose an **Autonomous Classification Head Selection** rule based on layer-averaged routing scores to achieve full deployment autonomy without test-time oracle leakage.

Evaluating on a 4-dataset Vision Transformer (ViT-Tiny) benchmark (MNIST, FashionMNIST, CIFAR-10, SVHN) under mixed-task streams, SLD-Merge maintains a stable joint accuracy of **63.87%** across all batch sizes, outperforming strong dynamic baselines by up to **+8.50%** in large-batch deployment, while reducing task-specific parameter storage by over **92.5%** and adding only **8.3%** computational overhead.

---

## Strengths and Weaknesses

### Soundness
- **Strength (Mathematical Clarity):** The mathematical formulation of offline SVD factorization, bounded cosine-similarity routing, and sample-wise parallel execution is exceptionally precise, clean, and easy to trace.
- **Strength (Rigorous Ablations):** The authors provide an extensive set of ablations, analyzing the effect of the truncation rank $r$, comparing zero-shot activation centroids with STE-optimized basis vectors, isolating SVD truncation error via a full-rank baseline, evaluating autonomous head selection, and verifying statistical robustness over sequence seeds and random data splits.
- **Weakness (Vectorized Parallel Forward Pass Complexity):** The paper claims that Top-1 hard gating achieves "extreme computational efficiency" and only adds 8.3% FLOPs. However, the vectorized parallel forward pass is expressed as:
  $$Y = X W_{\text{base}}^{(l)} + \sum_{k=1}^K \alpha_k \odot \left( (X A_k^{(l)}) B_k^{(l)} \right)$$
  If this equation is implemented as written in standard deep learning libraries (e.g., PyTorch), it will execute the low-rank forward pass for **all $K$ experts** before multiplying by the one-hot coefficient $\alpha_k$. For a batch size of $B$, this scales the practical compute cost by $O(K)$, defeating the sparse gating computational benefit. The authors should clarify if their implementation utilizes specialized conditional indexing or gather-scatter operations to achieve true $O(1)$ expert execution per sample during batch evaluation.
- **Weakness (Low-Data Scaling Confounder):** The experts are trained under extreme data constraints (only 256 training samples per dataset), which yields a very low standalone ceiling for SVHN (29.30%). The finding that SVD truncation acts as an implicit regularizer (outperforming full-rank by +1.38%) is highly interesting, but likely an artifact of these under-converged, overfitted low-shot experts. If the experts were trained on full datasets to convergence, SVD truncation would likely introduce a reconstruction penalty rather than a regularization benefit. This scaling assumption needs to be explicitly discussed.

### Presentation
- **Strength (Outstanding Writing):** The paper is beautifully written, utilizing clear academic terminology and a highly logical structural flow. Figure 1 and Figure 2 are of publication quality and greatly aid in understanding the proposed pipeline.
- **Weakness (Mismatched Narrative on Heterogeneity Collapse):** The authors claim that as batch size increases, dynamic baselines (Linear Router, QWS-Merge) suffer from catastrophic "heterogeneity collapse" and severe performance degradation due to batch averaging. However, looking at Table 1:
  - Linear Router drops from 59.28% ($B=1$) to 59.18% ($B=256$) (a microscopic decrease of 0.10%).
  - QWS-Merge actually rises slightly from 56.93% ($B=1$) to 57.03% ($B=256$).
  The empirical drop in accuracy *as batch size scales* is virtually non-existent. Instead, the baselines are already performing poorly at $B=1$. The primary bottleneck is **soft-routing task interference** at $B=1$ (applying a soft combination of all experts introduces parameter conflicts), and their performance remains flat as $B$ grows. The paper's narrative should be adjusted to reflect this nuance: Top-1 hard gating's primary strength is eliminating soft-routing task interference, and sample-wise routing preserves this benefit batch-independently.

### Significance
- **Strength (Edge-Deployment Utility):** For resource-constrained on-device deployment, the proposed framework is of high practical significance. Achieving a **92.5% reduction** in specialized block parameters (saving 3.66M parameters) and a **37.9% overall RAM reduction** with complete batch-independence and minimal latency impact is a major win for streaming AI applications.
- **Weakness (Scope of Evaluation):** The evaluation is restricted to a small ViT-Tiny backbone (5.7M parameters) on simple vision datasets. While appropriate as a proof-of-concept, modern model merging is heavily focused on Large Language Models (LLMs) and Vision-Language Models (VLMs). Discussing how SLD-Merge scales to these larger models would increase its potential impact.

### Originality
- **Strength (Creative Synthesis):** While the individual mathematical components are known, combining offline SVD, prototypical activation centroids, and Top-1 hard gating into a unified, batch-independent weight-merging framework is a highly creative and original synthesis that addresses a major real-world bottleneck.
- **Weakness (Lack of Historical Context & Attribution):** The paper presents several components as native, custom designs, neglecting their deep roots in classical and contemporary machine learning literature:
  - **Activation-Space Mean Initialization** is structurally identical to prototypical representation learning (e.g., Prototypical Networks, Snell et al., 2017) and nearest-centroid classifiers.
  - **SVD Task-Vector Decomposition** is conceptually related to post-hoc model compression (converting dense weights to LoRA adapters post-training, such as SVD-LoRA and LoRA-XS) and active 2025/2026 low-rank model merging research. 
  Properly citing and positioning the work relative to these lines of research is essential to establish scholarly rigor.

---

## Detailed Ratings

### Soundness: Good
The methodology is mathematically rigorous, reproducible, and supported by thorough empirical ablation studies. However, the paper leaves open questions about the true computational scaling of its PyTorch implementation and the generalizability of the SVD regularization benefit to high-resource, fully-converged experts.

### Presentation: Good
The paper is exceptionally well-written, clear, and structured. However, the narrative around "heterogeneity collapse" must be reconciled with the actual empirical data in Table 1 (where the baselines exhibit flat performance rather than an active drop as $B$ scales), and the related work section must be expanded to properly attribute foundational concepts.

### Significance: Good
The practical edge-deployment benefits (92.5% task-parameter storage savings, batch independence, and low computational overhead) are outstanding and highly relevant for streaming, on-device applications. The evaluation's focus on a tiny backbone (ViT-Tiny) and subsampled datasets is a minor limitation.

### Originality: Good
The integration of SVD weight decomposition with sample-wise metric-space activation routing is a highly clever and practical solution to a known bottleneck in dynamic model merging. However, the individual components (prototypical activation centroids, post-hoc SVD adaptation) are not sufficiently contextualized within the existing literature.

---

## Overall Recommendation: 4: Weak Accept
This is a technically solid, exceptionally well-written, and practically significant paper that successfully resolves the batch-dependency and soft-interference bottlenecks of dynamic model merging. The proposed SLD-Merge framework provides outstanding edge-deployment storage and computation benefits. 

However, before final publication, the authors must address some scholarly weaknesses regarding literature contextualization, baseline interpretation, and implementation details:
1. **Literature Contextualization:** The authors must explicitly acknowledge and cite the connections between Activation-Space Mean Initialization and Prototypical Networks (Snell et al., 2017) and nearest-centroid classification, as well as position their SVD-based task decomposition relative to post-hoc low-rank adapter construction (e.g., SVD-LoRA) and recent low-rank merging methods (e.g., DiDi-Merging, FRISM).
2. **Reconciling Heterogeneity Collapse Narrative:** The authors must clarify that the dynamic baselines' performance in Table 1 remains virtually flat as batch size scales from 1 to 256. The main bottleneck is **soft-routing task interference** at $B=1$, rather than a severe drop in performance as $B$ increases. The core benefit of Top-1 gating is eliminating soft interference, which is then preserved batch-independently.
3. **Implementation Details:** The authors should discuss whether their PyTorch implementation naively evaluates all $K$ low-rank paths in parallel, or if they utilize custom scatter/gather/indexing operations to achieve true $O(1)$ expert compute cost per sample during batch execution.
4. **Low-Data Scaling Discussion:** The authors should explicitly discuss the limitations of their extreme low-data subsampling (256 training samples) and acknowledge that the regularization benefits of SVD low-rank truncation may be specific to under-converged or low-resource regimes, whereas fully-saturated experts might experience a standard reconstruction trade-off.

Adressing these points will substantially elevate the scholarly quality and academic rigor of this highly promising work.
