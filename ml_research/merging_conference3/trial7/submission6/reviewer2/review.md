# Peer Review

## Summary of the Paper
The paper addresses the critical problem of **low-data calibration overfitting in dynamic weight-space model merging**. Weight-space merging is an efficient multi-task learning paradigm that directly combines fine-tuned expert parameters. While dynamic weight-space merging (predicting sample-specific ensembling weights on-the-fly) is highly expressive, parametric routers suffer from severe overfitting under extreme data scarcity ($B_{\text{cal}} \le 64$), leading to catastrophic generalization collapse on out-of-distribution (OOD) tasks.

To solve this, the authors reject ad-hoc, isotropic heuristics in favor of a mathematically rigorous approach. They derive the first-ever **Rademacher complexity generalization bound** for a dynamically merged hypothesis class under a coupled Softmax routing mechanism. The bound reveals that generalization error scales linearly with the product of the routing parameter norm and the task-vector norm (Frobenius or Spectral). 

Guided by this theory, they introduce **Spectral and Rademacher-guided Routing Regularization (SR3)**, which scales weight decay penalties proportionally to task-vector parameter-space norms (Frobenius or Spectral). They also propose:
1. A smoothed $L_1$ Group-Lasso variant (**SR3-L1**) to directly minimize the linear Rademacher bound.
2. A **Regularization Scheduling** warm-up scheme to bypass the non-smooth gradient barrier near the origin (the $L_1$ Group-Lasso Paradox).
3. A **Hybrid Adaptive Controller** (**SR3-Hybrid**) that dynamically modulates the regularization force based on backpropagated gradient norms, preventing the over-repression of complex tasks.

The framework is evaluated on a custom continuous weight-merging simulator featuring representation entanglement and structured singular value geometries, as well as on a physical PyTorch experiment (2-layer MLP on handwritten digits classification).

---

## Strengths and Weaknesses

### Strengths
1. **Pioneering Theoretical Foundation:** The derivation of the Rademacher complexity bound under a coupled Softmax routing layer is a major theoretical milestone for weight-space model merging. By using Maurer's vector-valued contraction theorem, the proof remains mathematically rigorous, avoiding the simplistic "independent sigmoid gating" assumptions common in literature.
2. **Computational & Scaling Efficiency (No Runtime Overhead):** The task-vector norms ($\|V_k\|_F$ and $\|V_k\|_{op}$) are static and precomputed offline, introducing **zero runtime or training overhead** during router training or inference.
3. **Power Iteration for LLM Scalability:** The authors proactively resolve the $O(D^3)$ complexity of Singular Value Decomposition (SVD) for large-scale models. By proposing a fast power-iteration approximation (reducing complexity to $O(D^2)$), they ensure the method scales seamlessly to modern giant LLMs.
4. **Insightful Algorithmic Solutions:** The paper successfully translates theoretical bounds into practical optimizers. The introduction of *Regularization Scheduling* to resolve the $L_1$ gradient barrier and the *Hybrid Adaptive Controller* to balance task specialization and generalization represent exceptionally clever, high-quality engineering.
5. **Outstanding Scientific Transparency:** The critical discussion section (Section 4.5) is incredibly honest and thorough, openly addressing evaluation circularity under analytical gap penalties, representation entanglement assumptions, and the optimization-complexity trade-offs of the proposed regularizers.

### Weaknesses
1. **The Inference Latency and GPU Batching Bottleneck:** 
   In dynamic weight-space merging, model parameters are assembled on-the-fly *for each individual input sample* ($W_{\text{merged}}(b) = W_{\text{base}} + \sum \alpha_{k, b} V_k$). In high-throughput industrial settings, this sample-specific parameter assembly is a massive memory-bandwidth and computational bottleneck. 
   - Standard GPU-accelerated deep learning relies on all samples in a batch sharing the same parameters to perform highly parallelized matrix multiplications (GEMMs). 
   - Doing sample-by-sample weight interpolation (such as `torch.einsum`) requires loading and interpolating massive parameter matrices in GPU memory for every forward pass. For multi-billion parameter models (e.g., LLaMA-3), this completely destroys inference throughput. 
   - *Mitigation/Discussion:* To make this method practically deployable in commercial pipelines, the routing granularity must be coarsened—for instance, performing routing at the **sequence/prompt level**, or at the **batch level (Homogeneous Batch Routing)**. The authors should explicitly discuss these deployment engineering trade-offs in their final draft.
2. **Evaluation Scale Gap:**
   While the physical validation on a 2-layer MLP (`TinyMLP`) on the toy digits dataset is a valuable addition to break the simulator's circularity, the scale of this setup is extremely small. The parameter geometries, representation manifolds, and optimization dynamics of a 2-layer MLP on 1797 digits do not reliably translate to modern multi-billion parameter foundation transformers. Evaluating on a medium-scale physical model (e.g., a Vision Transformer ViT-B/16 or RoBERTa) on standard multi-task benchmarks (e.g., GLUE or VTAB) would significantly enhance the practical significance of the empirical validation.
3. **Low-Data Calibration Variance:**
   Under extreme data scarcity ($B_{\text{cal}} \le 64$), the calibration process exhibits noticeable variance across random seeds, as shown by the standard deviations in Table 2 and Table 3. In industry pipelines, such high variance represents a deployment risk. The authors should discuss potential strategies to stabilize low-data calibration, such as incorporating metadata-based prior initialization or ensembling multiple sparse calibration subsets.

---

## Detailed Ratings

### Soundness
* **Rating:** **Excellent**
* **Justification:** The paper is exceptionally sound. The mathematical proofs for Theorem 1 and Theorem 2 are rigorous, complete, and correct. The authors are incredibly thorough and honest about evaluating both the strengths and weaknesses of their work, and they provide a transparent, detailed discussion profiling actual singular values to explain the "Spectral-Frobenius performance flip" in shallow physical settings.

### Presentation
* **Rating:** **Excellent**
* **Justification:** The paper is beautifully written, highly structured, and easy to follow. The transition from theoretical derivations (Section 3) to practical algorithm design (Section 3.4-3.6) and empirical verification (Section 4) is highly cohesive. The figures and tables are clean, clear, and informative.

### Significance
* **Rating:** **Excellent**
* **Justification:** The paper addresses an important, highly relevant problem in model ensembling and multi-task learning. By bridging the gap between empirical weight-space model merging (which has historically relied on ad-hoc heuristics) and first-principles statistical learning theory, this work has the potential to influence future research and inspire a new family of "geometry-aware" merging algorithms.

### Originality
* **Rating:** **Excellent**
* **Justification:** The work is highly original. The derivation of the Rademacher complexity bound for coupled Softmax routing using vector-valued contraction is a non-trivial theoretical innovation. Introducing asymmetric regularizers based on task-vector geometries, along with regularization scheduling and hybrid controllers, provides highly creative and novel combinations of optimization theory and statistical learning.

---

## Overall Recommendation
* **Recommendation:** **5: Accept**
* **Justification:** This is a technically solid, highly elegant paper that successfully bridges the gap between empirical weight-space model merging and first-principles statistical learning theory. The theoretical contributions are highly rigorous, the algorithmic solutions (scheduling and hybrid controllers) are incredibly clever, and the scientific transparency is commendable. While there is a scale gap in the physical experiments (toy 2-layer MLP) and an unaddressed GPU batching bottleneck for sample-specific dynamic weight assembly, these represent exciting avenues for future engineering work rather than flaws in the paper's core contribution. The work is of exceptionally high quality and warrants a clear acceptance.
