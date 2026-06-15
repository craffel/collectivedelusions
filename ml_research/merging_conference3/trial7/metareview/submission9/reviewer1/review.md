# Peer Review Report

## Summary of the Paper
The paper addresses the challenge of **test-time dynamic model merging** in heterogeneous streaming environments. Traditional parameter-space model merging methods suffer from **heterogeneity collapse** under mixed-task batches because they are forced to average routing coefficients over the batch dimension to maintain a set of merged weights. Prior state-of-the-art solutions like Micro-Batch Homogenization (MBH) mitigate this by wrapping the model in complex, stateful systems scheduling pipelines that buffer, sort, and partition streams, introducing significant Serving complexity and queuing latency.

To resolve this systems-centric bloat, the authors introduce **SABLE (Sample-wise Activation Blending of Low-Rank Experts)**. SABLE shifts the ensembling step from parameter space to activation space using the distributive property of matrix multiplication:
$$ Y_b = X_b W_{\text{base}} + \sum_{k=1}^K \alpha_{k, b} \cdot \left( (X_b A_k) B_k \right) $$
By performing activation blending on a per-sample basis during the forward pass, SABLE is natively immune to batch heterogeneity (achieving exactly **0.00% collapse**). SABLE is completely non-parametric, calibration-free, and runs in a single stateless forward pass.

The authors evaluate SABLE across a synthetic 14-layer Coordinate Sandbox, a physical 3-layer CNN, a physical 4-layer MLP, and high-dimensional ResNet-18 foundation features. SABLE Late Adaptation achieves **68.10%** joint accuracy in the sandbox, outperforming the systems-heavy PFSR+MBH pipeline (**67.20%**). Physical A100 GPU benchmarks demonstrate that SABLE reduces serving latency by **6.8$\times$** and saves **36.4%** peak VRAM memory compared to PFSR+MBH.

---

## Strengths and Weaknesses

### Strengths
1. **Mathematically Elegant & Stateless Alternative:** 
   The core mathematical relocation of ensembling from parameter space to activation space via the distributive property of matrix multiplication is highly elegant. It provides a clean, stateless network-level solution that completely eliminates the need for stateful systems-level queues, sorting steps, and temporal serving buffers, returning deep learning serving to its stateless, reproducible roots.

2. **Principled Mathematical Refinements:** 
   The paper introduces several highly clever and technically sound improvements to make activation ensembling practical:
   * **Refined Zero-Data Centroids:** Applying weight-space L2-normalization to class weight vectors before averaging successfully prevents vector cancellation, enabling robust zero-data routing with competitive accuracies.
   * **Layer-Dependent Hybrid-Rank Selection Protocol:** Ensembling low-dimensional output projections at full precision while hidden layers remain at aggressive low ranks ($r \le 2$). This uncovers the **Low-Rank Regularization Paradox**, where intermediate hidden layers of rank $r=2$ act as a powerful low-pass filter to prune cross-task representation noise, maximizing both parameter efficiency and joint accuracy.
   * **OOD Gating:** Introducing Soft Sigmoid Gating to eliminate hard-threshold sensitivity.

3. **Rigorous Hardware-Level Awareness:** 
   Unlike many machine learning works that focus solely on theoretical FLOP counts, SABLE provides a deeply grounded hardware and systems-level analysis. It explicitly addresses CUDA kernel launch overhead and GPU memory bandwidth limitations, showing how Top-$M$ expert pruning, Layer-Dependent Hybrid-Rank protocols, and vectorized multi-tenant serving engines (Punica, S-LoRA) combine to deliver actual physical wall-clock performance.

4. **Exhaustive Empirical Validation:** 
   The evaluation is exceptionally robust, validating SABLE across a 14-layer Coordinate Sandbox, a physical CNN, a physical Deep MLP, and a high-dimensional ResNet-18 foundation feature setup. The inclusion of physical NVIDIA A100 GPU benchmarks (proving a **6.8$\times$ latency speedup** and **36.4% VRAM savings** over MBH) provides definitive, high-signal proof of its real-world serving advantages.

5. **Exceptional Scientific Integrity and Transparency:** 
   The authors are highly commendable for their honesty and transparency. They do not over-hype their results; instead, they explicitly map out and analyze every theoretical and practical limitation of their method, including non-linear cumulative drift, early-feature loss in late-adaptation, dual-space mismatch, and input-space routing noise.

### Weaknesses
1. **Lack of Generative LLM Validation:** 
   SABLE's non-parametric routing is designed around task-specific classification heads. In generative LLMs, there are no task-specific heads; instead, they share a single vocabulary projection. While the authors propose a highly structured and actionable blueprint using a frozen semantic text embedder (e.g. MiniLM) and instruction-based centroids, this generative pathway remains unproven empirically in the current text.

2. **Dual-Space Mismatch of Zero-Data Centroids:** 
   Taking the cosine similarity between feature representations ($z$) and classification parameters ($w$) constitutes a dual-space manifold mismatch. This is reflected in SABLE's standard results, where Completely Zero-Data centroids suffer a 5.80% absolute accuracy drop compared to utilizing 16 support-split activation samples.

3. **Early Feature Loss under Mid-Layer Routing:** 
   Mid-Layer Routing (Late Adaptation) leaves the first $L_{\text{route}}$ layers unadapted. This represents a complete loss of any task-specific features learned in the early-to-mid layers of the experts during fine-tuning. SABLE is thus structurally restricted to experts whose adaptation is concentrated in late-stage layers.

4. **Input-Space Routing Constraints:** 
   Single-Pass Early-Routing is highly effective on starkly separable inputs (MNIST pixels vs. FashionMNIST pixels) but will suffer from severe, catastrophic routing noise on high-dimensional natural datasets where raw features lack semantic separability.

---

## Ratings

### Soundness
* **Rating:** Excellent
* **Justification:** SABLE is technically flawless. Its ensembling algebra is mathematically sound, and all identified trade-offs (non-linear cumulative drift, early-feature loss, dual-space mismatch, representational blurring) are explicitly analyzed and resolved with rigorous mathematical or architectural mitigations. The empirical validation is exhaustive, and the A100 GPU Serving benchmarks are highly rigorous.

### Presentation
* **Rating:** Excellent
* **Justification:** The paper is beautifully written, logically structured, and mathematically precise. The schematic diagrams (such as Figure 1) and tables are exceptionally detailed, making the complex concepts and hardware-level considerations highly accessible.

### Significance
* **Rating:** Excellent
* **Justification:** The paper addresses an important, highly relevant problem in dynamic model merging and multi-tenant serving. By shifting the ensembling step to activation space, it bypasses systems-centric complexity, returning serving to its stateless roots. It has substantial potential to influence future research and production deployment strategies for massive adapter pools.

### Originality
* **Rating:** Excellent
* **Justification:** Shifting test-time model merging from parameter space to activation space via the distributive property is highly original and clever. The proposed Refined Zero-Data Centroids construction and the Layer-Dependent Hybrid-Rank Selection Protocol (uncovering the Low-Rank Regularization Paradox) are highly original mathematical contributions that elevate SABLE beyond a naive linear-algebraic translation.

---

## Overall Recommendation

* **Overall Recommendation:** 5: Accept
* **Justification:** SABLE is an outstanding, technically solid, and highly complete paper. It introduces an elegant mathematical solution to a major systems-level streaming degradation (heterogeneity collapse), natively outperforming complex stateful scheduling pipelines while delivering a **6.8$\times$ wall-clock latency speedup** and **36.4% memory savings** on actual A100 hardware. The paper exhibits exemplary scientific integrity by thoroughly identifying and analyzing its own theoretical and physical boundaries, and the empirical evaluations are exceptionally comprehensive. The proposed Refined Zero-Data Centroids and Layer-Dependent Hybrid-Rank protocols are highly clever mathematical refinements. The paper will be of immense interest to researchers and practitioners working on parameter-efficient fine-tuning, model merging, and multi-tenant deep learning serving. I strongly recommend accepting this work.
