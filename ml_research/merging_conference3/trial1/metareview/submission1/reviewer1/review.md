# Peer Review of "QP-Merge: Quantization-Preserving Task Vector Merging"

## Summary of the Paper
This paper addresses the critical bottleneck of deploying merged, multi-task foundation models onto resource-constrained edge hardware. While model merging in high-precision (FP16 or FP32) successfully combines capabilities without joint training costs, subsequent Post-Training Quantization (PTQ) to low-bit integers (INT4/INT8) causes severe performance degradation. This is driven by heavy-tailed task outliers stretching quantization grids and severe activation scale mismatches between tasks.

To bridge this gap, the authors propose **QP-Merge** (Quantization-Preserving Merging), a framework that co-designs model merging and PTQ. QP-Merge introduces two core techniques:
1.  **Outlier-Residual Decoupling (ORD):** Identifies the top $\le 1\%$ highest-magnitude task-vector updates and routes them to a sparse, high-precision (FP16) tensor, leaving a tightly bounded dense remainder that is quantized to INT4 or INT8.
2.  **Quantization-Error Aware Scale Calibration (QE-Calib):** Uses an unlabeled calibration set ($M=128$) to perform a rapid (100 steps of Adam) optimization of layer-wise column-scaling diagonal parameters $D_l$ and learnable task merging weights $\lambda$. The scales $D_l$ are applied permanently to the weight updates without scaling activations during inference, bypassing runtime activation-alignment complexity.

Evaluating on dual-task vision classification (MNIST and SVHN) using a pre-trained `ViT-B-32`, QP-Merge INT4 achieves an average accuracy of 94.70% (within 0.42% of the optimized unquantized FP32 merged bound), while QP-Merge INT8 achieves 95.14% average accuracy, performing virtually lossless compared to the FP32 upper bound.

---

## Main Strengths
1.  **High Engineering Relevance:** The paper addresses a highly important, real-world deployment challenge. Enabling the compression of merged models to low-bit integers under tight edge-VRAM bandwidth constraints is of significant value to hardware and software engineers.
2.  **Unsupervised and Highly Data-Efficient Calibration:** The QE-Calib algorithm requires zero ground-truth downstream labels and calibrates scaling parameters in under 2 minutes over a tiny pool of just 128 samples. This makes the approach extremely easy to deploy in data-scarce enterprise settings.
3.  **Comprehensive Experimental Validation:** The authors provide thorough sensitivity analyses, including sweeps over outlier percentiles ($\gamma$) and calibration pool sizes ($M$). Additionally, they stress-test the model's out-of-distribution (OOD) generalization under synthetic corruptions and evaluate resilience to severe cross-domain calibration imbalance (e.g., calibrating purely on SVHN).
4.  **Exceptional Transparency and Candor:** The paper is highly commendable for its professional honesty. The detailed disclosure of high PyTorch runtime latency (resulting in a 5.8$\times$ slowdown) and the corresponding analytical DRAM-to-SRAM scaling analysis represent high-signal, rigorous engineering communication.

---

## Main Weaknesses
1.  **Heavy Reliance on Toy Datasets:** The entire evaluation is conducted on **MNIST** and **SVHN** digit classification. In modern machine learning, digit recognition on grayscale or low-resolution images is a solved toy problem. Real-world model merging is applied to large-scale, high-dimensional foundation models (e.g., LLaMA-7B or complex vision-language models) and multi-class downstream tasks. Evaluating purely on digit datasets fails to prove that the proposed non-equivalent weight scaling generalizes to complex, high-dimensional representation manifolds without causing catastrophic semantic drift.
2.  **Significant Out-of-the-Box Latency Overhead:** While the paper promises hardware-friendly execution, the physical GPU profiling reveals a major deployment bottleneck: QP-Merge INT4 is **5.8$\times$ slower than FP16** in PyTorch (60.92 $\mu$s vs 10.48 $\mu$s). This slowdown is driven by CUDA kernel launch overhead from running dense low-bit GEMM and sparse SpMM operators sequentially. While the authors argue that compile-time tools (Triton, TensorRT) can fuse these operations, they do not provide or demonstrate a fused runtime. Deploying a model that is significantly slower out-of-the-box is highly impractical for real-world production.
3.  **Risk of Representation Drift in Scaling:** Standard PTQ techniques (e.g., SmoothQuant) maintain mathematical equivalence by applying an inverse scale to activations. QP-Merge permanently scales the weight updates *without* adjusting activations. While this is a pragmatic workaround to resolve multi-task activation scaling conflicts, altering weight ranges permanently without inverse scaling introduces a high risk of representation drift or localized overfitting. While this risk is mitigated on simple digit tasks, it remains a severe potential flaw on larger, generative, or dense prediction tasks.
4.  **Unexplored Multi-Task Scaling:** The paper focuses on a simple, dual-task merge ($T=2$). In practical settings, model merging is most valuable when blending several tasks ($T \ge 8$). While the authors discuss a "global thresholding scheme" in their limitations to prevent disjoint outlier union from expanding sparse density, they did not implement or evaluate it. The scalability of the method to larger multi-task suites remains unverified.

---

## Detailed Ratings

### Soundness: Good
The underlying mathematical formulation is clean and logical. Decoupling outlier updates to protect symmetric quantization scales is a direct, sound response to the stretching problem. However, the lack of mathematical equivalence in QE-Calib and the high PyTorch execution overhead are significant practical caveats that limit immediate soundness under real-world deployment constraints.

### Presentation: Excellent
The paper is exceptionally well-written, direct, and professionally structured. The methodology is clearly defined, and the limitations section is refreshingly honest and comprehensive.

### Significance: Fair
While the conceptual contribution of co-designing merging and quantization is high, the immediate significance is constrained. Because the evaluation is limited to toy digit datasets and there is a 5.8$\times$ physical latency slowdown in PyTorch, the work operates as a promising proof-of-concept rather than an actionable, deployable solution for enterprise engineers.

### Originality: Good
The framework represents a highly creative combination of established PTQ concepts (hybrid dense-sparse formats, diagonal weight scaling) adapted specifically to parameter-space model merging. The decision to apply outlier decoupling to task-vector updates ($\Delta W$) rather than static weights is original and well-justified.

---

## Overall Recommendation

**Rating: 3: Weak Reject**

### Justification:
The paper has clear and notable merits, particularly in its conceptual approach of co-designing merging and compression, its high data-efficiency, and its exceptional presentation quality. However, the weaknesses currently outweigh these merits. 
Specifically, the heavy reliance on toy digit datasets (MNIST/SVHN) and the lack of a physically accelerated fused runtime to overcome the severe 5.8$\times$ PyTorch latency bottleneck mean that this work cannot yet be meaningfully deployed or built upon in practical settings. For a paper claiming to solve "real-world enterprise and edge deployment bottlenecks," evaluating on simple digit datasets and introducing massive physical latency overhead are significant shortcomings. 
To achieve publication grade, the paper requires revisions that:
1.  Evaluate the framework on more complex, non-toy benchmarks (such as standard multi-task vision suites or Large Language Models).
2.  Provide a basic compiled/fused runtime (e.g., Triton or TensorRT) to physically demonstrate the promised real-world edge execution speedups.

---

## Questions and Constructive Feedback for the Authors

1.  **Representation Drift & Scaling:** Since the diagonal scaling matrix $D_l$ does not maintain mathematical equivalence (lacking inverse scaling on activations), how does the model prevent representation drift or overfitting on more complex, high-dimensional datasets (e.g., CLIP/ViT-L on multi-class classification, or LLaMA-7B on text generation)? Have you conducted any preliminary experiments on larger-scale models/tasks?
2.  **Fused Runtime:** To resolve the 5.8$\times$ PyTorch latency slowdown caused by kernel launch overhead, have you explored exporting the hybrid model to compiled runtimes like `torch.compile` or exporting custom Triton/TensorRT kernels? Demonstrating even a basic fused kernel would significantly bolster the paper's edge-deployment claims.
3.  **Outlier Density at Scale:** In the proposed global thresholding scheme to handle outlier density when $T \ge 8$ tasks are merged, how does multi-task accuracy degrade as the global constraint prunes important task-specific outliers to maintain a strict 1.0% limit?
4.  **Unlabeled Data Source:** For the QE-Calib step, does the 128-sample calibration pool need to be strictly domain-balanced across all merged tasks to prevent representation drift on minority tasks? How does calibration performance degrade if the pool is heavily biased toward a single task (e.g., 90% MNIST, 10% SVHN)?
