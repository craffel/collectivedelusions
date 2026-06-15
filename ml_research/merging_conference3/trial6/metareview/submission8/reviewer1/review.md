# Peer Review

## Summary of the Paper
The paper introduces **Hybrid-Router**, a low-latency hybrid dynamic model merging framework designed to resolve the computational and memory-bandwidth bottlenecks of test-time parameter routing. While dynamic test-time model ensembling achieves high multi-task performance by adapting model weights on-the-fly, high-dimensional weight reconstruction at runtime introduces severe latency. Hybrid-Router resolves this by partitioning deep networks layer-wise: early task-agnostic layers are statically merged offline with uniform or AdaMerging-optimized weights (incurring zero runtime overhead), while only the final $k$ task-specific layers are dynamically routed and ensembled at test-time. 

Additionally, the paper explores **BSigmoid-Router** to analyze uncoupled task activations (revealing how conservative scaling limits explain its performance gap with Softmax) and introduces **Dynamic Batch Filtering (DBF)** to resolve representational collapse (*Batch Style Blur*) under heterogeneous streaming batches. The authors evaluate their framework using both a synthetic 14-layer Vision Transformer *Parameter-Space Representation Sandbox* proxy and a physical PyTorch-based SimpleCNN implementation on standard vision datasets (MNIST, FashionMNIST, CIFAR-10, SVHN).

---

## Strengths and Weaknesses

### Strengths
1. **Highly Practical, Hardware-Aware Systems Focus**: The paper addresses a critical but frequently overlooked real-world bottleneck in test-time model merging—computational latency and memory-bandwidth saturation during online weight reconstruction. The proposed solution is simple, elegant, and directly addresses active parameter memory footprint (providing $1 - k/L$ VRAM task-vector savings).
2. **Outstanding Systems-Level and Deployment Analyses**: The detailed CPU wall-clock latency breakdowns (Table 4) and the direct quantitative comparison against PEFT/LoRA serving frameworks (Punica, S-LoRA) in Table 5 provide exceptional engineering value. The inclusion of architectural blueprints for parallel execution (unified kernels, concurrent CUDA streams) and mixed-precision quantization demonstrates a deep understanding of hardware-level edge deployment constraints.
3. **Efficacy of Dynamic Batch Filtering (DBF)**: DBF represents a highly pragmatic, systems-level runtime buffering and clustering optimization that successfully resolves batch-level representational collapse under noisy heterogeneous streams. The empirical results show spectacular gains (e.g., boosting Linear Router from 63.54% to 93.77% accuracy at batch size $B=256$ in the sandbox, and providing over +27% absolute accuracy gains on physical CNN weights).
4. **Candidness and Scientific Transparency**: The authors are remarkably honest and explicit about their study's limitations, proactively detailing the "direct structural circularity" of their sandbox's penalty formulation and the "physical validation Pareto discrepancy" where their theoretical "paradox" was not observed.
5. **Excellent Writing and Clarity**: The manuscript is exceptionally well-written, logically structured, and highly detailed. The mathematical notation is rigorous, and the related work is comprehensive.

### Weaknesses
1. **Lack of Physical Validation on Deep, High-Capacity Architectures (ViTs)**:
   The primary quantitative high-accuracy results (such as the peak joint accuracy of 84.79% at $k=12$, the 71.3% ensembling speedup, and the sensitivity sweeps of $\eta$) are evaluated strictly within the synthetic, mathematically modeled Parameter-Space Representation Sandbox proxy environment, simulating a ViT-Tiny. There is no physical execution, weight merging, or routing conducted on actual, physical Vision Transformers (e.g., `vit_tiny_patch16_224` or `vit_base`) on real image datasets.
2. **"Overfitting-Optimizer Paradox" Remains Unproven on Physical Weights**:
   The "Overfitting-Optimizer Paradox"—where freezing early task-agnostic layers ($k < L$) acts as a structural regularizer that outperforms fully dynamic routing ($k=L$) under low-resource calibration splits—is a key theoretical claim. However, this was not observed in the physical CNN experiments, where performance increased monotonically with $k$. While the authors' capacity-based explanation is sound, the paradox remains a synthetic finding that has yet to be demonstrated on physical model weights.
3. **Limited Scale of Physical CNN Experts**:
   The physical CNN validation is conducted on extremely shallow, low-capacity networks (SimpleCNN with only 25k parameters and 4 layer groups) trained on highly subsampled datasets. Evaluating on such tiny models makes it difficult to draw definitive conclusions about the empirical behavior of deep, high-capacity architectures under layer partitioning.
4. **Small Statistical Scale**:
   All reported means and standard deviations (in both the sandbox and physical CNN sweeps) are computed across only 3 independent calibration/sampling seeds. Utilizing a larger statistical sample (e.g., 5 or more seeds) would provide greater statistical confidence in the robustness of the reported standard deviations.

---

## Soundness
**Rating: Good**

The methodology is mathematically sound and described with high clarity. The task-vector formulation, static-dynamic partition splits, and the DBF clustering logic are well-reasoned. The authors are careful to isolate scaling bounds via BL-Router and BSigmoid-Router to explain the Softmax-Sigmoid gap. However, the evaluation relies heavily on a synthetic sandbox environment that has a built-in representational penalty, which mathematically guarantees that early-layer freezing ($k < L$) outperforms fully dynamic routing ($k=L$). Since the "Overfitting-Optimizer Paradox" was not observed in the physical CNN validation, the soundness of this theoretical finding is partially limited by the lack of physical validation on real deep networks (such as ViTs).

---

## Presentation
**Rating: Excellent**

The paper is exceptionally clear, structured, and easy to follow. The mathematical equations are precise and properly contextualized. The tables are highly detailed and include standard deviations. The related work is exhaustive, covering static merging, test-time routing, PEFT serving runtimes, and Mixture of Experts (MoE), which successfully positions the work within the literature. The authors' transparency regarding potential "circularity" and limitations is commendable and highly professional.

---

## Significance
**Rating: Good**

The paper addresses an important, high-impact bottleneck in parameter-space model ensembling. Resolving the latency and memory overhead of dynamic weight reconstruction is essential for making dynamic merging practical. The systems-level contribution (DBF) and the quantitative deployment comparisons provide significant practical utility. However, the significance of the "Overfitting-Optimizer Paradox" and the primary accuracy sweeps is somewhat moderated because they have not yet been demonstrated on physical deep models.

---

## Originality
**Rating: Good**

The paper introduces a novel hybrid partition approach to parameter-space model ensembling, combining the advantages of static merging (zero overhead) and dynamic routing. The systems-level formulation of Dynamic Batch Filtering (DBF) to address Batch Style Blur is highly creative and original. While layer-wise freezing is a standard concept in representation learning, its application as a structural regularizer and VRAM compression tool for dynamic ensembling is highly innovative.

---

## Overall Recommendation
**Rating: 4: Weak Accept**

**Justification:** 
This is a highly practical, systems-aware, and exceptionally well-written paper that addresses a critical real-world bottleneck in test-time model merging. The systems-level contributions (such as Dynamic Batch Filtering and the quantitative comparisons with LoRA serving frameworks) are highly significant and provide great engineering value. 

However, from an empirical perspective, the evaluation is somewhat limited. The primary, high-accuracy results and the "Overfitting-Optimizer Paradox" are demonstrated strictly within a synthetic sandbox proxy environment with a built-in representational penalty, while the physical CNN validation fails to replicate the paradox and is conducted on extremely shallow, 25k-parameter networks. The paper would be significantly stronger with direct physical validation on a standard deep architecture like a physical Vision Transformer. Nevertheless, the paper's clarity, systems-level contributions, and exemplary transparency make it a highly valuable and solid contribution to the community.

---

## Questions and Suggestions for Authors

1. **Physical Validation on Vision Transformers**:
   While the SimpleCNN physical experiments provide an important grounding, the lack of physical validation on a deep, high-capacity model (such as a physical ViT-Tiny or ViT-Base) remains a key limitation. Do you plan to conduct a physical ensembling experiment on actual Vision Transformers pre-trained and fine-tuned on real image datasets to confirm the presence of the Overfitting-Optimizer Paradox on physical weights?
2. **Increasing the Statistical Scale**:
   The reported standard deviations and means are based on 3 independent seeds. Would it be possible to scale this up to 5 or 10 independent seeds in the final version to improve statistical rigor and confidence?
3. **Extending DBF to Parallel CUDA Streams**:
   In Section 4.6 (Hardware-Aware GPU Profiling), you propose executing the early layers and the late-layer weight assembly in parallel using asynchronous CUDA streams. Could the Dynamic Batch Filtering (DBF) online style-clustering step also be overlapped with early-layer execution in the default stream to completely mask the clustering latency?
