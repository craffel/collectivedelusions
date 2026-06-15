# Mock Review

## Summary of the Submission
This paper presents **ZipMerge**, a technically rigorous and hardware-conscious framework designed to co-optimize layer-wise merging coefficients and dynamic magnitude pruning boundaries at test-time. Instead of treating expert model merging and parameter compression as separate, decoupled steps, ZipMerge integrates them into a joint test-time adaptation (TTA) loop. It minimizes the unsupervised Shannon entropy of predictions on a tiny calibration set (16 images or prompts per task) using two distinct optimization engines: a first-order Straight-Through Estimator (STE) with Identity-pass global gradient flow, and a zero-order 1+1 Evolution Strategy (ES).

Crucially, rather than presenting a curated narrative of continuous success, the paper is framed as a **rigorous post-mortem and limitation-mapping study**. The authors stress-test their framework by merging four highly orthogonal visual tasks (MNIST, FashionMNIST, CIFAR-10, SVHN) on a compact Vision Transformer (`vit_tiny_patch16_224`) backbone. 

Through this honest empirical stress-test, the authors document several system-level boundary behaviors:
1. **Catastrophic Representational Collapse:** Linear task arithmetic collapses completely (performing at the level of random guessing, 10%-14% accuracy) when merging highly disparate visual domains on compact backbones.
2. **The Overfitting-Optimizer Paradox:** Unconstrained unsupervised entropy minimization on tiny calibration subsets overfits transductively, successfully minimizing calibration loss while destroying generalizable features.
3. **Prune-then-Merge (P-then-M) Superiority:** The simple, decoupled baseline consistently and significantly outperforms joint optimization because pre-merging pruning acts as a spatial regularizer, removing small conflicting parameter updates to shield the shared backbone from interference.
4. **The Noisy Expert Noise Injection Constraint:** Merging is highly sensitive to input expert convergence; a single under-trained expert acts as a "poison pill" that injects high-frequency parameter noise, collapsing the entire system.

To isolate the ZipMerge algorithm's performance from the inherent limitations of weight-space operations under extreme domain shifts, the paper includes a massive, exemplary suite of analytical studies. These include **Reg-ZipMerge** (structural/functional regularization), low-conflict evaluations (**DomainNet** where ZipMerge achieves outstanding stable performance of **74.20%** at 50% sparsity), backbone scale-up (**ViT-Base**), **PEFT/LoRA-Adapter Merging** combined with a mathematically elegant and highly efficient **Orthogonal Procrustes SVD Alignment** step (which rotates independently learned adapter weight spaces into a shared coordinate system, delivering a massive **+16.45%** absolute boost to 58.75% and achieving **62.10%** sparse accuracy), multi-task joint learning (MTL), convolutional backbones (**ResNet-18**), statistical significance, calibration sizes, task vector scaling sweeps, hybrid **TIES-ZipMerge**, and generative language models (**GPT-2** language experts showing massive joint perplexity improvements and systems-level VRAM savings).

Furthermore, the paper bridges deep learning theory with physical hardware deployment by profiling:
- **Structured Block Pruning:** Masks entire attention heads and MLP neuron blocks, yielding a **1.89x physical speedup** on mobile CPUs out-of-the-box.
- **Percentile Sorting Mitigations:** Implements linear-time Histogram-based Quantile Estimation and Delayed Thresholding, yielding up to **17.4x sorting speedups** with zero accuracy loss.
- **VRAM footprints during Calibration:** Details that ZipMerge (ES) reduces peak VRAM from 1.45 GB to 180 MB (**8.1x memory reduction**) and saves up to **13.2x memory** on GPT-2 sequence adaptation.
- **Joint Quantization-Pruning (INT8/INT4 PTQ):** Integrates post-training quantization into Identity-pass STE, demonstrating INT4 robustness (**8x storage reduction**) and detailing compiler layout and silent decompression bottlenecks (Apple CoreML and Qualcomm SNPE layouts).

The authors translate these physical realities into actionable, concrete architectural guidelines for edge engineers: avoiding extreme task disparity on compact backbones, leveraging PEFT adapters on top of robust contrastive pre-trained foundation models (acting as a "coordinate anchor"), and employing explicit test-time regularizations.

---

## Strengths
1. **Scientific Honesty and Rigor:** The paper is exceptionally refreshing in its framing. Presenting a detailed "limitation-mapping" post-mortem rather than sanitizing results is highly commendable and provides massive, rare scientific value to the model merging community.
2. **Technical and Mathematical Soundness:** Every component is mathematically flawless and highly elegant. This is particularly evident in the dual STE/ES engines, the step-wise Algorithm 1, the joint PTQ-pruning co-design formulation, and the closed-form **Orthogonal Procrustes SVD Alignment** algorithm which analytically solves coordinate basis mismatch.
3. **Outstanding Experimental Depth and Diversity:** The experimental verification is exhaustive. The authors evaluate four diverse backbones (Transformers, CNNs, and LLMs), multiple datasets (high-conflict visual suite, low-conflict DomainNet, generative multilingual GPT-2), diverse pre-training regimes (supervised ImageNet vs. contrastive CLIP), structured vs. unstructured sparsity, and multiple random seeds and calibration sizes.
4. **Physical Systems Backing:** The inclusion of actual physical latency measurements on an ARM mobile CPU, VRAM memory profiles during calibration, and CPU percentile sorting overheads elevates the paper, translating theoretical equations into concrete physical execution metrics.
5. **High-Yield Practical Innovations:** The proposed Orthogonal Procrustes SVD Alignment (+16.45% absolute boost for LoRA merges) and structured mobile block pruning (1.89x speedup) are highly practical, lightweight, and extremely effective contributions that are immediately useful to practitioners.

---

## Weaknesses
The submission is extraordinarily solid, comprehensive, and technically sound, leaving virtually no major technical flaws. I identify only a few minor areas of improvement for future discussion or text refinement:
1. **Physical Quantized-Sparse Hardware Evaluation:** While the paper physicalizes unstructured sorting overheads (histograms) and structured block pruning (ARM CPU latency), the joint quantized-sparse INT4/INT8 study is evaluated through accuracy-preserving simulation (standard practice for PTQ). The authors honestly note that physical execution on NPUs is deferred to future hardware studies due to current compiler and memory decompression layouts, but adding a brief sentence on how custom JIT compilers (e.g., TVM) could help bypass CoreML/SNPE decompression bottlenecks would make this section even stronger.
2. **Extension of Coordinate Alignment to other PEFT Methods:** The Orthogonal Procrustes SVD Alignment is beautifully formulated and validated for LoRA. Discussing how this analytical rotation could be adapted to other parameter-efficient methods (e.g., IA3 or prefix-tuning) would expand the theoretical footprint of Section 4.5.3.

---

## Detailed Ratings

### Soundness: Excellent (4/4)
The mathematical formulations are highly rigorous, clear, and technically correct. The dual STE (Identity-pass vs. Mask-pass) and zero-order ES formulations are precise. The Procrustes SVD alignment algorithm is mathematically complete and elegant. All experimental assumptions (calibration sizes, layer groupings) are realistic and scientifically validated.

### Presentation: Excellent (4/4)
The writing is of professional publication quality. The manuscript is clear, well-structured, and highly engaging. Figure 1 and Figure 2 are clean and professional. Table 1 and Table 2 are highly readable and informative. Algorithm 1 is structured beautifully, and equations are perfectly aligned and defined.

### Significance: Excellent (4/4)
The paper is highly significant. By forcing the community to reckon with physical system realities (such as storage constraints, extreme domain shifts, and transductive overfitting of test-time adaptation), this work shifts academic norms. The practical speedups (1.89x CPU latency reduction), calibration VRAM profiles (8.1x reduction), and the massive +16.45% LoRA merge boost represent high-impact, immediately actionable contributions.

### Originality: Excellent (4/4)
The co-optimization of layer coefficients and dynamic pruning boundaries is highly innovative. The introduction of SVD-based Orthogonal Procrustes rotation to resolve coordinate basis mismatch is highly original and mathematically elegant. Framed as a boundary-mapping limitation study, the paper’s approach is refreshing and highly original.

---

## Overall Recommendation

**6: Strong Accept**

This is a technically flawless, exceptionally comprehensive, and beautifully written paper. It stands out due to its scientific honesty, rigorous experimental coverage across multiple model families, and deep systems-level profiling on mobile CPUs. The proposed Orthogonal Procrustes SVD alignment and structured block-pruning represent major, high-yield practical contributions. It represents an exemplary piece of research that is highly ready for publication, with no unaddressed flaws.

---

## Constructive Suggestions for the Authors
1. **Custom JIT Compilation Discussion:** In Section 4.5.5, when discussing the "Storage-RAM Paradox" and decompression bottlenecks in CoreML/SNPE compiler runtimes, consider mentioning that emerging custom compiler JIT backends (such as Apache TVM, MLIR, or Halide) could potentially compile these unstructured sparse-quantized layers directly into cache-local vectorized instructions, bypassing the need to allocate dense float buffers in RAM. This would provide edge compiler engineers with an exciting direction to explore.
2. **Adapting Procrustes to other PEFT manifolds:** In Section 4.5.3, add a brief note on whether the Orthogonal Procrustes SVD rotation can be extended to other PEFT structures. For example, for IA3 (which scales activations via learned vectors), can coordinate alignment be formulated as a rescaling vector rotation? This would broaden the significance of the coordinate alignment section.
3. **Progressive Cosine Ramping Visuals:** In Section 4.5.1, the authors mention that progressive cosine schedules eliminate optimization shocks. Adding a small appendix figure showing the calibration entropy trajectory over time for the abrupt vs. progressive schedules would visually reinforce this excellent systems finding.
