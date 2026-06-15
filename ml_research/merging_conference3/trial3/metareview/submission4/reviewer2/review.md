# Peer Review of Conference Submission

## Paper Summary
The paper conducts a comprehensive post-mortem and limitation-mapping study of joint model merging and weight pruning under resource-constrained edge-deployment settings. The authors introduce **ZipMerge**, a framework designed to co-optimize layer-wise merging coefficients and magnitude-pruning boundaries at test-time on tiny, unlabeled calibration datasets using an unsupervised minimum entropy objective. To bypass the non-differentiable operators introduced by magnitude pruning, the paper evaluates and compares two optimization paradigms: (1) first-order gradient descent via a Straight-Through Estimator (ZipMerge-STE) and (2) zero-order search via a 1+1 Evolution Strategy (ZipMerge-ES).

Evaluating a compact Vision Transformer backbone (`vit_tiny_patch16_224`) across four highly disparate visual classification tasks (MNIST, FashionMNIST, CIFAR-10, SVHN) under extreme domain shift, the paper honest-reports a series of fundamental limits:
1. **Catastrophic Representational Collapse:** Every merged model (Uniform, AdaMerging, and standard ZipMerge) collapses to random guessing levels (~10% to 14% accuracy) due to severe representational and activation-pathway interference in the compact backbone.
2. **Prune-then-Merge (P-then-M) Outperformance:** The unoptimized, decoupled baseline, P-then-M, consistently outperforms joint test-time optimization because pre-merging pruning acts as a spatial regularizer, removing non-overlapping task updates and parameter noise.
3. **The Overfitting-Optimizer Paradox:** Unconstrained minimum-entropy TTA on tiny calibration sets overfits transductively, successfully minimizing entropy while destroying generalizable features and driving test set accuracy down.

To address these boundaries, the authors propose and evaluate several advanced solutions:
* **PEFT and Aligned Merging:** Restricting fine-tuning to low-rank manifolds (LoRA) dramatically limits representation shifts, boosting dense merge accuracy by +29% absolute. Introducing **Orthogonal Procrustes SVD Alignment**—which rotates separately trained LoRA coordinate spaces into a shared basis prior to merging—boosts accuracy further to 58.75% dense and 62.10% under 50% structured sparsity.
* **Structured Block Pruning:** Masking entire MLP neurons or attention heads delivers a 1.89$\times$ physical latency speedup on an ARM Cortex-A76 mobile CPU (34.2 ms down to 18.1 ms per image) with minimal optimization degradation.
* **Dynamic Sorting and Memory Optimizations:** Bypassing sorting overheads via Delayed Thresholding (10$\times$ speedup) or Histogram-based Quantile Estimation (17.4$\times$ speedup), and leveraging zero-order ES to realize an 8.1$\times$ peak RAM reduction (VRAM) during calibration.
* **Autoregressive Language Models:** Extending ZipMerge to GPT-2 (124M parameters), showing stable convergence and a 13.2$\times$ peak memory savings of ES over STE at larger context lengths.

---

## Strengths and Weaknesses

### Strengths
1. **Pragmatic and Highly Honest Post-Mortem:** The paper's willingness to publish a detailed "limitation mapping" study is incredibly refreshing and valuable. It saves real-world systems engineers and researchers massive amounts of time by clearly defining the boundaries of linear weight-space operations under extreme task shift.
2. **Exceptionally Comprehensive Empirical Evaluation:** The sheer volume and quality of the empirical sub-studies are outstanding. The paper includes multiple backbones (ViT-Tiny, ViT-Base, ResNet-18), modalities (Vision and Generative Language), extensive sensitivity checks (seeds, batch sizes, global scaling factors, quantization bit-widths), and rigorous ablations (expert convergence, TIES-ZipMerge).
3. **High-Value Systems and Hardware Profiling:** The paper directly addresses actual hardware constraints, providing physical latency measurements on an ARM Cortex-A76 mobile CPU, peak RAM/VRAM profiling during calibration (highlighting the ES memory advantage), and Xeon CPU latency measurements for sorting mitigations (Delayed Thresholding and Histogram-based Quantile Estimation).
4. **Elegant SVD-Based Orthogonal Procrustes Alignment:** The introduction of an analytical, post-hoc rotation to resolve coordinate basis misalignment in PEFT space is mathematically elegant and highly practical. It delivers a massive +16.45% absolute accuracy boost with zero data requirements and completely negligible computational overhead ($<1$ millisecond).
5. **Excellent Transparency and Reproducibility:** Includes complete mathematical formulations, step-by-step pseudo-code (Algorithm 1), explicit hyperparameters, and a Reproducibility Statement pointing to a public GitHub repository under the MIT License.

### Weaknesses
1. **Simulated Joint Quantization-Pruning:** While the joint post-training quantization (PTQ) and unstructured pruning co-design is highly promising and thoroughly simulated, actual physical execution latency and RAM measurements on specialized NPUs are not provided. The authors conduct a stellar, highly realistic discussion of edge compiler layouts and decompression bottlenecks (such as CoreML and SNPE), but empirical hardware execution studies remain an area of future work.
2. **High Information Density:** The paper pack-moves through a massive number of sub-studies and sweeps, which can make the draft feel highly dense. Streamlining some subsections or creating dedicated appendices for auxiliary sweeps (e.g., global scaling factor sweeps or calibration sample sensitivities) could help keep the main narrative focused, though the thoroughness of the current draft is a major asset.

---

## Detailed Dimension Evaluations

### Soundness
* **Rating:** Excellent
* **Justification:** The paper is technically solid and methodologically rigorous. Every mathematical formulation (task vectors, percentile thresholding, Identity-pass STE, 1+1 ES, Orthogonal Procrustes rotation, structured block pruning, and joint PTQ) is clearly stated, logically sound, and physically intuitive. The authors are extremely careful and honest about evaluating the limitations of their work, such as highlighting the Overfitting-Optimizer Paradox and the failure of unconstrained TTA. The experimental methodology is highly robust, utilizing 5 independent random seeds with extremely low standard deviations ($\pm 0.29\%$ to $\pm 0.52\%$), sweeping diverse backbones, and presenting comprehensive ablations.

### Presentation
* **Rating:** Excellent
* **Justification:** The submission is clearly written, exceptionally well-structured, and easy to follow. The overall narrative transitions smoothly from the initial "failure post-mortem" to systematic diagnostics, mathematical mitigations, and physical hardware profiling. The authors do an outstanding job positioning their work relative to existing literature and clearly explaining how their co-optimization framework differs. Captions on tables and figures are highly detailed and self-contained, and Algorithm 1 provides a complete, systems-level pseudocode trace.

### Significance
* **Rating:** Excellent
* **Justification:** The paper addresses a highly important, pressing problem in edge-device deployment: how to compress multi-task merged models to fit within tight on-device memory and latency budgets. By honest-mapping the limitations of linear merging under extreme domain shift, the paper provides immense practical utility, steering edge engineers away from naive, full-backbone merging and instead guiding them toward robust, modular solutions: parameter-efficient low-rank adapters (PEFT), SVD-based post-hoc coordinate pre-alignment, and hardware-friendly structured block-pruning. The findings regarding ES memory savings and CPU latency gains are highly significant and likely to influence future edge adaptions.

### Originality
* **Rating:** Excellent
* **Justification:** The co-optimization of layer-wise coefficients and dynamic pruning boundaries at test-time using STE or ES is highly original. Furthermore, the introduction of the SVD-based Orthogonal Procrustes alignment to rotate separately trained LoRA adapters into a shared basis is mathematically elegant and extremely novel in the context of model composition. The paper's overall framing as an honest, limitation-mapping post-mortem is highly creative and stands out significantly from the crowded field of standard "success-only" publications.

---

## Overall Recommendation

* **Recommendation:** 5: Accept
* **Justification:** This is a technically solid, highly thorough, and outstandingly comprehensive paper that makes a significant contribution to the applied machine learning and edge systems community. It provides immense practical utility by honestly mapping weight-space limits, profiling physical hardware metrics, and presenting elegant mathematical solutions (such as SVD-based coordinate alignment and structured block pruning) that directly resolve the massive performance and latency gaps. The evaluation is exceptionally thorough, the reproducibility is highly robust, and there are no unaddressed ethical considerations.

---

## Questions / Suggestions for the Authors
1. **Physical NPU Quantized-Sparse Execution:** In Section 4.5.4, you present a highly valuable simulated study of joint post-training quantization and pruning. Have you explored compiling these INT4/INT8 sparse-quantized models using specialized runtimes (such as Apache TVM or custom MLIR backends) that can generate cache-local instruction segments to bypass the CPU decompression bottleneck?
2. **Dynamic Procrustes in Adaptation:** Since the singular vectors ($U, V$) derived from the SVD of $C=W_1^T W_2$ are static (as the expert weights remain frozen during calibration), you correctly initialize the alignment once. Have you considered whether any dynamic rotation is beneficial if the blending coefficients $\Lambda$ scale the adapters unevenly across layers, or is the static coordinate rotation mathematically sufficient under all layer scaling configurations?
3. **Information Layout:** The draft is incredibly dense with sub-studies (which is a massive strength). To further enhance readability, consider moving the auxiliary sweeps (such as the calibration set size sensitivity and the global scaling factor sweeps) into a dedicated appendix, keeping the main text focused on the primary visual suite, PEFT alignments, and hardware profilings.
