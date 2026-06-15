# Mock Review: QP-Merge (Quantization-Preserving Merging)

**Recommendation:** 5 (Accept)  
**Soundness:** Good  
**Presentation:** Excellent  
**Significance:** Good  
**Originality:** Good  

---

## 1. Summary of the Paper

This paper introduces **QP-Merge** (Quantization-Preserving Merging), a training-free framework that co-designs parameter merging and post-training quantization (PTQ) to enable the lossless low-bit compression (INT4/INT8) of merged multi-task deep neural networks. 

Model merging (e.g., Task Arithmetic) combines specialized models without retraining, but standard PTQ applied to merged models causes catastrophic performance degradation. The authors identify two primary physical drivers of this failure:
1. **Heavy-tailed task outliers:** Subtracting base weights from fine-tuned weights inherently highlights sparse, high-magnitude parameter updates ($\Delta W_t$). Standard uniform quantization stretches the symmetric quantization scale $S_c$ to accommodate these outliers, squeezing the dense remainder weights into too few bins and introducing severe quantization noise.
2. **Activation scale mismatches:** Linearly blending task vectors ignores the fundamentally mismatched intermediate activation ranges of models fine-tuned on entirely different distributions.

QP-Merge addresses these bottlenecks through two lightweight, synergistic, training-free techniques:
* **Outlier-Residual Decoupling (ORD):** Isolates the top $\le 1\%$ (or 0.5%) highest-magnitude weight updates from each task vector into a highly sparse, high-precision FP16 tensor ($W_{\text{outlier}}$) while quantizing the tightly bounded dense remainder ($W_{\text{dense}}$) to INT4 or INT8.
* **Quantization-Error Aware Scale Calibration (QE-Calib):** Jointly optimizes column-wise diagonal weight scaling parameters $D_l$ and merging coefficients $\lambda$ using end-to-end backpropagation over a tiny, completely unlabeled set of $M = 128$ domain samples to minimize activation reconstruction MSE relative to the unquantized FP32 merged model's final embeddings.

**Primary Results:** Evaluated on dual-task vision classification (MNISTVal and SVHNVal) with a pre-trained `ViT-B-32`, QP-Merge INT4 achieves an average accuracy of **94.70% $\pm$ 0.13%** (within 0.42% of the optimized FP32 unquantized baseline), recovering over 88% of the performance drop caused by naive quantization. In INT8 mode, it achieves **95.14% $\pm$ 0.03%** average accuracy, representing a virtually lossless compression that slightly outperforms the uniform FP32 unquantized baseline (+0.21%) due to unsupervised target-domain task coefficient optimization.

---

## 2. Strengths

1. **First Co-Design of Merging and Quantization:** The paper is highly motivated, addressing a major practical deployment bottleneck. It represents the first framework to explicitly co-design model merging and post-training quantization, bridging two fields that have historically developed in silos.
2. **High Technical Rigor & Corrected Formulations:** The mathematical formulation is clean, consistent, and highly robust. The authors have correctly defined the column-wise scaling matrix multiplication on the right side of the dense task sum (resolving dimension conflicts present in earlier drafts) and correctly aligned the end-to-end Representation-Level Alignment loss in the text with the implementation.
3. **Thorough Empirical Evaluation & Baselines:** The evaluation compares against both standard naive quantization and a strong, optimization-based post-hoc PTQ baseline (SmoothQuant scale optimization). Including the unquantized "FP32 Merged Bound (Optimized)" baseline decouples the benefit of blending scale tuning from quantization-specific factors, providing a highly honest and clear scientific narrative.
4. **Thorough Robustness & Sensitivity Analyses:** The paper features comprehensive sweeps over outlier percentiles ($\gamma$) and calibration dataset sizes ($M$), alongside evaluations under synthetic out-of-distribution (OOD) corruptions and imbalanced/single-domain calibration constraints. These results demonstrate outstanding data-efficiency and cross-domain generalization.
5. **Honest Engineering Profile:** The inclusion of physical GPU memory profiling (demonstrating a **3.77$\times$ weight compression ratio** in INT4) and wall-clock latency analysis (transparently discussing PyTorch API/kernel launch overheads vs. low-level fused runtime projections) reflects high engineering maturity.
6. **Excellent Reproducibility:** An audit of the provided codebase confirms that the advanced evaluation scripts run cleanly and produce pristine, correct results, completely resolving previous state/variable leakage reporting bugs.

---

## 3. Weaknesses

While the paper is technically solid and highly practical, the following limitations should be addressed to maximize its scientific impact:

1. **Lack of Mathematical Equivalence in Weight Scaling (Approximation Risk):** Unlike traditional SmoothQuant-style approaches that apply inverse activation scaling to maintain strict mathematical equivalence during inference, QP-Merge permanently alters the weights by multiplying them by $D_l$ *without* any corresponding activation adjustments. While the optimization loop minimizes embedding drift over the 128-sample calibration set, this remains a heuristic weight-altering approximation that could theoretically risk representation drift on larger/more complex datasets.
2. **Highly Restricted Toy Benchmark Setup:** The evaluation is confined to a dual-task setup involving MNISTVal and SVHNVal using a `ViT-B-32` base model. MNIST and SVHN are saturated, low-resolution classification tasks. Standard model merging papers routinely evaluate on a much larger 8-task vision suite (including CIFAR-10, EuroSAT, GTSRB, etc.) or scale to large-scale autoregressive language model tasks (GLUE, GSM8K, Alpaca). Evaluating on only two simple digit datasets makes it difficult to assess how the framework scales to:
   * **Task Scaling:** Multi-task merging of 5 to 8+ tasks.
   * **Modality Scaling:** Severely outlier-heavy autoregressive NLP workloads.
3. **Projected vs. Realized GPU Latency Speedups:** On the profiling GPU node, the hybrid QP-Merge layer executes at 60.92 $\mu$s, which represents a 6$\times$ slowdown relative to the FP16 baseline (10.48 $\mu$s) due to sequential CUDA kernel launch overheads in PyTorch at batch size 1. Although the authors' analytical scaling analysis for LLM-sized layers and fused Triton/TensorRT kernel projections are highly logical, the paper lacks an empirical proof-of-concept fused kernel implementation to demonstrate actual wall-clock speedups at scale.

---

## 4. Rating Justifications

### Soundness: Good (4/4)
The methodology is mathematically sound, and the formulations in Section 3 are fully verified. The baseline comparisons are complete, utilizing an optimized FP32 upper bound to ensure clear scientific attribution of performance gains. The primary limitation is the lack of strict mathematical equivalence in weight scaling, which is a pragmatic approximation that appears to generalize well under the evaluated constraints but remains a theoretical drift risk.

### Presentation: Excellent (4/4)
The paper is exceptionally well-written, clearly structured, and easy to follow. Figure 1 provides a clean, intuitive overview of the hybrid execution path. The equations are mathematically valid, and the terminology is precise. The authors' discussion in Section 5.1 of edge hardware compatibility, outlier overlap, and scaling is outstandingly honest and mature.

### Significance: Good (3/4)
The work addresses a highly relevant, real-world deployment problem. It bridges the gap between academic model merging and practical low-cost edge Serving. The significance is slightly capped by the toy nature of the evaluation datasets (MNIST/SVHN), leaving some scaling questions open, but the physical compression ratio (3.77$\times$) and optimization efficiency make it a highly valuable contribution to practitioners.

### Originality: Good (3/4)
The individual components of QP-Merge are elegant adaptations of existing concepts in the PTQ literature (SqueezeLLM's dense/sparse outlier separation and AdaRound/SmoothQuant-style reconstruction loss minimization). The originality lies in the **creative, high-impact co-design**—applying these tools to delta task vectors and jointly calibrating scales and merging coefficients post-hoc.

---

## 5. Actionable Constructive Feedback

### Major Revisions & Clarifications:
* **Task and Benchmark Scaling:** Discuss or ideally run a larger-scale evaluation (even a 4-task vision suite) to empirically show how outlier density behaves when tasks are merged. If possible, show that weight outliers across tasks indeed cluster around identical feature projections (as hypothesized in Section 5.1), preventing linear density scaling.
* **Weight Scaling Drift Analysis:** Quantify the "representation drift" caused by non-equivalent scaling. You can do this by reporting the unquantized performance of the model after scaling is applied but *before* quantization. This will isolate the drift introduced by the permanent weight modification from the quantization noise.
* **Text-Table Result Synchronization:** Resolve minor reporting discrepancies in the manuscript. For example, Table 1 reports a 3-seed average INT8 accuracy of **95.14% $\pm$ 0.03%**, but the Abstract, Introduction, Table 2, and Section 5 refer to INT8 performance as **95.08%** (which is the result of single seed 2026). Ensure that the caption of Table 1 explicitly denotes "3-seed averages", and update the other references to either match Table 1 or clearly state they are reporting the representative seed 2026.

### Minor Suggestions:
* **Figure 1 Clarity:** Explicitly clarify in the caption of Figure 1 that the calibration parameters $D_l$ and $\lambda$ are optimized jointly using end-to-end representation-level backpropagation, rather than local layer-wise reconstruction, as "layer-wise scale calibration" might suggest local optimization to some readers.
* **Edge Hardware Compatibility Disclaimer:** Moderate the bold claims in the Abstract and Introduction regarding immediate deployment on "microcontrollers and edge IoT devices." Since QP-Merge requires executing a sparse floating-point (FP16) path alongside the dense path, it is incompatible with homogeneous fixed-point processors that lack floating-point units. Acknowledging this in the introduction aligns better with the excellent discussion in Section 5.1.
