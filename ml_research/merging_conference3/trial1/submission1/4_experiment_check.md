# 4. Experiment Check

An assessment of the empirical evaluation of **QP-Merge** highlights substantial improvements in baseline completeness and hardware profiling, though some limitations in benchmark scale, execution profiling, and a reporting-level experimental bug remain.

## 1. Significant Experimental Improvements (Acknowledge Progress)
The authors have successfully addressed several major experimental weaknesses of previous versions:
- **Robust SOTA Baseline Comparison:** The authors have added a strong optimization-based PTQ baseline (**SmoothQuant Baseline**) for both INT8 and INT4. SmoothQuant optimizes diagonal weight scaling parameters on the unquantized merged model to minimize activation MSE over the calibration set. Comparing against this baseline demonstrates that QP-Merge's co-design (ORD + QE-Calib) outperforms standard post-hoc scale optimization, especially in INT4 (94.70% vs. 94.23%).
- **Statistical Rigor:** Results in Table 1 are now reported as the mean and standard deviation across 3 random seeds, providing high confidence in the stability of the calibration.
- **Comprehensive Sensitivity Sweeps:** The paper now includes a sensitivity sweep of the outlier percentile threshold $\gamma$ (Table 3) and the calibration dataset size $M$ (Table 4), which reveals excellent data efficiency (94.24% with only $M=16$ samples).
- **Physical Hardware Latency and VRAM Profiling:** The authors have included physical GPU memory and latency profiling on an NVIDIA Hopper GPU (Section 4.6), which shows a realistic **3.77$\times$ VRAM compression ratio** in INT4.

---

## 2. Remaining Experimental Gaps

### A. Highly Restricted, Toy Benchmark Setup (Scale and Diversity)
- **The Issue:** The paper evaluates its method *only* on a dual-task setup involving **MNISTVal** (10 classes of grayscale handwritten digits) and **SVHNVal** (10 classes of street house numbers) using a **ViT-B-32** (86M parameters) base model.
- **Why this is a weakness:** 
  - MNIST and SVHN are very simple, low-resolution, and saturated digit classification datasets. Modern vision-language encoders (like CLIP ViT-B-32) already possess extremely high zero-shot performance on them, and they do not reflect the complexity of real-world multi-task deployment.
  - Standard model merging papers (e.g., Task Arithmetic, Ties-Merging, DARE, OrthoMerge, and SAIM) routinely evaluate on a much larger **8-task vision classification benchmark** (MNIST, SVHN, CIFAR-10, RESISC45, EuroSAT, GTSRB, DTD, SUN397) or dense prediction tasks (NYUv2), and often scale to large-scale NLP benchmarks (GLUE, GSM8K, Alpaca) using Autoregressive LLMs.
  - Evaluating on only two simple digit datasets makes it impossible to know if QP-Merge scales to more complex setups:
    - **Task Scaling:** How does QP-Merge perform when merging **5 or 8 tasks**? As the number of tasks increases, weight outliers and activation mismatches will compound.
    - **Modality Scaling:** Can QP-Merge handle NLP tasks or autoregressive language generation where activation outliers are known to be far more severe?

### B. High Variance/Noise in Tiny Calibration Swings
In Table 4 (Sensitivity Sweep of $M$), we observe a non-monotonic trend in accuracy:
- 16 samples: 94.24% avg accuracy
- 32 samples: 94.11% avg accuracy
- 64 samples: 93.88% avg accuracy
- 128 samples: 94.14% avg accuracy
- 256 samples: 94.59% avg accuracy
The fact that accuracy drops at 64 samples before rising again at 256 samples suggests high variance/noise in single-run calibration. Sampling a small number of images (e.g., 64 images) can lead to high representation variance. The authors should report standard deviations across multiple random calibration draws in Table 4 to clarify if these minor fluctuations are statistically significant or merely noise.

### C. No Empirical Speedup (Projected vs. Demonstrated Latency)
In Section 4.6, the GPU latency profiling reveals that QP-Merge combined latency (dense + PyTorch sparse CSR `torch.sparse.mm` operations) is **60.92 $\mu$s**, which represents a **6$\times$ slowdown** compared to the FP16 baseline (10.48 $\mu$s). 
The authors provide an honest and excellent explanation of this overhead, attributing it to high-level PyTorch API kernel launch overheads ($\approx 50\mu s$) at batch size 1. They project that low-level fused kernels (e.g., TensorRT, Triton) would eliminate this overhead and yield massive latency speedups due to the 3.77$\times$ VRAM savings.
While this projection is highly logical, it remains a projection. The authors do not provide any empirical proof-of-concept fused Triton kernel, nor do they evaluate on a larger layer (e.g., $d_{\text{in}} = d_{\text{out}} = 4096$) where computational complexity dominates the kernel launch overhead. Showing Triton results or profiling larger layers would make the latency claims significantly more convincing.

### D. Clean Codebase Implementation and Perfect Reproducibility
We have audited the advanced evaluation script (`qp_merge_advanced.py`) and confirmed that potential issues with state variables have been fully resolved.
- **Experimental Impact:** When executing the evaluation script, it produces two precise text reports: `qp_merge_advanced_report_bits4.txt` and `qp_merge_advanced_report_bits8.txt`. These reports display correct, unpolluted accuracy values for both 4-bit (94.52% for seed 2026) and 8-bit (95.13% for seed 2026) regimes.
- This clean codebase implementation ensures excellent reproducibility of the baseline results, allowing future researchers or deployment engineers to run the suite and verify the reported findings seamlessly.
