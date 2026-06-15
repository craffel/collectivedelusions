# 4. Experiment Check

An empirical analysis of the experimental design, datasets, baselines, and overall results highlights several severe limitations, biased comparisons, and missing statistical details that weaken the paper's claims.

## Evaluation of Experimental Setup & Datasets
- **Highly Artificial Datasets:** The "Isolating Coordinate Sandbox (ICS)" is a synthetic toy environment. The authors mention "simulated MNIST, FashionMNIST, and CIFAR-10 representational proxy samples," but these are not real image datasets. They are low-dimensional mathematical proxies. Real-world images or text data have high-dimensional manifolds with complex correlations, and evaluating on a toy proxy tells us very little about how the method will perform on real computer vision or natural language tasks.
- **Lack of Standard Benchmarks:** The paper does not evaluate LSPR on any standard, widely-accepted PEFT multi-task benchmarks (such as GLUE/SuperGLUE for LLMs, or VTAB/ImageNet-1K for Vision Transformers). This is a major gap for a paper claiming to establish a "new SOTA."

## Baseline Comparisons and Potential Biases

An Empiricist must look closely at how the baselines are evaluated and whether the comparisons are fair:

### 1. The "Co-Designed Adapter" Bias (Apples-to-Oranges Comparison)
LSPR's high accuracy (85.81%) and outstanding OOD detection (0.9755 AUROC) are evaluated on adapters trained with LSPR's joint classification-reconstruction loss.
- **Home-Court Advantage:** Training the adapters with the reconstruction loss specifically guides their column space to align with the activation subspace. Evaluating baseline methods (like SPS-ZCA, SABLE, PFSR) on these *co-designed* adapters is highly biased. It gives LSPR a massive home-court advantage.
- **No Evaluation on Standard Adapters:** The authors do not show how the baselines perform on standard, unaligned LoRA adapters (trained without the reconstruction loss) versus how LSPR performs. On standard adapters, LSPR's accuracy collapses to **19.79%** (as shown in Section 4.7). What about SPS-ZCA? If SPS-ZCA maintains high routing accuracy on standard adapters, then SPS-ZCA is a far more robust and practical post-hoc routing solution. The paper obscures this by comparing LSPR on its own co-designed adapters against baselines on the same co-designed adapters, presenting an incomplete picture of the method's practical limitations.

### 2. "Strawman" Systems Latency Baseline
In Figure 4, the authors compare LSPR's serving latency against "PFSR + MBH SOTA (Sequential Serving)" and show a massive physical speedup.
- **Unoptimized Sequential Loop:** The PFSR+MBH baseline is physically executed as a sequential `for` loop in PyTorch, which partitions the batch and launches sequential forward passes. In Python, launching sequential PyTorch execution blocks introduces severe interpreter and scheduling overhead.
- **Bypassing the Real Comparison:** The speedup LSPR achieves is primarily due to vectorizing the execution into a single parallel pass. However, any routing method (including PFSR) can be vectorized to execute in a single parallel pass if the systems layer is optimized. Comparing a vectorized parallel execution (LSPR) against an unoptimized sequential Python loop (PFSR) is a classic "strawman" systems comparison. A fair systems evaluation would compare LSPR to a vectorized parallel baseline.

### 3. Missing Empirical Proof for Sparse-LSPR Latency Scaling
The authors propose "Sparse-LSPR" (Top-$M$ gating) to solve the linear latency scaling with registry size $K$ (shown in Figure 5).
- **No Curve in Figure 5:** While the authors state in the text that "Sparse-LSPR Top-2 gating... decouples physical execution latency from the expert registry size $K$," they **do not include the empirical scaling curve of Sparse-LSPR in Figure 5**.
- **Unvalidated Systems Claim:** For a systems claim of "flat, constant-time scaling," an Empiricist expects to see the actual wall-clock latency curve of Sparse-LSPR plotted alongside LSPR and PFSR. Omitting this curve makes their scalability claims unconvincing and empirically unverified.

### 4. Significant Regression in "Post-Hoc Warm Alignment"
The authors present Post-Hoc Warm Alignment as a solution for public unaligned adapters.
- **Massive Performance Drop:** The reported accuracy of warm-aligned LSPR is only **66.02%**, which is a severe **19.79% absolute performance drop** from the 85.81% expert ceiling.
- **Understated Limitations:** The authors gloss over this 20% drop and claim it "completely restores LSPR's zero-shot serving compatibility... without sacrificing its original capabilities." An empirical perspective must point out that a 20% drop in accuracy is a failure state for most production systems, meaning warm alignment is practically non-viable in its current form.

## Missing Statistical Rigor
The paper is completely devoid of statistical significance details:
- **No Standard Deviations or Error Bars:** All reported accuracy metrics in Table 1 (e.g., 85.81%, 23.96%) and OOD AUROC (0.9755) are reported as exact point estimates. There are no standard deviations, confidence intervals, or error bars in any table or figure.
- **No Random Seeds:** The authors do not state how many random seeds were run to produce the results. Wall-clock measurements on host CPUs (Figures 4 and 5) are notoriously noisy and highly dependent on OS background processes. Reporting latency curves without error bars or averaging over multiple independent runs is statistically unsound.
- **Identical Points:** In Table 1, LSPR and Expert Ceiling are both exactly 85.81% for both homogeneous and heterogeneous streams, and PFSR is exactly 85.81%. This indicates that the authors are reporting exact ceiling recoveries rather than actual empirical means across multiple noisy training/evaluation runs.
