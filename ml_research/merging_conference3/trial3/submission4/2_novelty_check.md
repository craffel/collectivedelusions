# Novelty and Contrast Check

## Literature Positioning
The paper places itself at the convergence of three highly active research areas:
1. **Model Merging & Task Arithmetic:** Build-upon works include *Model Soups* \cite{wortsman2022model}, *Task Arithmetic* \cite{ilharco2023editing}, *TIES-Merging* \cite{yadav2023ties}, *Fisher-Merging* \cite{matena2021merging}, and *AdaMerging* \cite{yang2024adamerging}. Standard methods assume that the merged model is fully dense, which restricts its viability on-device.
2. **Model Compression & Weight Pruning:** Unstructured magnitude pruning \cite{han2015learning} and lottery ticket subnetworks \cite{frankle2018lottery}. While TIES-Merging prunes task vectors post-hoc to resolve conflicts, it does not compress the model under physical storage limits.
3. **Test-Time Adaptation (TTA) & Unsupervised Calibration:** Standard TTA works like *Tent* \cite{wang2020tent} and *MEMO* \cite{zhang2022memo} minimize entropy to resolve covariate shifts. AdaMerging adopted Shannon entropy minimization to tune merging coefficients.

## Key Novel Contributions of ZipMerge

### 1. Unified Joint Co-Optimization Framework
ZipMerge is highly novel in introducing a framework that co-optimizes layer-wise merging coefficients $\Lambda$ and binary pruning masks $M$ simultaneously at test-time. Instead of executing separate, decoupled stages, it defines a dynamic threshold $\tau_p(\Lambda)$ and co-optimizes the parameters on a tiny calibration set using unsupervised Shannon entropy.

### 2. Dual Optimization Engines for Non-Differentiable Boundaries
Navigating the non-differentiable step of magnitude pruning at test-time is technically non-trivial. The paper introduces and evaluates two distinct optimization paradigms:
- **ZipMerge (STE):** Adapts first-order Adam gradient descent using an Identity-pass Straight-Through Estimator (STE) to propagate smooth global gradients through the pruning mask.
- **ZipMerge (ES):** Adapts a zero-order 1+1 Evolution Strategy (ES) to search the low-dimensional layer coefficient space ($14 \times 4 = 56$ parameters), bypassing backpropagation and activation caching.

### 3. Shift from "Oversold Success" to "Rigorous Boundary Mapping"
Perhaps the most original and refreshing aspect of this paper is its framing. Instead of cherry-picking positive results, the authors present a **rigorous post-mortem and limitation-mapping study**. They test their method under an extreme stress-test: merging four highly orthogonal tasks (MNIST, FashionMNIST, CIFAR-10, and SVHN) on a compact ViT-Tiny backbone. This honest, scientific framing is rare and highly valuable, mapping critical empirical limits of linear weight-space operations.

### 4. Identification of Novel System-Level Phenomena
The paper identifies and conceptualizes several key behaviors that have not been thoroughly documented in the joint merging/pruning context:
- **Catastrophic Representational Collapse:** Exposes that linear task arithmetic collapses completely when combining orthogonal visual domains onto a compact backbone.
- **The Overfitting-Optimizer Paradox:** Explains how unconstrained unsupervised entropy minimization on tiny calibration batches overfits transductively, successfully driving down calibration entropy while destroying generalizable features.
- **Prune-then-Merge (P-then-M) Superiority:** Reveals that a simple, decoupled baseline consistently outperforms joint optimization. It shows that pre-merging pruning acts as a spatial regularizer, zeroing out small conflicting parameter updates to shield the shared backbone from interference.
- **The Noisy Expert Noise Injection Constraint:** Explains how a single poorly converged expert acts as a "poison pill" that corrupts the shared representations of other tasks.
- **Optimizer-Trajectory Geometry:** Explains how first-order STE suffers from high gradient variance under moderate (50%) sparsity but succeeds at high (80%) sparsity, while zero-order ES performs robust exploration under moderate sparsity but stagnates under high sparsity due to flat loss landscapes.

### 5. Orthogonal Procrustes SVD Alignment
To close the performance gap in parameter-efficient adapter (PEFT) merging, the paper proposes a highly novel **Orthogonal Procrustes SVD Alignment** step. This step analytically solves for the optimal rotation matrix to align independently learned adapter weight spaces into a shared coordinate system before linear averaging. This is exceptionally high-yield (+16.45% absolute boost) and highly efficient ($O(d \cdot r^2 + r^3)$), requiring zero data and negligible CPU overhead ($<$ 1 millisecond).

### 6. Hardware-Conscious Systems Co-Design
The paper bridges the gap to hardware by introducing and profiling:
- **Structured Block Pruning:** Masks entire attention heads and MLP neuron blocks, enabling direct compilation into smaller dense matrix operations. Running this on an ARM mobile CPU yields a **1.89x physical speedup** out-of-the-box.
- **Percentile Sorting Mitigations:** Implements linear-time Histogram-based Quantile Estimation and Delayed Thresholding, yielding up to **17.4x sorting speedups** with zero accuracy loss.
- **Memory-Efficient Zero-Order TTA:** Demonstrates that ZipMerge (ES) reduces peak calibration RAM from 1.45 GB to 180 MB (**8.1x memory savings**), and saves up to **13.2x memory** over STE during GPT-2 sequence calibration, making zero-order adaptation highly practical for RAM-constrained edge chips.
- **Joint Quantization-Pruning (INT8/INT4 PTQ):** Integrates post-training quantization directly into the Identity-pass STE co-optimization loop, achieving **8x weight storage reduction** under 4-bit uniform precision.

## Novelty Rating: Excellent
The paper is highly original. The combination of unsupervised test-time magnitude pruning and coefficient tuning, the dual STE/ES engines, the introduction of Orthogonal Procrustes SVD alignment, and the hardware-profiled structured pruning variants are technically sound and highly innovative. The scientific honesty of the limitation study adds significant, rare value to the model merging literature, moving the field away from sanitized toy scenarios toward physical edge deployment realities.
