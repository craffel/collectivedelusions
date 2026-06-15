# Experimental and Evaluation Audit

This document critical reviews the experimental setup, datasets, baselines, and whether the empirical results actually support the paper's core claims, through the eyes of a deployment-minded practitioner.

## 1. Experimental Setup and Sandbox Limitations

The core of the paper's empirical validation is conducted in the **Isolating Coordinate Sandbox**. While the authors justify this as a "highly controlled mathematical instrument" to audit individual routing behaviors (Section 3.1), a practitioner must raise severe concerns about this environment:
- **Disjoint, Orthogonal Block Coordinates:** The sandbox partitions the global feature space $z_b \in \mathbb{R}^{192}$ into 4 disjoint blocks of dimension 48, each containing noise or class prototypes. This represents an idealized, toy setup where there is absolutely no overlap or interference between task features.
- **Single-Layer Geometry:** The sandbox is a simple, single-layer linear classification model. 
- **Pre-normalized Classifier Weights (UNC):** Expert weights are pre-normalized, bypassing real-world learning and calibration challenges.

In real-world deployment (e.g., merging LoRA experts on a pre-trained Transformer), representation spaces are highly overlapping, non-orthogonal, and nested deep within multi-layer architectures. Features share dimensions, and model weights have significant parameter conflicts. Therefore, the sandbox is a highly contrived environment that does not represent realistic deep learning ensembling conditions.

## 2. Baseline Comparison

The paper compares the proposed methods against:
- **Static Merging:** Uniform Merging. (The authors mathematically prove that more advanced static ensembling methods like TIES-Merging and DARE reduce to Uniform Merging in this sandbox due to coordinate isolation, which is technically correct but further highlights the toy nature of the environment).
- **Parametric Routers:** Linear Router (Unreg/Reg), VR-Router, TSAR.
- **Non-Parametric Routers:** PFSR.

This is a reasonable list of ensembling-routing baselines. However, there are no comparisons to other streaming or multi-tenant serving baselines. S-LoRA and Punica are discussed qualitatively (Appendix D.5) but are not empirically compared in terms of latency, throughput, or memory under heterogeneous streams on actual GPU hardware.

## 3. Support for Claims

The empirical results in the paper do support the specific claims made *within the sandbox*:
- **Peak Performance Envelope (Fig 1):** The threshold sweep clearly shows a peak around $\gamma_{\text{conf}} \approx 0.85$ (for Max Probability) where CGHR achieves higher joint accuracy than pure parametric ($\gamma_{\text{conf}}=0$) or pure PFSR ($\gamma_{\text{conf}}=1$).
- **Data Scarcity Resilience (Fig 2):** At small calibration sizes ($N = 16$), CGHR maintains near-perfect flatline performance matching PFSR, while standard linear routers experience severe transductive overfitting and performance drops.
- **Heterogeneity Collapse Prevention (Fig 3):** Under mixed-task streams, standard routers suffer from catastrophic collapse as batch size $B$ scales, dropping to ensembling performance ($63.10\%$). Integrating MBH maintains a perfectly flat performance curve up to $B=512$, proving its absolute effectiveness in isolating tasks.
- **Systems optimizations (Tables 3, 5, 6):** 
  - Table 5 empirically validates that Fusion Weight Caching achieves a $2.87\times$ weight fusion speedup at discretization step $0.10$ with zero accuracy degradation.
  - Table 6 demonstrates that Warp Batch Padding under extreme skew improves effective GPU throughput by $1.63\times$ while reducing latency by $38.8\%$.

## 4. Major Empirical Gaps

Despite the strong numbers within the sandbox, there are critical gaps that a practitioner must highlight:

### A. Total Absence of Real-World Datasets and Models
There are **zero** evaluations on real-world datasets (such as GLUE, DomainNet, Decathlon) or actual neural network architectures (such as pre-trained Transformers, CNNs, or diffusion models). For a paper claiming to establish a "robust, deployment-ready framework for test-time dynamic model merging" (Abstract), this is an enormous omission. The proposed SVD projection scaling roadmap (Appendix A.3) is completely qualitative and lacks any empirical implementation.

### B. Lack of Real GPU Latency Benchmarking
The latency profiling in Table 3 is conducted on sequential, CPU-bound Python loops, where MBH shows almost no overhead due to Python loop overhead scaling linearly with $B$. 
- The "projected GPU latency" in Table 4, which shows a $1.57\times$ to $4.33\times$ latency penalty, is completely simulated.
- The authors did not actually implement the parallel GPU execution utilizing Triton Segmented-BGEMM kernels or parallel Radix Sort.
- Consequently, the paper provides absolutely no empirical proof of the systems-level viability or latency of MBH on actual parallel accelerators (e.g., NVIDIA GPUs).

### C. The Weak Expert Calibration Limit
Under SVHN (Task 3), where the expert is noisy and weak ($\sigma_3 = 1.25$, ceiling $26.40\%$), both PFSR ($18.40\%$) and CGHR ($18.80\%$) perform poorly. The gateway router frequently misclassifies SVHN inputs, routing them to cleaner experts (CIFAR-10) and ensuring downstream failure. This shows that the proposed method is highly sensitive to the standalone quality of the experts, and its performance can degrade significantly if the experts are weak.
