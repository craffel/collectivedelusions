# Systematic Mock Review: 1. Paper Summary

## 1.1 Overview of the Paper
This paper proposes **Gaussian Process Dynamic Routing (GP-DR)**, a training-free, non-parametric Bayesian framework for dynamic model merging in modular deep learning. Standard dynamic merging systems rely on parametric gating networks trained via gradient descent on small validation splits (calibration sets), which makes them highly prone to overfitting on representational noise. GP-DR solves this problem by modeling the parameter routing coefficients analytically as a closed-form posterior mean under a Gaussian Process (GP) prior with a Radial Basis Function (RBF) or other positive-definite kernels.

Additionally, the paper introduces **Micro-Batch Homogenization (MBH)**, a stream-level batch partitioning mechanism designed to prevent "vectorization collapse" (or representation averaging) under mixed-task inference batches, restoring localized dynamic routing capabilities.

The paper is motivated by two key limitations of existing dynamic model merging methods:
1. **The Overfitting-Optimizer Paradox:** Gating networks trained with standard backpropagation overfit heavily to spurious representations when calibrated on small validation splits (e.g., $N=64$ samples).
2. **Heterogeneity Stream Collapse:** Standard deep learning models process batches collectively, and standard dynamic routers that average features or apply uniform scaling across heterogeneous streaming batches experience representation averaging, which collapses performance back to static uniform merging.

## 1.2 Core Proposed Methodology
The proposed framework consists of:
* **Representational Projection and Normalization:** Extracts penultimate features $z(x)_b$, projects them to a low-dimensional space $d = K$ via a frozen matrix $P$ (such as class-specific prototypes or projection heads), and normalizes them onto the unit sphere to form the representation coordinate vector:
  $$\psi(x)_b = \frac{z(x)_b P}{\|z(x)_b P\|_2 + \epsilon}$$
  Highly out-of-distribution (OOD) samples are mapped to the origin $\mathbf{0}$ when they share zero similarity with any prototype.
* **Gaussian Process Prior:** Places independent GP priors on the routing coordinate of each task with a static, uniform prior mean function of $1/K$ and an RBF kernel:
  $$k(\psi, \psi') = \sigma_f^2 \exp\left(-\frac{\|\psi - \psi'\|_2^2}{2 \ell^2}\right)$$
  which simplifies directly to an exponential function of their cosine similarity under unit-sphere normalization.
* **Closed-Form Posterior Mean ($\mu$):** Resolves the dynamic blending coefficients in a single forward matrix multiplication:
  $$\mu(\psi_*) = \mathbf{m}(\psi_*) + \mathbf{k}_* \left( \mathbf{K} + \sigma_n^2 \mathbf{I} \right)^{-1} \left( \mathbf{Y} - \mathbf{m}(\mathbf{X}) \right)$$
  where $\mathbf{W}_{\text{GP}} = \left( \mathbf{K} + \sigma_n^2 \mathbf{I} \right)^{-1} \left( \mathbf{Y} - \mathbf{m}(\mathbf{X}) \right)$ is pre-computed offline.
* **Closed-Form Posterior Variance ($\sigma^2$):** Bounded within $[0, \sigma_f^2]$, providing a mathematically exact metric of epistemic uncertainty to trigger a safe fallback to a uniform prior blend when $\sigma^2(\psi_*) > \theta_{\text{OOD}}$:
  $$\sigma^2(\psi_*) = k(\psi_*, \psi_*) - \mathbf{k}_* \left( \mathbf{K} + \sigma_n^2 \mathbf{I} \right)^{-1} \mathbf{k}_*^T$$
* **Micro-Batch Homogenization (MBH):** Sorts incoming streaming batches into homogeneous micro-batches based on the argmax of their predicted routing preferences, forwarding each micro-batch separately to isolate representational spaces and bypass vectorization collapse.

## 1.3 Key Empirical Findings
* **Synthetic Evaluation (Isolating Coordinate Sandbox):**
  * On a 14-layer, 192-dimensional synthetic sandbox with 4 tasks (MNIST, F-MNIST, CIFAR-10, SVHN), GP-DR achieves **$72.40\%$** Joint Mean accuracy with zero optimization loops, representing a $+42.40\%$ absolute improvement over unregularized parametric linear baselines.
  * Exhibits extreme stability, whereas standard parametric Global Linear baselines collapse from $82.81\%$ train accuracy to $30.90\%$ test accuracy due to the Overfitting-Optimizer Paradox.
  * Soft parameter-space blending is shown to outperform a hard model selection baseline ($72.40\%$ vs. $71.50\%$).
* **Stream Recovery (MBH):**
  * Under a heterogeneous stream ($B=256$), dynamic routers collapse to uniform merging levels ($\approx 27.4\%$).
  * MBH restores GP-DR to **$70.20\%$** accuracy (a recovery margin of $+42.80\%$).
* **Hardware Profiling:**
  * CPU benchmarks show that MBH introduces a $1.75\times$ latency increase and a $44\%$ throughput reduction.
  * GPU benchmarks (NVIDIA A100) reveal a $2.26\times - 3.20\times$ latency increase and a $55\% - 68\%$ throughput drop due to warp underutilization at smaller batch sizes, stabilizing at larger batch sizes ($B=512$).
* **Real-World Validation (GLUE Benchmark with BERT-Tiny):**
  * On SST-2, CoLA, and MRPC tasks using a pre-trained `bert-tiny` backbone, GP-DR achieves a competitive Joint Mean accuracy of **$45.78\%$** (compared to $16.22\%$ for Static Uniform and $34.22\%$ for a Parametric Softmax Router).
  * MBH restores streaming performance from $14.06\%$ back to $45.76\%$ ($+31.70\%$ recovery margin).
* **OOD Rejection:**
  * Achieves $100.00\%$ AUROC and $0.00\%$ False Rejection Rate (FRR) on both synthetic and GLUE OOD evaluation when the OOD task is strictly orthogonal (projected to the origin).
  * Explicitly exposes and analyzes the severe limitation of GP variance under unit-sphere OOD noise (where it collapses and is blind to random noise) and representational overlap (where simpler distance-based heuristics like 5-NN outperform GPR posterior variance).
