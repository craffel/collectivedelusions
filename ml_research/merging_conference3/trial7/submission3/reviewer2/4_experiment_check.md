# Part 4: Experiment Check

## 1. Evaluation of Experimental Setup, Datasets, and Baselines
The paper conducts a highly rigorous, multi-faceted empirical evaluation across three distinct environments:
1. **Isolating Coordinate Sandbox:** A synthetic 14-layer, 192-dimensional framework designed to simulate vision backbone representations across $K=4$ tasks (MNIST, FashionMNIST, CIFAR-10, SVHN).
2. **Real-World GLUE Benchmark:** A multi-task language modeling setup merging three classification tasks (SST-2, CoLA, MRPC) using a pre-trained BERT-Tiny backbone.
3. **Generative LLM Pilot:** A focused evaluation of English-to-French translation and sentiment analysis on a pre-trained GPT-2 ($124$M) backbone.

The baselines are comprehensive and include:
* **Static Merging:** Static Uniform Merging (Weight Averaging).
* **Parametric Routers:** Global Linear (unregularized and regularized), L3-Linear, L3-Softmax, and QWS SOTA.
* **Non-Parametric SOTA:** Parameter-Free Subspace Routing (PFSR).
* **OOD Baselines:** Min Euclidean, 5-NN Euclidean, Min Cosine distance, Raw Mahalanobis, and Raw Energy-Based OOD.

## 2. Do the Results Support the Claims?
Yes, the empirical results broadly support the core claims, but with major, self-disclosed caveats:

* **Overfitting-Optimizer Paradox:** Supported. In low-data calibration ($N=64$), unregularized parametric routers achieve near-perfect training scores but collapse to poor test-time performance (~30% Joint Mean), while GP-DR remains stable at 72.40%.
* **Heterogeneity Stream Collapse and MBH Recovery:** Supported. In mixed-task streaming ($B=256$), representation-averaging collapse drags all dynamic routers down to uniform merging levels (~27%). Implementing MBH results in massive recovery (GP-DR recovers to 70.20%, a $+42.80\%$ recovery margin).
* **Bayesian OOD Rejection:** *Partially Supported/Severely Qualified.* While GP-DR achieves 100.00% AUROC on OOD samples mapped exactly to the origin, the authors' own extensive empirical sweeps (Table 5) prove that **GP-DR's posterior variance is blind to unit-sphere noise and is vastly outperformed by simpler local distance heuristics under representational coupling and overlap.**

## 3. Minimalist Critique of Experimental Findings
Analyzing the experimental section from a perspective that values simplicity and efficiency, several findings undermine the necessity of the proposed GP-DR framework:

1. **The In-Distribution Performance Gap:** PFSR SOTA (the simpler, parameter-free baseline without GPR) consistently outperforms GP-DR in-distribution:
   * Sandbox: PFSR SOTA achieves **77.60%** vs. GP-DR's **72.40%** (a $-5.20\%$ absolute penalty for GP-DR).
   * GLUE: PFSR SOTA achieves **50.22%** vs. GP-DR's **45.78%** (a $-4.44\%$ absolute penalty).
   The simpler baseline is more accurate because it avoids GPR's continuous Bayesian shrinkage which causes task-head interference.
2. **OOD Failure of GPR Posterior Variance:** In the representational overlap sweep (Table 5), **simpler coordinate-space distance heuristics (like 5-NN Euclidean distance) vastly outperform GPR posterior variance (both RBF and Cosine kernels)**:
   * At high representational overlap ($\beta = 0.75$), 5-NN Euclidean distance achieves an exceptional **99.77% AUROC and 30.40% FRR**. In contrast, GPR's RBF posterior variance collapses to **82.10% AUROC and 90.40% FRR** (virtually unusable in production).
   * Under pure unit-sphere OOD noise ($\beta = 0.00$), GPR posterior variance experiences severe variance collapse, resulting in an **80.80% False Rejection Rate**, whereas 5-NN distance achieves **99.98% AUROC with a tiny 4.40% FRR**.
   Since a simple local distance heuristic (which requires zero matrix inversions, zero Cholesky decompositions, and zero hyperparameter tuning) is vastly superior for OOD detection, the primary justification for adding the GPR layer over PFSR is empirically invalidated.
3. **The Sandbox Joint Evaluation Artifact:** The authors admit that under a *task-conditioned* evaluation protocol, ALL models—including Static Uniform Merging and GP-DR—recover their stand-alone expert ceilings (**83.00% Joint Mean**). This proves that the sandbox's joint evaluation is an artificial stress-test of "head muting" (task classification) rather than a physical evaluation of weight-space parameter-merging degradation.
4. **MBH Hardware Throughput Degradation:** On a high-performance NVIDIA A100 GPU, integrating MBH results in a **$55\% - 68\%$ drop in throughput** and up to a **$3.20\times$ increase in latency** (Table 6). Grouping and sequentially forwarding micro-batches introduces a massive execution penalty due to GPU warp underutilization, making it highly impractical for latency-sensitive real-time applications.
