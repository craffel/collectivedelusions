# 4. Experiment Check

## Experimental Setup & Datasets
The authors evaluate their proposed FIOSR framework across three distinct experimental settings, representing a gradient of realism:
1. **The Analytical Coordinate Sandbox (Primary Evaluation):** A highly synthetic 192-dimensional representation space designed to simulate 4 specialized task domains (MNIST, FashionMNIST, CIFAR-10, SVHN equivalents) with simulated coordinate subspace blocks ($d = 48$) and axis-aligned Gaussian noise.
2. **Real-World LoRA Activation Space Simulation:** A 64-dimensional representation space simulating $K=3$ expert tasks, where the coordinate noise and feature structures are estimated from real-world activation variances.
3. **End-to-End Physical ResNet-18 Validation:** A frozen physical ResNet-18 feature extractor ($512$-dimensional features) combined with three specialized linear classification heads trained on real image datasets: **MNIST**, **FashionMNIST**, and **SVHN**.

## Baselines
The paper compares FIOSR against a comprehensive suite of static and dynamic baselines:
- **Static Uniform Merging:** A standard baseline that merges weights uniformly with a fixed weight ($1/K$).
- **Linear Router (Unregularized) & QWS-Merge (Quantum Wave Superposition):** High-complexity parametric routers optimized at test-time over calibration splits.
- **L3-Softmax (Well-Regularized):** The state-of-the-art regularized parametric router.
- **PFSR + MBH (Flat Cosine):** The direct parameter-free predecessor that assumes a flat Euclidean space.
This choice of baselines is excellent and highly rigorous, allowing the authors to isolate the exact benefits of both parameter-free routing (over parametric optimization) and Fisher-weighted warping (over unweighted flat cosine similarity).

## Do the Results Support the Claims? (Critical Practitioner Analysis)

While the empirical evaluations are highly thorough and statistically sound (using 10 independent random seeds), a critical analysis reveals a major gap between the synthetic claims and actual physical utility:

### 1. The Synthetic Sandbox Over-Optimism
The primary quantitative claims in the paper—such as outperforming baselines by up to **40.7%** under heterogeneous streams and outperforming PFSR by **8.56%**—are evaluated entirely within the highly synthetic **Analytical Coordinate Sandbox**. 
- In this sandbox, coordinates have explicit, independent, axis-aligned noise. This is an environment that is *custom-designed* to make a diagonal Fisher (inverse-variance) coordinate filter succeed. 
- Because the synthetic noise is perfectly aligned with the axes and perfectly independent, diagonal covariance assumptions hold flawlessly, yielding highly optimistic routing accuracies ($\approx 100\%$ on MNIST/FashionMNIST equivalents) and joint ensembling accuracy ($76.86\%$).

### 2. The Physical ResNet-18 Performance Gap (The Real-World Reality Check)
The end-to-end physical validation on ResNet-18 (Section 4.8) is the most critical and honest part of the paper. However, the results in this realistic setting are underwhelming and reveal a massive performance gap:
- **Modest Routing Gain:** Under a physical ResNet-18 backbone, FIOSR's routing accuracy is **59.00%**, which is only a **+2.67%** absolute improvement over the flat Cosine baseline (**56.33%**).
- **Insignificant Joint Accuracy Gain:** The joint ensembling accuracy of FIOSR is **52.00%**, representing a minor **+1.33%** absolute improvement over flat Cosine (**50.67%**).
- **Substantial Oracle Gap:** Both methods perform extremely poorly compared to the Direct Expert Routing Oracle (**69.67%**), leaving a massive performance gap of **17.67%**.
- **Practical Interpretation:** This massive performance drop indicates that on actual physical networks (where activations are highly non-Gaussian, sparse due to ReLUs, and deeply correlated), the diagonal Fisher assumption provides very limited practical utility. The elegant Riemannian metric warping collapses to near-flat performance, casting doubt on whether the substantial mathematical overhead of FIM estimation and smoothing is practically justified in real-world deployments.

### 3. Thoroughness of Ablations
The authors must be highly commended for their exceptional empirical thoroughness in the appendix. They provide crucial data sweeps that reveal key limitations:
- **$N_c$ Sensitivity (Appendix B.3):** Demonstrates that the method overfits and collapses below the flat baseline if $N_c \le 4$.
- **Rotated Noise (Section 4.6 / Appendix B.1):** Reveals that diagonal Fisher collapses below the flat Cosine baseline under non-axis-aligned noise, and that full covariance estimation (which is computationally expensive) is necessary to restore gains.
- **Top-$M$ Gating (Appendix B.1):** Outlines that setting $M=1$ (hard Top-1 routing) eliminates sequential micro-batching overhead but sacrifices true weight-space ensembling.
These ablations are highly informative and provide a transparent, rigorous look at the boundaries of the proposed framework.
