# Experiment Check

**Evaluation Quality: Good**

The empirical evaluation of FlatMerge has been significantly elevated and is now highly rigorous. By supplementing their detailed Vision Transformer simulation with two physical deep learning validation benchmarks and direct hardware profiling, the authors have built a compelling empirical case for FlatMerge.

### 1. Robust and Diverse Baselines
The authors evaluate FlatMerge against a comprehensive suite of modern model-merging baselines:
- **Task Arithmetic:** The standard static merging baseline.
- **AdaMerging (Unconstrained):** Standard unconstrained first-order TTA.
- **AdaMerging + TV / AdaMerging + L2:** Standard gradient-based TTA with conventional smoothness penalties.
- **RegCalMerge:** The state-of-the-art regularized TTA framework featuring Class-Capacity Normalization and Elastic Spatial Regularization.
- **PolyMerge ($d=0, 1, 2, 3$):** Subspace-constrained TTA without flatness-aware optimization.

This selection allows the authors to dissect the specific contributions of both components of FlatMerge (the polynomial subspace constraint and the flatness optimization).

### 2. High Statistical Rigor in Simulation
The simulation results (Table 1 and Table 2) are exceptionally rigorous:
- Evaluated over **15 independent random seeds** (seeds 42 to 56 inclusive).
- Conducted on two distinct emulated loss surfaces calibrated from CLIP ViT-B/32 literature (Convex Sandbox and Coupled Non-Convex Stress-Test).
- Conducted across five different test-time noise scales ($\gamma \in \{1.0, 1.5, 2.0, 2.5, 3.0\}$).
- FlatMerge demonstrates clear state-of-the-art joint accuracy under noise while **reducing seed standard deviation by over 60%** (e.g., $0.63\%$ vs. $1.62\%$ for PolyMerge under moderate noise), proving outstanding optimization stability.

### 3. Empirical Validation on Physical Neural Networks
To bridge the simulation-to-real gap, the paper includes two physical neural network experiments:
- **3-Layer MLP (MNIST + FashionMNIST):** Evaluated under progressive Gaussian pixel noise ($\gamma \in \{0.0, 1.0, 2.0, 3.0\}$). Shows that first-order AdaMerging collapses under test-time noise due to transductive noise-fitting, while ZO-FlatMerge successfully prevents collapse (outperforming AdaMerging by $+8.72\%$ absolute under extreme noise).
- **5-Layer CNN (MNIST + FashionMNIST + KMNIST):** A more complex physical architecture (250K parameters) fine-tuned under a pre-training-and-fine-tuning paradigm. Under clean conditions ($\gamma=0.0$), first-order AdaMerging and PolyMerge collapse catastrophically to near-random accuracy (~16.67% and 14.27% respectively). In contrast, ZO-FlatMerge successfully prevents this representation collapse, achieving **48.57%** joint accuracy and outperforming first-order AdaMerging by over **30% absolute**. Under moderate-to-extreme noise, ZO-FlatMerge consistently and significantly outperforms all other adaptive baselines.

These physical experiments provide crucial empirical evidence that the "Overfitting-Optimizer Paradox" is a severe, real-world issue for physical weight merging, and that ZO-FlatMerge effectively resolves it.

### 4. Deep-Dive Sensitivity Analysis and Ablation Studies
The empirical analysis goes beyond simple benchmarking to provide deep qualitative and quantitative insights:
- **Figure 4 (Regularization Sweep):** Shows that conventional Total Variation and $L_2$ weight decay penalties are highly sensitive to hyperparameter tuning and fail to match FlatMerge's performance ceiling.
- **Figure 5 (Perturbation Radius Sweep):** Sweeps FlatMerge's perturbation radius $\rho \in [0.001, 0.2]$, demonstrating a broad, stable convex plateau around the default value of $\rho = 0.05$. This proves that FlatMerge does not require delicate on-device hyperparameter tuning.
- **Figure 6 (Qualitative Profile Analysis):** Visualizes the learned layer-wise blending coefficients, providing clear qualitative evidence that unconstrained AdaMerging produces highly jagged, overfitted profiles, PolyMerge suffers from low-frequency drift, and FlatMerge tracks the optimal ground-truth profile extremely closely.

### 5. Detailed Hardware Profiling
The authors provide direct hardware measurements (Section 3.5) comparing static memory overhead, peak activation caching, and step latency on physical hardware. This adds great pragmatic value, establishing the memory benefit (exactly 0.00 MB activation cache) and analyzing the latency-amortization of asynchronous background updates (reducing amortized latency to a mere 0.73%).

**Area for Improvement:**
The only remaining limitation is that the physical weight experiments are conducted on relatively small networks (MLP and 5-layer CNN on MNIST/FashionMNIST/KMNIST). Running a physical experiment using a full ViT backbone (e.g. CLIP ViT-B/32) on actual image datasets (such as ImageNet-C or DomainNet) under physical noise would make the empirical evaluation flawless. However, given the resource constraints of edge accelerators, the current combination of calibrated ViT simulations and physical toy model validations is extremely thorough and acceptable.
