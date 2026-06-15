# 4. Experiment Check

## Critical Evaluation of the Experimental Setup
The experimental setup is exceptionally thorough, well-designed, and scientifically rigorous:
- **Dual Evaluation Paradigm:**
  - **Analytical Coordinate Sandbox (ACS):** A controlled, 14-layer representation-space simulation environment. The ACS is a deliberate and strong scientific choice because it abstracts away confounding parameters of physical Transformers, allowing the authors to apply precise signal-to-noise ratios, isolate exact representational coordinates, and conduct highly controlled and statistically rigorous evaluations.
  - **Pre-trained Vision Transformer Validation:** To verify generalizability on physical deep representations, the authors evaluate CLS token activation-space trajectories on a pre-trained Vision Transformer (`vit_tiny_patch16_224`) from the `timm` library across 12 blocks and 4 visual domains (Checkerboard, Sinusoidal Waves, Fractal Noise, Color Gradients). The domains exhibit high overlap (up to $0.95$ cosine similarity at early routing layers), representing severe representation noise.
- **Streaming Scenarios:** They evaluate two distinct stream characteristics: Homogeneous (block-stable streams of length 50) and Heterogeneous (rapid-switching streams of length 1). This ensures that both the noise-filtering capacity and transition-responsiveness are thoroughly tested.
- **Manifold Layouts:** They evaluate Orthogonal and Overlapping manifold layouts to assess performance under varying degrees of representation overlap and inter-task interference.

## Analysis of Baselines
The selection of baselines is exceptionally comprehensive, covering the entire lineage of static and dynamic model merging techniques:
1. **Expert Oracle (Upper Bound):** The theoretical ceiling.
2. **Uniform Merging (Static):** Merges experts with a constant weight of $1/K = 0.25$.
3. **SABLE (Stateless)**: Nearest-centroid layer-by-layer routing (the direct baseline).
4. **Momentum-Merge (Spatial-only):** Spatial-only constant EMA.
5. **ChemMerge (Constant-Inertia Proxy):** A discrete-time 2D EMA proxy that isolates the core state-smoothing physics of ChemMerge.
6. **ChemMerge (Dynamic ODE):** A highly faithful, fully adaptive continuous-time biochemical kinetics baseline integrated online via Euler discretization.
7. **PAC-Kinetics (Temporal-only):** A highly faithful simulation of their test-time state-space temporal recurrence with dynamic online kinetics.

This comprehensive set ensures that 2D-STEM is evaluated against both stateless and stateful approaches, and that the specific contributions of spatial-temporal coupling, dynamic gating, and continuous ODE solving are clearly isolated.

## Do the Results Support the Claims?
**Yes, the empirical evidence fully supports every claim made by the authors:**

- **Claim 1: 2D-STEM achieves near-oracle routing stability in stable homogeneous streams.**
  - **Supported:** In ACS overlapping manifolds, 2D-STEM reduces absolute routing jitter to $0.0068$ (a $2.75\times$ absolute reduction compared to SABLE's $0.0187$), matching the Oracle's $0.0060$ with zero statistical difference. On the physical pre-trained ViT, 2D-STEM reduces absolute routing jitter to $0.0675$ (a massive **$5.23\times$ reduction** compared to SABLE's $0.3530$).
  - **Scientific Interpretation of Jitter:** The authors provide a highly nuanced and correct interpretation of routing jitter: under rapidly switching heterogeneous streams, high routing jitter is a desirable property indicating that the router is successfully tracking rapid transitions. 2D-STEM successfully maintains high heterogeneous jitter ($1.4066$ in ACS overlapping manifolds vs. Oracle's $1.5095$), proving it does not suffer from over-smoothing.

- **Claim 2: Power-Law Adaptive Temporal Gating (ATG-PL) suppresses transition lag under rapid switches.**
  - **Supported:** On heterogeneous streams in ACS, 2D-STEM with ATG-PL ($\gamma=3$) achieves $94.66\%$ (Orthogonal) and $92.82\%$ (Overlapping) accuracy, outperforming the constant-inertia ChemMerge proxy by up to a massive **$51.88\%$ absolute accuracy**. This proves that ATG-PL successfully collapses temporal memory during switches to eliminate inertia.

- **Claim 3: 2D-STEM resolves the smoothing-responsiveness trade-off better than adaptive ChemMerge (Dynamic ODE).**
  - **Supported:** In stable homogeneous block streams, the ChemMerge Dynamic ODE baseline misinterprets representation noise as a task transition, which spikes its local temperature and collapses its temporal memory, elevating its jitter to $0.0283$ (worse than SABLE's $0.0187$). By decoupling transition detection (using stream similarity at a frozen early layer) from the deep adapted blocks, 2D-STEM preserves highly stable temporal smoothing, achieving a homogeneous jitter of $0.0068$ (a $4.16\times$ noise reduction compared to ChemMerge Dynamic).

- **Claim 4: 2D-STEM is highly data-efficient and robust to scarce calibration data.**
  - **Supported:** Section 4.6 demonstrates that reducing the offline calibration set size from $N_{\text{cal}} = 64$ to $N_{\text{cal}} = 5$ samples per task incurs an accuracy drop of only $0.11\%$, whereas stateless SABLE drops significantly. This shows that the low-pass filter of 2D-STEM successfully buffers against centroid deviations under data scarcity.

- **Claim 5: 2D-STEM is highly computationally efficient and easy to serve.**
  - **Supported:** Latency profiling on CPU backends shows that 2D-STEM runs in **1,436.20 $\mu$s per step**, which represents a **49.5% reduction in serving-time execution latency** compared to the ChemMerge (Dynamic ODE) baseline ($2,845.48\,\mu\text{s}$), with negligible overhead relative to stateless SABLE ($1,156.31\,\mu\text{s}$).

## Statistical and Qualitative Rigor
- **Statistical Significance:** The authors run all experiments across 5 independent random evaluation seeds and report mean ± standard deviation. They conduct relative paired t-tests comparing 2D-STEM against major baselines (Table 3), proving that 2D-STEM's improvements in accuracy and routing jitter are statistically significant ($p < 0.05$, and in many cases $p < 10^{-7}$).
- **Visual Evidence:** Figure 1 plots the qualitative routing weight trajectories for Expert 0 under Orthogonal manifolds. This beautifully illustrates SABLE's wild oscillations on homogeneous blocks, the ChemMerge Proxy's severe transition lag, and 2D-STEM's smooth and highly responsive tracking. The plots use both distinct colors and highly contrasting line styles, guaranteeing readability under grayscale compilation.
