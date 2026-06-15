# 1. Summary of the Paper

This paper addresses the critical challenge of serving multi-task parameter-efficient experts (such as LoRA) on dynamic edge streams. The authors identify two distinct, orthogonal sources of representation and temporal noise:
1. **Intra-Sample Depth-Wise Noise (Routing Jitter):** Wild layer-to-layer ensembling coefficient oscillations within a single query's forward pass caused by representational variations across network depth.
2. **Inter-Sample Temporal Noise:** Unstable, erratic ensembling trajectories over consecutive, non-i.i.d. samples in sequential serving streams.

To solve this dual-noise problem, prior state-of-the-art frameworks rely on highly complex, parameterized formulations, such as non-equilibrium chemical reaction kinetics (ChemMerge) or learned first-order state-space models optimized via PAC-Bayesian bounds (PAC-Kinetics). Guided by Occam's razor, the authors deconstruct these frameworks and propose **2D-STEM** (2D Spatio-Temporal Exponential Moving Average), a training-free, parameter-efficient, and analytically simplex-preserving 2D recursive digital filter.

### Core Proposed Methods
1. **2D-STEM Recurrence:** A 2D bilinear Exponential Moving Average update equation that propagates ensembling coefficients across both network depth ($l$) and sequence time ($t$):
   $$\alpha_k^{(l)}(t) = \beta_{\text{depth}} \alpha_k^{(l-1)}(t) + \beta_{\text{temp}, t} \alpha_k^{(l)}(t-1) + \left(1 - \beta_{\text{depth}} - \beta_{\text{temp}, t}\right) w_k^{(l)}(t)$$
2. **Analytical Simplex Preservation:** A mathematical proof showing that under the simple linear inequality constraint $\beta_{\text{depth}} + \beta_{\text{temp}, t} \le 1$, the ensembling coefficients are guaranteed to lie on the probability simplex $\Delta^{K-1}$ without any explicit projection or re-normalization operations.
3. **Coordinate-Prior Spatial Boundary Condition:** A virtual boundary condition at the frozen layer computed via early task coordinates, preventing first-layer spatial momentum cancellation without inducing "accuracy drag."
4. **Adaptive Temporal Gating (ATG-PL) with Power-Law Sharpening:** A zero-overhead similarity gating mechanism that measures stream homogeneity at early frozen layers and scales down temporal momentum during task switches. By applying a power-law exponent ($\gamma \ge 2$, default $\gamma = 3$), ATG-PL squashes transition-similarity bias arising from non-negative coordinate overlap, eliminating transition lag.

### Key Claims and Empirical Findings
1. **Perfect Noise Filtering:** In homogeneous block-stable streams, 2D-STEM filters out high-frequency noise, reducing absolute routing jitter by up to $2.75\times$ compared to SABLE while recovering the oracle ensembling accuracy in simulated environments.
2. **Inertia Suppression:** On heterogeneous streams with rapid task switches, ATG-PL allows 2D-STEM to outperform constant-inertia stateful baselines by up to a massive $51.88\%$ absolute accuracy in simulated environments.
3. **Decoupled Low-Pass Filtering:** The authors identify that the highly adaptive ChemMerge (Dynamic ODE) baseline misinterprets representation noise as a task switch under stable streams, collapsing its temporal smoothing. 2D-STEM elegantly decouples transition detection (early frozen layers) from deep representational noise.
4. **Data-Scarce Calibration:** 2D-STEM is highly robust to calibration data scarcity, retaining full performance with as few as $N_{\text{cal}} = 5$ samples.
5. **Pre-Trained Representation Generalizability:** An activation-space serving trajectory validation on a pre-trained Vision Transformer (`vit_tiny`) shows that 2D-STEM successfully suppresses absolute routing jitter by over $5.2\times$ under highly overlapping real-world representation manifolds.
