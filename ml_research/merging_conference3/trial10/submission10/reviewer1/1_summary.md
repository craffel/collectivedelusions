# 1. Summary of the Paper

## Main Topic and Motivation
The paper addresses the challenge of serving multi-task parameter-efficient fine-tuning (PEFT) experts, such as LoRA or adapters, on dynamic, noisy, and non-i.i.d. edge streams. At test-time, dynamic routers typically blend expert parameters or activations layer-by-layer based on task similarity. This process faces two major noise sources:
1. **Intra-sample depth-wise noise (routing jitter):** Layer-to-layer representation fluctuations that cause routing coefficients to oscillate wildly.
2. **Inter-sample temporal noise:** High-frequency variations across sequential queries, causing erratic routing trajectories.

Prior state-of-the-art stateful serving methods (such as *PAC-Kinetics* or *ChemMerge*) address these noises using complex, heavily parameterized dynamical formulations (e.g., PAC-Bayesian state-space models requiring online optimization, or biochemical ODEs representing non-equilibrium chemical reaction kinetics). In contrast, this paper applies Occam's razor, arguing that the primary driver of performance is simply localized recursive filtering.

## Proposed Approach: 2D-STEM
To achieve stable and responsive dynamic merging, the authors propose **2D-STEM** (2D Spatio-Temporal Exponential Moving Average), which is training-free, highly parameter-efficient, and analytically simplex-preserving. The core components are:
1. **2D Bilinear Recurrence:** Smooths routing trajectories simultaneously across both network depth $l$ and sequence time $t$ via a simple bilinear recurrence equation:
   $$\alpha_k^{(l)}(t) = \beta_{\text{depth}} \alpha_k^{(l-1)}(t) + \beta_{\text{temp}, t} \alpha_k^{(l)}(t-1) + (1 - \beta_{\text{depth}} - \beta_{\text{temp}, t}) w_k^{(l)}(t)$$
2. **Analytical Simplex Preservation:** Proves that as long as $\beta_{\text{depth}} \ge 0, \beta_{\text{temp}, t} \ge 0,$ and $\beta_{\text{depth}} + \beta_{\text{temp}, t} \le 1$, the resulting routing weights $\boldsymbol{\alpha}^{(l)}(t)$ are guaranteed to lie on the probability simplex without requiring costly Euclidean projection or re-normalization.
3. **Adaptive Temporal Gating (ATG) with Power-Law Sharpening (ATG-PL):** To prevent transition lag when task distributions change abruptly, ATG measures stream homogeneity at an early frozen layer $L_{\text{frozen}}$. To resolve the upward bias of cosine similarity on non-negative spaces, ATG-PL introduces a sharpening exponent $\gamma \ge 1$ (default $\gamma = 3$):
   $$\beta_{\text{temp}, t} = \beta_{\text{temp}, 0} \cdot (Sim_t)^\gamma$$
4. **Coordinate-Prior Spatial Boundary Condition:** Defines a virtual boundary state using early task-coordinate representation vectors at $L_{\text{frozen}}$ to activate spatial smoothing starting from the first adapted layer, eliminating first-layer spatial momentum cancellation.

## Key Findings and Claims (with Evidence)
1. **Perfect Noise Filtering:** On stable homogeneous streams under overlapping task manifolds, 2D-STEM reduces absolute routing jitter by $2.75\times$ compared to stateless SABLE (Jitter of $0.0068$ vs. $0.0187$). It achieves $95.02\%$ (Orthogonal) and $95.00\%$ (Overlapping) homogeneous ensembling accuracy, recovering near-oracle routing performance ($95.05\%$) without any statistical difference.
2. **Effective Lag Suppression:** Under rapidly switching heterogeneous streams, 2D-STEM outperforms the constant-inertia ChemMerge proxy by up to a massive $51.88\%$ absolute accuracy on Orthogonal manifolds and $47.06\%$ on Overlapping manifolds. It accomplishes this by dropping sequence-level temporal momentum close to zero during transitions.
3. **ViT Trajectory Validation:** On physical pre-trained ViT representations under severe manifold overlaps, 2D-STEM reduces absolute routing jitter by over $5.23\times$ compared to SABLE (Jitter of $0.0675$ vs. $0.3530$) while matching SABLE's alignment accuracy.
4. **Negligible Overhead:** The method requires zero training parameters, zero online optimization, and only a microscopic active runtime memory of $K \times L$ floats for temporal history and $K$ floats for coordinates (240 bytes in a standard 4-expert, 14-layer setup). CPU profiling shows a $49.5\%$ reduction in per-step latency compared to ChemMerge (Dynamic ODE).
5. **Data-Scarce Robustness:** Robustness sweeps show that 2D-STEM achieves $94.88\%$ accuracy and $0.0087$ routing jitter even when the centroid calibration set is reduced to an extremely sparse $N_{\text{cal}} = 5$ samples per task.

## Summary of Explicitly Claimed Contributions
* **Minimalist Deconstruction:** Demonstrates that the success of prior stateful merging frameworks stems from their bilinear recursive filtering properties, and can be modeled with zero training overhead or ODE solvers.
* **2D-STEM Formulation:** Formulates a training-free 2D spatio-temporal EMA that is analytically simplex-preserving under a simple linear inequality constraint.
* **Adaptive Temporal Gating (ATG):** Introduces a zero-overhead gating mechanism that scales sequence momentum dynamically to eliminate transition lag.
* **Rigorous Sandbox and Physical Validation:** Evaluates the method in both a parameterized Analytical Coordinate Sandbox (ACS) and on pre-trained ViT representations, demonstrating major jitter reduction and transition accuracy.
