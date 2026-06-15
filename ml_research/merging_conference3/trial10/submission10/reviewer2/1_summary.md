# 1. Summary of the Paper

## Main Topic and Objective
The paper addresses the challenge of serving multi-task parameter-efficient fine-tuning (PEFT) experts, such as LoRA or adapters, on dynamic and non-i.i.d. edge-based sequential serving streams. In this environment, a dynamic router must route incoming tokens or samples through the network, dynamically blending expert parameters layer-by-layer. This process is corrupted by two types of noise:
1. **Intra-Sample Depth-Wise Noise (Routing Jitter):** High-frequency layer-to-layer representation-space variations causing routing coefficients to oscillate.
2. **Inter-Sample Temporal Noise:** Sample-to-sample fluctuations on sequential non-i.i.d. streams leading to unstable ensembling trajectories.

The objective is to achieve stable, high-accuracy dynamic expert ensembling that is robust to both sources of noise, has low latency/computational overhead, and is training-free—making it ideal for resource-constrained edge deployment.

## Proposed Approach
The authors introduce **2D-STEM** (2D Spatio-Temporal Exponential Moving Average), a training-free, highly parameter-efficient, and analytically simplex-preserving bilinear filter. Key elements of the approach include:
- **2D Bilinear Recurrence:** Smooths routing trajectories simultaneously across backbone depth and sequence history using a single, unified 2D recurrence equation.
- **Analytical Simplex Preservation:** Proves that under a simple linear inequality constraint on the momentum coefficients ($\beta_{\text{depth}} + \beta_{\text{temp}, t} \le 1$), the ensembling weights are analytically guaranteed to reside on the probability simplex $\Delta^{K-1}$ without requiring any projection or re-normalization operations.
- **Adaptive Temporal Gating (ATG) with Power-Law Sharpening (ATG-PL):** Measures stream homogeneity on-the-fly using representation coordinates at an early, frozen backbone layer. To avoid transition lag under task switches, it dynamically scales down temporal momentum via a power-law exponent ($\gamma \ge 2$, default $\gamma = 3$).
- **Coordinate-Prior Spatial Boundary Condition:** Rather than a raw-weight boundary (which cancels spatial momentum at the first adapted layer) or a uniform boundary (which introduces task-agnostic accuracy drag), it uses an early task-coordinate representation normalized to the simplex as the virtual boundary condition.

## Key Findings and Evidence
- **Noise Reduction in Homogeneous Streams:** On block-stable streams, 2D-STEM filters out high-frequency noise and achieves near-oracle routing stability. It reduces absolute routing jitter by up to $2.75\times$ compared to the stateless SABLE baseline under overlapping manifolds in the Analytical Coordinate Sandbox (ACS), and by over $5.23\times$ in an activation-space trajectory validation on a physical pre-trained Vision Transformer (`vit_tiny`).
- **Transition Lag Suppression:** Under rapidly switching heterogeneous streams, traditional stateful methods suffer from transition lag. 2D-STEM with ATG-PL ($\gamma=3$) instantly collapses temporal memory during switches, outperforming the constant-inertia ChemMerge proxy by up to a massive $51.88\%$ absolute accuracy in ACS.
- **Resolution of the Smoothing-Responsiveness Trade-off:** By decoupling stream similarity estimation (at an early frozen layer) from deep-block representations, 2D-STEM avoids the vulnerability of the adaptive ChemMerge Dynamic ODE baseline, which misinterprets representation noise as task transitions, collapsing its temporal smoothing under stable streams (increasing its jitter to $0.0283$ vs. 2D-STEM's $0.0068$).
- **Calibration Efficiency:** The low-pass filtering of 2D-STEM buffers against noisy centroids, allowing the method to maintain full performance even when the offline centroid calibration set size is reduced to extremely small scales ($N_{\text{cal}} = 5$ samples).

## Explicitly Claimed Contributions
1. **Minimalist Deconstruction:** Stripping away biochemical and learning-theoretic complexity from prior state-of-the-art stateful serving frameworks, showing that their performance stems primarily from local recursive filtering.
2. **2D-STEM Formulation:** Formulating a 2D Spatio-Temporal EMA that is analytically simplex-preserving.
3. **Adaptive Temporal Gating (ATG):** Introducing a zero-overhead, similarity-based gating mechanism with power-law sharpening to suppress transition lag.
4. **Empirical Evaluation:** Setting up the Analytical Coordinate Sandbox (ACS) and validating across multiple random seeds, along with an activation-space validation on a pre-trained Vision Transformer, detailed ablations of boundary conditions, sensitivity sweeps, complexity analyses, and hardware physical deployment roadmaps.
