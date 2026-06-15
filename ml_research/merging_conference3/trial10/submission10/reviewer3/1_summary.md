# Paper Summary

## Main Topic and Approach
The paper addresses the challenge of serving multi-task parameter-efficient experts (such as LoRA) on dynamic sequential edge streams. These streams suffer from two types of noise:
1. **Intra-Sample Depth-Wise Noise (Routing Jitter):** Representation-space variations across layers as a query propagates through the network, causing ensembling coefficients to oscillate wildly.
2. **Inter-Sample Temporal Noise:** High-frequency sequential noise over consecutive queries in non-i.i.d. streams.

To address both noise sources, the paper proposes **2D-STEM** (2D Spatio-Temporal Exponential Moving Average), a training-free, bilinear recursive filter that smooths routing trajectories across both backbone depth and sequence history simultaneously. Crucially, the method is designed to be highly parameter-efficient and analytically simplex-preserving, avoiding the need for computationally heavy online ODE solvers (as in ChemMerge), PAC-Bayesian optimization loops (as in PAC-Kinetics), or explicit Euclidean projection steps. To suppress the inertial transition lag (drag) inherent in temporal smoothing under task switches, the authors introduce **Adaptive Temporal Gating (ATG)**, which scales down temporal momentum when a task switch is detected. This is further refined using **Power-Law Gating (ATG-PL)** to counteract representational overlap bias.

## Key Findings
- **Routing Jitter Reduction:** In stable homogeneous streams, 2D-STEM reduces absolute routing jitter by up to $2.75\times$ compared to the stateless SABLE baseline on simulated overlapping task manifolds, and by over $5.23\times$ in a pre-trained Vision Transformer CLS-token representation simulation.
- **Simplex Preservation:** By enforcing a simple linear inequality constraint ($\beta_{\text{depth}} + \beta_{\text{temp}, t} \le 1$), 2D-STEM analytically guarantees that routing weights reside on the probability simplex at all layers and steps, requiring zero projection or re-normalization operations.
- **Transition Lag Suppression:** Under rapidly switching heterogeneous streams, ATG-PL ($\gamma=3$) collapses temporal memory instantly, outperforming the constant-inertia ChemMerge proxy by up to a massive $51.88\%$ absolute accuracy in the Analytical Coordinate Sandbox (ACS).
- **Physical ViT Alignment Discrepancy:** On the pre-trained ViT representation-space simulation, while 2D-STEM achieves substantial routing jitter reduction (reducing jitter from $0.3530$ to $0.0675$), it is actually outperformed in alignment accuracy and jitter by the temporal-only baseline **PAC-Kinetics** ($70.57\%$ accuracy, $0.0063$ jitter) and the **ChemMerge Proxy** ($65.83\%$ accuracy, $0.0419$ jitter), compared to 2D-STEM's $63.70\%$ accuracy and $0.0675$ jitter.

## Explicitly Claimed Contributions and Evidence
1. **Minimalist Deconstruction:** Deconstructing prior stateful merging frameworks to show that their performance stems from bilinear recursive filtering, allowing the removal of biochemical ODE and learning-theoretic complexity. 
   - *Evidence:* Successful simulation in ACS showing 2D-STEM's high performance compared to proxies.
2. **2D-STEM Formulation:** A training-free 2D Spatio-Temporal EMA that is analytically simplex-preserving.
   - *Evidence:* Formal mathematical induction proof of Theorem 3.1.
3. **Adaptive Temporal Gating (ATG):** A zero-overhead, similarity-based gating mechanism with Power-Law sharpening to suppress inertial lag.
   - *Evidence:* Quantitative simulation results showing high accuracy ($94.66\%$ and $92.82\%$) on heterogeneous streams compared to constant-inertia baselines ($42.78\%$ and $45.76\%$).
4. **Analytical Coordinate Sandbox & Trajectory Validation:** Evaluation under a 14-layer representation-space sandbox (ACS) and an activation-space serving validation on a pre-trained Vision Transformer CLS token.
   - *Evidence:* Results reported in Tables 1, 2, and 5, along with trajectory plots in Figure 1.
