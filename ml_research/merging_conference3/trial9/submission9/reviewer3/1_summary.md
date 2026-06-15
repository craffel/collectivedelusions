# 1. Summary of the Paper

## Main Topic
The paper addresses the challenge of **test-time dynamic model ensembling** (or model serving) under sequential, non-i.i.d. query streams. Specifically, when serving multiple parameter-efficient fine-tuning (PEFT) experts (e.g., LoRA adapters) on a shared pretrained backbone, the system must dynamically route and blend these experts in real-time. 

Existing stateless routers suffer from the **routing jitter paradox**—high-frequency switching between experts due to query-level noise, which destabilizes representation flows across deep layers and degrades performance (termed **cascading representation collapse**). Conversely, stateful heuristic routers (like ChemMerge) smooth the routing trajectories but lack mathematical stability/generalization guarantees and suffer from rigid memory that causes severe accuracy collapses (due to "inertial drag" or routing lag) when query streams shift rapidly.

## Proposed Approach
The authors propose **PAC-Kinetics**, a unified learning-theoretic and control-theoretic framework that models the stateful router as a continuous-time non-equilibrium chemical kinetics system.
- **Stateful Recurrence:** By integrating a continuous-time first-order ODE, the authors derive a discrete-time contractive state recurrence: 
  $$s_t = \mathbf{A} s_{t-1} + W \mathbf{e}_t$$
  where $s_t$ is the concentration state, $\mathbf{A} \in (0, 1)^K$ represents task-specific decay/retention rates, $W$ is the coupling matrix, and $\mathbf{e}_t$ represents the normalized task coordinate signals extracted from early layers.
- **Adaptive Online Kinetics:** To solve the "stateful-stateless" trade-off (routing lag under rapid task switches), they propose dynamically scaling the retention parameters based on local coordinate stream cosine similarity:
  $$a_{k, t} = a_k \cdot Sim_t$$
  where $Sim_t = \frac{\mathbf{e}_t^T \mathbf{e}_{t-1}}{\|\mathbf{e}_t\|_2 \|\mathbf{e}_{t-1}\|_2 + \epsilon}$.
- **PAC-Bayesian Generalization Bound:** To handle non-i.i.d., temporally dependent query streams, they model the sequence as a stationary $\beta$-mixing stochastic process. They derive a novel, parameter-space Catoni-type PAC-Bayesian generalization bound using the **Even/Odd Block Splitting** technique to avoid the exploding Total Variation penalty. The bound is directly minimized via gradient descent during a short calibration phase to optimize all kinetics and routing parameters.
- **Gating & Blending:** Active ensembling coefficients $\alpha_t$ are generated via a multi-temperature Gibbs Softmax policy over $s_t$, and the LoRA activations are blended sample-wise in a single, parallel forward pass.

## Key Findings
1. **Routing Jitter Reduction:** PAC-Kinetics slashes routing jitter by over **11.2$\times$** on orthogonal streams and up to **16.0$\times$** on overlapping streams in the Analytical Coordinates Sandbox compared to stateless SABLE.
2. **Robustness to Workload Shifts:** Under heterogeneous streams, while ChemMerge's accuracy collapses to **70.59%**, PAC-Kinetics remains highly robust, achieving **92.35%** (orthogonal) and **92.90%** (overlapping) accuracy, outperforming standard Stateful ERM by **5.21%** and **9.86%** respectively.
3. **Physical Validation:** On physical MNIST and Fashion-MNIST datasets with a 3-layer PyTorch MLP, PAC-Kinetics achieves **76.40%** classification accuracy under homogeneous streams (outperforming stateless PAC-ZCA at 71.20% and Uniform Merging at 54.90%) and slashes routing jitter by **2.59$\times$** compared to PAC-ZCA.
4. **Deterministic Surrogate Gap:** In both sandbox and physical settings, the randomized router required by pure PAC-Bayesian theory (sampling parameters $\Theta'_t \sim Q$) collapses to near-uniform accuracy ($\approx 31\%$-$33\%$ in sandbox, $\approx 43\%$ in physical) due to high-variance perturbations disrupting the stateful dynamics. In contrast, serving the deterministic surrogate (using the optimized mean $\Theta_{\text{opt}}$) restores stability and performance, reconciling theory and practice.

## Explicitly Claimed Contributions (with Evidence)
1. **First Learning-Theoretic Framework for Stateful, Sequential Ensembling:** The authors provide a complete learning theory for stateful routing (Theorem 3.1 and proofs in Appendix A).
2. **Provable Control-Theoretic Stability:** The stateful recurrence is proven to be Globally Asymptotically Stable (GAS) and Input-to-State Stable (ISS) under a quadratic Lyapunov function (Section 3.7). This is further extended to the time-varying Adaptive Online Kinetics system.
3. **Adaptive Online Kinetics to Suppress Inertial Drag:** The authors introduce a cosine-similarity multiplier to dynamically scale down retention coefficients during rapid task switches, which is shown to preserve high accuracy on heterogeneous streams (Tables 1, 2, and 3).
4. **Rigorous Handling of Non-i.i.d. Streams:** By partitioning streams into blocks and using Berbee's coupling, they establish a PAC-Bayesian bound for stationary $\beta$-mixing sequences (and extend it to piecewise-stationary sequences in Appendix B).
5. **Bridging the Simulation-to-Physical Gap:** They validate the method on real image datasets and deep networks in Appendix C, demonstrating strong classification accuracy and smoothness gains.
