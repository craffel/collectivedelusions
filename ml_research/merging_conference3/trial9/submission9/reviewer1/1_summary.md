# 1. Summary of the Paper

## Main Topic and Motivation
The paper addresses the challenge of serving multiple parameter-efficient fine-tuning (PEFT) task-specific experts (e.g., LoRA adapters) on sequential, heterogeneous multi-task query streams. While recent frameworks have enabled single-pass dynamic model ensembling/merging by blending activations in a single forward pass, they are primarily **stateless**. This stateless nature leads to a **routing jitter paradox**: query-level feature noise causes high-frequency oscillations in routing weights between consecutive queries. In deep cascaded architectures (like LLMs or deep ViTs), these minor early-stage fluctuations propagate and amplify, causing a catastrophic "downstream representation collapse."

To mitigate this, prior stateful work (like ChemMerge) used heuristic first-order chemical kinetics reaction equations to act as low-pass filters. However, these methods are purely heuristic, lack any stability or generalization guarantees, and perform poorly under heterogeneous workloads where their static parameters trigger severe routing lag ("inertial drag").

The paper proposes **PAC-Kinetics**, a learning-theoretic and dynamical systems framework that formalizes stateful, continuous-time model ensembling. It models the routing dynamics as a continuous-time non-equilibrium chemical kinetics system (discretized as a contractive linear recurrence), and derives a novel Catoni-type PAC-Bayesian generalization bound for stationary $\beta$-mixing stochastic processes. All routing and kinetics parameters are optimized by directly minimizing this PAC-Bayesian bound on a disjoint calibration stream.

---

## Core Technical Approach
1. **Unit-Norm PCA Coordinate Projection**: Extracts intermediate representation signals $z_t$ from an early layer, normalizes them to the unit sphere to enforce a strict dimension-free coordinate bound, and projects them onto task-specific subspaces using principal components extracted from a calibration split.
2. **Continuous-Time Non-Equilibrium Chemical Kinetics**: Maps routing concentrations to a state vector $s_t$ governed by a linear recurrence:
   $$s_t = \mathbf{A} s_{t-1} + W \mathbf{e}_t$$
   where the diagonal retention matrix $\mathbf{A} = \text{diag}(\sigma(u_k)) \in (0, 1)^K$ and coordinate injection matrix $W \in \mathbb{R}^{K \times K}$ are learnable. This linear state-space representation allows closed-form Lyapunov stability proofs.
3. **Adaptive Online Kinetics**: Addresses "inertial drag" under rapid task switches by dynamically scaling down the state-retention parameters $a_{k, t} = a_k \cdot Sim_t$, where $Sim_t \in [0, 1]$ is the cosine similarity of the current and previous task coordinate projection vectors.
4. **Gibbs Policy & Single-Pass Blending**: Translates state concentrations $s_t$ to active ensembling coefficients $\alpha_t$ using a multi-temperature Softmax, and blends the LoRA expert activations in a single parallel forward pass.
5. **PAC-Bayesian Bound for Stationary mixing sequences**: Employs **Even/Odd Block Splitting** over mixing sequences to derive a generalization bound without incurring a vacuous, exploding Total Variation (TV) penalty. The objective is directly optimized using PyTorch.

---

## Key Findings & Empirical Evidence
* **Jitter Reduction**: PAC-Kinetics reduces routing jitter by over **11.2$\times$** on orthogonal streams and up to **16.0$\times$** on overlapping streams relative to the stateless SABLE baseline, matching the smoothness of heuristic ChemMerge.
* **Accuracy-Stability Balance**: Under homogeneous streams, PAC-Kinetics matches or exceeds the expert Oracle (reaching up to **95.07%** accuracy in overlapping setups). Under heterogeneous streams, PAC-Kinetics achieves **92.35%** (orthogonal) and **92.90%** (overlapping) accuracy, heavily outperforming heuristic ChemMerge (which collapses to **70.59%** due to rigid parameters) and Stateful ERM (which achieves $87.14\%$ and $83.04\%$).
* **Physical Validation**: Evaluated on physical datasets (MNIST and Fashion-MNIST) with a deep 3-layer LoRA-blended MLP, PAC-Kinetics achieves **76.40%** classification accuracy under homogeneous streams (outperforming stateless PAC-ZCA's 71.20% and Uniform Merging's 54.90%) and slashes active jitter by **2.59$\times$**.
* **Systems Scalability**: Latency profiling shows CPU routing latency remains flat at **$\approx 10.4$ microseconds** across fleet sizes $K \le 8$, and GPU vectorized batch latency is under **3.5 microseconds** for batch size 128 under $K=8$.

---

## Explicitly Claimed Contributions
1. **Mathematical Rigor**: Replaces heuristic dynamic model serving parameters with provable contractive and Input-to-State Stability (ISS) guarantees under a Lyapunov candidate function.
2. **First Learning-Theoretic Bound for Dependent serving streams**: Derives a novel parameter-space Catoni-type PAC-Bayesian bound for stationary $\beta$-mixing stochastic processes.
3. **Adaptive Online Kinetics**: Proposes an elegant cosine-similarity mechanism to dynamically scale down retention coefficients during sudden task switches, successfully suppressing inertial drag.
4. **Comprehensive Sandbox & Physical Validation**: Introduces an Analytical Coordinate Sandbox for rigorous mixing-coefficient evaluation, alongside physical validation on PyTorch networks using real image datasets, resolving the "Uniform Merging Paradox."
