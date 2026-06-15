# Part 1: Summary of the Paper

## Core Motivation and Problem Statement
The paper addresses the **routing jitter paradox** in test-time dynamic model ensembling for parameter-efficient fine-tuning (PEFT) adapters (e.g., LoRA). Existing dynamic routers are primarily **stateless** (e.g., SABLE, SPS-ZCA, PAC-ZCA), processing incoming sequential queries independently. Under query-level observation noise, these routers exhibit high-frequency fluctuations in routing weights. In deep cascaded architectures, this jitter propagates and amplifies across layers, causing **cascading representation collapse** where intermediate features drift away from the pre-trained backbone manifold.

Conversely, existing **stateful** ensembling heuristics (e.g., ChemMerge) smooth routing trajectories using first-order chemical reaction kinetics differential equations, acting as a low-pass filter. However, they lack stability and out-of-sample generalization guarantees, and their static parameters lead to severe accuracy degradation under heterogeneous workload streams where task domains switch rapidly (referred to as **inertial drag** or routing lag).

## Proposed Methodology: PAC-Kinetics
To resolve this accuracy-stability trade-off, the authors introduce **PAC-Kinetics**, a learning-theoretic and control-theoretic framework for stateful, sequential model ensembling.
1. **Coordinate Projection**: Extracted pooled activations at a designated routing layer are unit-normalized (to enforce a strict, dimension-free coordinate bound) and projected using Unit-Norm PCA coordinates derived from a calibration split.
2. **Chemical Kinetics Recurrence**: The routing weights evolve as a continuous concentration state vector $s(t)$ in a continuous-time non-equilibrium chemical kinetics system:
   $$\frac{ds(t)}{dt} = - \mathbf{\Gamma} s(t) + \mathbf{\Phi} \mathbf{e}(t)$$
   Discretizing this yields the linear recurrence:
   $$s_t = \mathbf{A} s_{t-1} + W \mathbf{e}_t$$
   where $\mathbf{A} \in (0, 1)^K$ represents state-retention and $W \in \mathbb{R}^{K \times K}$ is the coordinate injection matrix. States are mapped to active ensembling coefficients $\alpha_t$ via a multi-temperature Gibbs Softmax policy.
3. **Adaptive Online Kinetics**: To mitigate inertial drag under rapid switches, the router computes the cosine similarity $Sim_t \in [0, 1]$ between consecutive coordinates and dynamically scales down retention: $a_{k, t} = a_k \cdot Sim_t$.
4. **PAC-Bayesian Optimization**: To learn stable parameters under non-i.i.d. conditions, the query stream is modeled as a stationary $\beta$-mixing stochastic process. The authors derive a novel, parameter-space Catoni-type PAC-Bayesian bound using block splitting:
   $$R(Q) \le \frac{\mathcal{L}_{\max}}{1 - e^{-\lambda}} \left[ 1 - e^{ -\frac{\lambda \hat{R}_T(Q)}{\mathcal{L}_{\max}} - 2 \frac{\text{KL}(Q \| P) + \ln(2/\delta)}{a} } \right]$$
   This bound is directly minimized over a short calibration stream using PyTorch autograd.

## Key Theoretical Results
1. **Global Asymptotic Stability (GAS) and Input-to-State Stability (ISS)**: Under a quadratic Lyapunov function $V(s) = \|s\|_2^2$, the authors prove that the state recurrence is GAS under zero coordinate input and ISS under bounded inputs. This stability is strictly preserved under the time-varying Adaptive Online Kinetics mechanism.
2. **Catoni's $\beta$-Mixing PAC-Bayesian Bound**: Theorem 3.1 provides out-of-sample generalization guarantees for the randomized Gibbs router under dependent sequences, utilizing even/odd block splitting to bypass TV penalty explosions.
3. **Deterministic Surrogate Approximation**: The authors analyze the theoretical-to-empirical gap between the randomized router $Q$ and the served deterministic surrogate $\Theta_{\text{opt}}$, proving a Lipschitz-based bound on their performance difference.

## Main Experimental Findings
1. **Jitter Reduction**: In a 14-layer, 192-dimensional Coordinates Sandbox (ICS) simulating 4 experts, PAC-Kinetics reduces routing jitter by over **11.2$\times$** on orthogonal streams and up to **16.0$\times$** on overlapping streams compared to stateless SABLE, matching the smoothness of heuristic ChemMerge.
2. **Robust Accuracy**: Under heterogeneous streams, where heuristic ChemMerge collapses to $70.59\%$ accuracy due to rigid parameters, PAC-Kinetics maintains a high joint accuracy of **92.35\%** (orthogonal) and **92.90\%** (overlapping).
3. **Generalization Benefit**: PAC-Kinetics outperforms standard Empirical Risk Minimization (Stateful ERM) by up to **9.86% absolute** under heterogeneous overlapping streams, validating the regularizing impact of the PAC-Bayesian KL complexity penalty.
4. **Physical Validation**: On a PyTorch MLP trained on MNIST and Fashion-MNIST subsets, PAC-Kinetics achieves the highest representation alignment ($87.61\%$) and downstream classification accuracy ($76.40\%$) among all stateful and stateless baselines under homogeneous streams, demonstrating robust physical transferability.
