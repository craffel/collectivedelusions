# 3. Soundness and Methodology

## Clarity of Description
The mathematical formulation in Section 3 is formally written and technically precise, presenting equations for force calculation, tangent projection, geodesic steps, and parallel transport. However, the description suffers from severe **mathematical obfuscation**. By casting a simple routing and weight-smoothing problem into a metaphorical "auxiliary physical cosmology" involving virtual spacecraft, celestial attractors, viscous medium drag, and orbital mechanics, the paper unnecessarily complicates the underlying operations. A simple problem is hidden behind dense, astrophysically-inspired terminology, making it difficult for an expert reader to quickly understand the actual, practical computations being performed.

## Appropriateness of Methods
From a minimalist engineering perspective, the methods are highly inappropriate and excessively over-engineered. The paper introduces a massive, state-dependent, second-order continuous dynamical system (requiring coordinate state trackers, velocity tracking, tangent projections, and parallel transport at every adapted layer) simply to compute a scalar ensembling weight $\alpha_k^{(l)}$ for each adapter. 

A standard neural network's layers are discrete, and the representations themselves are parameterized. Introducing an auxiliary continuous-time physical simulation (geodesic trajectory integration) running in parallel with the forward pass is highly unconventional and introduces unnecessary stateful overhead. 

## Potential Technical Flaws

### 1. The "Decoupling Illusion" in Default Mode
In the default "Decoupled GraviMerge" setup (which is used to achieve the main results in Table 1), the spacecraft coordinates $\mathbf{h}_{\text{sc}}^{(l)}$ and velocity $\mathbf{v}^{(l)}$ are updated based *only* on the initial early-layer activation $\mathbf{h}^{(3)}$ and the static centroids $\boldsymbol{\mu}_k^{(3)}$. 
Specifically, the dynamic masses $M_k$ are computed once at Layer 3, and the spacecraft coordinates are initialized at $\mathbf{h}_{\text{sc}}^{(3)}$. Throughout layers 4 to 14, the spacecraft moves deterministically in this static gravity field, completely independent of the actual propagating neural activations $\mathbf{h}^{(l)}$. 

This is a major conceptual flaw: **there is no closed-loop feedback or dynamic adaptation to representation drift in the deep layers.** The spacecraft's trajectory is completely pre-determined at Layer 3. This means that GraviMerge is not actually a "dynamic, stateful router" that responds to intermediate activations; it is simply a complex, deterministic interpolation function parameterized by the early activation $\mathbf{h}^{(3)}$. The entire layer-by-layer physical simulation is a computational charade—the entire sequence of weights $\alpha_k^{(4)} \dots \alpha_k^{(14)}$ could be pre-computed immediately at Layer 3.

While the authors mention "Coupled GraviMerge" as a closed-loop feedback alternative, it is not the default, introduces even more complexity (equation 10), and adds another hyperparameter ($\eta_{\text{feedback}}$) to tune.

### 2. Severe Hyperparameter Fragility
GraviMerge introduces an excessive number of hyperparameters that must be carefully calibrated:
- $G$ (gravitational constant)
- $\gamma_{\text{drag}}$ (viscous drag coefficient)
- $\Delta t$ (virtual step size)
- $\epsilon$ (softening factor)
- $\tau_{\text{grav}}$ (routing temperature)
- $\eta_{\text{feedback}}$ (feedback coupling)
- $\lambda_{\text{temporal}}$ (temporal decay)

This is a hyperparameter tuning nightmare. As shown in the ablation studies, the system exhibits severe "phase transitions" and "force singularities" (e.g., if $G \ge 0.05, \epsilon = 0.1$, jitter explodes to $0.0226$ due to velocity spikes). Such sensitivity makes the framework highly fragile and impractical for real-world deployment, where robustness to hyperparameter settings is critical. In contrast, SABLE and SPS-ZCA require almost no hyperparameter tuning, making them far more robust and practical.

## Reproducibility
Due to the extreme complexity of the parallel transport, tangent projections, and exponential maps, the method is exceptionally difficult to reproduce correctly. A slight error in the projection of the velocity vector or the parallel transport equations will lead to coordinate drift, state saturation, and catastrophic performance drops. The lack of standard, lightweight, and easily verifiable operations makes this framework a "black box" of complex math that is highly prone to implementation errors.
