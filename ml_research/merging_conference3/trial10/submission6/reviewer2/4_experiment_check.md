# 4. Experiment Check

This section provides a critical evaluation of the experimental setup, datasets, baselines, and whether the empirical findings actually support the authors' claims.

## Experimental Design and Setup
The authors evaluate PID-Merge in two distinct environments:
1. **The Isolating Coordinate Sandbox (ICS):** A custom simulation platform mimicking a deep ensembling system with $L=14$ layers, $D=192$, and $K=4$ experts. It models both "Orthogonal" and "Overlapping" coordinate manifolds, with query sequences served under Homogeneous and Heterogeneous (fast-switching) workloads.
2. **Physical GPU Validation:** A physical 12-layer GPT-2 Small model routing three actual task-specific adapters on an NVIDIA A100 GPU.

## Strengths of the Experimental Evaluation
- **Diverse Workloads:** Testing on both stationary (homogeneous) and rapid-switching (heterogeneous) workloads is excellent, as it clearly exposes the "inertial drag" failure mode of prior stateful routers.
- **Physical Validation:** Moving from a simulated sandbox to physical GPT-2 adapter serving on actual hardware (A100 GPU) is a major strength. It grounds the systems-level claims (such as latency overhead and GPU memory bottlenecks) in real-world measurements.
- **Strong Baseline Comparison:** The paper compares against a wide array of representative baselines: Uniform, SABLE (stateless raw), ChemMerge (stateful ODE kinetics), Momentum-Merge (EMA), and PAC-Kinetics (complexity-regularized stateful).

---

## Methodological Concerns and Limitations

### 1. Artificial Sandbox Stability and Noise Propagation Constraints
As the authors honestly acknowledge in Section 4.1, the simulated **Isolating Coordinate Sandbox (ICS)** suffers from a severe design limitation: representation noise is injected **only at the initial boundary layer** (Layer 3/4 boundary), with zero subsequent propagation across layers 5 to 14. 
- In actual deep networks, representation noise is generated and propagates dynamically at *every* layer.
- Because of this artificial constraint in the simulation, the stateless SABLE router does not exhibit any layer-wise oscillations across depth after its initial boundary transition. In Table 1, SABLE's depth-wise jitter in ICS is reported as a flat $0.136$, which represents only the single-step boundary transition penalty from the uniform initialization to the target one-hot vector. SABLE appears artificially stable in depth-wise simulation.
- While the authors use physical GPT-2 experiments to correct for this (where SABLE indeed exhibits a high depth-wise jitter of $0.724$), this sandbox limitation means that the ICS results are overly optimistic for stateless methods, and do not fully reflect the depth-wise filtering challenges.

### 2. Lack of Large-Scale Backbone Evaluation
The physical hardware validation is restricted to **GPT-2 Small (124M parameters)** with $K=3$ experts.
- Modern multi-tenant serving workloads typically operate on multi-billion parameter backbones (such as LLaMA-3 8B or 70B, Mixtral, etc.) with much larger expert pools (e.g., dozens of LoRA adapters).
- Systems-level issues like KV cache capacity, memory bandwidth bottlenecks, and GPU kernel launch overhead scale non-linearly with model size. While the authors present a theoretical scalability analysis in Appendix H, the lack of an empirical evaluation on a multi-billion parameter model is a weakness that limits the immediate applicability of their conclusions to production LLM environments.

### 3. Jitter-Accuracy Pareto sweep transparency
The paper claims in Section 4.4 that the calibrated configuration finds an "overdamped/critically damped" controller ($K_p \approx 0.35, K_i \approx 0.23, K_d \approx 0.13$) that slashes depth-wise jitter by 66% in simulation and 73% in physical GPT-2.
- While this is impressive, the transition to this overdamped state requires adding a differentiable depth-wise jitter penalty with coefficient $\beta = 1.0$ to the calibration loss.
- The paper lacks a comprehensive visualization of the Pareto frontier (Accuracy vs. Jitter) showing how accuracy degrades as the jitter penalty $\beta$ is scaled up. Since the transition from underdamped (high-speed) to overdamped (high-smoothing) reduces tracking speed, there must be a point where heavy smoothing leads to transition lag and accuracy degradation. This trade-off should be mapped out more systematically.

---

## Conclusion on Empirical Claims
The empirical findings are comprehensive and generally support the authors' claims. The physical GPT-2 validation is especially convincing, as it shows that PID-Merge slashes depth-wise jitter by over 73% while maintaining high accuracy and keeping latency overhead at an imperceptible 0.012 ms. However, the simulation sandbox constraints and the small size of the evaluated transformer limit the physical results.
