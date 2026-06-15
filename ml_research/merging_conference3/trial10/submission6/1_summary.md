# 1_summary.md: Comprehensive Summary of the Paper

## Main Topic and Motivation
The paper addresses the challenge of serving multi-tenant workloads in deep learning models using parameter-efficient fine-tuning (PEFT) expert adapters (specifically LoRA). While test-time ensembling of multiple expert adapters offers a flexible paradigm to handle heterogeneous streaming queries, existing dynamic routers suffer from a fundamental trade-off:
1. **Stateless Routers** (e.g., SABLE, SPS-ZCA) compute ensembling weights independently per layer and sample. While highly responsive, representation noise causes wild layer-to-layer ensembling weight oscillations across network depth (the *routing jitter paradox*), which corrupts representation alignment.
2. **Prior Stateful Routers** (e.g., ChemMerge, Momentum-Merge) attempt to smooth these trajectories via metaphorical chemical kinetics (continuous-time ODEs) or open-loop exponential moving averages (EMA). However, they accumulate past routing history too rigidly, introducing severe *inertial drag* (phase delay) under rapid task transitions, which causes performance to collapse under highly heterogeneous workloads. Furthermore, continuous-time ODE solvers introduce prohibitive execution latency that violates tight edge-device budgets.

## Proposed Approach: PID-Merge
To resolve this stability-responsiveness trade-off, the paper proposes **PID-Merge**, a closed-loop, discrete-time Proportional-Integral-Derivative (PID) controlled stateful routing framework. Rooted in classical control theory, PID-Merge treats raw early-layer similarity weights as the reference setpoint and active layer-wise ensembling coefficients as the plant output. 

Key architectural and control-theoretic features of PID-Merge include:
* **Incremental (Velocity-Form) PID State Update:** A lightweight recursive update running in $O(1)$ time ($15$ FLOPs per expert per layer) that updates unnormalized controller states across layers.
* **Derivative (D) Anticipation:** Measures error acceleration to instantly detect block switches and provide an anticipatory boost at boundary layers, overcoming depth-wise adaptation lag without temporal lag.
* **Numerical Normalization & Anti-Windup Safeguards:** Employs *scaled logit mean-centering* to prevent absolute floating-point overflow under expert-specific temperatures while maintaining mathematical translation invariance. Integrates *conditional integration (clamping)* using dynamic, $K$-scaled clamping thresholds ($\theta_{\text{high}} = 1 - \epsilon, \theta_{\text{low}} = \epsilon/K$) to prevent integrator windup and transition lag in deep topologies.
* **Multi-Temperature Gibbs Policy:** Maps controller states to simplex-bounded active ensembling weights, parameterized via a Softplus transformation to guarantee valid positive control gains.
* **Prefill-Locked Routing Policy:** Locks the ensembling weights $\alpha_k^{(l)}$ computed during the prefill phase, holding them static during autoregressive decoding. This guarantees complete KV Cache coherence and reduces the decoding routing overhead to absolutely zero.

## Key Empirical Findings
* **Isolating Coordinate Sandbox (ICS) Evaluations:** Calibrated PID-Merge achieves **94.82%** accuracy on challenging overlapping heterogeneous streams, outperforming ChemMerge by **+6.40%** and Momentum-Merge by **+8.65%** absolute accuracy, while matching the stateless ceiling within $0.12\%$.
* **Robust Zero-Shot Performance:** The training-free (zero-shot) mode with robust default gains ($K_p = 0.5, K_i = 0.15, K_d = 0.2$) achieves **93.35%** heterogeneous accuracy under extreme overlap, providing an immediate deployment option.
* **Physical Validation on GPT-2 (NVIDIA A100 GPU):** Validated on a physical 12-layer GPT-2 Small backbone routing 3 actual fine-tuned task adapters. PID-Merge slashes depth-wise layer-to-layer jitter by **over 73%** (from $0.7241$ to $0.1932$) while maintaining near-oracle serving accuracy ($88.64\%$), adding an imperceptible latency overhead of just **0.012 ms** ($40\times$ faster than SOTA ChemMerge at $0.482$ ms).
* **Robustness to Overfitting & Calibration Bias:** Calibrated on a tiny sequence of $32$ samples, the controller generalizes perfectly to out-of-sample streams, even when calibrated on a extremely biased or purely homogeneous single-task sequence.

## Explicitly Claimed Contributions
1. **Control-Theoretic Formulation:** The first closed-loop, discrete-time PID control framework for dynamic stateful model serving, featuring derivative anticipation and dynamic conditional integration clamping.
2. **Extensive Simulated Evaluation:** Empirical proof on the ICS sandbox demonstrating exceptional accuracy-stability performance and detailing the speed-stability Pareto trade-off.
3. **Physical Validation & Latency Profiling:** Physical validation on a 12-layer GPT-2 model on an NVIDIA A100 GPU demonstrating a $40\times$ latency reduction over ChemMerge and over 73% jitter reduction.
4. **Systems-Level Production Blueprint:** Practical solutions for multi-tenant security/privacy constraints, KV cache coherence (prefill-locking), and GPU execution bottlenecks (fused Triton/CUDA kernels).
5. **Analytical Stability Proof:** Jury's stability criterion analysis establishing tight, closed-loop BIBO stability bounds with a soft optimization penalty to enforce stable parameters.
