# Intermediate Evaluation 1: Summary of the Paper

## Main Topic and Motivation
The paper addresses the challenge of test-time dynamic ensembling of multiple task-specific expert adapters (such as LoRA) on a shared frozen base transformer model under sequential, heterogeneous, and non-stationary query streams. Modern serving environments must handle queries from various tasks (e.g., sentiment analysis, text summarization, machine translation) in real-time. 

While **stateless routers** (like SABLE or SPS-ZCA) compute ensembling weights on-the-fly sample-by-sample and layer-by-layer, they suffer from the **routing jitter paradox**: representation noise causes the dynamic ensembling weights to oscillate wildly across layers within a single query's forward pass, corrupting representation manifolds and degrading accuracy.

To solve this, prior **stateful routers** (such as ChemMerge, which uses continuous-time chemical kinetics ODEs, and Momentum-Merge, which uses open-loop Exponential Moving Averages (EMA)) smooth these trajectories but introduce **inertial drag** (phase/group delay) during sudden task transitions. Consequently, their performance collapses under rapidly switching (heterogeneous) workloads. Additionally, continuous-time chemical solvers (ChemMerge) introduce prohibitive serving latency, violating strict latency budgets.

## Proposed Approach: PID-Merge
The authors propose **PID-Merge**, a closed-loop, discrete-time Proportional-Integral-Derivative (PID) controlled stateful routing framework. Rooted in classical control theory, PID-Merge treats raw early-layer similarity-based routing weights as the reference setpoint and the active layer-by-layer ensembling coefficients as the controlled plant output. It computes the tracking error at each adapted layer and updates the routing state using an incremental velocity PID control algorithm.

The core components of PID-Merge include:
1. **Discrete-Time Closed-Loop Error Feedback:** Measures the discrepancy between the raw routing setpoint and the previous layer's active ensembling coefficient.
2. **Incremental (Velocity) State Update:** Computes unnormalized state increments using Proportional (P), Integral (I), and Derivative (D) terms to ensure both smooth transitions (via low-pass filtering) and rapid adaptation (via error acceleration anticipation).
3. **Scaled Logit Mean-Centering Numerical Safeguard:** A translation-invariant centering technique that bounds logit magnitudes to prevent floating-point overflow over deep network topologies without distorting ensembling probabilities.
4. **Conditional Integration (Anti-Windup Clamping):** Freezes the integral state accumulator whenever the active weights saturate near the boundary limits ($\theta_{\text{high}}$ and $\theta_{\text{low}}$) to eliminate saturation-induced transition lag in deep architectures.
5. **Practical Serving Integration:**
   - **Zero-Shot (Training-Free) Mode:** Employs robust heuristic default gains ($K_p = 0.5, K_i = 0.15, K_d = 0.2$).
   - **Calibrated (Optimized) Mode:** Optimizes gains and expert-specific routing temperatures via backpropagation on a tiny 32-sample sequence.
   - **Prefill-Locked Routing Policy:** Locks routing weights during autoregressive decode steps to the values computed during the prefill phase, guaranteeing KV Cache coherence and eliminating decoding latency overhead.

## Key Findings and Empirical Evidence
- **Isolating Coordinate Sandbox (ICS) Results:** 
  - On heterogeneous, overlapping manifolds, calibrated PID-Merge achieves **94.82%** accuracy, outperforming ChemMerge by **+6.40%** and Momentum-Merge by **+8.65%** absolute.
  - Zero-Shot PID-Merge achieves **93.35%** accuracy, offering a strong training-free alternative.
  - Out-of-sample parameter generalization is outstanding; gains calibrated on a biased or purely homogeneous 32-sample sequence generalize perfectly to a balanced heterogeneous test stream.
- **Physical Validation on GPT-2 (NVIDIA A100 GPU):**
  - Fine-tuned on IMDB, SAMSum, and WMT16 English-to-German translation.
  - Calibrated PID-Merge achieves **88.64%** average accuracy (virtually matching SABLE's 89.14% ceiling) while slashing depth-wise layer-to-layer jitter by **73.3%** (from 0.724 to 0.193).
  - PID-Merge introduces a negligible latency overhead of just **0.012 ms** (over $40\times$ faster than ChemMerge's 0.482 ms).

## Explicitly Claimed Contributions
1. **First Closed-Loop Control-Theoretic Framework:** Introduces discrete-time PID control to dynamic, stateful model ensembling, resolving the stability-responsiveness trade-off and eliminating tracking lag via the Derivative (D) term.
2. **Scalability Safeguards:** Introduces scaled logit mean-centering and conditional integration (anti-windup clamping) to guarantee absolute numerical stability and prevent transition delays in deep topologies.
3. **Empirical and Physical Validation:** Demonstrates state-of-the-art performance on the Isolating Coordinate Sandbox (ICS) and physically validates the framework on a 12-layer GPT-2 model on an NVIDIA A100 GPU, confirming dramatic jitter reduction and negligible latency overhead.
4. **Systems Blueprint:** Provides a concrete systems blueprint for integration into high-throughput multi-tenant serving frameworks (e.g., S-LoRA/Punica) with prefill-locked routing to guarantee KV cache coherence.
