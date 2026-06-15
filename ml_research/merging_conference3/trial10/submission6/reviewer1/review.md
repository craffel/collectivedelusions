# Peer Review

## Summary of the Paper
This paper addresses the challenge of serving multi-tenant workloads under sequential, non-stationary query streams by dynamically ensembling multiple task-specific expert adapters (e.g., LoRA) on a shared frozen base model at test-time. 

While stateless dynamic routers suffer from a "routing jitter paradox" (wild weight oscillations across layers within a single query's forward pass due to representation noise), prior stateful routers (e.g., ChemMerge, Momentum-Merge) smooth these trajectories but introduce "inertial drag" (phase delay) under rapidly switching streams, causing accuracy collapse.

To resolve this trade-off, the paper proposes **PID-Merge**, a closed-loop, discrete-time Proportional-Integral-Derivative (PID) controlled stateful routing framework. Rooted in classical control theory, PID-Merge treats raw early-layer similarity-based routing weights as the reference setpoint and active layer-wise ensembling coefficients as the controlled plant output. By feeding back the tracking error utilizing Proportional (P), Integral (I), and Derivative (D) gains, PID-Merge filters out representation noise while completely suppressing tracking phase lag. 

Crucially, the Derivative (D) term measures error acceleration to instantly detect task transitions and provide an anticipatory boost driving weights to the target within 2--3 layers. The framework is co-designed with several ML systems safeguards:
1. **Scaled Logit Mean-Centering:** Prevents logit drift and absolute overflow over deep networks while maintaining mathematical translation invariance under task-specific temperatures.
2. **Conditional Integration (Anti-Windup Clamping):** Dynamically freezes error accumulation near boundary saturation to ensure responsiveness in deep topologies.
3. **Prefill-Locked Routing Policy:** Locks weights during autoregressive decode steps to those computed during the prefill phase, mathematically guaranteeing KV Cache coherence and reducing decoding latency overhead to zero.

The authors validate PID-Merge on both the simulated Isolating Coordinate Sandbox (ICS) and a physical 12-layer GPT-2 Small backbone routing three task adapters on an NVIDIA A100 GPU. Calibrated PID-Merge achieves **94.82%** accuracy on the ICS heterogeneous overlapping stream (outperforming ChemMerge by **+6.40%** and Momentum-Merge by **+8.65%**) and **88.64%** accuracy physically (virtually matching stateless Oracle performance) while slashing depth-wise jitter by over **73%** and introducing a negligible latency overhead of just **0.012 ms** ($40\times$ faster than ChemMerge).

---

## Strengths and Weaknesses

### Strengths
1. **Exceptional Originality & Conceptual Leap:** The core conceptual contribution is outstanding. Treating the *depth* of a deep neural network as the temporal dimension of a discrete-time dynamical system is a brilliant and elegant analogy. Applying closed-loop discrete-time PID control to the ensembling weights themselves represents a massive conceptual leap over naive, open-loop, and continuous-time ODE kinetics models (e.g., ChemMerge).
2. **Co-Design of Control Theory and Deep Learning Constraints:** The paper does not merely copy-paste PID control equations. It introduces highly thoughtful, custom-tailored mathematical safeguards that directly address the specific properties of deep architectures, such as:
   - **Scaled Logit Mean-Centering** to prevent absolute unnormalized logit drift/overflow while maintaining mathematical translation invariance under a multi-temperature Softmax policy.
   - **Conditional Integration (Anti-Windup Clamping)** using dynamic, $K$-scaled thresholds to prevent saturation-induced transition delays in deep topologies.
   - **Prefill-Locked Routing** to guarantee KV Cache coherence across autoregressive generation steps and reduce decoding latency to absolutely zero.
3. **Rigorous Mathematical Grounding:** The control-theoretic formulation is exceptionally solid. The authors derive the discrete-time transfer functions of the closed-loop system, apply **Jury's Stability Criterion** to obtain tight analytical stability bounds, and enforce these bounds via a differentiable stability penalty ($\mathcal{L}_{\text{stab}}$) during optimization to prevent convergence to unstable, underdamped regions.
4. **Exemplary Empirical & Physical Validation:** The paper features a stellar dual-pronged evaluation. The simulated ICS sandbox enables extensive parameter sensitivity sweeps and scalability testing up to $K=64$ experts, showing a dramatic **567.3$\times$ latency reduction** over ChemMerge. The physical GPU validation on GPT-2 with real tasks (IMDB Sentiment, SAMSum Summarization, and WMT16 English-to-German Translation) validates that PID-Merge translates seamlessly to real-world neural representations and hardware.
5. **Outstanding Writing, Structure, and Transparency:** The narrative flow is logical, and the equations are clean. The appendices are rich and cover hardware blueprints, implementation details, parameter sensitivity sweeps, and theoretical scaling dynamics. The authors are refreshingly honest and self-aware regarding the limitations of their work, which significantly enhances its scientific credibility.

### Weaknesses
1. **Scale of Physical GPU Validation:** The physical hardware experiments are conducted on GPT-2 Small (12 layers, 117M parameters). While this is a highly appropriate proof-of-concept for verifying layer-wise activation blending and profiling latency, testing on a modern, multi-billion parameter model (e.g., LLaMA-3 8B with 32 layers) would further elevate the empirical section. (Note: The authors discuss scaling challenges and anti-windup clamping in deeper architectures theoretically in Appendix A, which mitigates this concern).
2. **Lack of Empirical Evaluation for Online Self-Tuning:** In Appendix D, the authors propose an incredibly clever, autocorrelation-based online self-tuning PID controller to dynamically scale gains on-the-fly under non-stationary streams. However, this adaptive formulation is only presented theoretically and lacks empirical evaluation. Including even a small simulated comparison for this self-tuning variant would have been a phenomenal addition.

---

## Ratings and Justifications

### Soundness: Excellent
The paper's soundness is exceptional. All claims are fully backed by rigorous control-theoretic derivations and comprehensive empirical validation. The mathematical simplified velocity-form PID updates under constant anchored setpoints are correct. The $z$-domain linearized stability analysis and Jury's Stability Criterion bounds are mathematically solid. The dual-pronged evaluation (ICS sandbox and physical GPT-2 Small backbone) is clean, properly utilizes standard baselines, and reports means and standard deviations across 5 distinct random seeds to establish statistical robustness.

### Presentation: Excellent
The presentation quality is outstanding. The paper is beautifully structured, and the narrative easy to follow. The figures (trajectory tracking and layer-wise convergence plots) are intuitive and immediately convey the qualitative behavior of the different routing techniques. The tables are clean and comprehensive. The inclusion of a production-ready PyTorch wrapper blueprint in Appendix G ensures that the mathematical methodology is highly reproducible and actionable for systems engineers.

### Significance: Excellent
The significance of this work is very high. In the context of model serving, PID-Merge offers a highly practical, computationally lightweight ($O(1)$ updates, $15$ FLOPs per expert-layer), and deployment-ready solution that addresses a major bottleneck in multi-tenant LLM serving engines. Beyond model serving, this paper introduces a powerful paradigm of treating internal state trajectories or routing weight paths as closed-loop dynamical systems. This control-theoretic feedback framework has broad potential to influence future research in Mixture-of-Experts (MoE) routing, multi-modal alignment, neural ODEs, and neural representation filtering.

### Originality: Excellent
The originality is the crowning achievement of this paper. The conceptual shift from open-loop, continuous-time chemical metaphors (ChemMerge) to a discrete-time, closed-loop PID control loop operating strictly depth-wise across network layers is a massive, highly original contribution. The co-design of control-theoretic safeguards (scaled logit mean-centering, conditional integration anti-windup) and LLM serving constraints (prefill-locked routing for KV cache coherence) is exceptionally creative and represents a significant conceptual leap that stands out from typical incremental papers in the field.

---

## Overall Recommendation

**6: Strong Accept**

**Summary Justification:**
This is an exceptionally strong, complete, and technically flawless paper that introduces a highly novel, elegant, and mathematically rigorous closed-loop control-theoretic paradigm to dynamic model serving. By treating network depth as time and applying discrete-time PID control to weight trajectories, PID-Merge filters out layer-wise representation noise while completely suppressing tracking lag under non-stationary task streams. 

The paper excels across all dimensions:
- It introduces an elegant and original conceptual leap.
- It co-designs control-theoretic safeguards with deep learning specific structures (scaled logit mean-centering, conditional integration clamping, and prefill-locked routing for KV Cache coherence).
- It provides rigorous mathematical stability proofs via Jury's Criterion in the $z$-domain.
- It features exemplary empirical validation across both simulated sandboxes and physical GPU hardware profiling on actual LLMs, proving massive latency and accuracy advantages over state-of-the-art baselines.

This work represents a rare and beautiful intersection of classical control theory and machine learning systems. It is of exceptional quality and has the potential to influence how the community thinks about routing and state trajectories across deep topologies. I recommend a Strong Accept with the highest enthusiasm.

---

## Constructive Feedback and Questions for the Authors

1. **Physical Validation Scale:** Have the authors explored deploying PID-Merge on larger topologies like LLaMA-3 8B? If so, did the conditional integration (clamping) and scaled logit mean-centering safeguards successfully manage the increased depth (32 layers) under physical representations? Any preliminary results would be highly valuable.
2. **Empirical Self-Tuning:** The autocorrelation-based dynamic gain adaptation proposed in Appendix D is a brilliant concept. Do the authors have any plans to evaluate this adaptive variant empirically in the future? It would be interesting to see if it outperforms the calibrated static gains on streams that shift between long homogeneous blocks and chaotic step-to-step switches.
3. **Robustness to Overlapping Manifolds in Large Pools:** In Table 6 (Appendix F), the tracking accuracy of SABLE and PID-Merge is evaluated as a function of the active expert pool size $K \in \{4, 8, 16, 32\}$. SABLE's accuracy remains flat at $94.93\%$, and PID-Merge's accuracy rises from $93.38\%$ to $94.93\%$ before dropping slightly to $94.33\%$ at $K=32$. Could the authors provide more intuition as to why PID-Merge's accuracy actually improves as $K$ scales from 4 to 16, and what causes the minor decline at $K=32$?
4. **Minor Detail on Temperature Sensitivity:** In the calibrated mode, how sensitive is the optimization to the initialization of the unconstrained temperature parameters $w_k$? Does starting from a high temperature risk converging to poor local minima during backpropagation?
