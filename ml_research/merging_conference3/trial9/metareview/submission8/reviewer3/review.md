# Peer Review: GraviMerge

## 1. Summary of the Paper
The paper presents **GraviMerge**, a novel, physics-informed model merging paradigm for test-time ensembling of parameter-efficient expert adapters (e.g., LoRA) under dynamic, non-stationary serving workloads.
To address the "accuracy-stability dilemma"—where stateless dynamic merging methods (like SABLE) suffer from layer-to-layer ensembling weight jitter, and existing state-dependent first-order kinetic smoothers (like ChemMerge or EMA) introduce severe phase lag that destabilizes representational propagation—GraviMerge models representation routing as a second-order classical multi-body gravitational system. 

Specifically, intermediate activations are mapped to a virtual stateful spacecraft probe traveling on the unit hypersphere $\mathbb{S}^{D-1}$, while pre-trained task expert centroids act as stationary celestial stellar attractors. GraviMerge incorporates three main physical and geometric components:
1. **Arrhenius Mass Activation (AMA):** Dynamically computes task-attractor masses at test-time using zero-shot similarity with max-subtraction stabilization.
2. **Geodesic Trajectory Integration (GTI):** Integrates net gravitational forces, velocity, and viscous drag strictly on the sphere's tangent spaces, using exact spherical **Exponential Maps** (geodesic step) and closed-form **Parallel Transport** to prevent coordinate scale drift and numerical energy accumulation.
3. **Gravitational Influence Blending (GIB):** Derives continuous, differentiable ensembling weights proportional to the relative magnitude of softened gravitational forces (using an Arctangent potential to eliminate division-by-zero singularities).

Evaluating GraviMerge across 10 random seeds on a Projected Digit Representation Space (RDS) Proxy benchmark demonstrates that it achieves the highest joint serving accuracy (**88.69%**) while reducing layer-wise ensembling weight jitter by **6.01$\times$** compared to ChemMerge, **5.47$\times$** compared to EMA, and **2.40$\times$** compared to SABLE. Extensive evaluations in the Appendix validate that the framework scales to GPT-2 ($D=768$) and Llama-3 ($D=4096$) dimensions under layer-specific representational drift (achieving up to $1.06 \times 10^6\times$ jitter reduction), handles representational noise robustly, and supports out-of-distribution streams via Sentinel Attractor Dynamics (SAD) and low-latency serving via parallelized pre-computation.

---

## 2. Overall Recommendation
**Recommendation:** **5: Accept**
*Justification:* GraviMerge represents an exceptionally well-thought-out, mathematically rigorous, and highly original contribution to the active area of test-time model ensembling. The paper successfully resolves a fundamental trade-off (accuracy vs. stability) using a second-order dynamical system, backed by a clean control-theoretic explanation. The empirical results are robust (evaluated across 10 seeds with thorough baselines), and the Appendix provides outstandingly thorough scaling, noise, OOD, and hardware latency analyses. While there is a simulation-to-reality gap (evaluating on projected digit representations rather than actual text-generation tasks on a physical LLM), the authors openly acknowledge this and frame it as an immediate research milestone. This is a solid, high-impact paper that the community will build upon.

---

## 3. Soundness
**Rating:** **Excellent**
*Justification:* The methodology is mathematically flawless and physically consistent. By mapping routing state updates onto the curved manifold $\mathbb{S}^{D-1}$ through local tangent projections, geodesic Exponential Maps, and Parallel Transport, the authors avoid the radial scale drift and numerical instabilities that would plague naive Euclidean operations on a sphere. The control-theoretic proof in Section 3.5 establishes a rigorous justification for *why* a second-order spring-mass-damper system ($m\ddot{\mathbf{x}} + c\dot{\mathbf{x}} + k\mathbf{x} = \mathbf{F}$) is optimal: its transfer function decays at $-40$ dB/decade (perfectly filtering high-frequency noise) while its active force-driven nature avoids the severe phase lag (time delay) that causes first-order filters (EMA, ChemMerge) to overshoot and oscillate in closed-loop systems. The evaluations are extremely thorough, featuring 10-seed standard deviations, deep parameter ablations, noise robustness sweeps, OOD safeguards, and physical wall-clock latency profiling.

---

## 4. Presentation
**Rating:** **Excellent**
*Justification:* The paper is written with exceptional clarity, structuring its physical-to-mathematical analogies logically and elegantly. The equations are self-contained, mathematically rigorous, and easy to trace. Figure 1a is highly informative, beautifully illustrating the smooth, stable ensembling weight trajectory of GraviMerge against SABLE's high-frequency oscillations and ChemMerge's volatile drifts. Figure 1b provides a clean and striking visual of the Pareto Frontier. Sibling procedural details (Algorithm 1) and LLM-scale integration blueprints (addressing sequence-level memory scaling, low-dimensional projections, and block-wise geodesic updates) are beautifully presented in the Appendix, resolving any reproducibility or implementation questions.

---

## 5. Significance
**Rating:** **Good**
*Justification:* Multi-task edge serving under resource constraints is a highly significant and active problem in distributed AI. Dynamic model merging allows a single backbone model to combine specialized adapters (like LoRA) on-the-fly, bypassing the computational cost of keeping multiple model instances active. By breaking the accuracy-stability bottleneck of dynamic model merging, GraviMerge introduces a highly stable routing controller that is immediately applicable to modern served architectures. While the empirical validation is currently grounded in a coordinate sandbox proxy (using scikit-learn digits) rather than downstream generative LLM tasks (such as MMLU or GSM8k), the mathematical scaling analyses ($D=4096$, $K=64$ experts) and system blueprints prove the high feasibility and practical utility of GraviMerge in production-scale model servers.

---

## 6. Originality
**Rating:** **Excellent**
*Justification:* The originality of this work is outstanding. Most physics-informed deep learning papers (such as Neural ODEs, Hopfield Networks, or continuous diffusion flows) operate as first-order gradient-like concentration kinetics. GraviMerge is the first framework to introduce continuous, second-order Newtonian mechanics (incorporating geodesic exponential maps and parallel transport on curved manifolds) specifically designed to govern parameter-space blending. Rather than relying on simple signal-processing smoothers, the paper introduces a highly elegant combination of classical mechanics, differential geometry, and control theory to solve an open, practical deep learning serving problem.

---

## 7. Major Strengths and Weaknesses

### Strengths
* **High Mathematical Rigor:** The integration of Riemannian geometry (tangent projection, Exponential Map, Parallel Transport) ensures absolute geometric consistency, preventing coordinate scale drift and numerical energy accumulation on curved manifolds.
* **Rigorous Control-Theoretic Foundation:** Shifting the physical analogy from a mere "heuristic metaphor" to a mathematically necessary system architecture via a spring-mass-damper transfer function analysis is highly elegant and convincing.
* **Outstanding Empirical Gains:** Achieving BOTH the highest joint serving accuracy (**88.69%**) and a dramatic **6.01$\times$** reduction in layer-wise routing weight jitter ($0.00190$ MAD) completely dominates existing state-of-the-art baselines.
* **Exhaustive Scaling and Robustness Analyses (Appendix):** Sibling analyses in the Appendix are exceptionally comprehensive, covering Llama-3-scale scaling ($D=4096$, $K=64$ experts), noise robustness studies, out-of-distribution safeguards (Sentinel Attractor Dynamics), and GPU memory-optimization blueprints (Low-Dimensional Spacecraft Projection and Block-Structured Geodesic Integration).

### Weaknesses
* **Simulation-to-Reality Gap:** The primary empirical evaluation is conducted on a projected digit coordinate simulation proxy (RDS). While mathematically robust for isolating the routing equations, evaluating the framework on actual token/sentence representations extracted from a physically deployed LLM (such as Llama-3) performing standard downstream generation tasks (like GLUE or MMLU) would make the evaluation much more compelling.
* **Sequential Serving Bottlenecks in Coupled Mode:** In Coupled GraviMerge ($\eta_{\text{feedback}} > 0$), there is a sequential layer-to-layer dependency where the feedback force depends on the actual activation of the previous layer. While Decoupled Mode allows complete parallel pre-computation of all layer weights immediately after Layer 3, Coupled Mode is strictly sequential, which may introduce latency overheads on physical GPUs and prevent kernel parallelization.

---

## 8. Detailed Scholarly Positioning and Citation Check
The submission does an excellent job of tracing its lineages to parameter-efficient fine-tuning (PEFT, LoRA), static model merging (Task Arithmetic, TIES-Merging, DARE), and physical analogies in deep learning (Neural ODEs, Hopfield Networks). 
However, to fully ground this work within the current state of the art in dynamic, test-time adapter ensembling, we suggest that the authors discuss or compare GraviMerge relative to three emerging real-world instance-level dynamic LoRA selection and merging frameworks:
1. **LoRA on the Go (LoGo):** A training-free, instance-level dynamic LoRA selection and merging framework (ACL 2026) that leverages signals from a single forward pass to identify relevant adapters and determine merging weights on-the-fly. Comparing GraviMerge's second-order stateful routing to LoGo's single-pass signal-based routing would greatly enrich the literature contextualization.
2. **DA-MergeLoRA:** A hypernetwork-based approach for few-shot test-time domain adaptation that generates per-column merging factors to fuse source LoRA modules.
3. **TTMM (Test-Time Model Merging):** An approach used in local mixtures of experts that dynamically selects a subset of LoRA adapters and merges their parameters before inference to reduce memory overhead.

Integrating a brief discussion of these emerging test-time LoRA ensembling paradigms in the Related Work (Section 2) or Discussion will highlight a deeper, more nuanced understanding of the historical and contemporary landscape of the field.

---

## 9. Constructive Suggestions for Future Improvement
1. **Bridge the Evaluation Gap with Physical LLMs:** We highly encourage the authors to execute a physical serving benchmark where GraviMerge (configured in Decoupled Mode) is integrated into Llama-3-8B paired with diverse specialized LoRA experts, evaluating downstream text-generation accuracies on benchmarks like MMLU or GSM8k.
2. **Address Coupled Mode GPU Latency:** We recommend exploring fused CUDA/Triton kernels to compile and parallelize the sequential Geodesic Trajectory Integration operations in Coupled Mode, thereby minimizing kernel launch latencies and intermediate DRAM read/write overheads on physical GPUs.
3. **Hyperparameter Sensitivity Discussion:** While Section 5 and Appendix 6.7 demonstrate that GraviMerge is highly robust to $G$ (as it cancels out of GIB), we suggest including a practical recipe in the main text advising practitioners on how to quickly select optimal base parameters ($\epsilon = 0.8, \gamma_{\text{drag}} = 0.9, \Delta t = 1.0$) for any high-dimensional representation space.
