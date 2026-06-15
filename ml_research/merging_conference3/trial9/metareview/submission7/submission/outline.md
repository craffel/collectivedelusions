# Paper Outline: Lyapunov-Stable Active Representation Coupling (L-ARC)

## Title
Lyapunov-Stable Active Representation Coupling for Dynamic Model Serving

## Author Information (Camera-Ready)
- **Author:** Lysander Sterling
- **Affiliation:** Department of Computer Science, Princeton University
- **Email:** lsterling@princeton.edu

## Abstract (sections/00_abstract.tex)
- Context: Large-scale serving of dynamic model adapters (e.g., LoRA) for heterogeneous workloads.
- Problem: "Heterogeneity collapse" and "vectorization collapse" in dynamic ensembling due to high-frequency routing weight jitter and cascading representational drift. Standard techniques (like ChemMerge) use constant feedback rates ($\eta$) which are unstable under mixed streams and pull activations off-manifold.
- Solution: L-ARC (Lyapunov-Stable Active Representation Coupling). Formulate a candidate Lyapunov function representing the weighted cosine distance error of hidden representations with respect to catalytic task centroids.
- Key Theory: Derivation of a discrete-time Lyapunov difference equation. Propose an analytical "Dissipation Guard" that dynamically computes if the coupling is error-reducing ($A > 0$) or destabilizing ($A \le 0$), scaling the step size proportionally or setting it to zero on-the-fly.
- Results: In the 14-layer Coordinate Sandbox (ICS) across 10 seeds, L-ARC achieves **78.36% ± 0.72%** joint mean accuracy, outperforming SABLE and ChemMerge, recovering **99.19%** of the Expert Oracle ceiling, and showing extreme resilience to task manifold entanglement.

## 1. Introduction (sections/01_intro.tex)
- Deep learning serving systems increasingly rely on parameter-efficient adapters (e.g., LoRAs) dynamically routed per sample to handle diverse tasks.
- Explaining the stateless vs. stateful ensembling paradigm: stateless ensembling (e.g., SABLE, SPS-ZCA) suffers from high routing volatility (jitter) across layers. Stateful ensembling (e.g., ChemMerge) introduces ordinary differential equations (ODEs) to act as spatial low-pass filters.
- Introducing the major bottleneck: **Cascading Representational Drift**. Active coupling in stateful routing uses a heuristic constant feedback step size ($\eta$) to pull intermediate features towards active task centroids. While beneficial for homogeneous streams, this is catastrophically unstable under heterogeneous streaming, where routing errors compound layer-by-layer and pull activations off-manifold.
- The Theorist's core philosophy: Heuristics are insufficient. We need formal control-theoretic guarantees to ensure feature propagation remains stable and on-manifold.
- Introduce **L-ARC**: A training-free closed-loop control system that mathematically guarantees dissipation of representational error.
- Summarize our key contributions:
  1. Formal control-theoretic Lyapunov framework for continuous-depth active model serving.
  2. The analytical Dissipation Guard and Closed-Loop Control Law.
  3. Extensive empirical evaluation on ICS with state-of-the-art results, showing recovery of the oracle performance and robustness under high manifold entanglement.

## 2. Related Work (sections/02_related_work.tex)
- **Parameter-Efficient Tuning & Adapter Merging:** LoRA, weight-space merging (linear, spherical), and their limitations under dynamic multi-task workloads.
- **Dynamic Model Serving & Routing:** SABLE, SPS-ZCA, and activation-space ensembling. Focus on the issue of high-frequency layer-wise jitter.
- **Continuous-Depth Architectures & Physics-Inspired Models:** Neural ODEs, reaction-decay models (ChemMerge). Point out how prior continuous-depth ensembling models lack rigorous control-theoretic guarantees for feedback stability, falling back on manual heuristic tuning or disabling feedback entirely.

## 3. Mathematical Formulation & Control Law (sections/03_method.tex)
- *This is the core theoretical section of the paper, reflecting the Theorist persona.*
- **System Model:** The 14-layer Coordinate Sandbox (ICS). Expert LoRAs targeting self-attention query/value projections.
- **Continuous Concentration Kinetics (NEKR ODE):**
  - Arrhenius collision rate equation.
  - Discretized Euler update for concentrations.
  - Mass-action ensembling weights.
- **Active Representation Coupling Update:**
  - Formula for the warped representation.
  - Layer-wise propagation equations.
- **Lyapunov Stability Analysis (Theoretical Derivations & Proofs):**
  - Definition of the system-level candidate Lyapunov function $V(C, h)$. Prove that $V \ge 0$.
  - Discrete difference $\Delta V^{(l)} = V(C^{(l)}, h^{(l-1)\text{ warped}}) - V(C^{(l-1)}, h^{(l-2)\text{ warped}})$.
  - Linearization via first-order Taylor expansion around $\eta = 0$.
  - Directional derivative $D_k^{(l)}$.
  - Derivation of the Drift-Accumulation Coefficient $B_b^{(l)}$ and Dissipation Coefficient $A_b^{(l)}$.
  - Proving that $\Delta V^{(l)} \le 0 \iff \eta_b^{(l)} A_b^{(l)} \ge B_b^{(l)}$.
- **Closed-Loop Lyapunov-Stable Control Law:**
  - Dissipation Guard logic: If $A_b^{(l)} \le 0 \implies \eta_b^{(l)} = 0$.
  - Adaptive Step Size logic: If $A_b^{(l)} > 0 \implies \eta_b^{(l)} = \min(\eta_{\max}, \gamma A_b^{(l)})$.
  - Rigorous exposition of *why* this guarantees stability and prevents cascading representational drift.

## 4. Experimental Setup & Results (sections/04_experiments.tex)
- **Experimental Environment:** Analytical Coordinate Sandbox (ICS), 14 layers, $D = 192$, $K = 4$ visual task manifolds (MNIST, Fashion-MNIST, CIFAR-10, SVHN) fine-tuned via LoRA (rank 8).
- **Baselines:** Expert Ceiling, Uniform Merging, Linear Router, SPS-ZCA, SABLE, ChemMerge (Decoupled, $\eta = 0$), ChemMerge (Coupled, constant $\eta = 0.05$).
- **Main Results Table:** Clear performance comparison under heterogeneous serving. Discuss how L-ARC achieves **78.36% ± 0.72%** (recovering 99.19% of Oracle) and outperforms SABLE SOTA.
- **Visualizations & Deep Analysis:**
  - **Layer-wise Concentration Trajectories (`trajectories.png`):** Show how NEKR/L-ARC smooth out the layer-to-layer volatility of SABLE.
  - **Coupling Feedback Ablation (`coupling_ablation.png`):** Analyze the trade-off of constant $\eta$ under homogeneous vs. heterogeneous workloads, and how L-ARC adaptively solves both.
  - **Manifold Entanglement Resilience (`entangled_robustness.png`):** Discuss the sweep of entanglement parameter $\rho \in [0.0, 0.5]$ and show L-ARC's extreme robustness.

## 5. Conclusion & Future Work (sections/05_conclusion.tex)
- Summarize L-ARC and its theoretical and empirical advantages.
- Conclude with how this demonstrates that bringing control-theoretic rigor to deep learning heuristics is a powerful paradigm for designing next-generation adaptive AI serving systems.
- Outline future avenues: extending Lyapunov control to larger multi-modal models and exploring stochastic Lyapunov control under noisier environments.
