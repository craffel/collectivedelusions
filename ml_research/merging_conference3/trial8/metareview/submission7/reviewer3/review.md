# Peer Review: ChemMerge

## 1. Summary of the Paper
This paper addresses the critical challenge of dynamic model ensembling and merging under streaming, heterogeneous workloads on edge devices. Standard static parameter-space model merging (e.g., Task Arithmetic, TIES-Merging, DARE) achieves constant $O(1)$ latency but suffers from *Heterogeneity Collapse* under mixed streams due to parameter interference. Existing dynamic activation-space ensembling methods (e.g., SABLE, SPS-ZCA) dynamically route inputs inside the forward pass, but they treat layers as decoupled, independent execution blocks. Under noisy or rapidly switching streams, this stateless layer-wise routing leads to high-frequency ensembling coefficient oscillations (*routing weight jitter*). On the other hand, systems-level scheduling queues (e.g., Micro-Batch Homogenization) restore representation stability but introduce a prohibitive $O(K)$ sequential latency penalty on edge hardware.

To resolve this latency-accuracy-stability trade-off, the authors present **ChemMerge**, a training-free, continuous-time paradigm that models the flow of representations through a deep network's depth as a multi-component chemical reactor governed by non-equilibrium reaction kinetics. By viewing task experts (LoRA adapters) as reactive species and early-layer centroids as catalytic enzymes, ChemMerge tracks continuous, sample-wise expert concentration states that evolve across layers governed by a system of first-order kinetics differential equations (Non-Equilibrium Kinetic Routing, NEKR). Blending coefficients are derived from active concentrations via the Law of Mass Action (Catalytic Activation Blending, CAB) inside a single-pass parallel forward pass.

The authors present two numerical schemes to discretize the ODE: an Explicit Euler step with projection clipping and a mathematically elegant, exact Exponential Integrator that guarantees concentration bounds inside $[0, 1]$ via convex combinations. They evaluate their method inside a 14-layer, 192-dimensional Analytical Coordinate Sandbox (ICS) across 10 random seeds, recovering **98.81%** of the Expert Oracle ceiling and outperforming stateless nearest-centroid baselines by up to **+8.22%** under constant $O(1)$ latency. They also conduct a routing-only validation using real-world high-dimensional activation manifolds extracted from a pre-trained Vision Transformer ($\text{ViT-B/16}$) across a geometric shape stream, demonstrating a dramatic **9.9$\times$ reduction** in layer-to-layer routing weight jitter compared to nearest-centroid routing and over **2.15$\times$** compared to SABLE (under equivalent routing sensitivities).

---

## 2. Strengths and Weaknesses

### Soundness
* **Strengths:** 
  * The mathematical and physical formulation is exceptionally sound. The authors do not simply rely on a qualitative chemical metaphor; they back it up with rigorous ordinary differential equations (ODEs), steady-state convergence analysis, and precise analytical step-size bounds ($\Delta t < 1.538$) that match their empirical sweeps perfectly.
  * The derivation of the exact analytical Exponential Integrator is highly elegant and robust. It completely bypasses the need for heuristic projection clipping, maintaining absolute numerical stability across extreme step sizes.
  * The authors demonstrate outstanding scientific honesty and integrity. They prominently and repeatedly include **CRITICAL SCIENTIFIC DISCLOSURES** clarifying that their task-stream evaluations are entirely simulated within a sandbox (ICS), and that their pre-trained model validation is a routing-only simulation on frozen activations. This transparent reporting prevents any false or misleading claims of real-world adapter execution and sets an exemplary standard for the community.
* **Weaknesses:**
  * While the simulated Analytical Coordinate Sandbox (ICS) and routing-only validation are excellent for isolating ensembling and routing mechanics with high scientific transparency, the paper lacks end-to-end evaluation with actual physical adapter training, loading, and activation blending on real-world multi-task benchmarks (e.g., VTAB-1k or GLUE). The authors have provided a highly thorough 5-step roadmap in the future work section to execute this, but the absence of physical adapter blending on real pixels/tokens remains a slight limitation.

### Presentation
* **Strengths:**
  * The paper is beautifully written, articulate, and grammatically flawless. The overall narrative flows logically and is highly engaging.
  * The mathematical notation is highly consistent, precise, and clean across both the methodology and the appendix.
  * Visualizations are exceptional. Figures 1, 4, 5, 6, 7, and 8 provide clear, high-signal representations of the performance curves, concentration trajectories, parameter sensitivity sweeps, and trajectory-smoothing behavior.
* **Weaknesses:**
  * None of significance. The presentation and structure follow the standard ICML template perfectly.

### Significance
* **Strengths:**
  * Resolving the trade-off between representational stability, multi-task accuracy, and edge-serving latency is a highly important problem for on-device deployment of foundational models.
  * The demonstrated NumPy-vectorized latency scaling shows that ChemMerge's parallel matrix formulation is highly hardware-friendly, executing 42.1% faster than SABLE and 49.4% faster than SPS-ZCA at $K=16$.
  * The concept of tracking routing weights as a continuous stateful trajectory across depth represents a major paradigm shift that could inspire a new class of continuous-depth ensembling architectures, particularly for token-by-token generation in autoregressive LLMs.
* **Weaknesses:**
  * The latency evaluations are conducted on CPU-bound NumPy runtimes. Physical edge hardware profiling (such as NPUs or low-power neuromorphic runtimes) with direct power-metering would be necessary to fully confirm and quantify the practical, real-world serving advantages.

### Originality
* **Strengths:**
  * High level of conceptual originality. Bridging biochemistry, dynamical systems, and test-time model ensembling is a highly creative and successful synthesis of ideas.
  * While Neural ODEs and related works model representations as continuous trajectories, ChemMerge is highly novel in modeling the **ensembling routing weights themselves** as continuous physical concentration variables.
  * The mathematical duality establishing that the reversible reaction ODE is equivalent to a state-dependent adaptive Exponential Moving Average (EMA) smoothing filter is highly elegant. It provides a principled, biochemically grounded mechanism that generalizes static signal-processing filters.
* **Weaknesses:**
  * The Related Work section, while thorough, misses some very recent concurrent and prior literature specifically dedicated to routing stability, gating oscillations, and jitter mitigation in Mixture-of-Experts. Incorporating these works would significantly enrich the paper's scholarly context.

---

## 3. Soundness Rating
**Rating: Excellent**
*Justification:* The mathematical derivations, continuous convergence proofs, discretization stability bounds, and exact Exponential Integrator equations are technically flawless and rigorous. All empirical evaluations are backed by multi-seed statistics. The authors are exceptionally honest, transparent, and precise about the simulated nature of their sandbox and routing-only validation, leaving no room for misleading claims.

---

## 4. Presentation Rating
**Rating: Excellent**
*Justification:* The paper is outstandingly clear, well-structured, and highly articulate. The mathematical notations are clean, and the diagrams and multi-panel figures are of exceptional publication quality, providing deep physical intuition.

---

## 5. Significance Rating
**Rating: Good**
*Justification:* The paper addresses a highly relevant bottleneck in edge model serving. Its training-free, single-pass parallel formulation achieves constant $O(1)$ latency while suppressing representation jitter. Although the current validation is simulated/routing-only, the conceptual impact and hardware-friendly vectorized scaling are substantial.

---

## 6. Originality Rating
**Rating: Excellent**
*Justification:* The connection between systems biochemistry, non-equilibrium reaction kinetics, and deep model ensembling is highly creative and novel. The derivation of the state-dependent adaptive EMA duality is exceptionally elegant and distinguishes the method from simple static low-pass filtering.

---

## 7. Overall Recommendation
**Rating: 5: Accept**
*Justification:* ChemMerge is an exceptionally solid, beautifully written, and conceptually original paper that successfully bridges systems biochemistry and deep neural inference. Its mathematical derivations are technically rigorous, and its empirical sweeps are extensive and robust. The authors set a stellar example of scientific honesty and integrity through their clear disclosures of the simulated environments. Although the lack of physical adapter ensembling on real-world image pixels represents a minor limitation, the rigorous trajectory analysis on real-world pre-trained ViT-B/16 manifolds, combined with a highly comprehensive future work roadmap, makes this a highly valuable and high-signal contribution to the conference.

---

## 8. Detailed Comments and Constructive Feedback for Authors

### 1. Positioning in the Broad MoE Routing Jitter and Gating Oscillation Literature
While the paper situates itself very well within PEFT, parameter-space merging, and classic Neural ODEs, it would benefit significantly from citing and discussing recent, highly relevant concurrent/prior work on dynamic routing stability and gating oscillations in Mixture-of-Experts:
* **ReMoE (2025/2026):** Investigates bidirectional coupling to introduce temporal smoothing in sequence routing, reducing abrupt expert switches.
* **Dense Backpropagation (2025):** Explores applying EMA smoothing to gating logits during training to stabilize convergence.
* **FourierMoE (2026):** Reformulates expert adaptation in the spectral domain using frequency-adaptive routing.
Discussing these works in Section 2.2 or 2.3 will ground ChemMerge even more firmly in the broader machine learning literature on routing stabilization.

### 2. Minor Mathematical and Typographical Points
* In Equation 12 (Section 3.4, непрерывная сходимость), there is a small typo in the continuous ODE recurrence error expansion. Expanding $e^{(l)} = C_{k}^{(l)} - C_{k}^*$ shows that the continuous time convergence is stable, but please ensure that the continuous ODE in Eq. 10 is clearly distinguished from its discretized error recurrence in Eq. 13.
* In Equation 17, the authors introduce the state-dependent adaptive smoothing factor $\beta^{(l)} \equiv \Delta t (k_k^{(l)} + k_{\text{decay}})$. To ensure strict convex combination behavior in the expanded explicit Euler recurrence, please explicitly state the bounds required on $\beta^{(l)}$ (i.e., $\beta^{(l)} \in [0, 1]$), which connects back to the step-size stability boundary discussed in Section 3.4.

### 3. Transitioning to Real-World Adapter Blending
The 5-step roadmap presented in Section 5.2 for transitioning to actual physical adapter training, loading, and activation blending (CAB) across VTAB-1k and GLUE is excellent. The paper would be significantly strengthened if the authors could execute a mini-version of this roadmap for a two-adapter vision merging setup (e.g., merging two LoRA experts trained on distinct domains of DomainNet) and report the classification accuracy. This would empirically confirm that Catalytic Activation Blending (CAB) holds up under physical parameter updates.

### 4. Questions for the Authors
* **Question 1:** Under the *Layer-Specific Centroid Routing* (Multi-Centroid Mode) in extremely deep architectures (e.g., 32-layer LLMs), the centroid calibration overhead scales as $O(L \cdot K \cdot D)$. Have the authors considered *centroid interpolation*, where centroids are only calibrated and stored at a sparse subset of layers (e.g., every 4th layer) and intermediate centroids are computed via smooth geometric interpolation? How would this affect routing accuracy and jitter?
* **Question 2:** For the *Temporal Cross-Sample Propagation* in autoregressive LLMs, if concentrations are carried over across consecutive tokens, there is a physical transition lag when sudden topic shifts occur. Have the authors considered implementing a *similarity-driven temporal reset*, where a large drop in cosine similarity (or a spike in perplexity/surprise) instantly resets concentrations to uniform $1/K$, bypassing transition lag completely?
