# Active Inference Routing (AIR) - Paper Outline

## Title: Active Inference Routing: Stateful Model Serving via Variational Free Energy Minimization

## Fictional Identity:
- **Author:** Dr. Julian Vance
- **Affiliation:** Department of Engineering Science, University of Oxford, UK
- **Email:** julian.vance@eng.ox.ac.uk

---

## 1. Abstract
- **Context:** Large-scale machine learning systems increasingly serve heterogeneous query streams using specialized model experts. Dynamic routing mechanisms blend these experts at runtime based on incoming query activations.
- **Problem:** Existing dynamic routers face a fundamental trade-off: stateless routers react immediately to task changes but suffer from high-frequency sensory jitter (routing noise), while stateful routers (e.g., biochemical ODEs, EMA) smooth trajectories but introduce severe representational lag (inertial drag) at task boundaries.
- **Proposed Solution:** Active Inference Routing (AIR). We model the multi-expert routing layer as an active, self-organizing cognitive agent performing test-time perception and action.
- **Mechanism:** The routing state is represented as a Gaussian variational belief updated by minimizing Variational Free Energy, which balances top-down temporal expectations and bottom-up sensory prediction errors.
- **Results:** Evaluated on the Analytical Coordinate Sandbox, AIR achieves state-of-the-art classification accuracy (matching Oracle ceilings) and slashes routing jitter by over 2.4$\times$. It eliminates representational lag, adapting near-instantaneously to rapid task shifts.
- **Ablation:** A non-negative ablation verifies that negative inhibitory coupling in the generative mapping is mathematically required to actively suppress old task states.

---

## 2. Introduction
- **Paradigm Shift:** Moving beyond feed-forward heuristics and static temporal filters. We propose viewing the router as a brain-inspired, active-inference agent.
- **The Serving Stream as a Dynamical System:** Let the sequential query stream be driven by a latent task trajectory. The router's goal is to track this hidden context under noisy observations.
- **The Jitter-Lag Dilemma:**
  - Stateless approaches (e.g., SABLE) are highly sensitive to noise, causing high-frequency ensembling oscillations (jitter) which disrupt representations.
  - Stateful methods (e.g., ChemMerge, Momentum-Merge) smooth routing weights but are slow to respond to actual task transitions (representational lag), causing severe accuracy drops during transitions.
- **Our Vision:** Active Inference provides a unified, mathematically rigorous theory (originating from neuroscience) to balance prior expectations and incoming sensations.
- **Contributions:**
  1. *Brain-Inspired Router:* The first active-inference formulation of dynamic model ensembling.
  2. *Variational Free Energy Formulation:* Analytical derivation of precision-weighted prediction errors.
  3. *Elimination of the Jitter-Lag Trade-Off:* Rapid adaptation without sacrificing steady-state smoothness.
  4. *Inhibitory Pathways:* Identification and validation of active suppression as a requirement for sequential serving.

---

## 3. Related Work
- **Dynamic Model Merging & Gating:** Discussion of MoE, LoRA ensembling, and routing layers.
- **Stateful vs. Stateless Routers:** Contrast SABLE (nearest-centroid angular similarity) with ChemMerge (biochemical ODE kinetics) and PAC-Kinetics.
- **Active Inference & Cognitive Control:** Introduce the Free Energy Principle (Friston et al.) and its applications in predictive coding, state estimation, and adaptive control.

---

## 4. Methodology
- **System Formulation:** Setup of intermediate routing layers ($l_{\text{route}}$) and deep expert ensembling across adapted layers ($l > l_{\text{route}}$).
- **The Generative Model:**
  - Transition prior: $p(\mathbf{s}_t | \mathbf{s}_{t-1}) = \mathcal{N}(\mathbf{A}\mathbf{s}_{t-1}, \mathbf{\Sigma}_s)$.
  - Observation likelihood: $p(\mathbf{e}_t | \mathbf{s}_t) = \mathcal{N}(\mathbf{W}\mathbf{s}_t, \mathbf{\Sigma}_e)$.
- **Variational Free Energy Formulation:**
  - Analytical derivation: $\mathcal{F}_t = \frac{1}{2} (\mathbf{e}_t - \mathbf{W}\mathbf{\mu}_t)^T \mathbf{\Pi}_e (\mathbf{e}_t - \mathbf{W}\mathbf{\mu}_t) + \frac{1}{2} (\mathbf{\mu}_t - \mathbf{A}\mathbf{\mu}_{t-1})^T \mathbf{\Pi}_s (\mathbf{\mu}_t - \mathbf{A}\mathbf{\mu}_{t-1})$.
  - Mathematical definition of precision terms ($\mathbf{\Pi}_e, \mathbf{\Pi}_s$).
- **Belief Update (Perception):**
  - Unrolling gradient descent steps at test-time to find $\mathbf{\mu}_t$ that minimizes $\mathcal{F}_t$.
  - Adaptive filtering interpretation: how precision weights dynamically shift attention.
- **Action (Policy):** Multi-temperature Gibbs Softmax policy.
- **Calibration and Training:** Backpropagation through the unrolled minimization graph over a tiny calibration subset.

---

## 5. Experiments
- **Experimental Environment:** 14-layer, 192D Analytical Coordinate Sandbox with 4 specialized experts.
- **Query Stream Paradigms:**
  - *Homogeneous Stream:* Long, stable periods of a single task. Tests noise filtering.
  - *Heterogeneous Stream:* Rapid, chaotic, high-frequency task switching. Tests adaptation latency.
- **Quantitative Evaluation:** Detailed comparison against Oracle, Uniform, SABLE, Momentum-Merge, ChemMerge, and PAC-Kinetics across Orthogonal and Overlapping manifolds.
- **Mechanistic Insights (Trajectory Analysis):**
  - Tracking weight trajectories over task transitions (`fig1_weight_trajectories.png`).
  - Analysis of ChemMerge's inertial lag vs. AIR's rapid overcoming of the temporal prior.
- **Ablation Study (The Need for Biochemical Inhibition):**
  - Investigating $W \ge 0$ constraint (passive decay).
  - Showing how without negative weights (inhibition), Task A cannot be actively suppressed, leading to a 0.87% drop in accuracy.

---

## 6. Conclusion and Future Directions
- **Summary:** AIR establishes a new class of active-perception routers that naturally resolve the stability-accuracy trade-off.
- **Future Scope:** Expanding to non-linear deep transition priors, scale-up to large language models, and active exploration strategies.
- **Aesthetic Note:** The visionary beauty of combining neuroscience with system serving.
