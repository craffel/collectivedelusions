# Novelty and Literature Assessment: LDS-Kinetics

This document evaluates the key novel aspects, the exact "delta" from prior literature, and the overall characterization of novelty in the proposed Layer-Decoupled Stateful Kinetics (LDS-Kinetics) framework.

---

## 1. Characterization of Novelty
The novelty of LDS-Kinetics is **significant**. It is not merely an incremental patch on existing stateful routing frameworks, but a conceptual advancement that bridges two previously separate paradigms in deep learning:
1. **Temporal Stateful Kinetics:** Using memory states (like continuous-time chemical kinetics or state-space models) to smooth temporal routing trajectories and suppress high-frequency ensembling jitter (e.g., ChemMerge, PAC-Kinetics).
2. **Spatial Depth Dynamics:** The structural reality that different layers of deep networks extract features at different semantic scales and temporal tempos (e.g., shallow layers capture transient alignments, deep layers refine class logits).

By unifying these concepts, the paper introduces **depth-decoupled stateful ensembling**, showing both theoretically and empirically that the network naturally organizes itself into a spatial-temporal "tempo-gradient."

---

## 2. The "Delta" from Prior Work

### A. Delta from Static Model Merging (e.g., Task Arithmetic, TIES-Merging, DARE)
* **Prior Work:** Blends model weights statically offline. This is parameter-free at inference but suffers from severe representational interference, task deletion, and representation collapse when merging highly dissimilar tasks.
* **LDS-Kinetics Delta:** Performs dynamic, test-time ensembling of specialized task experts (LoRAs) on a sample-by-sample basis, preventing inter-task interference and representation collapse.

### B. Delta from Stateless Dynamic Routing (e.g., SABLE, SPS-ZCA)
* **Prior Work:** Uses stateless projection coordinates (like task-specific PCA) to blend experts sample-by-sample. This is highly vulnerable to coordinate query noise, resulting in extreme ensembling weight oscillations—the **routing jitter paradox**—which degrades serving stability and disrupts downstream representation alignment.
* **LDS-Kinetics Delta:** Integrates temporal memory states across network depths. Compared to stateless SABLE (Raw) which exhibits massive jitter (up to $1.1362$ in heterogeneous streams), LDS-Kinetics suppresses this jitter by up to $46.6\%$ on physical models and maintains stable representation-space pathways across layers.

### C. Delta from Stateful Routers with Spatial Homogeneity (e.g., ChemMerge, Momentum-Merge, PAC-Kinetics)
* **Prior Work:** Incorporates a stateful memory trajectory but enforces **spatial homogeneity**, meaning a single, global ensembling weight vector is shared uniformly across all network layers. This forces an unfavorable systems-level trade-off between adaptation speed (needed at early layers to track workload transitions) and decision stability (needed at deep layers to shield the classifier from noise).
* **LDS-Kinetics Delta:** Breaks the spatial homogeneity assumption by partitioning layers into $M$ disjoint blocks, each maintaining separate state variables and block-specific learnable retention rates and temperatures. This is the first framework where network depth acts as an active variable in stateful ensembling.

### D. Delta from Standard Layer-wise Depth Dynamics Literature
* **Prior Work:** Analyzes feature representations at different network depths or applies layer-wise learning rates/decoupled optimization schemes during static training.
* **LDS-Kinetics Delta:** The first to explore and validate the impact of layer-wise depth dynamics in the context of *dynamic, test-time stateful model merging*. It provides the first empirical proof that network depth maps directly to distinct optimal ensembling tempos (shallow layers learning low-retention/high-decay tempos, deep layers learning high-inertia/low-decay tempos).

---

## 3. Notable Strengths in Novelty & Contextualization
* **Symmetry-Breaking Optimization Insight:** The discovery that unregularized decoupled routing fails due to a sign-symmetry optimization pathology under Adam (driven by shared coordinate inputs) is a highly original and sophisticated insight. The demonstration that Catoni's PAC-Bayesian bound's KL gradient acts as a principled symmetry-breaker—while simultaneously bounding statistical complexity—adds exceptional theoretical depth.
* **Rigorous Baseline Construction:** To isolate whether the benefits of LDS-Kinetics arise from spatial variation or temporal kinetics, the paper constructs two highly customized stateless spatial baselines (*Static Layer-Wise Decay* and *Static Block-Wise Constant*). This rigorous control proves that spatial variation alone cannot solve the routing jitter paradox, and that stateful temporal ensembling is mathematically necessary under non-linear propagation (such as GELU + LN).

---

## 4. Citation and Presentation Vulnerabilities
* **Introduction Citation Bug:** A noticeable flaw in the manuscript's presentation is that key SOTA references are cited as `anonymous` in the Introduction:
  * *"For instance, SABLE~\cite{anonymous} projects intermediate representations..."* (Section 1, Paragraph 2, Line 2)
  * *"To suppress jitter, stateful routing frameworks such as ChemMerge~\cite{anonymous} and PAC-Kinetics~\cite{anonymous} were introduced..."* (Section 1, Paragraph 3, Line 1)
  
  While the Related Work and Bibliography sections correctly cite these works as `sable_2024`, `chemmerge_2026`, and `pac_kinetics_2026`, using the `anonymous` citation key in the Introduction is a significant presentation oversight that would lead to compilation warnings/errors and represents a lack of polish in referencing. Proper attribution of SABLE, ChemMerge, and PAC-Kinetics must be maintained consistently across the entire manuscript.
