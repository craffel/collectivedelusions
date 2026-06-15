# Peer Review: Layer-Decoupled Stateful Kinetics (LDS-Kinetics) for Dynamic Model Merging

---

## 1. Summary of the Paper
This paper addresses the challenge of **dynamic model merging (test-time ensembling)** of low-rank adapters (LoRAs) under sequential multi-task workloads. Dynamic model merging serves as a scalable paradigm for executing multi-task streams under constrained resource footprints by ensembling specialized adapters on-the-fly. Current state-of-the-art stateful routing methods (such as ChemMerge or PAC-Kinetics) smooth ensembling trajectories to suppress high-frequency coordinate noise (the "routing jitter paradox"), but they enforce **spatial homogeneity**: a single global ensembling weight vector is applied uniformly across all network depths.

The authors introduce **Layer-Decoupled Stateful Kinetics (LDS-Kinetics)** to challenge the spatial homogeneity assumption. LDS-Kinetics treats network depth as an active variable, partitioning layers into $M$ disjoint blocks. Each block maintains an independent concentration state vector that evolves according to its own block-specific parameters. To mitigate the threat of transductive overfitting on short calibration sequences ($T=32$) due to overparameterization, the authors integrate a unified learning-theoretic complexity penalty derived from Catoni's $\beta$-mixing PAC-Bayesian bound.

Through extensive sweeps inside a 14-layer coordinate sandbox and physical validation on a 6-layer sequence Transformer, the paper deconstructs spatial-temporal routing dynamics. The key findings are:
1. **The Spatial-Temporal "Tempo-Gradient":** Network depth maps directly to distinct optimal temporal scales. Early blocks learn high-decay (low-retention, $a \approx 0.32$) and high-temperature ($\tau \approx 0.18$) dynamics to act as responsive trackers. Late blocks learn low-decay (high-retention, $a \approx 0.94$) and low-temperature ($\tau \approx 0.04$) dynamics to act as stable low-pass filters, shielding final logits from representation noise.
2. **The Optimization-Level Lockstep Pathology:** Unregularized Empirical Risk Minimization (ERM) fails because of a sign-symmetry pathology under the Adam optimizer (driven by shared coordinate inputs). The PAC-Bayesian KL gradient acts as a natural, uniform bias that breaks this starting sign-symmetry during optimization, while simultaneously bounding statistical complexity.
3. **Non-linear Propagation & Latency Neutrality:** Under non-linear activation propagation (GELU + LN), stateful ensembling completely bridges the "stateful accuracy penalty," outperforming stateless SABLE by up to $0.70\%$ in absolute accuracy. By packing updates into a single parallelized batched tensor operation, the systems overhead of decoupled routing is completely eliminated, achieving virtual latency neutrality.

---

## 2. Strengths and Weaknesses

### A. Soundness
* **Strength (Principled Theoretical Grounding):** The paper does an outstanding job grounding the complexity penalty in Catoni's PAC-Bayesian bound for stationary mixing processes. Rather than relying on heuristic $L_2$ weight decay, the mathematical derivation connects posterior parameter drift and sample complexity directly.
* **Strength (Intellectually Honest Analysis of Non-Stationarity):** The authors address a critical theoretical critique: real-world serving involves abrupt, non-stationary switches, which technically violate the stationarity assumption of Catoni's bound. The authors establish a clean, robust separation of roles: the PAC-Bayesian complexity penalty acts as a robust offline regularization prior, while their online similarity scaling ($Sim_t$) physically flushes memory and manages non-stationarity during serving.
* **Strength (Optimization-Level Depth):** The diagnosis and resolution of the Adam sign-symmetry lock pathology is a highlight. It explains why unregularized models collapse to global baselines, and how the KL gradient naturally resolves it.
* **Strength (Exhaustive Controls & Baseline Construction):** To isolate whether the benefits arise from spatial-only variations or temporal stateful kinetics, the paper constructs two highly specialized stateless spatial control baselines (*Static Decay* and *Static Block*). The results under non-linear GELU + LN propagation demonstrate that temporal ensembling recurrences are mathematically necessary to maintain stable, cohesive representational pathways across layers.
* **Strength (Physical Implementation & Latency Neutrality):** Validating the framework on a physical 6-layer Transformer backbone with pre-trained LoRA experts and demonstrating virtual latency neutrality through a packed matrix-vector formulation is a substantial systems-level engineering contribution.
* **Weakness:** The primary evaluations are conducted within a simulated coordinate sandbox, which simplifies the physical representation spaces of large language models. However, the authors successfully bridge this gap via physical Transformer validation and a complete PyTorch proof-of-concept.

### B. Presentation
* **Strength:** The paper is exceptionally well-written, structured, and polished. The mathematical notation is precise, clear, and consistent throughout the manuscript.
* **Weakness (Critical Citation Bug):** A noticeable presentation flaw occurs in the Introduction (Section 1). Key state-of-the-art frameworks are cited using the placeholder key `anonymous`:
  * *"For instance, SABLE~\cite{anonymous} projects intermediate representations..."* (Section 1, Paragraph 2)
  * *"To suppress jitter, stateful routing frameworks such as ChemMerge~\cite{anonymous} and PAC-Kinetics~\cite{anonymous} were introduced..."* (Section 1, Paragraph 3)
  
  While the Related Work, Bibliography, and Experimental sections correctly cite these works as `sable_2024`, `chemmerge_2026`, and `pac_kinetics_2026`, using the `anonymous` citation key in the Introduction is a significant presentation oversight that would lead to compilation warnings/errors and represents a lack of polish in referencing.
* **Weakness (Manuscript Text Truncation):** At the end of Section 4.3.5 (Adaptive Block Grouping and Optimal Depth Boundaries), the text abruptly truncates in the middle of a sentence: *"...In contrast, in the... [truncated]"*. This must be repaired to ensure the section is grammatically complete.

### C. Significance
* **Strength (Highly Relevant serving Challenge):** High-throughput serving of diverse expert adapters (LoRAs) under tight latency/resource footprints is a major, active challenge in deep learning. Suppressing ensembling jitter is critical for serving stability.
* **Strength (Systems-Conscious Insights):** The deconstruction of how deep blocks learn low-decay tempos (high inertia) acts as a low-pass filter that is essential for preserving KV-cache coherence during autoregressive token generation. This is a highly significant systems contribution.
* **Strength (Practical Engineering Guidelines):** Framing the Tri-Block ($M=3$) configuration as the primary recommendation for production is highly valuable. Under non-linear propagation, $M=3$ acts as a robust spatial regularizer that preserves representational trajectory cohesion while reducing execution latency by $73.1\%$ over the $M=11$ model.

### D. Originality
* **Strength (Challenging Spatial Homogeneity):** Treating network depth as an active variable in temporal stateful model merging is a highly original concept.
* **Strength (Creative Combating of Optimization Pathology):** Identifying the sign-symmetry update pathology under Adam and using Catoni's PAC-Bayesian KL gradient to break it is exceptionally creative.

---

## 3. Dimension Ratings & Justification

### Soundness: Excellent
The methodology is exceptionally sound. The theoretical PAC-Bayesian derivations are mathematically correct, and the empirical sandbox simulations, rigorous control baselines, and physical validation provide exhaustive and statistically significant evidence to support the claims.

### Presentation: Good
The writing quality, mathematical clarity, and structure are excellent. However, the rating is adjusted to "Good" due to the critical citation bug (`anonymous` key in the Introduction) and the manuscript text truncation at the end of Section 4.3.5. These referencing and formatting errors must be corrected.

### Significance: Excellent
The paper addresses a highly important, high-impact systems challenge (stable LoRA expert serving), introduces depth-decoupled stateful ensembling with virtual latency neutrality, and provides key systems-conscious insights regarding KV-cache coherence and practical production recommendations.

### Originality: Excellent
Decoupling stateful kinetics along network depths is a novel and significant conceptual leap that merges temporal ensembling kinetics and spatial depth dynamics. The diagnosis and resolution of the Adam sign-symmetry pathology are highly original.

---

## 4. Overall Recommendation
**5: Accept**

LDS-Kinetics is a technically solid, exceptionally thorough paper that advances the active field of dynamic model merging. It elegantly challenges the spatial homogeneity assumption, bridges temporal kinetics and spatial depth dynamics, and provides strong systems-conscious guidelines (batched recurrences, KV-cache stability, and Tri-Block production recommendations) with virtual latency neutrality. 

The paper contains an outstanding level of empirical and theoretical depth. It would be a Strong Accept (6) if the minor presentation and referencing bugs (the undefined `anonymous` citation keys in Section 1, and the manuscript text truncation in Section 4.3.5) are corrected. The authors are strongly urged to fix these bugs prior to publication.
