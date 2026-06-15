# Conference Meta-Review & Decision Report

This report summarizes the meta-review process and final decisions for the 10 paper submissions evaluated in this cycle. Following rigorous examination of both the numeric scores and the qualitative contents of all 30 peer reviews, exactly **three (3) submissions** have been selected for acceptance. 

The entire `submission/` subdirectories of the accepted papers have been copied to:
- `accepted_papers/submission1/`
- `accepted_papers/submission5/`
- `accepted_papers/submission10/`

---

## 1. Meta-Review Process & Selection Methodology

The meta-review process was conducted with the highest scientific hygiene and rigor. Rather than relying solely on raw numerical score averages, each submission was evaluated on:
1. **Theoretical Soundness & Mathematical Rigor:** Whether the proposed models are supported by correct proofs, realistic assumptions, and robust derivations.
2. **Empirical hygiene:** Dedication to statistical confidence, including multiple independent random seeds, proper standard deviations, and fair synchronized baselines.
3. **Practical & Systems-Level Utility:** Practical deployment readiness, compatibility with edge-hardware compilation chains (ONNX/TensorRT), and wall-clock latency/energy profiling.
4. **Originality & Paradigm-Shifting Contributions:** Moving away from incremental heuristics to introduce creative mathematical structures (e.g., Probabilistic Graphical Models, Information Geometry, or Control Theory).

### Complete Submission Score & Decision Registry

Below is the complete registry of all 10 submissions, their reviewer scores, averages, and final decisions:

| Submission ID | Reviewer 1 Rating | Reviewer 2 Rating | Reviewer 3 Rating | Score Average | Final Decision | Key Conceptual Theme |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Submission 1** | 5: Accept | 5: Accept | 6: Strong Accept | **5.33** | **Accept** | Markovian Path-Integral Ensembling (QPathMerge) |
| **Submission 2** | 3: Weak Reject | 3: Weak Reject | 5: Accept | **3.67** | Reject | Stateful Depth-Decoupled Kinetics |
| **Submission 3** | 5: Accept | 2: Reject | 3: Weak Reject | **3.33** | Reject | Biologically-Grounded Population Routing |
| **Submission 4** | 4: Weak Accept | 5: Accept | 5: Accept | **4.67** | Reject | Integer Apportionment & AEF on MCUs |
| **Submission 5** | 6: Strong Accept | 5: Accept | 4: Weak Accept | **5.00** | **Accept** | Unit Geodesic Routing (UGR) |
| **Submission 6** | 6: Strong Accept | 3: Weak Reject | 5: Accept | **4.67** | Reject | Control-Theoretic PID-Merge |
| **Submission 7** | 5: Accept | 5: Accept | 4: Weak Accept | **4.67** | Reject | Slot-Kinetics & Virtual Task Caching |
| **Submission 8** | 3: Weak Reject | 3: Weak Reject | 4: Weak Accept | **3.33** | Reject | Discrete Cosine Trajectory Merging |
| **Submission 9** | 3: Weak Reject | 5: Accept | 5: Accept | **4.33** | Reject | Active Inference Serving Controllers |
| **Submission 10** | 6: Strong Accept | 6: Strong Accept | 3: Weak Reject | **5.00** | **Accept** | 2D Spatio-Temporal Bilinear Filter (2D-STEM) |

---

## 2. Detailed Justification for Accepted Papers

### 🏆 Submission 1: Markovian Path-Integral Ensembling (QPathMerge)
* **Final Ratings:** 5 (Accept), 5 (Accept), 6 (Strong Accept)
* **Average Score:** 5.33
* **Decision:** **Accept**

#### Meta-Review Synthesis:
Submission 1 is the strongest paper in this cohort, presenting a flawless intersection of Probabilistic Graphical Models (PGMs) and edge-serving systems. To resolve the accuracy-stability dilemma under dynamic streams (avoiding layer-wise routing jitter of stateless models and historical lag of stateful models), the authors model network depth as a 1D lattice and routing as a discrete Euclidean path integral. This maps the ensembling weight optimization to a 1D chain Markov Random Field (MRF), solved exactly via Forward-Backward belief propagation in $O(L K^2)$ linear time. 

To make this viable for resource-constrained edge hardware, they introduce **QPathMerge-Single**, recursively computing backward messages on-the-fly over a truncated horizon. They formally prove the exponential convergence of this truncation using **Dobrushin's contraction theorem** and address non-monotonicity with linear trend projections.

#### Key Strengths:
- **Outstanding Empirical Rigor:** Extremely robust statistical design, reporting means and standard deviations across **5 random seeds** (synthetic sandbox) and **3 random seeds** (physical pre-trained ResNet-18 validation).
- **Strong & Honest Baselines:** Evaluated against **seven state-of-the-art baselines** (including SABLE and ChemMerge). The authors audited prior stateful frameworks, evaluating them under a consistent protocol that preserves history, exposing their severe temporal lag.
- **Hardware-Level Profiling:** Profiles CPU/NPU latency, showing that QPathMerge-Single adds only **$1.35$ ms ($5.35\%$)** overhead on ResNet-18, and provides a clear physical argument of how spatial trajectory smoothing actively prevents DRAM memory transactions.

---

### 🏆 Submission 5: Unit Geodesic Routing (UGR) for Curved Manifold Ensembling
* **Final Ratings:** 6 (Strong Accept), 5 (Accept), 4 (Weak Accept)
* **Average Score:** 5.00
* **Decision:** **Accept**

#### Meta-Review Synthesis:
Submission 5 introduces an elegant geometric paradigm shift that rejects the unconstrained flat-space assumptions of previous temporal routers in favor of curved manifolds. Drawing on Information Geometry, the authors formulate stateful ensembling via Fisher-Rao geodesic flows on the probability simplex. By utilizing the square-root homeomorphism (Born's rule), they map the simplex to the positive orthant of a unit sphere, executing torque-driven Rodrigues-like spherical linear interpolation (Slerp) updates. This natively suppresses routing oscillations without overshoot, resolving high-dimensional measure concentration through local sub-manifold projections.

#### Key Strengths:
- **Exemplary Scientific Hygiene:** The authors conducted a detailed code-level audit of prior baselines. They discovered a critical initialization bug in *Momentum-Merge (Advanced)*—where the boundary prior was overwritten with the target vector, artificially suppressing its jitter—and corrected it to ensure a perfectly fair comparison.
- **Outstanding Statistical Confidence:** Evaluations are run across **10 independent random seeds** (sandbox) and **5 independent seeds** (real-world text classification), reporting complete standard deviations for all tables.
- **Completeness:** The extensive appendices provide analytical backpropagation gradients, positive orthant persistence proofs, and local sub-manifold routing formulations.

---

### 🏆 Submission 10: 2D Spatio-Temporal Bilinear Filter (2D-STEM)
* **Final Ratings:** 6 (Strong Accept), 6 (Strong Accept), 3 (Weak Reject)
* **Average Score:** 5.00
* **Decision:** **Accept**

#### Meta-Review Synthesis:
Submission 10 represents a major triumph of minimalist engineering, applying Occam's razor to deconstruct prior state-of-the-art stateful ensembling models (such as non-equilibrium biochemical kinetics in ChemMerge, or learned state-space models in PAC-Kinetics). The authors prove that the noise-reduction of these highly complex systems is driven primarily by local recursive filtering. Driven by this, they propose **2D-STEM** (2D Spatio-Temporal Exponential Moving Average), a zero-parameter, zero-training, and single-line 2D bilinear recursive filter. 

They provide an inductive mathematical proof that under a simple linear inequality constraint ($\beta_{\text{depth}} + \beta_{\text{temp}, t} \le 1$), the ensembling weights are analytically guaranteed to reside on the probability simplex, completely bypassing expensive Softmax or Euclidean projections. To eliminate transition lag under overlapping task manifolds, they introduce Adaptive Temporal Gating (ATG) with a cubic Power-Law Sharpening exponent ($\gamma = 3$) and formulate a Coordinate-Prior spatial boundary condition.

#### Discussion of the Reviewer Debate:
Reviewers 1 and 2 rated the paper as a **Strong Accept (6)**, praising its minimalist philosophy, zero-parameter overhead, and a massive **$49.5\%$ reduction in serving execution latency** relative to ChemMerge. However, Reviewer 3 gave a **Weak Reject (3)**, raising three core concerns:
1. *Representation Space Discrepancy:* 2D-STEM performs worse than PAC-Kinetics on a pre-trained ViT representation space ($63.70\%$ vs $70.57\%$ accuracy).
2. *Training-Free Contradiction:* The optional 2-layer MLP coordinate mapper introduced in Appendix B requires training, contradicting the "training-free" claim.
3. *Downstream Classification:* Lack of actual task accuracy evaluation of merged physical LoRA experts.

**Meta-Reviewer Resolution:**
The meta-reviewer overrides the Weak Reject in favor of acceptance. While Reviewer 3's points are valuable, they are addressable limitations rather than fundamental flaws:
- The ViT performance discrepancy is due to the extreme simplicity of 2D-STEM ($O(K \cdot L)$ bilinear filter with 0 parameters and 0 training) compared to PAC-Kinetics, which is a heavily parameterized state-space model requiring offline training and online backpropagation. A $49.5\%$ speedup and a microsecond-level execution latency make 2D-STEM far more deployable on real-world edge gateways.
- The 2-layer MLP mapper is explicitly presented as an *optional extension* for extremely fine-grained domains, meaning the core 2D-STEM filter remains completely training-free.
- The zero-overhead, projection-free, and branch-free nature of 2D-STEM allows it to be compiled into a single fused CUDA kernel (ONNX/TensorRT), which represents an exceptional, high-impact systems contribution that heavily outweighs the sandbox-to-physical evaluation gap.

---

## 3. Discussion on Competitive Non-Accepted Submissions

### ❌ Submission 4: Integer Apportionment & Activation Error Feedback (QA-Merge)
* **Ratings:** 4 (Weak Accept), 5 (Accept), 5 (Accept)
* **Average Score:** 4.67
* **Decision:** **Reject**

#### Comparative Analysis:
Submission 4 is a very strong, highly practical systems-level paper that maps dynamic model ensembling to low-precision edge-serving environments (INT8/INT4). It introduces a branchless, sorting-free Permutation-Invariant Single-Pass Apportionment (PI-SPA) algorithm to solve the integer apportionment bottleneck in vector pipelines, and validates it on a physical STM32 microcontroller (ARM Cortex-M7), achieving a **5.2x latency speedup and 42% power reduction**. 

While highly accomplished, it was ultimately not accepted because its core ensembling mathematics are relatively straightforward integer rounding feedback loops. Unlike Submissions 1, 5, and 10, which introduce fundamental, paradigm-shifting representations of depth ensembling trajectories (MRFs, Fisher-Rao geodesic flows, and 2D bilinear filters), Submission 4's contribution lies primarily in hardware optimization. Since we can accept only 3 papers, the higher theoretical innovation and structural novelty of Submissions 1, 5, and 10 were prioritized.

---

### ❌ Submission 7: Slot-Kinetics & Virtual Task Caching (TDSR)
* **Ratings:** 5 (Accept), 5 (Accept), 4 (Weak Accept)
* **Average Score:** 4.67
* **Decision:** **Reject**

#### Comparative Analysis:
Submission 7 addresses a critical systems bottleneck: state contamination in multi-tenant edge serving. When a stateful global router processes interleaved multi-user streams, user states contaminate each other. The authors resolve this by introducing **Slot-Kinetics state decoupling** and Virtual Task Caching (the Slot-Tenant-Task Triad), mapping user sessions to separate hardware slots on-the-fly and performing sub-nanosecond lookups. 

This is an exceptionally polished and practical paper. However, it was not accepted because the evaluation is conducted **entirely** within the synthetic Analytical Coordinate Sandbox (ICS) without any physical hardware or real-world LLM/PEFT framework validation (as noted by Reviewer 3). Because Submissions 1 (ResNet-18 physical validation), 5 (NLP text classification), and 10 (pre-trained ViT validation) successfully bridged the sandbox-to-physical gap in their evaluations, they were prioritized over Submission 7's synthetic-only validation.

---

### ❌ Submission 6: Control-Theoretic PID-Merge
* **Ratings:** 6 (Strong Accept), 3 (Weak Reject), 5 (Accept)
* **Average Score:** 4.67
* **Decision:** **Reject**

#### Comparative Analysis:
Submission 6 proposes a highly original concept of treating network depth ensembling as a closed-loop dynamical system, applying discrete-time PID control to weight trajectories. While exceptionally creative, Reviewer 2 identified a **fundamental mathematical omission in the stability analysis**: the Jury stability proof completely omits the fourth necessary and sufficient condition for 3rd-order discrete systems, rendering the corresponding optimization penalty $\mathcal{L}_{\text{stab}}$ mathematically incomplete. Furthermore, the linear time-invariant (LTI) assumption for a state-dependent, layer-varying plant gain is a severe oversimplification. Because these theoretical guarantees require revision before they can be built upon, the paper is rejected.
