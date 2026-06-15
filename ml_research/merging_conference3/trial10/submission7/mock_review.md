# Synthesized Mock Peer Review

## 1. Summary of the Submission

This paper addresses a critical, real-world deployment hurdle in test-time dynamic model merging: the **state contamination (cross-talk) bottleneck** of stateful routing policies. While stateful routers (e.g., *ChemMerge* and *PAC-Kinetics*) act as temporal low-pass filters to suppress sample-level feature noise and routing jitter, they are designed under the unrealistic assumption that query streams belong to a single, continuous user session. In high-throughput, multi-tenant cloud gateways or edge servers, requests are heavily interleaved. Consequently, standard stateful routers bleed memory states across unrelated tenants, causing incorrect expert blending, routing lag, and representation bleeding.

To resolve this major deployment blocker, the authors introduce **Tenant-Decoupled Stateful Routing (TDSR)**, powered by a **Slot-Kinetics** state-decoupling mechanism. TDSR maintains a highly compact, decoupled pool of virtual routing state slots and centroids. The authors propose two practical operation modes:
1. **Explicit Session Tagging (Metadata-Tagged):** Maps queries directly to virtual state slots based on system-provided session/tenant metadata with zero computational overhead.
2. **Implicit Tagless Clustering (Dynamic Inference):** Dynamically infers task affinity on-the-fly via online cosine similarity between query activation coordinates and fixed orthogonal detector centroids ($\mathbf{c}_{m, m} = 1.0$), eliminating centroid drift and clustering collapse.

To prevent memory washout of sparse user sessions under interleaved streams, the authors introduce **Tenant-Specific Session-Step Decay** (local decay). To release inactive memory and prevent leaks without garbage-collection overhead, they propose an elegant **Dual-Clock Decay** policy that holds states constant on logical steps while decaying inactive slots on a physical background wall-clock timer.

The authors evaluate TDSR inside a 14-layer high-fidelity **Analytical Coordinate Sandbox (ICS)**. Under realistic, interleaved multi-tenant streams, TDSR Explicit resolves state contamination, achieving up to **70.60%** classification accuracy on Orthogonal Manifolds (+1.90% over the contaminated Global PAC-Kinetics baseline) and **70.85%** on Overlapping Manifolds (within 0.50% of the isolated clean-stream baseline ceiling). Furthermore, their proposed **intra-session jitter** analysis shows that TDSR slashes high-frequency ensembling oscillations by up to **2.4$\times$** relative to stateless SABLE.

---

## 2. Key Strengths

1. **High Practical Relevance:** The paper is written with a commendable focus on real-world systems engineering and production constraints. Exposing the state contamination bottleneck in interleaved multi-tenant environments addresses a major practical blocker that previously made stateful model merging completely undeployable in production infrastructures (such as S-LoRA or Punica).
2. **Elegant Virtual Task Caching:** Under the implicit tagless clustering mode, utilizing fixed orthogonal detector centroids to cluster queries based on task affinity rather than physical user IDs represents an exceptionally clever design. This "Slot-Tenant-Task Triad" decouples the router's memory footprint from the scale of active concurrent tenants ($M$), mapping requests to a small, fixed pool of $K$ slots ($K \ll M$), allowing infinite scaling with zero memory explosion.
3. **Rigorous and Clean Metric Correction:** Prior work evaluated routing jitter globally over interleaved query sequences, resulting in high jitter metrics for any router tracking interleaved task context switches. The paper makes an important metrological correction by distinguishing **inter-session jitter** (which must be high to track active user task switches) from **intra-session jitter** (which correctly isolates and measures temporal smoothing within each user's context).
4. **Pragmatic Systems-Level Architecture Focus:** Incorporating systems considerations such as sub-microsecond latency (< 1.5 microseconds), register-level memory footprint (64 bytes for a $4 \times 4$ array), and a dual-clock decay policy to prevent session memory leaks makes the framework highly appealing to ML systems practitioners.
5. **Rigorous Baseline Comparison and Evaluation:** The paper reports mean and standard deviation metrics across **5 independent random seeds** on both orthogonal and overlapping manifold configurations, comparing against standard uniform, stateless SABLE, global PAC-Kinetics, and an Oracle clean-stream ceiling. The inclusion of clear pseudocode in Algorithm 1 makes the algorithmic pipeline highly reproducible.

---

## 3. Weaknesses and Areas for Improvement

While the paper is technically excellent and substantially complete, addressing the following limitations would elevate its scientific weight and clarity:

### A. Coordinate-Argmax Simplification in Implicit Mode (Conceptual Over-Engineering)
* **Critique:** In Section 3.2, the authors describe the "Implicit Tagless Clustering" mode by initializing and fixing the slot centroids $\mathcal{C}$ as orthogonal coordinate detector vectors ($\mathbf{c}_{m, m} = 1.0$), which corresponds to standard unit basis vectors $\mathbf{e}_m$. Consequently, the online cosine similarity simplifies mathematically to:
  $$\text{Sim}(\mathbf{e}_t, \mathbf{c}_m) = \frac{\mathbf{e}_t^T \mathbf{c}_m}{\|\mathbf{e}_t\|_2 \|\mathbf{c}_m\|_2 + \epsilon_{\text{sim}}} = \frac{e_{m, t}}{\|\mathbf{e}_t\|_2 \cdot 1.0 + \epsilon_{\text{sim}}}$$
  Since $\|\mathbf{e}_t\|_2 + \epsilon_{\text{sim}}$ is a positive scalar constant identical for all candidate slots $m \in \{0, \dots, M-1\}$, the argmax similarity operation in Equation 5 is mathematically identical to selecting the slot with the maximum raw coordinate value:
  $$m^*_t = \arg\max_{m \in \{0, \dots, M-1\}} e_{m, t}$$
  While this coordinate-argmax selection is highly practical and computationally optimal, framing it as an "Implicit Tagless Clustering via online cosine similarity against fixed centroids" is conceptually over-engineered and slightly oversold as unsupervised clustering. Acknowledging that this is mathematically a simple argmax coordinate assignment would improve scientific transparency and conceptual clarity.
* **Actionable Suggestion:** Explicitly show this mathematical simplification in Section 3.2. This shows reviewers that you are mathematically thorough and makes the actual system-level execution look even simpler and more elegant (by highlighting the zero-overhead lookup speed).

### B. Lack of Empirical Scaling and Dual-Clock Decay Evaluation
* **Critique:** While the proposed **Dual-Clock Decay** policy is conceptually elegant and resolves the logical contradiction of holding state indefinitely, it is not quantitatively evaluated in the experiments. Furthermore, the simulation experiments are currently restricted to a relatively small pool ($M = 4$ tenants and $K = 4$ tasks) with logical step decay ($\Delta t_m = 0$). Evaluating how the system behaves under larger concurrency scales (e.g., $M = 16, 64, 256$) under varying degrees of interleaving sparsity, and verifying the behavior of the secondary physical wall-clock timer (e.g., how the timeout threshold affects state recovery and eviction) would substantially strengthen the systems-oriented claims.
* **Actionable Suggestion:** Include a simulation experiment or sweep under larger $M$ (e.g., 16 and 64 tenants) and vary the sparsity of tenant requests. Provide a short discussion or plot of the impact of the physical timeout threshold on classification accuracy, confirming the sub-microsecond latency profile remains robust under larger state pools.

### C. Lack of Real-World LLM Serving Evaluation
* **Critique:** The entire quantitative evaluation is conducted within the simulated Analytical Coordinate Sandbox (ICS) environment. While the ICS is a high-fidelity and mathematically clean testbed that effectively isolates representation alignment dynamics, validating the framework on a real-world LLM serving framework (e.g., serving 4 task-specialized LoRAs on a base LLaMA-3-8B model using Punica or S-LoRA on actual user query streams) would dramatically increase the paper's empirical weight and appeal to systems-focused reviewers.
* **Actionable Suggestion:** Even a small-scale real-world test on a smaller base model (e.g., GPT-2 or LLaMA-3-8B) with standard datasets (such as GSM8K, Alpaca, and HumanEval) would validate that the coordinate projection and state decoupling dynamics generalize perfectly to actual Transformer activations.

---

## 4. Technical Ratings and Overall Recommendation

* **Soundness: Excellent**  
  The mathematical formulation is exceptionally rigorous. The diagonal recurrence matrix bounded within $(0,1)$ guarantees BIBO stability, and the temperature threshold $\tau_{\min} = 0.01$ ensures Lipschitz continuity. The choice to fix centroids as orthogonal detectors is an elegant and robust solution to online clustering collapse, and the Dual-Clock Decay policy solves the logical contradiction of infinite state carryover.
* **Presentation: Excellent**  
  The paper is beautifully written, logical, and highly professional. The tables are clean and comprehensive, and the visualizations in Figures 1 and 2 directly support the central claims. Algorithm 1 is structured exceptionally well.
* **Significance: Good**  
  By resolving the state contamination bottleneck with negligible overhead, this framework makes stateful dynamic model merging viable in multi-tenant gateways for the first time. The "Slot-Tenant-Task Triad" is a valuable systems contribution.
* **Originality: Excellent**  
  The identification of cross-tenant state contamination and the slot-based recurrence decoupling approach represent a significant and original contribution at the intersection of PEFT ensembling and MLSys.

**Overall Recommendation: Accept (5/6)**  
The paper is technically solid, addresses an important practical deployment blocker, and is highly complete. Resolving the minor suggestion regarding the argmax simplification and providing a small scalability discussion/experiment will make this an exceptionally strong paper.

---

## 5. Detailed Questions and Constructive Comments for the Authors

1. **Dual-Clock Decay Implementation:** In Section 3.6, you describe the "Dual-Clock Decay" policy where active slots use logical session steps and inactive slots use a secondary physical wall-clock timer for eviction. Is this dual-clock mechanism fully integrated and evaluated in your sandbox, or is it a conceptual design for production? If integrated, what is the impact of different physical timeout thresholds on the classification accuracy of sparse tenants?
2. **Soft Slot Assignment / Gumbel-Softmax Gating:** For the tagless implicit mode, you suggest implementing soft slot assignment as a future direction to resolve coordinate overlap on overlapping manifolds. Since a soft assignment distributes updates proportionally, wouldn't it introduce a minor form of state contamination (cross-talk) across slots, slightly compromising session isolation? Have you explored the math behind this trade-off?
3. **Integration with S-LoRA/Punica Schedulers:** In real-world serving frameworks, KV cache memory allocation is highly dynamic. If we map TDSR states as sequence-level metadata, can these states be bundled directly into the physical KV-cache block metadata, or should they reside in a separate register pool in the scheduler? It would be helpful to discuss this systems mapping in more detail in Section 4.5.
