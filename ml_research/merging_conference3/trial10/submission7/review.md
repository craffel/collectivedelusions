# Mock Peer Review

## 1. Summary of the Submission

This paper addresses a critical, real-world deployment hurdle in test-time dynamic model merging: the **state contamination (cross-talk) bottleneck** of stateful routing policies. While stateful routers (e.g., *ChemMerge* and *PAC-Kinetics*) act as temporal low-pass filters to suppress sample-level feature noise and routing jitter, they are designed under the unrealistic assumption that query streams belong to a single, continuous user session. In high-throughput, multi-tenant cloud gateways or edge servers, requests are heavily interleaved. Consequently, standard stateful routers bleed memory states across unrelated tenants, causing incorrect expert blending, routing lag, and representation bleeding.

To resolve this major deployment blocker, the authors introduce **Tenant-Decoupled Stateful Routing (TDSR)**, powered by a **Slot-Kinetics** state-decoupling mechanism. TDSR maintains a highly compact, decoupled pool of virtual routing state slots and centroids. The authors propose two practical operation modes:
1. **Explicit Session Tagging (Metadata-Tagged):** Maps queries directly to virtual state slots based on system-provided session/tenant metadata with zero computational overhead.
2. **Implicit Tagless Clustering (Dynamic Inference):** Dynamically infers task affinity on-the-fly via online cosine similarity between query activation coordinates and fixed orthogonal detector centroids ($\mathbf{c}_{m, m} = 1.0$), eliminating centroid drift and clustering collapse.

To prevent memory washout of sparse user sessions, the authors introduce **Tenant-Specific Session-Step Decay** (local decay). To release inactive memory and prevent leaks without garbage-collection overhead, they propose a **Dual-Clock Decay** policy.

The authors evaluate TDSR inside a 14-layer high-fidelity **Analytical Coordinate Sandbox (ICS)**. Under realistic, interleaved multi-tenant streams, TDSR Explicit resolves state contamination, achieving up to **70.60%** classification accuracy on Orthogonal Manifolds (+1.90% over the contaminated Global PAC-Kinetics baseline) and **70.85%** on Overlapping Manifolds (within 0.50% of the isolated clean-stream baseline ceiling). Furthermore, their proposed **intra-session jitter** analysis shows that TDSR slashes high-frequency ensembling oscillations by up to **2.4$\times$** relative to stateless SABLE.

---

## 2. Key Strengths

1. **High Practical Relevance:** The paper is written with a commendable focus on real-world systems engineering and production constraints. Exposing the state contamination bottleneck in interleaved multi-tenant environments addresses a major practical blocker that previously made stateful model merging completely undeployable in production infrastructures (such as S-LoRA or Punica).
2. **Elegant Virtual Task Caching:** Under the implicit tagless clustering mode, utilizing fixed orthogonal detector centroids to cluster queries based on task affinity rather than physical user IDs represents an exceptionally clever design. This "Slot-Tenant-Task Triad" decouples the router's memory footprint from the scale of active concurrent tenants ($M$), mapping requests to a small, fixed pool of $K$ slots ($K \ll M$), allowing infinite scaling with zero memory explosion.
3. **Rigorous and Clean Metric Correction:** Prior work evaluated routing jitter globally over interleaved query sequences, resulting in high jitter metrics for any router tracking interleaved task context switches. The paper makes an important metrological correction by distinguishing **inter-session jitter** (which must be high to track active user task switches) from **intra-session jitter** (which correctly isolates and measures temporal smoothing within each user's context).
4. **Systems-Level Architecture Focus:** Incorporating systems considerations such as sub-microsecond latency (< 1.5 microseconds), register-level memory footprint (64 bytes for a $4 \times 4$ array), and a dual-clock decay policy to prevent session memory leaks makes the framework highly appealing to ML systems practitioners.
5. **Statistical Rigor and Reproducibility:** The paper reports mean and standard deviation metrics across **5 independent random seeds** on both orthogonal and overlapping manifold configurations. The inclusion of clear pseudocode in Algorithm 1 makes the algorithmic pipeline highly reproducible.

---

## 3. Weaknesses and Critical Areas for Improvement

While the paper is technically excellent and substantially complete, addressing the following limitations would elevate its scientific weight and clarity:

### A. Lack of Real-World LLM serving Evaluation
* **Critique:** The entire quantitative evaluation is conducted within the simulated Analytical Coordinate Sandbox (ICS) environment. While the ICS is a high-fidelity and mathematically clean testbed that effectively isolates representation alignment dynamics, validating the framework on a real-world LLM serving framework (e.g., serving 4 task-specialized LoRAs on a base LLaMA-3-8B model using Punica or S-LoRA on actual user query streams) would dramatically increase the paper's empirical weight and appeal to systems-focused reviewers.
* **Actionable Suggestion:** Even a small-scale real-world test on a smaller base model (e.g., GPT-2 or LLaMA-3-8B) with standard datasets (such as GSM8K, Alpaca, and HumanEval) would validate that the coordinate projection and state decoupling dynamics generalize perfectly to actual Transformer activations.

### B. Scaling of the State Pool
* **Critique:** The simulation experiments are currently restricted to a relatively small pool ($M = 4$ tenants and $K = 4$ tasks). While this is sufficient to prove the core concept, evaluating how the system behaves under larger concurrency scales (e.g., $M = 16, 64, 100$) under varying degrees of interleaving sparsity would demonstrate the practical scalability of the Tenant-Specific Session-Step Decay policy under extremely sparse workloads.
* **Actionable Suggestion:** Run a quick simulation sweep under larger $M$ (e.g., 16 and 64 tenants) and include a small paragraph in the discussion describing how performance or memory scaling scales with the tenant count, confirming the sub-microsecond latency profile.

### C. Logical Inconsistency on Inactive Slot Decay
* **Critique:** There is a minor mathematical and logical discrepancy between Section 3.3 (Methodology) and Section 3.6 (Systems Analysis) regarding inactive slot decay:
  * In Section 3.3, the authors state that in their evaluation, they set $\Delta t_m = 0$ for inactive slots, meaning inactive slots hold their state perfectly between active queries and do not decay.
  * In Section 3.6, the authors praise the "automatic and continuous passive exponential decay" ($A_{\text{decay}}$) as a "self-cleaning property" that prevents memory leaks without active garbage collection.
  * These two statements are contradictory: if $\Delta t_m = 0$ when inactive, there is no passive decay for inactive slots, meaning stale states will linger indefinitely unless a separate eviction policy is implemented.
* **Actionable Suggestion:** Clarify this contradiction. Either explain that global step-wise decay is a theoretical alternative while logical step-decay was used in evaluation, or propose a concrete eviction policy (such as Least-Recently-Used) that handles inactive tenant memory in actual production deployments.

---

## 4. Technical Ratings and Overall Recommendation

* **Soundness: Excellent**  
  The mathematical formulation is exceptionally rigorous. The diagonal recurrence matrix bounded within $(0,1)$ guarantees BIBO stability, and the temperature threshold $\tau_{\min} = 0.01$ ensures Lipschitz continuity. The choice to fix centroids as orthogonal detectors is an elegant and robust solution to online clustering collapse.
* **Presentation: Excellent**  
  The paper is beautifully written, logical, and highly professional. The tables are clean and comprehensive, and the visualizations in Figures 1 and 2 directly support the central claims. Algorithm 1 is structured exceptionally well.
* **Significance: Good**  
  By resolving the state contamination bottleneck with negligible overhead, this framework makes stateful dynamic model merging viable in multi-tenant gateways for the first time. The "Slot-Tenant-Task Triad" is a valuable systems contribution.
* **Originality: Excellent**  
  The identification of cross-tenant state contamination and the slot-based recurrence decoupling approach represent a significant and original contribution at the intersection of PEFT ensembling and MLSys.

**Overall Recommendation: Accept (5/6)**  
The paper is technically solid, addresses an important practical deployment blocker, and is highly complete. Resolving the minor logical discrepancy regarding inactive slot decay will make this an exceptionally strong paper.

---

## 5. Detailed Questions and Constructive Comments for the Authors

1. **Dual-Clock Decay Implementation:** In Section 3.6, you describe the "Dual-Clock Decay" policy where active slots use logical session steps and inactive slots use a secondary physical wall-clock timer for eviction. Is this dual-clock mechanism fully integrated and evaluated in your sandbox, or is it a conceptual design for production? If integrated, what is the impact of different physical timeout thresholds on the classification accuracy of sparse tenants?
2. **Soft Slot Assignment / Gumbel-Softmax Routing:** For the tagless implicit mode, you suggest implementing soft slot assignment as a future direction to resolve coordinate overlap on overlapping manifolds. Since a soft assignment distributes updates proportionally, wouldn't it introduce a minor form of state contamination (cross-talk) across slots, slightly compromising session isolation? Have you explored the math behind this trade-off?
3. **Integration with S-LoRA/Punica:** In real-world serving frameworks, KV cache memory allocation is highly dynamic. If we map TDSR states as sequence-level metadata, can these states be bundled directly into the physical KV-cache block metadata, or should they reside in a separate register pool in the scheduler? It would be helpful to discuss this systems mapping in more detail in Section 4.5.
