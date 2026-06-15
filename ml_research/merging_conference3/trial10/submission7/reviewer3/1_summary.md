# Paper Evaluation: 1_summary.md

## 1. Main Topic of the Paper
The paper addresses a critical deployment challenge in **test-time dynamic model merging** (expert ensembling) for Parameter-Efficient Fine-Tuning (PEFT) adapters (like LoRA). Specifically, it targets **stateful or kinetic routers** (such as ChemMerge and PAC-Kinetics) which act as temporal low-pass filters to smooth routing ensembling weight trajectories and suppress sample-level feature noise. While mathematically elegant, existing stateful routers assume a continuous, highly correlated single-user stream. Under realistic production systems (such as high-throughput multi-tenant LLM gateways or edge servers), the workloads are interleaved across independent users. The paper exposes and analyzes the **state contamination (cross-talk) bottleneck**, where standard stateful routers bleed memory states across unrelated tenants, causing catastrophic routing lag, representational bleeding, and severe serving accuracy drops (up to 9.0% absolute classification accuracy drop).

---

## 2. Proposed Approach
To resolve this deployment blocker, the paper introduces **Tenant-Decoupled Stateful Routing (TDSR)**, also known as **Slot-Kinetics**. Instead of a single global state, TDSR maintains a highly compact pool of virtual routing state slots and proposes two highly pragmatic session routing modes:
1. **Explicit Session Tagging (Metadata-Aware):** If the serving framework (e.g., S-LoRA, Punica) provides tenant metadata, the router maps the query directly to its corresponding state slot, achieving perfect isolation with zero computational overhead.
2. **Implicit Tagless Clustering (Dynamic Inference):** When metadata is unavailable, the router dynamically infers the session context on-the-fly. By computing online cosine similarity between the current query coordinate vector and fixed orthogonal coordinate detector centroids ($\mathbf{c}_{m, m} = 1.0$), the assignment mathematically simplifies to a simple **coordinate-argmax assignment** (selecting the slot corresponding to the task with the maximum activation coordinate). This eliminates vector dot-products, norm computations, and division, allowing slot assignment to be implemented as a single, sub-nanosecond integer index lookup.

### Additional Practical Systems Mechanisms:
- **Tenant-Specific Session-Step Decay:** Inactive slots do not decay on global steps. Instead, a slot's state is held constant ($\Delta t_m = 0$) or decayed on its own logical session steps to prevent state washout in sparse workloads.
- **Dual-Clock Decay:** Combines logical clock decay during active interleaved serving with a secondary background physical wall-clock timer (e.g., evicting/decaying a session if it remains inactive for longer than a specified timeout like 5 seconds) to prevent memory leaks of obsolete sessions in register memory.
- **Gibbs Softmax Policy Mapping:** Maps the winning slot's updated state to ensembling weights via a multi-temperature Gibbs Softmax policy.

---

## 3. Key Findings and Claims
- **Exposing State Contamination:** Standard stateful routing (Global PAC-Kinetics) under interleaved workloads suffers from state contamination, dropping up to 5.00% absolute accuracy compared to the Oracle clean stream.
- **Accuracy Improvements:** TDSR Explicit completely resolves state contamination, achieving up to **70.60%** classification accuracy on Orthogonal Manifolds (outperforming contaminated Global PAC-Kinetics by **+1.90%** absolute) and **70.85%** on Overlapping Manifolds (outperforming contaminated Global PAC-Kinetics by **+1.75%** absolute, within 0.50% of the isolated clean-stream baseline).
- **Intra-Session Routing Stability:** TDSR Explicit slashes high-frequency routing jitter (intra-session jitter) by up to **2.4$\times$** relative to stateless SABLE, recovering near-Oracle routing stability.
- **Microscopic Resource Footprint:** Storing a small state pool requires a microscopic $M \times K$ tensor array (64 bytes for $M=4, K=4$), allowing the state to be pinned in CPU/GPU registers or fast L1 cache, completing in **less than 1.5 microseconds** per query without any database, disk, or network lookups.
- **Scalability and Self-Cleaning Validity:** Under high-concurrency scaling sweeps (up to $M=256$ tenants), TDSR Local consistently outperforms Global PAC-Kinetics by up to **+1.60%** absolute accuracy with minimal latency overhead (67.39 microseconds in unoptimized PyTorch CPU sweeps). Sweeping the Dual-Clock timeout threshold confirms that a background physical timer successfully purges stale, inactive states to prevent memory leaks with negligible accuracy degradation.
