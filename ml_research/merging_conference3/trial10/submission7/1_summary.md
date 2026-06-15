# Summary of the Submission

## 1. Context and Core Problem
The paper, **"Tenant-Decoupled Stateful Routing: Resolving State Contamination in Multi-Tenant Serving of Dynamic Model Merging,"** addresses a critical gap between the theoretical design and the practical deployment of stateful ensembling routers in Parameter-Efficient Fine-Tuning (PEFT) and dynamic model merging systems. 

Recently, stateful/kinetic routers (such as *ChemMerge* and *PAC-Kinetics*) have been introduced to smooth routing weight trajectories, acting as temporal low-pass filters to suppress high-frequency sample-level representation noise and routing jitter. However, these frameworks rely on a highly limiting assumption: that incoming queries belong to a continuous, single-user stream. 

In actual production environments (e.g., LLM gateways, cloud serving infrastructures), workloads are multi-tenant and heavily interleaved across independent users. Deploying standard stateful routers in these settings leads to the **state contamination (cross-talk) bottleneck**, where the global routing state $\mathbf{s}_t$ is blended and contaminated across unrelated tenants, resulting in high routing lag, representational bleeding, and catastrophic accuracy drops.

---

## 2. Proposed Solution: Tenant-Decoupled Stateful Routing (TDSR)
To resolve the state contamination bottleneck, the authors introduce **Tenant-Decoupled Stateful Routing (TDSR)**, also known as **Slot-Kinetics**. TDSR decouples the temporal smoothing dynamics across tenant contexts by maintaining a pool of $M$ active *state slots* $\mathcal{S} = \{\mathbf{s}_0, \dots, \mathbf{s}_{M-1}\}$ and centroids $\mathcal{C} = \{\mathbf{c}_0, \dots, \mathbf{c}_{M-1}\}$. 

The key architectural components of TDSR are:
1. **Subspace Coordinate Projection:** Normalizes early boundary-layer activations and projects them onto task-specific subspaces (derived via offline PCA) to form a coordinate vector $\mathbf{e}_t \in [0, 1]^K$, representing task-affinity profiles.
2. **Multi-Tenant State Decoupling:**
   - **Explicit Session Tagging (Metadata-Tagged):** When tenant session IDs or metadata are provided (as in S-LoRA or Punica), the router assigns queries directly to their corresponding slot with zero overhead.
   - **Implicit Tagless Clustering (Dynamic Inference):** When metadata is unavailable, the router assigns queries by computing the online cosine similarity between $\mathbf{e}_t$ and fixed orthogonal task detector centroids ($\mathbf{c}_{m, m} = 1.0$), avoiding centroid drift and clustering collapse.
3. **Stateful Recurrence with Flexible Decay:**
   - **Active update:** The selected slot is updated via a first-order recurrence: $\mathbf{s}_{m^*_t, t} = \mathbf{A} \mathbf{s}_{m^*_t, t-1} + W \mathbf{e}_t$.
   - **Inactive decay:** Inactive slots undergo either **Global Decay** (passive decay on every step) or **Tenant-Specific Session-Step Decay** (local decay, where inactive states are held constant between their own local active steps to prevent memory washout, i.e., setting $\Delta t_m = 0$ when inactive).
4. **Gibbs Softmax Policy Mapping:** The active slot's state is converted to ensembling weights $\boldsymbol{\alpha}_t$ using a multi-temperature Softmax policy.

---

## 3. Main Contributions
- **Exposes State Contamination:** Identifies and analyzes the cross-talk bottleneck that degrades stateful model merging under interleaved workloads.
- **Formulates Slot-Kinetics (TDSR):** Introduces a zero-overhead decoupling mechanism supporting both metadata-tagged (Explicit) and tagless (Implicit) stream partitioning.
- **Introduces Tenant-Specific Session-Step Decay:** Solves the sparse-query memory washout problem in high-throughput cloud settings.
- **Disentangles Jitter Metrics:** Clarifies that global inter-session jitter is mathematically forced to be high to track rapid user task switches, and introduces **intra-session jitter** to isolate and measure true temporal smoothing within a tenant's context.
- **Empirical Validation:** Demonstrates near-Oracle classification accuracy (70.60% ± 2.81% on Orthogonal Manifolds, outperforming contaminated Global PAC-Kinetics by +1.90% absolute) and up to a **2.4$\times$ stability improvement** (slashing intra-session jitter from 0.552 to 0.232) over stateless SABLE in the high-fidelity Analytical Coordinate Sandbox (ICS) across 5 independent seeds.
