# Novelty and Originality Check

## 1. Conceptual Novelty
The core conceptual contribution of this paper—identifying and addressing the **state contamination (cross-talk) bottleneck** in stateful dynamic model merging—is highly original and of significant practical importance. 

While the machine learning community has recently seen major advancements in:
1. **Stateful routing policies** (e.g., *ChemMerge* \cite{chemmerge25}, *PAC-Kinetics* \cite{pac_zca_2026}) that use temporal smoothing to suppress representation jitter, and
2. **Multi-tenant deep learning serving infrastructures** (e.g., *Punica* \cite{chen2023punica}, *S-LoRA* \cite{s-lora}) that optimize low-level memory layout and continuous batching,

these two lines of research have remained completely siloed. Stateful routers have been designed under an unrealistic, isolated single-user assumption. Meanwhile, multi-tenant serving infrastructures have focused entirely on GPU execution efficiency (GEMM tiling, PagedAttention) while being agnostic to the mathematical recurrence of dynamic ensembling. This paper is the first to explicitly bridge this gap, exposing how interleaved workloads from independent users corrupt global stateful ensembling, rendering them less effective than stateless baselines in actual deployments.

---

## 2. Methodological Originality
While slot-based state tracking is a known pattern in traditional systems architecture, its mathematical and algorithmic formulation within the context of PEFT and dynamic model merging introduces several highly innovative features:
- **The Slot-Tenant-Task Triad (Virtual Task Caching):** In the implicit tagless mode, utilizing fixed orthogonal coordinate detector centroids ($\mathbf{c}_{m, m} = 1.0$) to group queries based on task-affinity rather than physical session IDs is an exceptionally elegant design. It decouples the router's memory footprint from the scale of concurrent tenants ($M$), mapping queries to a fixed pool of $K$ task slots (where $K \ll M$). This allows scaling to thousands of concurrent users with zero memory explosion while still preventing cross-talk between unrelated tasks.
- **Tenant-Specific Session-Step Decay (Local Decay):** Formulating a decay rate that is conditioned on a slot's own local active clock (rather than global serving steps) is a pragmatic and clever solution to the sparse-query memory washout problem in high-concurrency settings.
- **Separating Inter-Session vs. Intra-Session Jitter:** Prior work on stateful routing evaluated jitter globally over consecutive serving steps. This paper exposes a major statistical misinterpretation: in interleaved streams, a successful router *must* change weights between interleaved user queries, resulting in high global (inter-session) jitter. Isolating true smoothing via **intra-session jitter** is a key conceptual correction that future papers in this sub-area will build upon.

---

## 3. Differentiation from Prior Work
The paper positions itself very clearly and fairly against existing work:
- Unlike **SABLE** (stateless), TDSR retains temporal history to suppress activation-level noise, slashing intra-session jitter by up to $2.4\times$.
- Unlike **PAC-Kinetics** or **ChemMerge** (global stateful), TDSR maintains virtual slots to isolate temporal dynamics, avoiding cross-talk.
- Unlike **S-LoRA / Punica** (infrastructure-focused), TDSR operates at the routing scheduler layer and is fully complementary to their low-level batching/scheduling engines.

Overall, the novelty is high, and the paper makes significant, well-contextualized conceptual and algorithmic contributions that go far beyond a simple incremental adaptation of PAC-Kinetics.
