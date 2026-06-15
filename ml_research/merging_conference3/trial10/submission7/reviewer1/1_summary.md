# Paper Summary

## Main Topic
The paper addresses the **state contamination (cross-talk) bottleneck** in test-time dynamic model merging and expert adapter ensembling under multi-tenant workloads. Standard stateful ensembling methods (e.g., PAC-Kinetics) maintain a single global routing state, which behaves correctly under homogeneous single-user streams but suffers severe routing lag, activates incorrect adapters, and incurs catastrophic accuracy drops when multi-tenant queries are interleaved. To resolve this blocker, the paper proposes **Tenant-Decoupled Stateful Routing (TDSR)**, also known as **Slot-Kinetics**.

## Approach
The TDSR framework introduces a pool of virtual routing states (slots) to isolate temporal smoothing dynamics within respective tenant or session contexts. Two distinct deployment modes are presented:
1. **Explicit Session Tagging:** When tenant metadata or session IDs are provided by the serving infrastructure (e.g., S-LoRA, Punica), the router retrieves and updates the corresponding virtual state slot directly with zero computational overhead.
2. **Implicit Tagless Clustering:** When session metadata is unavailable, the router dynamically infers the session context on-the-fly. This is achieved by computing online cosine similarity between the activation coordinate vector and fixed orthogonal task detector centroids. Crucially, the authors show that this online cosine similarity simplifies mathematically to a highly efficient **coordinate-argmax assignment** ($m^*_t = \arg\max_{m} e_{m, t}$).

Once the slot is assigned, its state is updated via a first-order diagonal recurrence. To prevent memory washout of sparse tenants, the authors introduce **Tenant-Specific Session-Step Decay** (local decay), where inactive slots hold their state constant during global steps. For production deployments, they propose a **Dual-Clock Decay** policy that integrates physical wall-clock timers to evict obsolete sessions and prevent memory leaks. The active slot's state is mapped to ensembling weights via a multi-temperature Gibbs Softmax policy.

## Key Findings
1. **State Contamination Exists:** Global PAC-Kinetics loses significant classification accuracy under rapidly interleaved multi-tenant streams due to cross-talk between unrelated queries.
2. **TDSR Recovers Performance:** Under interleaved workloads inside the high-fidelity Analytical Coordinate Sandbox (ICS), TDSR Explicit recovers performance close to the clean-stream Oracle baseline. Specifically, under Orthogonal Manifolds, TDSR Explicit achieves **70.60%** accuracy (outperforming Global PAC-Kinetics by **+1.90%** absolute). Under Overlapping Manifolds, TDSR Explicit achieves **70.85%** accuracy (performing within 0.50% of the isolated clean-stream baseline).
3. **Drastic Jitter Reduction:** Under intra-session routing jitter analysis, TDSR Explicit slashes high-frequency activation noise, reducing intra-session jitter by **2.4$\times$** relative to the stateless SABLE baseline.
4. **Systems Efficiency:** Storing slot states requires a negligible memory footprint (64 bytes for a $4 \times 4$ array), allowing register-level pinning. Execution of updates and assignments completes in under **1.5 microseconds**, introducing zero disk/network lookup overhead.

## Explicitly Claimed Contributions (with Evidence)
- **Conceptual Formulation of the State Contamination Bottleneck:** Demonstrating that standard stateful ensembling degrades heavily under multi-tenant interleaved workloads (supported by quantitative comparisons in Section 4.3).
- **Tenant-Decoupled Stateful Routing (Slot-Kinetics) Framework:** Proposing TDSR to isolate temporal dynamics across sessions, with both explicit and implicit modes (Section 3).
- **Coordinate-Argmax Simplification:** Exposing that implicit tagless clustering against fixed basis vectors simplifies to a sub-nanosecond coordinate-argmax assignment, avoiding expensive vector calculations (Section 3.2).
- **Tenant-Specific Session-Step Decay & Dual-Clock Decay:** Formalizing local clock decay to prevent premature state washout of sparse tenant queries while enabling physical-timer based memory reclamation (Section 3.3).
- **Statistical Disentanglement of Jitter:** Exposing that global inter-session jitter must remain high to track interleaved streams, and proposing intra-session jitter to correctly evaluate temporal low-pass filtering stability (Section 4.4).
- **Empirical Validation:** Rigorous evaluation across 5 seeds on the ICS showing significant improvements (+1.90% orthogonal, +1.75% overlapping) and verifying scalability up to 256 concurrent tenants (Section 4.3 and Section 4.6).
