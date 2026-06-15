# Experimental Evaluation Check

## Experimental Setup
The experimental evaluation is highly detailed and structurally rigorous:
- **Environment:** The evaluations are performed inside the 14-layer high-fidelity **Analytical Coordinate Sandbox (ICS)**. Although a simulation environment rather than real-world large language models (LLMs) on standard benchmarks, the sandbox environment is highly appropriate here as it isolates coordinate and representation alignment dynamics from confusing downstream generation noise.
- **Configurations:** The paper evaluates across two manifold geometry settings: Orthogonal Manifolds (disjoint expert subspaces) and Overlapping Manifolds (introducing inter-task coordinate interference).
- **Tenant-Task Decoupling:** In early sandbox configurations, tenant IDs were conflated with task IDs (`tenant_id == task_id`). Commendably, the authors broke this conflation in their updated evaluation, running randomized streams where each tenant's target task shifts dynamically and independently over time.
- **Statistical Significance:** All dynamic routing and baseline methods are evaluated across **5 independent random seeds** with standard deviations reported. This resolves potential seed-noise constraints and confirms statistical significance.

## Baselines
The paper compares TDSR against an appropriate, comprehensive set of baselines:
1. **Static Uniform Merging:** A static baseline providing a reference point for non-dynamic expert blending.
2. **Stateless SABLE:** The state-of-the-art stateless dynamic router. Highly responsive but prone to extreme jitter.
3. **Global PAC-Kinetics:** The standard global stateful ensembling baseline, which is contaminated under multi-tenant streams.
4. **Oracle Stateful Routing (Clean-Stream Ceiling):** A theoretical upper-bound where tenants are served in isolated, clean sequential streams.

This baseline suite covers all relevant paradigms: static, stateless dynamic, and global stateful, allowing a clean ablation of the proposed tenant-decoupled approach.

## Do the Results Support the Claims?
Yes, the quantitative results strongly and consistently support all central claims:
1. **State Contamination exists and is resolved:** Tables 1 and 2 show that Global PAC-Kinetics loses significant accuracy compared to Oracle. TDSR Explicit recovers this performance, outperforming Global PAC-Kinetics by **+1.90%** (Orthogonal) and **+1.75%** (Overlapping), and performing within 0.50% of the Oracle ceiling.
2. **Intra-Session Routing Stability:** The paper proves that TDSR Explicit slashes intra-session routing jitter by up to **2.4$\times$** relative to SABLE (dropping from ~0.55 down to ~0.23 in Table 1). 
3. **Implicit Mode Robustness:** TDSR Implicit achieves strong results, outperforming Global PAC-Kinetics and Stateless SABLE on both orthogonal and overlapping manifolds, validating the slot-kinetics task-specific virtual caching concept.
4. **Systems Scalability:** Table 3 (Concurrency Scaling Sweep) confirms that TDSR scales up to 256 tenants, consistently outperforming Global PAC-Kinetics. It also shows a sub-microsecond register-level design profile, as the unoptimized CPU PyTorch forward pass takes only 67 microseconds at M=256 concurrency.
5. **Memory Reclamation via Dual-Clock Decay:** Table 4 (Timeout Sweep) verifies that background physical timers successfully evict inactive slots to prevent memory leaks with negligible accuracy drop (74.90% vs 75.10% at a 100-step threshold).
