# Evaluation: Soundness and Methodology

## Clarity of Description
The methodology is exceptionally clear, precise, and well-structured.
- **Mathematical Formulations:** Every step of the pipeline—from coordinate projection (Eq. 1-2) and implicit/explicit slot assignment (Eq. 3-8) to recurrent updates (Eq. 9-10) and Gibbs Softmax mapping (Eq. 11-13)—is formalized with exact mathematical notation.
- **Simplification Insight:** The authors explain the step-by-step mathematical reduction of online cosine similarity against orthogonal centroids (Eq. 5-6) into a simple, sub-nanosecond coordinate-argmax assignment (Eq. 8). This transparent explanation bridges high-level clustering theory and register-level execution.
- **Algorithmic Pseudocode:** Algorithm 1 provides a detailed, line-by-line pipeline of the entire Tenant-Decoupled Stateful Routing (TDSR) framework. It perfectly aligns with the text description, specifying inputs, outputs, and logical transitions.
- **Systems Integration Details:** Section 4.5 provides outstanding clarity on how the virtual slots should be integrated with real-world continuous batching engines and scheduler-level register pools, demonstrating strong systems-level engineering insight.

## Appropriateness of Methods
The methods proposed are highly appropriate, creative, and tailored to the practical constraints of modern serving systems:
- **Subspace Projections:** Projecting early-layer activations onto pre-computed PCA task-specific subspaces to extract coordinates is a standard, theoretically-grounded technique for dynamic ensembling.
- **Fixed Orthogonal Centroids:** Fixing centroids as orthogonal task detectors is an extremely clever way to completely bypass the traditional failures of unsupervised online clustering (clustering collapse and runaway slot attraction) without requiring ground-truth labels during deployment.
- **Virtual Task Caching (Triad):** Grouping queries by task affinity in the tagless mode decouples the slot pool size from the concurrency scale ($K \ll M$). This is highly appropriate for high-throughput cloud environments because it keeps the stateful memory footprint constant regardless of the number of concurrent users.
- **Local (Session-Step) Decay & Dual-Clock decay:** Maintaining a constant state for inactive slots prevents premature state washout under sparse interleaved traffic. Resolving this with a dual-clock physical timeout for stale session eviction is mathematically sound and prevents memory leaks.
- **Lipschitz Continuity:** Defining task-specific temperatures with a strict minimum threshold ($\tau_{\min} = 0.01$) ensures Lipschitz continuity, preventing division-by-zero or numerical instabilities in the Gibbs Softmax policy.

## Potential Technical Flaws and Limitations
The authors are commendably transparent about the limitations of their work, which we evaluate below:
1. **The Task-Transition Tracking Failure in Implicit Mode:** In the implicit tagless clustering mode, centroids are bound to task-expert domains. When a user transitions from Task 1 to Task 2, their consecutive queries are routed to separate slots (Slot 1 and Slot 2). Consequently, the temporal stateful smoothing is localized *within-task* rather than *within-session*, rendering the router effectively stateless at the moment of task transitions. This is a fundamental limitation of tagless routing. The authors openly acknowledge and discuss this, noting that the explicit mode is required to maintain perfect sequence-level smoothing across task switches.
2. **Coordinate Contamination under Overlapping Manifolds:** Under the overlapping manifold configuration, shared dimensions interfere with coordinate projection accuracy. This causes TDSR Implicit to occasionally assign queries to incorrect slots, resulting in a minor performance drop (-0.70% compared to TDSR Explicit Local). The authors' proposed solutions (Soft Slot Assignment or Dynamic Centroid Learning) are theoretically sound, though they correctly note that soft assignment introduces a fundamental trade-off: distributing updates across slots can introduce cross-talk, compromising the session isolation.
3. **Sandbox Simulation Constraints:** The Analytical Coordinate Sandbox (ICS) is a simplified simulation environment that uses linear coordinate spaces and Gaussian/orthogonal manifolds. While highly effective for isolating representational alignment dynamics and verifying state contamination, it does not evaluate full text-generation workloads or complex non-Gaussian activation noise of a physical LLM in production. However, the authors' detailed systems-level discussion in Section 4.5 mitigates this by bridging the gap to real-world frameworks.

## Reproducibility
The reproducibility of this methodology is **excellent**:
- The paper provides exact parameter values: depth $L = 14$ layers, dimension $D = 192$, routing layer $l_{\text{route}} = 3$, experts $K = 4$, and slots $M = 4$.
- The test stream setups (calibration stream $N_{\text{cal}} = 100$, test stream $N_{\text{test}} = 400$, interleaving patterns) are clearly defined.
- The mathematical formulations and Algorithm 1 are detailed and complete, making it highly straightforward for an expert reader to re-implement and reproduce the results.
- The experimental results are evaluated across 5 independent random seeds with reported standard deviations, demonstrating statistical rigor.
