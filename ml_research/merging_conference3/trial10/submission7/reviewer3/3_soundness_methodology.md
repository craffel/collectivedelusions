# Paper Evaluation: 3_soundness_methodology.md

## 1. Clarity of the Description
The methodology description is **excellent and highly transparent**. 
- The paper details each mathematical formulation clearly, from the intermediate feature normalization and PCA projection to the stateful diagonal recurrence update and the final multi-temperature Gibbs Softmax policy mapping.
- The mathematical derivation showing how the online cosine similarity with fixed orthogonal centroids simplifies to a simple **coordinate-argmax assignment** (Eq. \ref{eq:cosine_sim} to Eq. \ref{eq:argmax_coord}) is elegant and easy to follow.
- **Algorithm 1 (Tenant-Decoupled Stateful Routing)** provides clear, step-by-step pseudocode that outlines the full execution flow of a single incoming query, making the algorithm highly accessible for systems developers.

---

## 2. Appropriateness of Methods
The proposed techniques are highly appropriate and reflect a deep understanding of practical systems constraints:
- **Explicit Mode:** Leverages existing metadata-aware scheduling frameworks (like Punica, S-LoRA) to perform zero-overhead, perfect context-switching. This is highly aligned with modern enterprise cloud designs.
- **Implicit Mode with Fixed Orthogonal Centroids:** Unsupervised clustering is notoriously unstable and prone to collapse or drift during online inference. Keeping centroids fixed as orthogonal task detectors is an exceptionally practical and appropriate design choice to guarantee stability.
- **Tenant-Specific Session-Step Decay:** Standard global exponential decay is completely inappropriate for sparse workloads because a tenant's state is washed out by other users' traffic. Restricting state decay to a slot's active logical session steps ($\Delta t_m = 0$ for inactive slots) is mathematically sound and highly appropriate for interleaved settings.
- **Dual-Clock Decay:** Memory leak is a fatal issue in multi-session servers. Introducing a background physical clock to evict slots inactive for longer than a specified timeout (e.g., 5 seconds) is a very robust and production-ready solution to reclaim register memory.

---

## 3. Potential Technical Flaws and Limitations

### A. Non-Orthogonal Manifolds (Coordinate Interference)
In implicit mode, the slot assignment depends on the PCA coordinate vector $\mathbf{e}_t$. If the underlying expert task subspaces have significant overlap (non-orthogonal manifolds), the task coordinates suffer from representation-space interference. This contamination degrades PCA task coordinate accuracy, leading to hard misrouting where queries are assigned to incorrect slots. The authors honestly acknowledge this limitation, noting that TDSR Implicit trails TDSR Explicit by a minor 0.70% drop on Overlapping Manifolds. 

### B. The Task-Transition Stateful Tracking Failure in Implicit Mode
Under implicit tagless mode, because slots specialize in task domains rather than physical tenants, a single user's sequence that transitions across task boundaries (e.g., from Task 1 to Task 2) is split across separate slots. As a result, the router has zero temporal memory of the user's previous query at the moment of transition, rendering it effectively stateless during task switches. In contrast, TDSR Explicit maintains perfect sequence-level smoothing across task switches because virtual slots are bound to physical tenant session IDs. This represents a fundamental limitation of the implicit mode that is transparently exposed by the authors.

### C. Sensitivity of Calibration
The diagonal state retention matrix $\mathbf{A}$, the injection coupling matrix $W$, and log-temperatures $\mathbf{w}$ are trained on a joint calibration stream of $N_{\text{cal}} = 100$ samples. While the paper shows robust generalization when scaling concurrent tenants to $M=256$ at test-time, there is a risk of calibration sensitivity. If the calibration stream is highly homogeneous or lacks representation of specific tasks, the learned coupling weights might become biased, causing gating collapse at scale. Incorporating a load-balancing loss ($\mathcal{L}_{\text{balance}}$) during calibration helps mitigate this, which is a commendable addition.

---

## 4. Reproducibility
- **Mathematical Completeness:** The paper provides complete mathematical formulations for all steps, enabling an expert reader to reproduce the core algorithms.
- **Algorithm Details:** Algorithm 1 is highly detailed and ready for implementation.
- **Missing Resource:** The authors do not provide a link to an open-source repository or supplementary code containing the Analytical Coordinate Sandbox (ICS). Although the equations are clear, providing the sandbox codebase would dramatically improve reproducibility and facilitate direct community benchmarking.
