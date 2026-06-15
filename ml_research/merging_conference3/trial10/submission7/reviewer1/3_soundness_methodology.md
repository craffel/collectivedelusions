# Soundness and Methodology Evaluation

## Clarity of the Description
The methodology is exceptionally well-described, clear, and mathematically rigorous:
- **Formalization:** The paper provides a clear mathematical progression from subspace coordinate projection to multi-tenant slot decoupling, stateful recurrence updates, decay policies, and finally Gibbs Softmax mapping and dynamic representation blending.
- **Algorithm:** The step-by-step TDSR pipeline is laid out in high-fidelity pseudocode (Algorithm 1), mapping out the exact operational lifecycle of an incoming query.
- **Simplification Proof:** The mathematical proof showing that online cosine similarity against fixed orthogonal centroids simplifies to coordinate-argmax assignment is highly clear and extremely satisfying.

## Appropriateness of Methods
Through a minimalist design philosophy, the chosen methods are highly appropriate, elegant, and effective:
- **First-Order Diagonal Recurrence:** A first-order recurrence is the simplest possible stateful tracking mechanism. It acts as an effective temporal low-pass filter with a single scalar multiplication and addition, introducing minimal computational complexity.
- **State Pool Slot-Kinetics:** Decoupling the global state vector into a virtual state slot pool is the most direct and elegant way to isolate tenant history, completely avoiding complex session-tracking architectures.
- **Explicit Metadata Tagging:** Leveraging existing scheduler metadata (session/tenant IDs) from punica or s-lora is the most efficient and robust way to achieve perfect session isolation with zero overhead.
- **Implicit Coordinate-Argmax Assignment:** Bypassing heavy online K-means or uninterpretable neural network classifiers by fixing slot centroids as orthogonal basis task detectors is a brilliant, zero-overhead design. It utilizes the model's own activation-space projection coordinates as a natural clustering basis.
- **Logical Session-Step Decay & Dual-Clock Decay:** Standard global-step decay is inappropriate for sparse tenant traffic because other tenants' steps wash out inactive states. Setting inactive steps' decay to zero ($\Delta t_m = 0$) preserves history. Reconciling this with physical timeout-based decay to prevent memory leaks is highly practical and theoretically sound.

## Potential Technical Flaws & Limitations
The authors are commendably honest and transparent about several key limitations of their approach, which they discuss in depth:
1. **Task-Transition Stateful Tracking Failure in Implicit Mode:** In the implicit tagless mode, because slots are bound to orthogonal task detectors, requests are routed based on task affinity rather than physical user identity. If a user transitions from Task 1 to Task 2, their queries are routed to separate slots. Consequently, the router cannot smooth states across a user's task transitions, making the router effectively stateless at the exact moment of task switch. This is a fundamental limitation of tagless routing.
2. **The Overlapping Manifold Bottleneck for Implicit Tagless Clustering:** Under overlapping manifolds, coordinate interference can cause queries to be assigned to the wrong slots, introducing minor state contamination. While the authors propose future solutions (soft slot assignment/Gumbel-Softmax, online centroid learning, slot-repelling losses), they note a critical trade-off: soft assignment distributes updates proportionally across slots, which can re-introduce cross-tenant cross-talk.
3. **Policy Collapse during Calibration:** In initial experiments, standard gradient descent converged to a constant gating policy (Expert 3). The authors resolved this "policy collapse" by adding a standard MoE load-balancing entropy term ($\mathcal{L}_{\text{balance}}$).

## Reproducibility
The reproducibility of this paper is **excellent**:
- Every parameter is specified, including network depth ($L=14$), hidden dimension ($D=192$), boundary routing layer ($l_{\text{route}}=3$), expert fleet ($K=4$), slot size ($M=4$), calibration stream ($N_{\text{cal}}=100$), and test stream ($N_{\text{test}}=400$).
- The manifold geometry details (dimension sizes, overlap sizes) are explicitly listed.
- The step-by-step pseudo-code in Algorithm 1 makes re-implementation extremely straightforward.
- All baseline routing methods are standard and easily reproducible.
