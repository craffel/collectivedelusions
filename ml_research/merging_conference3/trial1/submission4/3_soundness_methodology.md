# Soundness and Methodology: FluidMerge

## 1. Technical Soundness Rating: Good / Fair
The technical soundness is generally **Good**, as the paper is well-structured, mathematically rigorous, and carefully deconstructs its physical metaphors into established ML concepts (Task Arithmetic, EWC, and distillation). However, there are several key inconsistencies, ambiguities, and potential methodological flaws that must be addressed.

---

## 2. Critical Methodological Flaws and Inconsistencies

### A. Mathematical Inconsistency in Virtual Time vs. Discretization Step Size
In Section 3.4, the authors define the virtual time trajectory $\theta(t)$ over the interval $t \in [0, T]$ (where $T = N \cdot \Delta t = 10.0$ in primary experiments). However, in Section 1 (Introduction), the virtual time horizon is still described as $t \in [0, 1]$.
This creates a narrative discrepancy:
- If the virtual time horizon is truly $t \in [0, 1]$, then for $N=100$ steps, the step size must be $\Delta t = 0.01$.
- If the step size is indeed $\Delta t = 0.1$ and $N=100$, the total integrated virtual time is $T = 10.0$.
While Section 3.4 is mathematically consistent in explaining $T = 10.0$ and $\Delta t = 0.1$, the authors should correct the residual reference to $t \in [0, 1]$ in Section 1 to ensure absolute clarity.

### B. The Out-of-Distribution (OOD) Teacher Soft-Label Noise Flaw
According to Section 3.1 & 3.4, for every unlabeled batch $X_{\text{unlabeled}}$, the composite advection force is calculated by summing the gradients across all $K$ tasks:
$$\mathbf{F}(\theta_n) = \sum_{k=1}^K w_k(t_n) \mathbf{F}_k(\theta_n) = -\sum_{k=1}^K w_k(t_n) \nabla_{\theta_n} \mathcal{H}\left(P_k^{\text{merged}}(\theta_n), P_k^{\text{ft}}\right)$$

In Section 3.2, the authors elegantly resolve cross-task OOD soft-label noise by stating that they route task-specific unlabeled batches $\{X_1, X_2, \dots, X_K\}$ to their native fine-tuned teacher experts $\theta_k$.
However, the paper should clarify:
- At test-time, how are these task-specific batches $X_k$ routed during evaluation of a single dataset stream?
- If the model adapts on a single dataset stream (e.g. adapting on SVHN), are the other 7 teachers still active on their own native streams, or are they frozen?
- While the confidence-based entropy filtering mechanism:
  $$\tilde{P}_k^{\text{ft}} = P_k^{\text{ft}} \cdot \mathbb{I}\left(\mathcal{H}(P_k^{\text{ft}}) \le \tau\right)$$
  is theoretically sound, the authors never specify the value of $\tau$ in Section 4, nor do they ablate its sensitivity.

### C. Stationary Fisher Curvature in a Non-Convex Landscape
The Fisher-Information-based Viscosity operator uses the empirical diagonal Fisher coordinates $F_i^{(0)}$ computed strictly at the initial state $\theta(0) = \theta_{\text{TA}}$:
$$[\mathbf{D}_{\text{Fisher}}(\theta)]_i = - F_i^{(0)} \left( \theta_i - \theta_i(0) \right)$$

While computationally necessary to avoid $O(d^2)$ matrix storage and frequent recalculations, keeping the Fisher coordinates stationary is problematic:
- As the parameters $\theta(t)$ evolve over 100 integration steps, they drift away from the initial Task Arithmetic state $\theta(0)$.
- In a highly non-convex loss landscape, the local Hessian/Fisher curvature shifts rapidly. The static $F_i^{(0)}$ becomes an increasingly poor approximation of the actual functional sensitivity of the parameters.
- This limitation of static EWC-based anchoring is well-known in continual learning but is not discussed or mitigated by the authors.

### D. Hybrid Continuous-Discrete Optimization Inconsistency
The shared image encoder $\theta$ evolves under discretized Euler integration (which is mathematically equivalent to standard Gradient Descent), while the task-specific classification heads $C_k$ are updated concurrently using the Adam optimizer with momentum buffers (Section 3.5).
- This introduces an optimization mismatch: the representation extractor (encoder) and the decision boundaries (classifiers) are optimized under completely different optimization geometries and learning rate schedules.
- While the authors provide a reasonable qualitative justification for this hybrid coupling, the lack of uniformity departs from a pure continuous-time physical flow, making the "pure ODE" formulation less mathematically rigorous than claimed.
