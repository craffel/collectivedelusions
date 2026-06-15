# 2. Novelty Check & Related Literature Delta

## Characterization of Novelty
The novelty of this paper is **significant and conceptually refreshing**. It bridges theoretical cognitive science/neuroscience (the Free Energy Principle and Active Inference) with high-speed deep learning systems engineering (dynamic parameter-efficient expert ensembling and Mixture-of-Experts serving). Instead of proposing incremental, heuristic step-size scaling or static exponential smoothing rules, the paper re-conceptualizes routing as an active perceptual and action-taking process governed by a single, first-principles information-theoretic optimization framework.

---

## Key Novel Aspects
1. **First-Principles Variational Formulation for Expert Serving:** The derivation of the Variational Free Energy ($\mathcal{F}_t$) specifically for multi-expert parameter ensembling. It shows that under a linear-Gaussian state-space generative model, minimizing free energy reduces exactly to balancing precision-weighted sensory and prior prediction errors.
2. **Exact Closed-Form Serving Solver:** Standard active inference control models typically rely on iterative gradient unrolling (e.g., gradient descent on free energy) or variational message passing, which are too slow and unstable for microsecond-level model serving. AIR's formulation is quadratic in the belief mean, yielding an exact analytical closed-form solution. By using Cholesky factorization of the constant Hessian, it reduces test-time updates to backward-substitution of quadratic $\mathcal{O}(K^2)$ complexity, making it extremely fast ($8$--$39\,\mu\text{s}$).
3. **Mechanistic Verification of Excitatory-Inhibitory Pathways:** The discovery and empirical verification that inhibitory pathways in the generative coordinate mapping matrix ($\mathbf{W} \in \mathbb{R}^{K \times K}$) are mechanically necessary to form negative feedback loops and actively suppress obsolete expert beliefs. Restricting $\mathbf{W} \ge 0$ (mimicking passive systems like chemical reaction kinetics) introduces a localized 15-step transient lag at task boundaries, highlighting the importance of biological excitatory-inhibitory balance.
4. **Systems-Level Equivalence Proof:** Mathematically deriving that under static variational covariance, this brain-inspired free energy minimizer is equivalent to a classical linear state observer (Kalman filter), establishing a beautiful theoretical bridge.

---

## The 'Delta' from Prior Work
The paper positions its contributions relative to:
- **Stateless Routers (e.g., SABLE \cite{sable2024}):** SABLE evaluates each query in absolute isolation. AIR's delta is its stateful variational belief, which incorporates prior temporal context. This allows AIR to filter out high-frequency sensory noise during homogeneous periods, slashing routing jitter by up to $2.49\times$.
- **Stateful Heuristics (e.g., Momentum-Merge \cite{momentummerge2025}, ChemMerge \cite{chemmerge2025}):** These methods apply rigid, fixed exponential smoothing or biochemical ODE simulation to ensembling weights, which introduces severe representational lag during task transitions. AIR's delta is that its prior constraints are dynamically balanced with bottom-up sensory predictions through precision weighting. When a task transition occurs, the resulting prediction error spike acts as an informational force that instantly overcomes prior constraints, allowing lag-free adaptation (within 1--2 steps).
- **Recurrent / Optimization-Based Methods (e.g., PAC-Kinetics \cite{packinetics2025}):** PAC-Kinetics optimizes continuous-time gating but requires step-size scheduling or iterative unrolled gradient optimization. AIR's delta is its exact, single-step closed-form linear-Gaussian solver, which is 100% numerically stable, has zero approximation error, and is computationally instantaneous.

---

## Critical Gaps in Scholarly Literature Context (Key Area for Improvement)
While the paper does a commendable job situating itself within parameter-efficient ensembling and control-theoretic routing, it suffers from a **significant omission of contemporary literature at the intersection of the Free Energy Principle (FEP) and Mixture-of-Experts (MoE) routing / dynamic ensembling**. Specifically:

1. **Omission of Wong (2026) - *"Affinity Is Not Enough: Recovering the Free Energy Principle in Mixture-of-Experts"*:**
   - **Context:** Wong (2026) explicitly critique standard MoE routing for being reactive and "affinity-based," proposing three FEP-inspired mechanisms: Temporal Memory ($\beta$) based on Leaky Integrate-and-Fire (LIF) dynamics, Precision-Weighted Gating ($\Pi$), and Anticipatory Routing to handle domain transitions.
   - **Delta:** Wong's model is formulated for token-level MoE routing using biological spiking LIF dynamics in discrete state-spaces. In contrast, AIR targets sequence-level adapter ensembling using a continuous-state linear-Gaussian formulation solved in a single closed-form analytical step. 
   - **Why this is a gap:** The authors claim to propose "the first multi-expert serving routing layer as an active-inference cognitive agent." This claim of absolute primacy is inaccurate and overstated because Wong (2026) represents a concurrent/prior FEP-inspired routing framework. The paper MUST cite and discuss Wong (2026) to properly attribute ideas and clearly delineate how their closed-form linear-Gaussian adapter-level approach differs from Wong's LIF-based token-level gating.

2. **Omission of the ODAR Framework (2025/2026):**
   - **Context:** Frameworks like **ODAR** (*Principled Adaptive Routing for LLM Reasoning via Active Inference*) use amortized active inference and variational free energy minimization to perform difficulty-based expert routing (e.g., routing between "Fast" and "Slow" agents or fusing multi-expert outputs).
   - **Why this is a gap:** Omitting ODAR means the paper fails to accurately describe the landscape of active inference applications in deep learning routing. Discussing ODAR would help contextualize AIR's systems-focused, Cholesky-factorized, microsecond-level parameter ensembling contribution within the broader landscape of cognitive compute routing.

### Recommendation:
The authors must temper their claim of being the "first" to apply active inference to expert routing, and instead properly position AIR as **the first to derive a single-step exact closed-form analytical solver for continuous-state linear-Gaussian active inference specifically for dynamic adapter-level serving**. They must cite and discuss Wong (2026) and ODAR (2025/2026) in Section 2 (Related Work) to establish a nuanced, honest, and historically complete scholarly context.
