# Paper Summary: GraviMerge

## 1. Main Topic and Problem Statement
The paper addresses the challenge of **dynamic model merging (or test-time ensembling)** of multiple specialized parameter-efficient expert adapters (specifically Low-Rank Adaptation, or LoRA) under dynamic, non-stationary workloads for resource-constrained multi-task edge serving. 

While static merging techniques (e.g., Task Arithmetic, TIES-Merging, DARE) combine weights offline, they cannot adapt to real-time, non-stationary input streams. On the other hand, existing dynamic model merging methods suffer from a severe **stability-accuracy bottleneck**:
* **Stateless Ensembling (e.g., SABLE):** Computes routing similarities independently at each layer, resulting in catastrophic layer-to-layer ensembling weight jitter (rapid parameter oscillation through weight space across sequential depth), disrupting the smooth propagation of activations and causing representational incoherence.
* **First-Order State-Dependent Systems (e.g., ChemMerge):** Modeled on first-order non-equilibrium chemical reaction kinetics to smooth updates. However, they lack physical inertia/momentum, and under competitive reaction rates (low-temperature regimes) they suffer from volatile concentration oscillations and overreactions, which amplifies weight jitter instead of resolving it.

---

## 2. Proposed Approach: GraviMerge
To break this stability-accuracy bottleneck, the paper introduces **GraviMerge**, a physics-informed model merging paradigm based on second-order Newtonian gravitational dynamics on spherical manifolds. 

### Key Technical Concepts:
* **Analogy and Coordinates:** Decouples neural representation propagation from routing dynamics. It models intermediate activations as a virtual spacecraft coordinate probe traveling on the unit hypersphere $\mathbb{S}^{D-1}$ (to prevent unbounded feature explosion), while task-specific expert adapters act as stationary, high-mass celestial attractors (stars).
* **Virtual Time:** Layer depth $l \in [4, L]$ acts as discrete virtual time steps ($\Delta t$).
* **Second-Order Inertia:** The probe maintains a velocity vector $\mathbf{v}^{(l)} \in \mathbb{R}^D$ and is subject to viscous medium drag ($\gamma_{\text{drag}}$) and gravitational pull, introducing true physical momentum.
* **Three Novel Routing Mechanisms:**
  1. **Arrhenius Mass Activation (AMA):** Dynamically computes gravitational masses $M_k \in (0, 1]$ of task attractors at test-time based on zero-shot similarity at early layers (using an Arrhenius/Gibbs factor).
  2. **Geodesic Trajectory Integration (GTI):** Project net gravitational force $\mathbf{a}^{(l)}$ and velocity $\mathbf{v}^{(l)}$ onto the local tangent space of the sphere to preserve constraints, update coordinates via the exact spherical **Exponential Map (geodesic step)**, and **parallel transport** the velocity state vector from the tangent space at step $l-1$ to step $l$.
  3. **Gravitational Influence Blending (GIB):** Translates localized gravitational pull magnitudes directly into continuous ensembling weights $\alpha_k^{(l)}$ to govern adapter weight/activation blending.
* **Advanced Extensions:**
  * **Coupled GraviMerge (Closed-Loop):** Introduces a feedback force $\mathbf{F}_{\text{feedback}}^{(l)}$ pointing from the spacecraft toward the live normalized activation at layer $l-1$, coupling the trajectory directly to live representation changes.
  * **Temporal State Carryover:** Carries over exit velocity $\mathbf{v}^{(L)}$ to initialize the velocity of the next incoming query ($\mathbf{v}^{(3)}_{\text{next}} = \lambda_{\text{temporal}} \mathbf{v}^{(L)}_{\text{prev}}$) to support non-stationary streams.
  * **Sentinel Attractor Dynamics (SAD):** Integrates confidence-based gating to safeguard against Out-of-Distribution (OOD) task streams, pulling the spacecraft toward the geometric barycenter for a uniform fallback expert blend.
  * **Adaptive Gravitational Scheduling (AGS) & Adaptive Viscous Drag Scheduling:** Auto-tuning mechanisms to modulate the gravitational constant $G$ and drag coefficient $\gamma_{\text{drag}}$ dynamically.

---

## 3. Key Findings and Quantitative Results
The proposed approach is evaluated on a **Projected Digit Representation Space (RDS) Proxy benchmark** (projecting scikit-learn's `load_digits` dataset to $D = 192$ dimensions) and validated across three edge-serving workloads:
* **Accuracy-Stability Resolution:** GraviMerge achieves the highest joint serving accuracy (**$88.69\% \pm 1.68\%$**) across homogeneous, heterogeneous ($B = 256$), and real-time ($B = 1$) serving workloads, while reducing layer-to-layer ensembling weight jitter (measured in Mean Absolute Deviation, MAD) to **$0.00190 \pm 0.00012$**.
* **Baselines Outperformed:**
  * **SABLE (Stateless):** Accuracy $87.65\%$, Jitter $0.00456$ (GraviMerge reduces jitter by **$2.40\times$**).
  * **ChemMerge (First-Order Kinetics):** Accuracy $78.17\%$, Jitter $0.01141$ (GraviMerge reduces jitter by **$6.01\times$** and improves accuracy by **$10.52\%$**).
  * **EMA Smoothing (First-Order Filter):** Accuracy $79.70\%$, Jitter $0.01040$ (GraviMerge reduces jitter by **$5.47\times$**).
  * **WMomentum (Weight-space Momentum):** Accuracy $87.09\%$, Jitter $0.02763$ (GraviMerge reduces jitter by **$14.54\times$**).
  * **Kalman Filter Baseline (State-space Tracking):** Accuracy $87.97\%$, Jitter $0.00447$ (performs similarly to SABLE, failing to stabilize).
* **Scale and Geometric Robustness (GPT-2 dimensions, $D = 768$):** When tested on layer-specific representational drift, SABLE's jitter explodes to $0.16862$ MAD, whereas GraviMerge maintains a stable, smooth trajectory ($1.59 \times 10^{-7}$ MAD), achieving a **$1.06 \times 10^6\times$ jitter reduction**.
* **OOD Safeguard Verification:** Under OOD task streams, standard GraviMerge has a highly skewed weight allocation (Std Dev $0.2323$), whereas incorporating SAD reduces the ensembling weight standard deviation to **$0.0578 \pm 0.0024$**, ensuring a robust uniform fallback blend.

---

## 4. Explicitly Claimed Contributions
The authors explicitly claim four core contributions in Section 1:
1. **A Physics-Informed Routing Paradigm:** Introducing GraviMerge, modeling activation trajectories using multi-body gravitational physics on spherical manifolds to provide interpretable, robust representational stability.
2. **Novel Physical and Geometric Mechanisms:** Formulation of Arrhenius Mass Activation (AMA), Geodesic Trajectory Integration (GTI), and Gravitational Influence Blending (GIB) to maintain spherical constraints and parallel transport.
3. **Rigorous Stability and Robustness:** Exhaustive evaluations across 10 independent seeds demonstrating the resolution of the accuracy-stability dilemma (beating SABLE, ChemMerge, EMA, WMomentum, and Kalman Filter).
4. **Resilience to Dynamic Workloads:** Demonstrating flat, optimal accuracy profiles under Homogeneous, Heterogeneous, and Vectorized ($B = 1$) environments, verifying suitability for resource-constrained edge serving.
