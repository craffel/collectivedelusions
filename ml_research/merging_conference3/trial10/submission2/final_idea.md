# Layer-Decoupled Stateful Kinetics (LDS-Kinetics) for Dynamic Model Merging

## 1. Persona Alignment
This project is deeply aligned with **The Empiricist** persona:
*   **Empirical Deconstruction of Depth Dynamics:** Instead of assuming a single global routing coefficient is optimal across all layer depths, LDS-Kinetics treats the network's depth as an active variable. We empirically test this by decoupling the temporal-spatial ensembling dynamics.
*   **Massive Parallel sweeps:** We design a high-dimensional experimental sweep covering:
    1.  *Decoupling scales ($M$):* Global routing ($M=1$), Tri-Block routing ($M=3$), and Fully Layer-Decoupled routing ($M=11$).
    2.  *Initialization seeds & calibration sizes:* $T \in \{32, 64, 128\}$ to stress-test generalization bounds under data scarcity.
    3.  *Regularization strengths:* Posterior variance sweeps $\sigma_0^2 \in \{1.0, 5.0, 10.0\}$.
*   **Robust Ablation Studies:** We implement modular ablation switches to analyze the performance contribution of decoupling specific blocks (e.g., decoupling only early layers vs. only late layers) to isolate where stateful memory is most beneficial.
*   **Overwhelming Empirical Validation:** We evaluate across Orthogonal and Overlapping manifold layouts under Homogeneous and Heterogeneous sequential query streams over 5 independent random seeds.

---

## 2. Core Techniques
We introduce **Layer-Decoupled Stateful Kinetics (LDS-Kinetics)**, which modifies and extends the following foundational techniques:
1.  **Subspace routing coordinate projection:** Sourced from SABLE \cite{sable_2024} and PAC-ZCA \cite{pac_zca_2026}. We project unit-normalized representations at routing layer $l_{\text{route}} = 3$ onto orthonormal task-specific PCA subspaces to obtain scale-free affinity coordinates $\mathbf{e}_t$.
2.  **Continuous-time chemical kinetics state space modeling:** Sourced from ChemMerge \cite{chemmerge_2026} and PAC-Kinetics \cite{pac_kinetics_2026}. However, we generalize the state recurrence to be block-specific or layer-specific.
3.  **Adaptive Online Kinetics:** Sourced from PAC-Kinetics \cite{pac_kinetics_2026}. We scale down block-specific state retention using the rolling cosine similarity of incoming coordinate vectors to suppress inertial drag during abrupt switches.
4.  **PAC-Bayesian Generalization Bound:** Sourced from PAC-Kinetics \cite{pac_kinetics_2026}. We apply a unified Gaussian complexity penalty to all $M$ sets of decoupled parameters simultaneously, preventing transductive overfitting.

---

## 3. Mathematical Formulation

Let $K=4$ be the number of task-specific experts. Let the dynamic ensembling layers (Layers 4 to 14) be partitioned into $M$ disjoint blocks: $B^{(1)}, \dots, B^{(M)}$. Let $m(l)$ denote the block index containing layer $l$.

### Coordinate Signals
At time step $t$, the intermediate representation $z_t \in \mathbb{R}^D$ is extracted at layer $l_{\text{route}} = 3$ and unit-normalized:
\begin{equation}
    \tilde{z}_t = \frac{z_t}{\|z_t\|_2 + \epsilon}
\end{equation}
The coordinates are projected onto $K$ task PCA subspaces:
\begin{equation}
    e_{k, t} = \|P_k \tilde{z}_t\|_2 \in [0, 1]
\end{equation}

### Decoupled Stateful Recurrence
For each block $m \in \{1, \dots, M\}$, we maintain an independent concentration state vector $s^{(m)}_t \in \mathbb{R}^K$ evolving as:
\begin{equation}
    s^{(m)}_t = \mathbf{A}^{(m)}_t s^{(m)}_{t-1} + W^{(m)} \mathbf{e}_t
\end{equation}
where:
*   $\mathbf{A}^{(m)}_t = \text{diag}(a^{(m)}_{1, t}, \dots, a^{(m)}_{K, t})$ is the dynamic state-retention matrix.
*   To suppress phase delay, we compute the local workload similarity $Sim_t$:
    \begin{equation}
        Sim_t = \frac{\mathbf{e}_t^T \mathbf{e}_{t-1}}{\|\mathbf{e}_t\|_2 \|\mathbf{e}_{t-1}\|_2 + \epsilon}
    \end{equation}
    and scale the learnable retention rates $a^{(m)}_k$:
    \begin{equation}
        a^{(m)}_{k, t} = a^{(m)}_k \cdot Sim_t, \quad \text{with} \quad a^{(m)}_k = \sigma(u^{(m)}_k) \in (0, 1)
    \end{equation}
*   $W^{(m)} \in \mathbb{R}^{K \times K}$ is the unconstrained coordinate injection (coupling) matrix for block $m$.

### Gibbs Policy
For each block $m$, the active ensembling coefficients $\alpha^{(m)}_t \in \Delta^{K-1}$ are:
\begin{equation}
    \alpha^{(m)}_{k, t} = \frac{\exp(s^{(m)}_{k, t}/\tau^{(m)}_k)}{\sum_{j=1}^K \exp(s^{(m)}_{j, t}/\tau^{(m)}_j)}
\end{equation}
where the learnable block-specific temperatures are $\tau^{(m)}_k = e^{w^{(m)}_k} + \tau_{\min}$, with $\tau_{\min} = 0.01$.

### Layer-Wise Dynamic Activation Blending
For layer $l \in [4, 14]$, activations are ensembled sample-wise in a single pass using the coefficients of block $m(l)$:
\begin{equation}
    h_t^{(l)} = h_t^{(l-1)} W_{\text{base}}^{(l)} + \sum_{k=1}^K \alpha^{(m(l))}_{k, t} \left( h_t^{(l-1)} A_k^{(l)} B_k^{(l)} \right)
\end{equation}

### Learning-Theoretic Regularization
The full parameter vector is $\Theta = \{ \mathbf{u}^{(m)}, W^{(m)}, \mathbf{w}^{(m)} \}_{m=1}^M \in \mathbb{R}^{M \times (2K + K^2)}$. 
We minimize Catoni's $\beta$-mixing PAC-Bayesian bound, which simplifies to regularized Empirical Risk Minimization centered at SABLE-grounded prior defaults $\Theta_0$ (where $\mathbf{u}^{(m)}_0 = \mathbf{0}$, $W^{(m)}_0 = I_K$, $\mathbf{w}^{(m)}_0 = \ln(0.05) \cdot \mathbf{1}$):
\begin{equation}
    \mathcal{J}(\Theta) = \frac{\lambda}{\mathcal{L}_{\max}} \hat{R}_T(\Theta) + \frac{1}{a \sigma_0^2} \sum_{m=1}^M \left( \|\mathbf{u}^{(m)} - \mathbf{u}_0\|_2^2 + \|W^{(m)} - W_0\|_F^2 + \|\mathbf{w}^{(m)} - \mathbf{w}_0\|_2^2 \right)
\end{equation}
where $\hat{R}_T(\Theta)$ is the truncated Cross-Entropy loss ($\mathcal{L}_{\max} = 5.0$), $\lambda = 0.5$, $\sigma_0^2 = 5.0$, and $a = T/4$ is the block count for a sequence of length $T$.

---

## 4. Architecture Specifications
*   **Backbone Structure:** 14 layers, hidden dimension $D = 192$.
*   **Routing Coordinate extraction:** Occurs at Layer 3 ($l_{\text{route}} = 3$). 
*   **Dynamic Blending Range:** Layers 4 to 14 ($L=14$), with LoRA rank $r = 8$ and scaling factor $\gamma_V = 0.05$.
*   **Decoupled Blocks Configurations:** We parameterize and evaluate three distinct scales of $M$:
    1.  *Global ($M=1$):* A single block spanning Layers 4-14.
    2.  *Tri-Block ($M=3$):* 
        *   Block 1 (Early, L4-7): Focuses on transient representational alignment.
        *   Block 2 (Middle, L8-11): Focuses on semantic task integration.
        *   Block 3 (Late, L12-14): Focuses on output logit refinement.
    3.  *Fully Decoupled ($M=11$):* Each of the 11 ensembling layers maintains its own independent state recurrence and parameters.
*   **Inputs:** Incoming sequence of raw query activations $\mathbf{x}_t \in \mathbb{R}^D$ and true task labels $y_t$.
*   **Intermediate representations:** Dynamic blending states $h_t^{(l)} \in \mathbb{R}^D$.
*   **Outputs:** Distance-based classification logits mapping final representations $h_t^{(L)}$ to task signatures.

---

## 5. Baselines
We compare **LDS-Kinetics** against the following robust baselines:
1.  **Expert Oracle:** Hypothetical ceiling with 100% routing accuracy.
2.  **Uniform Merging (Static):** Static, parameter-free average ($\alpha_k = 1/K$).
3.  **SABLE (Raw) / SPS-ZCA:** Stateless dynamic routing using raw, unregularized coordinates.
4.  **Stateless PAC-ZCA:** SOTA learning-theoretic baseline with temperature-only Gibbs routing.
5.  **Heuristic ChemMerge:** Stateful baseline with static, hand-tuned global chemical kinetics parameters.
6.  **Global PAC-Kinetics (Trial 9, Submission 9):** State-of-the-art stateful kinetics with global, unified parameters ($M=1$).
7.  **Decoupled ERM:** Decoupled architecture with $M \in \{3, 11\}$ but optimized with zero regularization ($\sigma_0^2 \to \infty$). This isolates the benefit of our PAC-Bayesian complexity penalty in high-dimensional decoupled spaces.

---

## 6. Step-by-Step Interaction

1.  **Inference Query Arrival:** A raw test sample $\mathbf{x}_t$ arrives at time step $t$.
2.  **Early Propagation & Extraction:** The sample propagates through the frozen backbone layers 1 to 3. At Layer 3, raw activation $z_t \in \mathbb{R}^D$ is extracted.
3.  **Coordinate Normalization:** $z_t$ is normalized to $\tilde{z}_t$ on the unit sphere to ensure strict scale bounds.
4.  **PCA Subspace Projection:** $\tilde{z}_t$ is projected onto $K$ task-specific orthonormal PCA projection matrices $P_k$, producing coordinate signal vector $\mathbf{e}_t$.
5.  **Workload Cosine Similarity:** Cosine similarity $Sim_t$ between $\mathbf{e}_t$ and $\mathbf{e}_{t-1}$ is computed to measure current stream homogeneity.
6.  **Decoupled State Evolution:** For each block $m \in \{1,\dots, M\}$, the previous concentration state $s^{(m)}_{t-1}$ is decayed by the dynamic retention coefficient $a^{(m)}_{k, t} = a^{(m)}_k \cdot Sim_t$ and injected with the new coordinate vector via the coupling matrix $W^{(m)}$, yielding $s^{(m)}_t$.
7.  **Multi-Temperature Gibbs Softmax:** For each block $m$, the state $s^{(m)}_t$ is passed through the Gibbs policy with block-specific learned task temperatures $\tau^{(m)}_k$, yielding the active ensembling coefficients $\alpha^{(m)}_t$.
8.  **Layer-by-Layer Blending:** The representation propagates through layers 4 to 14. At each layer $l$, the expert low-rank adapter activations are dynamically scaled and blended sample-wise using the coefficients $\alpha^{(m(l))}_t$ of its corresponding block $m(l)$.
9.  **Distance Classifier & Loss:** The final representation $h_t^{(14)}$ is evaluated against target signatures to compute alignment distance, final logits, and the corresponding classification loss.
