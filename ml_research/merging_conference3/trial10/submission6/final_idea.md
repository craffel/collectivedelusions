# PID-Merge: Closed-Loop PID-Controlled Stateful Routing for Dynamic Model Serving

## 1. Persona Alignment
PID-Merge is perfectly aligned with **The Pragmatist** persona. Instead of relying on complex, uninterpretable, and fragile metaphorical formulations (like continuous-time non-equilibrium biochemical reaction kinetics ODEs), PID-Merge leverages **classical, closed-loop PID (Proportional-Integral-Derivative) control theory**—the absolute cornerstone of real-world industrial automation. 
* **Real-World Impact and Robustness:** In actual production deployments, model ensembling is subject to high-frequency representation noise and rapid workload transitions. PID-Merge uses a closed-loop system to smooth out layer-to-layer oscillations (routing jitter) while ensuring instant responsiveness.
* **Deployment Constraints and Ease of Integration:** PID-Merge is computationally lightweight and trivial to implement. It runs in $O(1)$ parallel serving time and can be operated in a **completely training-free (zero-shot) mode** with robust heuristic default parameters (bypassing any offline calibration), or optimized on a tiny sequence using standard gradient descent.
* **Aesthetic and Conceptual Simplicity:** It replaces continuous virtual-time discretization and uninterpretable activation energies with three well-understood, physical control gains ($K_p$, $K_i$, $K_d$) that govern proportional responsiveness, integral smoothing, and derivative anticipation.

---

## 2. Core Techniques
PID-Merge introduces a closed-loop, discrete-time PID controller to update the ensembling weights of task-specific expert adapters (e.g., LoRA) across the depth of a deep neural network. It builds on and modifies the following foundational frameworks:
* **SABLE (Stateless Calibrated Routing) / SPS-ZCA \cite{sable}:** PID-Merge uses nearest-centroid cosine similarity routing anchored at intermediate layers as the reference input signal (the "setpoint").
* **Momentum-Merge \cite{momentum_merge}:** While Momentum-Merge acts as a simple, first-order open-loop Exponential Moving Average (EMA) filter (equivalent to a 1st-order lag/Proportional-Integral action), it introduces severe "inertial drag" (phase delay) during rapid task switches. PID-Merge resolves this by introducing the **Derivative (D) term**, which measures error acceleration to anticipate task switches and instantly suppress tracking lag.
* **Discrete-Time Process Control \cite{astrom_pid}:** PID-Merge adapts the classical incremental (velocity) PID algorithm to execute stateful recursive updates across sequential adapted layers $l \in [L_{\text{frozen}}+1, L]$.

---

## 3. Mathematical Formulation

Let $w_k^{(l)} \in [0, 1]$ be the raw nearest-centroid similarity routing weight for expert $k$ at adapted layer $l$. This raw weight serves as our reference input (setpoint). Let $\alpha_k^{(l)}$ denote the active ensembling coefficient applied to expert $k$ at layer $l$ (the controlled plant output).

### 3.1. Closed-Loop Tracking Error
We define the tracking error for expert $k$ at layer $l$ as:
\begin{equation}
e_k^{(l)} = w_k^{(l)} - \alpha_k^{(l-1)}
\end{equation}
where $\alpha_k^{(l-1)}$ is the normalized ensembling weight from the previous layer. At the boundary layer $L_{\text{frozen}}$, we initialize:
\begin{equation}
\alpha_k^{(L_{\text{frozen}})} = \frac{1}{K} \quad \text{and} \quad e_k^{(L_{\text{frozen}})} = 0, \quad e_k^{(L_{\text{frozen}}-1)} = 0
\end{equation}

### 3.2. Incremental PID State Update
To update the unnormalized routing state (virtual concentration) $s_k^{(l)} \in \mathbb{R}$ of expert $k$ at layer $l$, we employ the standard discrete-time velocity (incremental) PID formulation:
\begin{equation}
s_k^{(l)} = s_k^{(l-1)} + \Delta s_k^{(l)}
\end{equation}
where the state increment $\Delta s_k^{(l)}$ is computed as:
\begin{equation}
\Delta s_k^{(l)} = K_p \left( e_k^{(l)} - e_k^{(l-1)} \right) + K_i e_k^{(l)} + K_d \left( e_k^{(l)} - 2 e_k^{(l-1)} + e_k^{(l-2)} \right)
\end{equation}
Here, $K_p \ge 0$, $K_i \ge 0$, and $K_d \ge 0$ represent the proportional, integral, and derivative control gains, respectively.

### 3.3. Normalized Gibbs Routing Policy
The unnormalized states are mapped onto the probability simplex using a multi-temperature Gibbs Softmax policy to yield the final active ensembling weights:
\begin{equation}
\alpha_k^{(l)} = \frac{\exp\left(s_k^{(l)} / \tau_k\right)}{\sum_{j=1}^K \exp\left(s_j^{(l)} / \tau_j\right)}
\end{equation}
where $\tau_k = e^{w_k} + \tau_{\min}$ is the task-specific routing temperature.

### 3.4. Gain Parameterization & Constrained Learning
To support both training-free and optimized modes, the gains are globally shared across tasks (enforcing simplicity and preventing overfitting) and mapped via a Softplus function to guarantee non-negativity:
\begin{equation}
K_p = \ln(1 + e^{u_p}), \quad K_i = \ln(1 + e^{u_i}), \quad K_d = \ln(1 + e^{u_d})
\end{equation}
where $u_p, u_i, u_d \in \mathbb{R}$ are unconstrained log-parameters. For the training-free mode, the parameters are statically fixed to robust defaults: $K_p = 0.5$, $K_i = 0.15$, and $K_d = 0.2$.

---

## 4. Architecture Specifications
* **Representation Dimensions:** Works seamlessly across standard hidden dimensions (e.g., $D = 192$ in our sandbox, or $D = 768$ in ViT/LLaMA architectures) with $K$ task-specific experts.
* **Routing Layer:** Cosine similarity centroid routing is anchored at early layer $l_{\text{route}} = L_{\text{frozen}}$.
* **Expert Adaptation Space:** Low-rank adapter projections (e.g., LoRA rank $r = 8$) target Query ($W_Q$) and Value ($W_V$) matrices in adapted layers $l \in [L_{\text{frozen}}+1, L]$.
* **Inputs:** A sequential stream of unlabeled queries $X_1, X_2, \dots$ sample-by-sample ($B=1$ or $B=16$).
* **Intermediate Representations:**
  1. Early shared layers $1 \to L_{\text{frozen}}$ extract representation $z_t$.
  2. Cosm-similarity centroids compute $w_k^{(l)}$.
  3. PID controller recursively calculates states $s_k^{(l)}$ and coefficients $\alpha_k^{(l)}$ across network depth.
* **Outputs:** Single-pass, parallel activation blending at layer $l$:
  \begin{equation}
  h^{(l)} = h_{\text{base}}^{(l)} + \sum_{k=1}^K \alpha_k^{(l)} \left( h^{(l-1)} A_k^{(l)} B_k^{(l)} \frac{s_{\text{LoRA}}}{r} \right)
  \end{equation}

---

## 5. Baselines
We evaluate PID-Merge against the following representative prior methods:
1. **Expert Oracle:** The hypothetical ceiling assuming 100% correct expert routing.
2. **Uniform Merging (Static):** Static weight averaging where $\alpha_k^{(l)} = 1/K$, representing a parameter-free baseline.
3. **SABLE (Raw) \cite{sable}:** A stateless nearest-centroid dynamic routing baseline that suffers from high routing jitter under representation noise.
4. **ChemMerge \cite{chemmerge}:** The stateful chemical kinetics reactor model that uses continuous ODE simulations. This baseline represents the complex prior SOTA.
5. **Momentum-Merge \cite{momentum_merge}:** The training-free constant Exponential Moving Average (EMA) model, which acts as a first-order lag filter and suffers from phase delay.
6. **PAC-Kinetics \cite{pac_kinetics}:** The learning-theoretic PAC-Bayesian stateful router. Comparing with PAC-Kinetics evaluates whether PID-Merge can match or exceed its performance while using a much simpler, more interpretable control-theoretic framework.

---

## 6. Step-by-Step Interaction
For each incoming query sample $X$ at sequence step $t$:
1. **Base Feature Extraction:** Propagate $X$ through the shared frozen layers $1 \to L_{\text{frozen}}$ to extract the intermediate representation $z \in \mathbb{R}^D$.
2. **Subspace Coordinate Projection:** Compute the task coordinate projection vector $\mathbf{e} = [e_1, \dots, e_K]^T$ by measuring directional alignment (cosine similarity) to early-layer centroids $\mu_k$.
3. **Softmax Setpoint Computation:** Normalize coordinates into raw similarity routing weights $w_k^{(l)}$ using a gated Softmax with temperature $\tau_0$.
4. **Error Calculation:** At each adapted layer $l \in [L_{\text{frozen}}+1, L]$:
   * Calculate tracking error: $e_k^{(l)} = w_k^{(l)} - \alpha_k^{(l-1)}$.
5. **PID Control Update:**
   * Compute the state increment $\Delta s_k^{(l)} = K_p ( e_k^{(l)} - e_k^{(l-1)} ) + K_i e_k^{(l)} + K_d ( e_k^{(l)} - 2 e_k^{(l-1)} + e_k^{(l-2)} )$.
   * Update unnormalized controller state: $s_k^{(l)} = s_k^{(l-1)} + \Delta s_k^{(l)}$.
6. **Ensembling Weight Generation:** Pass $s_k^{(l)}$ through the multi-temperature Gibbs Softmax policy to obtain simplex-bounded ensembling weights $\alpha_k^{(l)}$.
7. **Single-Pass Parallel Activation Blending:** Blend the expert LoRA outputs with the frozen base model activation using $\alpha_k^{(l)}$, and propagate the blended representation to the next layer $l+1$.
