# PAC-Kinetics: PAC-Bayesian Non-Equilibrium Chemical Kinetics for Provably Stable Dynamic Model Merging

## 1. Persona Alignment
In accordance with **The Theorist** persona, **PAC-Kinetics** completely rejects empirical heuristics (such as the unregularized, parameter-free nearest-centroid schemes or the heuristic chemical-kinetics ODEs of prior work) in favor of a rigorous, learning-theoretic and dynamical systems framework. We approach test-time model ensembling under noisy, heterogeneous workloads through the dual lenses of:
1. **Continuous-Time Stochastic Dynamical Systems:** Formulating representation flows as stateful first-order reaction kinetics to filter out high-frequency observation noise and minimize routing weight jitter.
2. **PAC-Bayesian Generalization Theory for Dependent Streams:** Resolving the standard, highly fragile i.i.d. assumption of standard learning bounds by deriving a strict, provable PAC-Bayesian generalization bound for stationary $\beta$-mixing stochastic processes.

By doing so, we establish the first learning-theoretic foundation for stateful, sequential model ensembling. Every parameter in our system—from the kinetic decay rates to the temperature scaling—is optimized by directly minimizing our derived generalization bound, providing provable guarantees of out-of-sample performance and trajectory stability.

---

## 2. Core Techniques
Our framework introduces and unifies four core techniques:
1. **Subspace Energy Projection (SEP) & Unit-Norm PCA Normalization (UN-PCA-SEP):** From PAC-ZCA \cite{pac_zca_2026}, we extract task coordinates from early frozen representations ($z_b \in \mathbb{R}^D$ at layer $l_{\text{route}}$) using orthogonal projection matrices $P_k = V_{k, d} V_{k, d}^T \in \mathbb{R}^{D \times D}$ computed from a disjoint subspace extraction calibration split $\mathcal{C}^{\text{sub}}$. We normalize representations to the unit sphere ($\tilde{z}_b = z_b / \|z_b\|_2$) prior to projection to guarantee a strict, dimension-free coordinate bound $\|\mathbf{e}_b\|_\infty \le 1$.
2. **First-Order Chemical Kinetics Dynamical State Update (Kinetics):** Inspired by ChemMerge \cite{chemmerge_2026}, we model the sample-specific routing weights as a continuous concentration state vector $s_t \in \mathbb{R}^K$ evolving across sequential queries. This acts as an online low-pass filter to resolve the temporal routing jitter paradox under heterogeneous streams.
3. **Catoni-type PAC-Bayesian Bounds for Stationary Mixing Processes (PAC):** We derive a formal generalization bound for dependent streams under the $\beta$-mixing framework, using the parameter-space Kullback-Leibler (KL) divergence as a powerful complexity control over the dynamical parameters.
4. **Single-Pass Activation Blending (SPS):** From SPS-ZCA \cite{sps_zca_2026}, we perform continuous activation-space blending of low-rank LoRA experts in a single, parallel forward pass, maintaining flat $O(1)$ backbone execution latency.

---

## 3. Mathematical Formulation

### 3.1 Problem Setup and Streaming Dynamics
Let $f_\theta$ be a pre-trained frozen backbone, and $\{E_1, \dots, E_K\}$ be $K$ task-specific LoRA adapters. We receive a sequential stream of queries $(\mathbf{x}_1, y_1), (\mathbf{x}_2, y_2), \dots, (\mathbf{x}_T, y_T)$, where $\mathbf{x}_t$ is the input at time step $t$ and $y_t \in \{1,\dots, K\}$ is its latent task identity.

At early layer $l_{\text{route}}$, we extract the normalized coordinate vector $\mathbf{e}_t = [e_{1, t}, \dots, e_{K, t}]^T \in [0, 1]^K$ via Unit-Norm PCA:
\begin{align}
    \tilde{z}_t &= \frac{z_t}{\|z_t\|_2 + \epsilon}, \quad z_t = \text{Pool}\left( f_{\theta, 1 \to l_{\text{route}}}(\mathbf{x}_t) \right) \in \mathbb{R}^D \\
    e_{k, t} &= \|P_k \tilde{z}_t\|_2 = \|V_{k, d}^T \tilde{z}_t\|_2 \in [0, 1]
\end{align}

### 3.2 Continuous-Time Chemical Kinetics Model
We treat the ensembling process as a multi-component chemical reactor where task concentrations $s(t) \in \mathbb{R}^K$ evolve in continuous-time $t$:
\begin{align}
    \frac{ds(t)}{dt} = - \mathbf{\Gamma} s(t) + \mathbf{\Phi} \mathbf{e}(t)
\end{align}
where $\mathbf{\Gamma} = \text{diag}(\gamma_1, \dots, \gamma_K) > 0$ represents the task-specific decay/reaction rates, and $\mathbf{\Phi} \in \mathbb{R}^{K \times K}$ is the coupling matrix projecting raw coordinate signals. Integrating this ODE over a discrete interval $\Delta t = 1$ yields the recurrence:
\begin{align}
    s_t = \mathbf{A} s_{t-1} + \mathbf{B} \mathbf{e}_t
\end{align}
where $\mathbf{A} = \text{diag}(a_1, \dots, a_K) = \text{diag}(e^{-\gamma_1}, \dots, e^{-\gamma_K}) \in (0, 1)^K$ represents state retention, and $\mathbf{B} = \mathbf{\Gamma}^{-1}(I - \mathbf{A})\mathbf{\Phi} \in \mathbb{R}^{K \times K}$ represents the coordinate injection matrix. The active ensembling coefficients $\alpha_t \in \Delta^{K-1}$ are given by:
\begin{align}
    \alpha_{k, t} = q_k(s_t; \boldsymbol{\tau}) = \frac{\exp(s_{k, t}/\tau_k)}{\sum_{j=1}^K \exp(s_{j, t}/\tau_j)}
\end{align}
We collect the log-parameters of our router into $\Theta = \{\mathbf{u}, W, \mathbf{w}\} \in \mathbb{R}^{2K + K^2}$, where:
\begin{align}
    u_k = \ln\left(\frac{a_k}{1 - a_k}\right) \in \mathbb{R}, \quad W = \mathbf{B} \in \mathbb{R}^{K \times K}, \quad w_k = \ln \tau_k \in \mathbb{R}
\end{align}
This mapping ensures that the state retention coefficients $a_k = \sigma(u_k)$ are strictly bounded in $(0, 1)$ during gradient-based optimization.

### 3.3 Learning-Theoretic Bound for Dependent Streams
Since sequential streams violate the i.i.d. assumption, we assume the input coordinate sequence $(\mathbf{e}_t, y_t)_{t=1}^T$ is a stationary $\beta$-mixing process with mixing coefficients $\beta(j)$. Under $\beta$-mixing, the dependency between segments of the stream decays exponentially as their temporal separation increases.

Let $P = \mathcal{N}(\Theta_0, \sigma_0^2 I)$ be our data-independent routing prior centered at SABLE-grounded defaults:
\begin{align}
    \mathbf{u}_0 = \ln(0.5 / 0.5) \cdot \mathbf{1} = \mathbf{0}, \quad W_0 = I_K, \quad \mathbf{w}_0 = \ln(0.05) \cdot \mathbf{1}
\end{align}
Let $Q = \mathcal{N}(\Theta, \sigma_0^2 I)$ be our learned posterior. The parameter complexity penalty is the Gaussian KL divergence:
\begin{align}
    \text{KL}(Q \| P) = \frac{\|\mathbf{u} - \mathbf{u}_0\|_2^2 + \|W - W_0\|_F^2 + \|\mathbf{w} - \mathbf{w}_0\|_2^2}{2 \sigma_0^2}
\end{align}
To bound the expected sequential Cross-Entropy risk $R(\Theta) = \mathbb{E}[\mathcal{L}_{\text{CE}}(q_t(\Theta), y_t)]$ over unseen streams, we partition a calibration stream of length $T$ into $a$ blocks of size $b$ ($T = ab$). By applying Catoni’s PAC-Bayesian inequality generalized to $\beta$-mixing dependent processes \cite{alquier2013pac}, we establish that for any $\beta > 0$ and $\delta \in (0, 1)$, with probability at least $1 - \delta$ over the choice of calibration stream:
\begin{align}
    R(\Theta) \le \frac{1}{1 - e^{-\beta}} \left( 1 - \exp\left( -\beta \hat{R}_T(\Theta) - \frac{\text{KL}(Q \| P) + \ln(1/\delta)}{a} \right) \right) + 2 \beta(b)
\end{align}
where $\hat{R}_T(\Theta) = \frac{1}{T} \sum_{t=1}^T \mathcal{L}_{\text{CE}}(q_t(s_t; \Theta), y_t)$ is the empirical sequential Cross-Entropy loss over a decoupled calibration stream $\mathcal{C}^{\text{opt}}$ of length $T$. In our implementation, we set $\beta = 0.5$, $\delta = 0.05$, and block size $b=4$ (giving $a = T / 4$), and assume a fast mixing rate such that the residue $2 \beta(b)$ is a negligible constant. We directly minimize this bound using the Adam optimizer to learn the optimal parameters $\Theta^*$.

### 3.4 Bounded Lipschitz and Contractive Trajectory Guarantees

#### Lemma 1 (Lipschitz Continuity of the State Trajectory)
*Let $\{\mathbf{e}_t\}_{t=1}^T$ and $\{\mathbf{e}'_t\}_{t=1}^T$ be two coordinate sequences bounded in $[0, 1]^K$. Under our log-retention parameterization where $a_k = \sigma(u_k) \in (0, 1)$, the routing state trajectory is Lipschitz continuous with respect to the injection weights $W$ and retention parameters $\mathbf{u}$. Specifically, the state difference is bounded at any step $t$ by:*
\begin{align}
    \|s_t - s'_t\|_2 \le \sum_{i=1}^t \|\mathbf{A}\|^{t-i}_2 \|W\|_2 \|\mathbf{e}_i - \mathbf{e}'_i\|_2
\end{align}
*Furthermore, the state transition operator is strictly contractive if the spectral norm satisfies $\|\mathbf{A}\|_2 = \max_k \sigma(u_k) < 1$, which is guaranteed since $\sigma(u_k) \in (0, 1)$ for all $u_k \in \mathbb{R}$.*

#### Lemma 2 (Lyapunov Stability of PAC-Kinetics)
*The dynamical routing state recurrence $s_t = \mathbf{A} s_{t-1} + W \mathbf{e}_t$ represents a globally asymptotically stable (GAS) discrete-time dynamical system. Under the quadratic Lyapunov candidate function $V(s) = s^T P s$ with positive definite matrix $P = I$, the state trajectory is bounded and contracts toward an invariant ellipsoid scaled by the input coordinate magnitude. The trajectory is strictly input-to-state stable (ISS).*

#### Proof of Lyapunov Stability
Let us evaluate the Lyapunov difference $\Delta V = V(s_t) - V(s_{t-1})$ under zero coordinate input ($\mathbf{e}_t = \mathbf{0}$):
\begin{align}
    \Delta V &= s_t^T s_t - s_{t-1}^T s_{t-1} \nonumber \\
    &= s_{t-1}^T \mathbf{A}^T \mathbf{A} s_{t-1} - s_{t-1}^T s_{t-1} \nonumber \\
    &= s_{t-1}^T (\mathbf{A}^2 - I) s_{t-1}
\end{align}
Since $\mathbf{A} = \text{diag}(a_1, \dots, a_K)$ with $a_k = \sigma(u_k) \in (0, 1)$, the matrix $\mathbf{A}^2 - I = \text{diag}(a_1^2 - 1, \dots, a_K^2 - 1)$ is strictly negative definite. Specifically, let $a_{\max} = \max_k a_k < 1$. Then:
\begin{align}
    \Delta V \le (a_{\max}^2 - 1) \|s_{t-1}\|_2^2 < 0, \quad \forall s_{t-1} \ne \mathbf{0}
\end{align}
Thus, the system is globally asymptotically stable under zero input. For bounded coordinate inputs $\|\mathbf{e}_t\|_2 \le \sqrt{K}$, by applying the triangle inequality, the state magnitude is strictly bounded:
\begin{align}
    \|s_t\|_2 \le \frac{\|W\|_2 \sqrt{K}}{1 - a_{\max}}
\end{align}
This proves the system is input-to-state stable (ISS), meaning that the state trajectory remains strictly bounded and is completely immune to chaotic divergence or unbounded representation spikes. $\blacksquare$

---

## 4. Architecture Specifications

### 4.1 Input and Backbone Specs
- **Input Stream:** Sequential batches of images or text: $\mathbf{x}_t$.
- **Backbone Model:** Frozen pre-trained transformer (e.g., $\text{ViT-B/16}$) or CNN (e.g., $\text{ResNet-18}$).
- **Early Routing Layer:** $l_{\text{route}} \in \{1,\dots, L\}$ (typically layer 3 or 4 of the backbone).

### 4.2 Routing Mechanics
1. **Feature Extraction:** Pull intermediate pooled representations at layer $l_{\text{route}}$: $z_t = \text{Pool}\left( f_{\theta, 1 \to l_{\text{route}}}(\mathbf{x}_t) \right) \in \mathbb{R}^D$.
2. **Unit-Norm PCA Normalization:** Normalize $z_t$ and project using the disjoint-split projection matrix $P_k = V_{k, d}V_{k, d}^T$:
\begin{align}
    e_{k, t} = \|V_{k, d}^T \tilde{z}_t\|_2 \in [0, 1], \quad \text{where } \tilde{z}_t = \frac{z_t}{\|z_t\|_2 + \epsilon}
\end{align}
3. **Kinetics Stateful Recurrence:** Maintain the continuous-time routing state $s_t$:
\begin{align}
    s_t = \mathbf{A} s_{t-1} + W \mathbf{e}_t \in \mathbb{R}^K, \quad \text{with } s_0 = \mathbf{0}
\end{align}
4. **Gibbs Policy Softmax:** Compute active ensembling weights:
\begin{align}
    \alpha_{k, t} = \frac{\exp(s_{k, t} / e^{w_k})}{\sum_{j=1}^K \exp(s_{j, t} / e^{w_j})}
\end{align}

### 4.3 Subsequent Ensembling Specs (SPS)
For layers $l > l_{\text{route}}$, blend LoRA expert activations sample-wise in a single parallel pass:
\begin{align}
    h_t^{(l)} = h_t^{(l-1)} W_{\text{base}}^{(l)} + \sum_{k=1}^K \alpha_{k, t} \left( h_t^{(l-1)} A_k^{(l)} B_k^{(l)} \right)
\end{align}
where $A_k^{(l)} \in \mathbb{R}^{d_{\text{in}} \times r}$, $B_k^{(l)} \in \mathbb{R}^{r \times d_{\text{out}}}$ are the low-rank LoRA matrices for expert $k$.

---

## 5. Baselines
We evaluate PAC-Kinetics against five highly appropriate and diverse baselines:
1. **Expert Oracle:** The hypothetical ceiling where inputs are routed directly to their true task-specific adapter (100% routing accuracy).
2. **Uniform Weight Merging (Static):** The default parameter-free baseline where all expert weights are averaged statically ($\alpha_k = 1/K$ for all samples).
3. **Stateless Nearest-Centroid Routing (SPS-ZCA / SABLE):** The standard non-parametric dynamic routing baseline, which is highly susceptible to temporal jitter.
4. **Stateless Temperature-Only PAC-ZCA:** The learning-theoretic state-of-the-art baseline, which resolves overfitting but is stateless and suffers from temporal oscillations.
5. **Heuristic Continuous-Time Chemical Kinetics (ChemMerge):** The stateful continuous-time ensembling baseline, which reduces jitter but lacks any generalization bounds or formal stability guarantees.

---

## 6. Step-by-Step Interaction

The flow of data through the PAC-Kinetics serving system is structured as follows:

```
[Incoming Sample Stream: x_t]
         │
         ▼
[Frozen Backbone: Layers 1 -> l_route]
         │
         ▼
[Pooled Activation: z_t]
         │
         ▼
[Unit-Norm Normalization: \tilde{z}_t] ──► Guarantees bounded coordinate range [0, 1]
         │
         ▼
[Subspace Energy Projection: e_{k,t} = ||V_{k,d}^T \tilde{z}_t||_2]
         │
         ▼
[Kinetics State Update: s_t = A * s_{t-1} + W * e_t] ──► Filters out temporal jitter
         │
         ▼
[Gibbs Softmax Router: \alpha_{k,t} = Softmax(s_t / \tau)]
         │
         ▼
[Single-Pass Activation Blending: h^{(l)} = h^{(l-1)}W_base + \sum \alpha_{k,t} (h^{(l-1)}A_k B_k)]
         │
         ▼
[Classification Output: y_hat_t]
```

1. **Streaming Input Processing:** The system receives a vectorized, heterogeneous stream of samples $\mathbf{x}_t$ sequentially over time steps $t=1, 2, \dots, T$.
2. **Early Representation Extraction:** Each sample is fed into the frozen pre-trained backbone up to the shared, adapter-free routing layer $l_{\text{route}}$, outputting pooled activation feature vector $z_t \in \mathbb{R}^D$.
3. **Unit-Norm Coordinate Extraction:** The representation $z_t$ is normalized to unit-length $\tilde{z}_t$ and projected onto each task's principal subspace $V_{k, d}$ (pre-computed from a disjoint calibration split $\mathcal{C}^{\text{sub}}$). The L2 norm of the projection yields the coordinate vector $\mathbf{e}_t \in [0, 1]^K$.
4. **Stateful Kinetics Update:** The stateful router updates its internal concentration state $s_t \in \mathbb{R}^K$ by combining the decayed prior state $\mathbf{A} s_{t-1}$ with the coordinate injection $W \mathbf{e}_t$. The retention rates $a_k = \sigma(u_k)$ and injection matrix $W$ are strictly regularized by our PAC-Bayesian complexity penalty.
5. **Ensembling Weight Computation:** The concentration state $s_t$ is fed into a Gibbs Softmax scaled by the optimized temperatures $\boldsymbol{\tau} = e^{\mathbf{w}}$ to generate the sample-wise ensembling coefficients $\alpha_t \in \Delta^{K-1}$.
6. **Single-Pass Parallel Execution:** Subsequent backbone layers $l > l_{\text{route}}$ execute in a single forward pass, blending the LoRA expert adapter activations dynamically using coefficients $\alpha_{k, t}$, producing stable, high-accuracy task predictions.
