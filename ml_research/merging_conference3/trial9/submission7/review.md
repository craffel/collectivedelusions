# Peer Review: Lyapunov-Stable Active Representation Coupling for Dynamic Model Serving

**Overall Recommendation:** 5: Accept  
**Reviewer Confidence:** 5: Expert  

---

## 1. Summary of the Paper
The paper presents **Lyapunov-Stable Active Representation Coupling (L-ARC)**, a training-free, closed-loop control framework designed to stabilize continuous-depth representation flow and expert ensembling when dynamically serving Parameter-Efficient Fine-Tuning (PEFT) adapters (e.g., LoRA adapters) under heterogeneous streaming workloads.

Dynamic model ensembling on edge devices is fundamentally challenged by **routing volatility (jitter)** and **cascading representational drift**. Stateless ensembling methods compute weights independently at each layer, ignoring spatial dependencies and leading to severe routing fluctuations. Stateful ensembling models (like ChemMerge) smooth trajectories using continuous-depth reaction-decay ordinary differential equations (ODEs), but they rely on a heuristic, constant feedback rate ($\eta > 0$) to warp intermediate hidden features toward static task centroids. Under practical serving workloads where centroids are extracted at an early stage, this constant feedback pulls highly refined late-layer activations backward toward noisier early-stage coordinates—a phenomenon termed **representational backward-shift**—which degrades representation quality and downstream task performance.

L-ARC resolves these issues through a dual-shielded, closed-loop control system:
1. **State-Space Shielding (ECG-Reset):** It tracks sample-wise expert concentrations via explicit Euler discretization and introduces **Entropy-Gated Concentration Gating (ECG-Reset)**. ECG-Reset monitors the normalized Shannon routing entropy ($H^{(l)}$) and freezes the ODE kinetics updates ($\Delta t = 0$) under high routing uncertainty or transient failures ($H^{(l)} > 0.95$), preventing state-space memory corruption.
2. **Representation-Space Shielding (Dissipation Guard):** It models the representation similarity error as a candidate Lyapunov function. Using first-order linearization, it derives an analytical **Dissipation Guard** ($A^{(l)}$) and proves its unconditional non-negativity (Theorem 3.3). If $A^{(l)} \le \theta_G = 0.04$ (deadband), feedback warping is gated off ($\eta = 0$) to prevent noise propagation. If $A^{(l)} > \theta_G$, the step size is adaptively scaled ($\eta^{(l)} = \min(\eta_{\max}, \gamma A^{(l)})$) to guarantee that representation warping is strictly error-decreasing (dissipative).
3. **Efficiency Optimization (ET-L-ARC):** Bypasses Dissipation Guard calculations under low routing uncertainty ($H < 0.15$ or $H > 0.95$), collapsing the relative serving latency overhead to only **99.85%** (adding just $0.06$ ms of wall-clock latency per sample).
4. **State Correction (RASC):** Addresses systematic feedforward router corruption by closing the loop between feedforward predictions and feedback representation-space coordinate tracking, completely neutralizing the "state-locking" failure of stateful kinetics under persistent routing bias.

The authors evaluate L-ARC within the 14-layer **Analytical Coordinate Sandbox (ICS)** across four distinct scenarios: Setting A (static centroids), Setting B (layer-specific centroids), Setting C (transient routing failures), and Setting D (confident systematic routing bias).

---

## 2. Strengths (Theoretical Rigor & Analytical Brilliance)

The paper is exceptionally strong, characterized by high mathematical rigor, creative control-theoretic formulation, and scientific transparency.

### 2.1. Mathematical Rigor of the Lyapunov Stability Framework
The theoretical formulation of the representation warping update as a state-space dynamical system is highly elegant. 
- **Theorem 3.3 (Unconditional Non-Negativity of the Dissipation Coefficient):** The proof that the dissipation coefficient $A = c \| P_{h^\perp}(\bar{\mu}) \|_2^2 \ge 0$ is mathematically flawless and represents an exceptional theoretical contribution. It rigorously reveals that the rate of error dissipation is governed strictly by the magnitude of the ensembled centroid's projection orthogonal to the current hidden state.
- **Proposition 3.1 & Remark 3.2 (Zero-Error Incompatibility Under Multi-Expert Activation):** The authors show outstanding theoretical depth by identifying that when multiple orthogonal experts are active, the system cannot reach a zero-error state $V = 0$ due to Bessel's inequality on the unit sphere ($\sum_k S(h, \mu_k)^2 \le 1$). Deriving the analytical lower bound:
  $$V \ge \sum_{k=1}^K C_k^{(l)} - \max_{k} C_k^{(l)} > 0$$
  demonstrates a deep, expert-level understanding of the geometry of the representation space. They correctly note that this does not invalidate the Dissipation Guard, which only requires a dissipative directional update ($\Delta V^{(l)} \le 0$) to drive representations towards the ensembled centroid.

### 2.2. Rigorous Approximation Error Bounding (Theorem 3.2)
To justify the **Layer-Identity Approximation** ($S(h^{(l-2) \text{ warped}}, \mu_k^{(l-1)}) \approx S(h^{(l-1)}, \mu_k^{(l)}$)), which assumes adjacent layers are close to an identity mapping modified only by a small residual block, the authors prove Theorem 3.2. This theorem bounds the approximation error by:
$$e^{(l)} \le 2 \| r^{(l-1)} \|_2 + \| \mu_k^{(l)} - \mu_k^{(l-1)} \|_2$$
Under the unified early-stage centroid approximation, the second term vanishes, tightening the bound to $2 \| r^{(l-1)} \|_2$. This proof is highly complete, correct, and establishes a solid theoretical guarantee on the validity of the linearized discrete-time Lyapunov difference equation.

### 2.3. Cohesive Integration of Gating Filters
The theoretical unification of state-space shielding (ECG-Reset) and representation-space shielding (Dissipation Guard) is highly satisfying. Section 3.7 provides a brilliant discussion showing that ECG-Reset is a mathematically necessary theoretical precondition. Without ECG-Reset, transient failures accumulate routing noise, forcing concentrations to a uniform state and collapsing the Lyapunov function. ECG-Reset freezes the kinetics, ensuring the concentrations remain valid coordinate weights and preserving the positive semi-definiteness of the Lyapunov function.

### 2.4. Unprecedented Scientific Transparency and Honesty
The authors exhibit an outstanding level of scientific integrity, which is highly commendable. 
- They explicitly report that under clean serving workloads (Setting A), active representation feedback warping is **not statistically significant** ($p = 0.0969$, t-statistic of $1.8531$) and honestly advise edge-device practitioners to simply use decoupled kinetics ($\eta = 0$) to save latency.
- They clearly document the limitations of L-ARC's kinetics lag, the Layer-Identity assumption, and the synthetic evaluation environment.

---

## 3. Weaknesses & Critical Theoretical Limitations

While the paper is outstanding and highly complete, there are still a few areas that can be improved to further elevate the submission:

### 3.1. Stylized Synthetic Evaluation Sandbox (ICS)
The primary limitation of this work is that the entire empirical evaluation is conducted within the 14-layer Analytical Coordinate Sandbox (ICS). While the sandbox is mathematically controlled and helps isolate variables for control-theoretic analysis, it remains a highly stylized, synthetic simulation. There is no evaluation on full-scale transformer backbones (such as LLaMA, Mistral, or ViT) executing high-dimensional NLP or computer vision tasks. Real transformers introduce complex multi-head attention and highly non-linear feed-forward (MLP) layers, which could introduce significant noise and challenge the stability of the control loop. Although Section 4.1 outlines a clear, plug-and-play mapping to real transformers, the lack of actual validation on full-scale models remains a notable empirical constraint.

### 3.2. Fragility of the Layer-Identity Assumption under Large Residual Updates
The derivation of the discrete Lyapunov difference equation relies on the **Layer-Identity Approximation** ($S(h^{(l-2) \text{ warped}}, \mu_k^{(l-1)}) \approx S(h^{(l-1)}, \mu_k^{(l)})$).
- In the proof of Theorem 3.2, the step $\|h^{(l-1)} - h_w\|_2 \le 2\|r^{(l-1)}(h_w)\|_2$ assumes that $\|h_w + r^{(l-1)}(h_w)\|_2 \ge 1$, which is justified by noting that residual updates are designed to be constructive or near-orthogonal (i.e., $h_w \cdot r^{(l-1)}(h_w) \ge 0$). While this holds in stable networks with layer normalization, highly destructive or contractive transformations could violate this, which represents a subtle physical assumption that must be noted.
- Furthermore, in actual deep neural networks (especially the early or middle layers of transformer models), the residual block outputs can be highly transformative and non-linear, meaning the residual scale $\|r^{(l-1)}\|_2$ can be substantial. The authors' sweep over the residual scale $\gamma$ in Section 4.3 empirically confirms that large residual updates violate the Layer-Identity assumption and degrade the controller's active feedback gains toward the decoupled baseline. The theoretical framework would benefit from a more detailed discussion of how this assumption holds up under highly transformative deep layers.

### 3.3. Trade-off between Kinetics Lag and Representation-Space Distortion (Setting C)
Under transient routing failures (Setting C, Table 2), the stateless **SPS-ZCA SOTA** baseline achieves a final-layer Semantic Similarity of **0.8270 ± 0.0042**, which is substantially superior to full L-ARC (**0.7813 ± 0.0075**), even though L-ARC has a higher ensembling accuracy ($73.97\%$). 
- This occurs because SPS-ZCA operates strictly at early stages and is completely stateless. It completely avoids the kinetics propagation lag and representation warping of noisy, corrupted signals across depth, keeping its activations closer to the clean, early task manifolds (yielding higher similarity), although its ensembling weights are less refined. This highlights that active feedback warping, even when guarded, can still introduce a minor representational distortion relative to simple early-stage static routing, and is a trade-off that deserves explicit discussion.

---

## 4. Questions and Suggestions for the Authors

1. **Transformer MLP Block Analysis:** In a standard transformer layer, representations undergo highly non-linear transformations through the MLP/FFN blocks. Would you recommend applying L-ARC's warping globally across all sub-layers, or restricting feature warping solely before the self-attention blocks where ensembled adapters are active? This could minimize the Layer-Identity Approximation error.
2. **Kinetics Lag Mitigation:** To address the "kinetics propagation lag" in early layers under Setting B, have you considered utilizing a dynamic step-size schedule (e.g., larger $\Delta t$ in earlier layers to accelerate convergence, followed by smaller $\Delta t$ in later layers to stabilize routing)?
3. **Real-World Calibration Overhead:** How sensitive are the task centroids $\mu_k^{(3)}$ to the choice of calibration split size? Have you evaluated the performance under extremely low-data calibration splits (e.g., fewer than 10 samples) to see if representational drift increases?
