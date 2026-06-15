# Peer Review

## Summary of the Paper
The paper presents a rigorous methodological audit and experimental deconstruction of state-of-the-art (SOTA) activation-space dynamic model merging methods—specifically, SABLE (Sample-wise Activation Blending of Low-Rank Experts) and ChemMerge (which uses continuous-time chemical kinetics). The authors investigate the widely held consensus that classical parametric linear routers catastrophically fail in low-data calibration regimes ($N_{\text{cal}} \le 128$). They hypothesize that these reported failures are not due to fundamental representational limitations of linear gating, but are confounding artifacts of weak experimental methodology, specifically random weight initialization and a lack of proper regularization.

To audit these claims, the authors introduce a properly regularized, maximum-entropy zero-initialized classical linear router. They evaluate these gating systems within a 14-layer, 192-dimensional synthetic Analytical Coordinate Sandbox (ICS) under controlled representation anisotropy, and validate their findings on a real pre-trained BERT-Tiny model.

---

## Overall Recommendation

**Rating: 5: Accept**

**Justification:** 
The submission is a highly rigorous, well-reasoned, and methodologically sound audit of dynamic model merging. It applies Occam's razor to show that complex, physically-inspired SOTA ensembling methods have been evaluated against weak, under-regularized, and poorly initialized "straw-man" classical baselines. The paper's strength lies in its diagnostic clarity: it isolates the "small-sample bottleneck" as a standard overfitting artifact and provides a beautiful, mathematically sound control-theoretic deconstruction of stateful kinetics. While there are some minor theoretical gaps in the synthetic sandbox and some terminological inflation of standard practices, the paper performs an invaluable service to the machine learning community and will significantly raise evaluation standards in dynamic model ensembling.

---

## Strengths
1. **Rigorous Methodological Auditing:** The paper exposes a critical flaw in the current literature: comparing new routing methods against unregularized, randomly-initialized linear baselines on tiny datasets is a straw-man comparison.
2. **Elegant Control-Theoretic Deconstruction:** The deconstruction of ChemMerge's ODE kinetics as a closed-loop feedback controller acting as a temporal low-pass filter (and the associated analysis of the numerical instability of its discretized Euler solver under $\Delta t = 1.5$) is mathematically outstanding and highly insightful.
3. **Thorough Empirical Sweeps:** The systematic exploration of sample sizes ($N_{\text{cal}} \in [32, 4000]$), representation anisotropy ($\rho \in [0.0, 0.5]$), and regularization strengths ($\lambda$) provides a complete, multi-dimensional view of the ensembling landscape.
4. **Transparent Limitation Reporting:** The authors are highly honest about their evaluation limits, specifically discussing the simplicity of the synthetic sandbox, the direct logit blending shape constraints, and the under-fitted nature of the BERT-Tiny experts.

---

## Weaknesses & Technical Critiques

### 1. Mathematical Simplification of the Synthetic Sandbox (ICS)
The "Analytical Coordinate Sandbox (ICS)" is a 14-layer, 192-dimensional simulated network. To evaluate its soundness, we analyze its underlying recurrence relation.
Let $h_b^{(l)} \in \mathbb{R}^D$ be the representation at layer $l$, and let $v'_k$ be the target task prototypes. The blending layers $l \in [4, 14]$ evolve via:
$$h_b^{(l)} = h_b^{(l-1)} + \sum_{k=1}^K \alpha_{k, b} \gamma_V (v'_k - h_b^{(l-1)})$$
where $\gamma_V = 0.05$ is a constant scaling factor. Let $s_b = \sum_{k=1}^K \alpha_{k,b} v'_k$ represent the blended prototype vector, and let $\bar{\alpha}_b = \sum_{k=1}^K \alpha_{k,b}$ be the sum of ensembling weights.
We can rewrite the recurrence as:
$$h_b^{(l)} = \left( 1 - \gamma_V \bar{\alpha}_b \right) h_b^{(l-1)} + \gamma_V s_b$$

Under Softmax gating, the ensembling weights satisfy the partition of unity: $\bar{\alpha}_b = 1$. The recurrence simplifies to:
$$h_b^{(l)} = \left( 1 - \gamma_V \right) h_b^{(l-1)} + \gamma_V s_b$$
Subtracting $s_b$ from both sides yields:
$$h_b^{(l)} - s_b = \left( 1 - \gamma_V \right) \left( h_b^{(l-1)} - s_b \right)$$
Solving this recurrence from Layer 3 (the frozen boundary) to Layer 14 (the final layer) over $14 - 3 = 11$ steps:
$$h_b^{(14)} - s_b = \left( 1 - \gamma_V \right)^{11} \left( h_b^{(3)} - s_b \right)$$
$$h_b^{(14)} = \left( 1 - \gamma_V \right)^{11} h_b^{(3)} + \left[ 1 - \left( 1 - \gamma_V \right)^{11} \right] s_b$$

For $\gamma_V = 0.05$:
$$\left( 1 - \gamma_V \right)^{11} = 0.95^{11} \approx 0.5688$$
Therefore, the final representation is:
$$h_b^{(14)} \approx 0.5688 h_b^{(3)} + 0.4312 s_b$$

**Critique:**
- This mathematical derivation reveals that the sandbox operates as a **strictly linear contracting mapping** where $56.88\%$ of the final representation is determined by the initial noisy feature vector $h_b^{(3)}$, and only $43.12\%$ is determined by the blended task prototypes.
- While mathematically clean, this linear recurrence is a highly simplified model of actual deep networks. In real models, the blending of activations occurs layer-by-layer and is recursively transformed by non-linear attention and MLP blocks. The paper lacks a theoretical proof showing that this linear, non-hierarchical contraction behaves similarly to actual pre-trained transformers.

### 2. Numerical Instability of ChemMerge ODE Solver
The paper notes a "numerical hack" in ChemMerge's discretized implementation where concentration values are hard-clamped to $[0, 1]$ to prevent divergence under a large Euler step size ($\Delta t = 1.5$). We formally derive the stability bounds of this system.

ChemMerge models the concentration $C_k(t)$ of task species $k$ evolving via the ordinary differential equation:
$$\frac{dC_k(t)}{dt} = R_k(t) (1 - C_k(t)) - K_{\text{decay}} C_k(t)$$
Assuming the reaction rate $R_k(t) = R_k$ is constant, this is a first-order linear ODE:
$$\frac{dC_k(t)}{dt} = -\left( R_k + K_{\text{decay}} \right) C_k(t) + R_k$$

The continuous solution is stable and converges to a steady-state concentration:
$$C_k^*(t) = \frac{R_k}{R_k + K_{\text{decay}}}$$

The discretized Euler solver with step size $\Delta t$ updates the system as:
$$C_k(t+\Delta t) = C_k(t) + \Delta t \left[ -\left( R_k + K_{\text{decay}} \right) C_k(t) + R_k \right]$$
$$C_k(t+\Delta t) = \left[ 1 - \Delta t \left( R_k + K_{\text{decay}} \right) \right] C_k(t) + \Delta t R_k$$

For the discretization to be stable and avoid high-frequency oscillations or divergence, the coefficient of the homogeneous part must be non-negative:
$$1 - \Delta t \left( R_k + K_{\text{decay}} \right) \ge 0 \implies \Delta t \le \frac{1}{R_k + K_{\text{decay}}}$$

Under the paper's parameters ($K_{\text{decay}} = 0.3$ and $R_k \in [0, 1]$):
- In the worst-case (strongest reaction rate $R_k = 1.0$), we require:
  $$\Delta t \le \frac{1}{1.0 + 0.3} \approx 0.769$$

**Critique:**
- Since the actual implementation of ChemMerge uses $\Delta t = 1.5$, which is nearly double the stability limit ($1.5 > 0.769$), the discretized Euler update is **mathematically unstable**. It results in negative coefficients (e.g., $1 - 1.5(1.3) = -0.95$), which causes the concentration to oscillate and overshoot the physical boundary.
- If $C_k(t) = 0$ and $R_k = 1.0$, the update yields $C_k(t+1.5) = 1.5 > 1.0$.
- The authors' observation that this requires hard-clamping to $[0.0, 1.0]$ after each step is **mathematically correct and highly insightful**. The clamping is a crude stabilization mechanism to prevent chaotic divergence.
- However, the paper's methodology would be stronger if the authors had evaluated a mathematically sound discretization scheme (e.g., implicit Euler, or a smaller, stable step size $\Delta t \le 0.5$) to determine if ChemMerge's performance ceiling is an artifact of the unstable discretization or a true physical property of the continuous-time kinetics.

### 3. Lack of Formal Generalization and Sample-Complexity Bounds
The authors claim that the optimal regularization parameter $\lambda$ scales inversely with $N_{\text{cal}}$, citing a "bias-variance trade-off."
- **Theoretical Gap:** From a statistical learning theory perspective, this is consistent with classical generalization bounds. For a linear model optimized via Empirical Risk Minimization with L2 regularization, the Rademacher complexity bound on the generalization error suggests that the optimal regularization weight $\lambda^*$ is of order $O(1/N_{\text{cal}})$.
- Under $N_{\text{cal}} = 64$, the optimal $\lambda^* \approx 10^{-2}$, whereas under $N_{\text{cal}} = 4000$, the optimal $\lambda^* \approx 10^{-4}$.
- **Critique:** While this matches the empirical sweeps, the paper does not derive a formal generalization error bound or sample complexity bound for this specific multi-layer ensembling architecture. It relies on qualitative references to the "bias-variance trade-off," leaving a gap between the general statistical theory and the specific deep-routing application.

### 4. terminological Inflation of Standard Principles
The paper employs high-level terminology to describe standard, elementary deep learning practices:
- Rebranding standard zero-initialization as "Maximum-Entropy Zero-Initialization."
- Rebranding standard $L_2$ weight decay as "Proper L2 Regularized Calibration."
While the authors provide a physical/information-theoretic justification for this framing (i.e., complete starting symmetry and restricting the hypothesis space), a theory-minded reader may view this as unnecessary conceptual inflation. The paper should use standard, widely-accepted terminology while maintaining its conceptual arguments.

### 5. Limited Pre-trained Validation Scale
While validating on BERT-Tiny is a useful proof-of-concept, the model is too small (4 layers, hidden size 128) and evaluated on under-fitted experts ($58.80\%$ on SST-2 and $65.60\%$ on QQP). It fails to serve as a convincing generalizability proof for modern multi-billion parameter large language models (LLMs) or vision transformers (ViTs). Evaluating on a standard, medium-sized model (e.g., RoBERTa-base or LLaMA-3-8B with LoRA adapters) would significantly bolster the empirical claims.

---

## Detailed Ratings

### Soundness: Good
The empirical evaluations are meticulous and the deconstruction of baseline hyperparameter configurations is highly rigorous. The deconstruction of ChemMerge's numerical instability is mathematically sound. However, the theoretical gap regarding formal generalization bounds and the extreme simplicity of the synthetic sandbox prevent an "Excellent" rating.

### Presentation: Excellent
The paper is exceptionally well-written, clearly structured, and mathematically precise. The notation is consistent across all sections, and the figures and tables are well-designed and integrated cleanly into the narrative. The "Discussion and Methodological Guidelines" section is particularly strong, translating the empirical findings into actionable recommendations.

### Significance: Good
The paper performs a critical service to the community by correcting a widespread methodological error in the dynamic model merging literature. The concrete "Deployment Decision Matrix" has significant practical value for practitioners deploying multi-task systems on edge devices.

### Originality: Good
The paper's novelty is primarily diagnostic rather than constructive. It applies standard tools ($L_2$ regularization, zero-initialization) to expose baseline issues. However, the control-theoretic interpretation of stateful kinetics is highly creative and provides a novel conceptual perspective on trajectory smoothing.

---

## Questions for the Authors
1. **Mathematical Validation of Sandbox:** Can you provide a formal proof or mathematical analysis demonstrating that the linear attraction dynamics in your 14-layer synthetic sandbox (ICS) bound or structurally represent the generalization error or optimization landscape of real, non-linear deep networks?
2. **Stable ODE Solvers:** Have you evaluated ChemMerge using a mathematically sound, stable numerical integration scheme (e.g., implicit Euler or a smaller step size $\Delta t \le 0.5$ without hard-clamping)? Does its closed-loop feedback premium persist when the discretization is stable?
3. **Generalization Bounds:** Can you derive a formal Rademacher complexity bound or generalization error bound for your linear router inside a multi-layer activation blending architecture to theoretically justify the $O(1/N_{\text{cal}})$ scaling of $\lambda$?
4. **Scale up BERT Validation:** Have you considered scaling up your pre-trained validation to a medium-sized model (e.g., RoBERTa-base or LLaMA-3-8B) with fully converged LoRA adapters?
