# Experimental Check and Evaluation - LDS-Kinetics

## 1. Experimental Setup and Baselines
The experimental setup is highly structured, evaluating:
- Two manifold geometries (Orthogonal and Overlapping).
- Two workload stream patterns (Homogeneous and Heterogeneous).
- Extensive noise and bias levels.
- A wide range of competitive baselines, including static average merging, stateless SABLE, static decay baselines, and Global PAC-Kinetics ($M=1$).

## 2. Verification of Claims vs. Empirical Evidence

### Claim 1: LDS-Kinetics achieves superior serving performance over global baselines.
- **Fact (Linear Sandbox):** On linear sandboxes (Tables 1 and 2), the absolute joint classification accuracy gains of LDS-Kinetics over Global PAC-Kinetics are virtually non-existent:
  - *Orthogonal Heterogeneous:* $66.79\%$ ($M=11$) vs. $66.73\%$ ($M=1$) — an improvement of **$0.06\%$**.
  - *Overlapping Heterogeneous:* $66.84\%$ ($M=11$) vs. $66.81\%$ ($M=1$) — an improvement of **$0.03\%$**.
  - *Homogeneous Workloads:* Exactly identical accuracy ($66.22\%$ and $66.25\%$) for both global and decoupled models.
- **Jitter Check:** On these same workloads, LDS-Kinetics actually increases ensembling jitter:
  - *Orthogonal Jitter:* $0.8002 \to 0.9269$ (a **$15.8\%$ increase** in jitter).
  - *Overlapping Jitter:* $0.8460 \to 0.8997$ (a **$6.3\%$ increase** in jitter).
- **Critique:** The reported accuracy gains are incredibly marginal and far smaller than the sequence-dependent workload standard deviation of $\sim 3.8\%$. Although the authors run paired $t$-tests to claim statistical significance, an absolute improvement of $0.03\%$ in a simulator is practically meaningless, especially when it comes at the cost of worsening the ensembling jitter by up to $15.8\%$.

### Claim 2: Under non-linear propagation (GELU + LN), stateful ensembling completely bridges the "stateful accuracy penalty."
- **Fact (Non-Linear Sandbox):** 
  - *Orthogonal Heterogeneous:* Global PAC-Kinetics gets $69.10\%$. Tri-Block ($M=3$) gets $69.40\%$ ($+0.30\%$ gain). Fully Decoupled ($M=11$) gets $69.30\%$ ($+0.20\%$ gain).
  - *Overlapping Heterogeneous:* Global PAC-Kinetics gets $68.40\%$. Tri-Block ($M=3$) gets $68.50\%$ ($+0.10\%$ gain). Fully Decoupled ($M=11$) suffers a **regression** to $68.00\%$ (**$-0.40\%$ loss** compared to the global baseline!).
- **Critique:** This represents a major empirical contradiction to the paper's core premise. Fully decoupling the ensembling kinetics across layers ($M=11$) actually *regresses* performance in overlapping manifolds under non-linear propagation. The authors have to backpedal and use a coarser Tri-Block ($M=3$) configuration as "spatial regularization" to preserve representation trajectory cohesion. This demonstrates that spatial homogeneity (or near-homogeneity) is actually beneficial for maintaining stable representational paths, and that fully decoupling the kinetics introduces destructive inter-layer representation drift.

### Claim 3: The model scales seamlessly to large expert pools ($K$).
- **Fact (Table 4):** As $K$ scales to $8, 12, 16$:
  - At $K=8$: Heterogeneous accuracy is $56.17\%$ (Global) vs. $56.18\%$ (LDS-Kinetics $M=11$) — **$+0.01\%$ gain**.
  - At $K=12$: Heterogeneous accuracy is $51.41\%$ (Global) vs. $51.45\%$ (LDS-Kinetics $M=11$) — **$+0.04\%$ gain**.
  - At $K=16$: Heterogeneous accuracy is $48.31\%$ (Global) vs. $48.34\%$ (LDS-Kinetics $M=11$) — **$+0.03\%$ gain**.
- **Critique:** The accuracy gains completely vanish at scale. The authors explain that the PAC-Bayesian complexity penalty constrains the $M=11$ parameters closely to the global default prior to prevent catastrophic overfitting. If the regularizer forces the model to behave almost exactly like the global baseline to survive, then the entire multi-block decoupled architecture becomes functionally redundant at scale.

## 3. Computational and Latency Overhead
Table 3 reports CPU execution latencies:
- **Global ($M=1$):** $29.72\ \mu\text{s}$ per step.
- **Tri-Block ($M=3$):** $88.45\ \mu\text{s}$ per step (**$+197.6\%$ overhead**).
- **Fully Decoupled ($M=11$):** $328.75\ \mu\text{s}$ per step (**$+1006.2\%$ overhead**).

While $328\ \mu\text{s}$ is low in absolute terms compared to large LLM forward passes, introducing a **10-fold serial routing slow down** for an accuracy improvement of $0.03\%$ (and a regression of $-0.40\%$ under non-linear overlapping streams) is an extremely poor trade-off. This represents a classic "high cost, zero gain" over-engineered solution.
