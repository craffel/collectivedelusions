# Experimental Evaluation Critique: Lyapunov-Stable Active Representation Coupling (L-ARC)

## 1. Marginal and Statistically Insignificant Accuracy Gains on Clean Workloads
In Table 1, under **Setting A: Static Centroids** (which represents the highly practical, resource-constrained serving scenario):
*   **L-ARC (Ours):** **74.38% $\pm$ 0.31%** Joint Mean Accuracy.
*   **ChemMerge (Decoupled, $\eta=0$):** **74.33% $\pm$ 0.34%** Joint Mean Accuracy.
*   **The Delta:** A microscopic **0.05%** improvement in accuracy.
*   **Statistical Significance:** As the authors admit in Section 4.3, a paired t-test over 10 seeds yields a p-value of **0.0969** ($p > 0.05$), meaning **this improvement is not statistically significant**.
*   **Comparison to Heuristic:** A simple linear decay of the feedback step size (**Decay-ChemMerge**) achieves **74.38% $\pm$ 0.30%** accuracy, which is **identical** to full L-ARC.
*   **Critical Implication:** The core proposed novelty of the paper—the Lyapunov feedback warping and Dissipation Guard—is practically redundant under standard clean serving conditions. It performs identically to a simple, computationally lightweight linear decay heuristic and is not statistically superior to disabling feedback warping altogether.

---

## 2. Underperformance under Ideal Centroids (Setting B)
In Table 1, under **Setting B: Layer-Specific Centroids** (where unshifted, high-quality centroids are available for each layer):
*   **SABLE SOTA:** **74.82% $\pm$ 0.32%** Joint Mean Accuracy.
*   **EMA-SABLE (Heuristic):** **75.00% $\pm$ 0.33%** Joint Mean Accuracy.
*   **L-ARC (Ours):** **74.46% $\pm$ 0.31%** Joint Mean Accuracy.
*   **Critical Implication:** 
    L-ARC is **outperformed** by stateless SABLE by $0.36\%$ and a simple EMA smoothing heuristic by $0.54\%$. 
    The authors blame this on the "kinetics propagation lag" (inertial delay of the continuous-time ODE). While control-theoretically interesting, this lag is a fundamental physical limitation of their stateful kinetics model. In practice, it means that under high-quality centroids, the proposed mathematically complex model actually degrades performance compared to a simple, low-overhead moving average (EMA-SABLE).

---

## 3. Redundancy of the Feedback Controller under Failures (Setting C)
In Table 2, under **Setting C: Transient Routing Failures** (20% random dropouts):
*   **L-ARC (Ours):** **73.97% $\pm$ 0.39%** Joint Mean Accuracy.
*   **L-ARC (ECG-Reset Only, $\eta=0$):** **73.93% $\pm$ 0.41%** Joint Mean Accuracy.
*   **The Delta:** A minuscule **0.04%** improvement.
*   **Statistical Significance:** A paired t-test yields a p-value of **0.3443**, which is **extremely statistically insignificant**.
*   **Critical Implication:** 
    Under transient failures, the entire Lyapunov Feedback Controller and Dissipation Guard provide **zero meaningful classification accuracy benefits**. 
    The massive accuracy improvement (+5.14% over open-loop ChemMerge's 68.79%) is driven **entirely by ECG-Reset** (the entropy gating of the integration step size). The actual "active representation feedback" contributes nothing to accuracy, meaning the paper's core theoretical apparatus is completely redundant for fault-tolerant serving.

---

## 4. Contradictory Representational Distortion under Noise
In Table 2 (Setting C), the authors report the "Semantic Similarity to $v_k$," which measures representation quality directly:
*   **SPS-ZCA SOTA:** **0.8270 $\pm$ 0.0042** Semantic Similarity.
*   **L-ARC (Ours):** **0.7813 $\pm$ 0.0075** Semantic Similarity.
*   **Critical Implication:** 
    The stateless SPS-ZCA baseline achieves a **significantly higher** semantic similarity to the true task signatures than L-ARC under noisy serving. 
    This indicates that despite L-ARC's "Dissipation Guard" gating off non-dissipative updates, the remaining active feedback warping still introduces cumulative representational distortion (pulling features off-manifold under noise). This directly undermines the authors' claims that L-ARC provides "absolute representation-space and semantic-space protection."

---

## 5. Unfavorable Latency vs. Accuracy Trade-off
In Section 4.4, the authors profile the average execution latency of the methods:
*   **ChemMerge (Decoupled, $\eta=0$):** **60.29 ms**
*   **L-ARC (Ours, with ET-L-ARC):** **120.50 ms**
*   **Critical Implication:** 
    L-ARC **doubles** (100% increase) the ensembling routing latency compared to Decoupled ChemMerge. 
    While the authors argue that this is "only 0.06 ms per sample," in resource-constrained edge serving, a 2x latency overhead for a routing layer that provides no statistically significant accuracy benefits under standard or failing conditions (over simple decoupled or pure state-gated baselines) is a highly unfavorable engineering trade-off. 
    The high-dimensional dot-products and Einstein contractions required on-the-fly inside the Dissipation Guard represent a substantial computational tax for zero statistical gain.
