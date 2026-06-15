# 4. Experimental Evaluation Check

## Evaluation of the Experimental Setup
The authors evaluate their proposed Lotka-Volterra Competitive Serving (LVCS) model using a two-tiered evaluation strategy:
1. **The Coordinates Sandbox (CS):** A highly controlled, synthetic 14-layer representation simulation testbed representing mutually orthogonal or overlapping task manifolds. It simulates streams of 1000 queries under both homogeneous (long stationary blocks) and heterogeneous (rapid, step-by-step task transitions) serving conditions.
2. **Real-World Sequence Classification (BERT-Tiny on GLUE):** A high-fidelity sequence classification pipeline using SST-2, MRPC, and CoLA. This evaluates downstream ensembled classification accuracy on a mixed sequence stream of 1200 total queries, utilizing fine-tuned LoRA experts.

### Strengths of the Setup
*   **Controlled Complexity:** The Coordinates Sandbox allows the authors to study isolated geometric properties (overlapping vs. orthogonal representations) and temporal dynamics (switching frequency) in a clean environment, free from confounding training variables.
*   **Rigorous Statistical Validation:** All quantitative results in the sandbox are averaged over 5 independent random seeds ($42$ to $46$ inclusive), reporting both the mean and standard deviation. This ensures that the reported accuracy improvements are statistically significant and robust.
*   **Diverse and Competitive Baselines:** The paper compares LVCS against a highly comprehensive suite of 9 baselines, covering:
    *   Stateless baselines (SABLE, Uniform)
    *   Stateful continuous/constant baselines (ChemMerge, Momentum-Merge)
    *   Stateful linear baselines (PAC-Kinetics (Vanilla))
    *   Static learned baselines (Softmax, MLP)
    *   Unconstrained non-linear recurrent baselines (GRU Router)
*   **Exceptional Fairness in Comparisons:** 
    *   **Dynamic SABLE:** Unlike prior static-copying simulations, the authors implement a dynamic, layer-wise rollout for SABLE, allowing it to exhibit its actual depth-wise routing jitter.
    *   **Ablative PAC-Kinetics (Augmented):** To decouple the benefit of Lotka-Volterra non-linear recurrence from the benefit of temporal stream similarity tracking, the authors implement PAC-Kinetics (Augmented), which adds their proposed Adaptive Niche Plasticity gating ($Sim_t$) to the baseline. This isolates the Lotka-Volterra recurrence as the source of the remaining performance gains.

---

## Evaluation of Results and Claims Support
The empirical results strongly and consistently support the authors' central claims:

### 1. Robustness under Overlapping Manifolds
On Overlapping Manifolds (where representation leakage is severe), LVCS outperforms all state-of-the-art baselines. Under heterogeneous streams, LVCS (Static) achieves **90.06%** accuracy, outperforming PAC-Kinetics (Vanilla) (**88.72%**) by **+1.34%** absolute and PAC-Kinetics (Augmented) (**88.68%**) by **+1.38%** absolute. Under homogeneous streams, LVCS (Dynamic) achieves **89.26%** accuracy, outperforming PAC-Kinetics (Vanilla) by **+1.20%**. This supports the claim that the non-linear coupled self-regulation (Winner-Take-All competitive dynamics) of the Ricker recurrence successfully prunes minor representational leaks and isolates expert representations.

### 2. Generalization to Real-World Representations
On the real-world BERT-Tiny GLUE evaluation, LVCS (Static) achieves **61.25%** downstream sequence accuracy, outperforming stateless SABLE and stateful PAC-Kinetics (**60.25%**), Softmax (Static) (**60.08%**), and MLP (Static) (**61.00%**), while being highly competitive with the overparameterized GRU Router (**61.42%**). Crucially:
*   **Parameter Efficiency:** LVCS achieved this high performance using only **24 parameters**, which is $5\times$ to $16\times$ fewer than MLP (Static) (115 parameters) and GRU Router (404 parameters).
*   **Regularization Benefit:** Black-box MLPs overfit the clean, synthetic sandboxes but lose their advantage in messy, real-world embeddings. The structured ecological constraints of LVCS act as an effective inductive bias that prevents representation-space overfitting.

### 3. Systems-Level Utility
*   **Static Coordinate Efficiency:** The static approximation (LVCS (Static)) achieves virtually identical accuracy to the theoretically pure dynamic model (LVCS (Dynamic))—within 0.1% to 0.2% accuracy—while reducing serving latency by **over 51%** (1626.34 $\mu$s vs 3335.69 $\mu$s).
*   **Throughput Scalability:** Vectorized CPU benchmarking demonstrates that query throughput scales super-linearly (from 703 QPS at $B=1$ to 86,933 QPS at $B=1024$), and the computational overhead of the sequential Ricker loop collapses from $51.88\%$ to $20.37\%$. This proves that the implementation avoids serialization bottlenecks and is highly viable for production.

---

## Nuanced Analysis of Routing Jitter
The authors provide an excellent, theoretically rigorous discussion on routing jitter:
*   Stateless SABLE reports an extremely low spatial Jitter of $\sim 0.010$, which is an artifact of stateless soft-gating where weights remain close to uniform at every layer (causing severe representation leakage).
*   LVCS reports a spatial Jitter of $\sim 0.070$, which is the exact mathematical signature of **Active Competitive Sharpening** (transitioning systematically from a uniform starting state at Layer 3 to a sharp, low-entropy selection across depth).
*   Therefore, the higher depth-wise "jitter" of LVCS is a directed convergence that isolates expert adapters, whereas its temporal (query-wise) trajectory is highly smoothed and stable compared to SABLE's high temporal jitter.
