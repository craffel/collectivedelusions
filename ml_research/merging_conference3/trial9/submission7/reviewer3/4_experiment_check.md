# 4. Experiment Check

## Critical Evaluation of the Experimental Setup
The authors evaluate their proposed L-ARC framework within the **Analytical Coordinate Sandbox (ICS)**. The ICS simulates representation propagation across a 14-layer deep network in a $D=192$ dimensional coordinate space with $K=4$ distinct task manifolds (MNIST, Fashion-MNIST, CIFAR-10, SVHN) mapped to orthogonal sub-spaces. 

### Strengths of the Setup:
1. **Isolated Variables:** The coordinate sandbox is an excellent and mathematically controlled environment for evaluating control-theoretic principles. It avoids confounding factors (such as complex attention heads or dense FFN layers) to isolate the impact of representational drift, continuous kinetics, and closed-loop control.
2. **Realistic Serving Configurations:**
   * **Setting A (Static Centroids):** Centroids are extracted once at Layer 3 and reused across depth. This simulates realistic serving constraints where storing/loading layer-wise centroids is memory-prohibitive.
   * **Setting B (Layer-Specific Centroids):** Storing unique centroids for all layers, representing ideal tracking but with high-overhead.
3. **Rigorous Stress-Testing of System Faults:**
   * **Setting C (Transient Routing Failures):** Simulated dropouts with $p_{\text{fail}} = 0.20$ yielding uniform noise, evaluating kinetics memory corruption.
   * **Setting D (Confident Router Bias):** Confident but systematically biased routing updates, evaluating "state-locking" under persistent corruption.
4. **Complementary and Non-Circular Metrics:** By evaluating both **Joint Mean Accuracy** (classification proxy) and **Semantic Similarity to $v_k$** (direct activation-space cosine similarity to the true task manifold coordinate), the authors avoid circular reasoning and directly assess whether feedback warping causes semantic corruption.
5. **Real-World Pilot Study on LLaMA-3-8B:** To bridge the gap between sandbox simulations and real-world massive models, the authors fine-tune three specialized LoRA adapters on SST-2, AG-News, and GSM8K, demonstrating that extracted centroids are indeed orthogonal in 4096 dimensions ($\mu_i \cdot \mu_j \le 0.08$) and that L-ARC recovers 98% of Oracle serving accuracy (+5.00% over SABLE).

---

## Evaluation of Baselines
The paper compares L-ARC against a strong set of six baselines, including:
* **Stateless SOTA:** SPS-ZCA SOTA [spszca2025] and SABLE SOTA [sable2025].
* **Stateful Open-Loop:** ChemMerge (Decoupled, $\eta = 0$) [chemmerge2025].
* **Heuristics suggested by peer reviewers:** EMA-SABLE (smoothing heuristic) and Decay-ChemMerge (unconditional linear decay of feedback).
This comprehensive suite of baselines allows the authors to successfully isolate and demonstrate the value of each component of L-ARC (stateful kinetics, closed-loop feedback, ECG-Reset, and RASC).

---

## Do the Results Support the Claims?
Yes, the empirical results strongly and rigorously support the paper's claims:

### 1. Robustness under Practical Static Centroids (Setting A)
* **Claim:** L-ARC prevents representational backward-shift and out-performs stateless models under practical memory constraints.
* **Support:** In Table 1 (Setting A), SABLE's semantic similarity collapses to $0.7590$ due to representational drift, while L-ARC maintains a high semantic similarity of **0.7937 ± 0.0059** and Joint Mean Accuracy of **74.38% ± 0.31%**.

### 2. Statistical Redundancy under Clean Serving
* **Claim:** Active feedback is redundant under clean workloads.
* **Support:** The authors honestly report a paired t-test comparing L-ARC and decoupled ChemMerge showing $p = 0.0969$ (not statistically significant). They advise edge practitioners that simply running decoupled kinetics is preferred under pristine serving, enhancing the paper's scientific credibility.

### 3. State-Space Shielding via ECG-Reset (Setting C)
* **Claim:** ECG-Reset prevents memory corruption under transient dropouts.
* **Support:** Under Setting C (Table 2), open-loop ChemMerge collapses to $68.79\%$ due to persistent memory faults. L-ARC (ECG-Reset Only, $\eta = 0$) achieves **73.93% ± 0.41%** (a massive **+5.14%** absolute improvement), confirming that the sample-and-hold gating successfully shields the ODE state space.

### 4. Overriding Systematic Bias via RASC (Setting D)
* **Claim:** RASC resolves state-locking under confident systematic bias.
* **Support:** Under Setting D (Table 3), ChemMerge collapses to $68.52\%$ due to state-locking. L-ARC (Ours, with RASC) achieves an outstanding **73.59% ± 0.39%** accuracy and **0.7467 ± 0.0082** similarity, yielding extreme statistical significance ($p = 0.0000$) over open-loop kinetics.

---

## Experimental Gaps and Critiques (Theorist Lens)

Despite the overall excellence, two key experimental nuances should be highlighted:

### 1. Accuracy vs. Representational Distortion Trade-off under Failures
In Table 2 (Setting C), although L-ARC achieves the highest downstream task classification accuracy ($73.97\% \pm 0.39\%$), the stateless **SPS-ZCA SOTA** baseline achieves a significantly superior final-layer Semantic Similarity (**0.8270 ± 0.0042** vs. L-ARC's **0.7813 ± 0.0075**). 
* **Critique:** This exposes a fundamental trade-off: stateful ensembling (L-ARC) actively warps activations toward the blended centroids to maximize routing tracking, which means any routing noise that leaks through the filters still introduces minor cumulative representational distortion relative to early-stage, single-pass stateless routing. The paper should discuss this trade-off more explicitly: while stateful ensembling maximizes classification performance, it carries a slight representational distortion penalty under severe transient noise compared to early-stage static routing.

### 2. Linear Accuracy Proxy
The main Joint Mean Accuracy is mapped using a linear interpolation proxy from final-layer ensembling weights. Real-world adapters can have non-linear threshold effects. Although the authors address this by sweeping a non-linear step-threshold activation mapping in Section 3.8 and showing L-ARC retains a consistent $+3.81\%$ gain, full-scale end-to-end task accuracies on standard transformer benchmarks would make the empirical results even more robust.
