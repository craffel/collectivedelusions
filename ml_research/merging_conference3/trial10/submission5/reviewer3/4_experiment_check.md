# Critical Experimental Evaluation

## Experimental Setup and Dataset Representativeness
The experimental evaluation is divided into two parts: a synthetic coordinate sandbox (ICS) and a real-world multi-task text classification benchmark.

1. **Analytical Coordinate Sandbox (ICS):**
   * *Critique:* The sandbox is a 14-layer, 192-dimensional synthetic environment. While highly controlled and ideal for isolating routing and trajectory dynamics across network depth, its dimensions are exceptionally small compared to contemporary deep learning architectures (e.g., modern Transformer blocks have hidden dimensions of 4096+ and dozens of layers). It remains unclear if the spatial-temporal dynamics observed in this toy coordinate space translate to the complex activation manifolds of massive networks.

2. **Real-World NLP Text Classification Benchmark:**
   * *Critique:* The "real-world" benchmark uses the classic **20newsgroups** dataset with TF-IDF features ($D=1024$) and a simple **2-layer MLP** (128 hidden units).
   * *Critical Concern:* A 2-layer MLP on TF-IDF features is a highly outdated, toy NLP setup. It does not represent contemporary Mixture-of-Experts serving challenges, which typically involve autoregressive, token-by-token generation in Large Language Models (LLMs) like LLaMA or Mistral, or vision adapters in Vision Transformers (ViT). Autoregressive decoding features distinct generation dynamics, token-level task transitions, and multi-turn semantic coherence. 
   * Furthermore, the expert pool size is extremely small ($K=4$), and the experts are trained on biased subsets (80% of samples belong to domain $k$) rather than being fully specialized or jointly optimized. The paper discusses scaling to $K=16$ or $K=32$ experts and "centroid crowding," but offers **zero empirical results** demonstrating UGR's performance under larger expert pools. An empirical validation with a larger expert pool is crucial to prove that "concentration of measure" on high-dimensional spheres does not degrade the self-regulating torque feedback loop.

---

## Evaluation of Baselines and Fairness of Comparisons

1. **Baselines Selection:**
   * The paper compares UGR against a reasonable set of baselines: *Expert Ceiling (Oracle)*, *Uniform Merging (Static)*, *SABLE (Stateless)*, *ChemMerge (SOTA Biochemical)*, and *Momentum-Merge (SOTA Euclidean EMA)*.
   * However, are there other dynamic model ensembling or routing baselines that should have been included? For example, dynamic gating networks, or simple Softmax-based sliding-window routing? SABLE is the only stateless dynamic routing baseline compared.

2. **Fairness of Comparisons and Baseline Tuning:**
   * *The Momentum-Merge Jitter Bias:* The authors deserve credit for identifying an initialization discrepancy in previous reports of the *Momentum-Merge (Advanced)* baseline (where the boundary prior at layer 3 was being overwritten with the target vector of layer 4, artificially suppressing its transition jitter). Once corrected, its Jitter ($L \ge 5$) rises from $2.73 \times 10^{-4}$ to $68.64 \times 10^{-4}$. This code-level audit represents high scientific hygiene.
   * *The ChemMerge Step Size Constraints:* The paper states that ChemMerge exhibits high routing jitter ($4.098 \times 10^{-3}$) because virtual-time numerical ODE integration becomes unstable under a large step size of $dt=1.5$ (necessary for low-latency serving). 
   * *Critique:* Why is $dt=1.5$ strictly necessary? Is there an empirical wall-clock latency comparison showing that a smaller, stable $dt$ (e.g., $dt=0.1$ with multiple steps) actually introduces an "unacceptable computational bottleneck" for such a lightweight MLP or sandbox? Without wall-clock latency curves for varying ODE step sizes, the claim that ChemMerge's instability is an unavoidable latency-accuracy trade-off is an unproven assumption.

---

## Statistical Soundness
* **Multiple Random Seeds:** Standard results are evaluated across **10 independent random seeds** (synthetic sandbox) and **5 independent random seeds** (NLP text classification), which is a major strength.
* **Confidence Intervals:** Mean values and standard deviations are reported (e.g., 92.25% $\pm$ 0.90% accuracy). This provides excellent statistical transparency.
* **Symmetry in Memory Configurations:** Evaluating both "Reset" and "Coupled" configurations for all stateful baselines (ChemMerge and Momentum-Merge) is a highly commendable decision. It isolates the performance gains of the curved geodesic flow itself from the simple cross-query memory-coupling mechanism, ensuring absolute scientific symmetry.

---

## Do the Empirical Results Support the Claims?

1. **Claim: UGR achieves state-of-the-art accuracy and stability.**
   * *Supported?* **Yes, but within a limited scope.** In both the synthetic sandbox and the 20newsgroups MLP stream, UGR consistently achieves the highest classification accuracy and suppresses intra-query routing jitter ($L \ge 5$) compared to other coupled/reset stateful baselines.
   * However, the massive +21.60% absolute accuracy margin over ChemMerge on the text classification task is highly suspicious. Why does ChemMerge perform so poorly (70.67% accuracy, barely better than static uniform at 64.95%) on the text classification stream? The authors note that ChemMerge's state variables oscillate violently and clip at boundaries under $dt=1.5$. This suggests that ChemMerge was evaluated in a highly suboptimal, unstable regime, raising questions about whether it was tuned fairly.

2. **Claim: Torque-Driven Agility resolves the stability-plasticity dilemma.**
   * *Supported?* **Yes.** The decomposed jitter analysis (Section 4.4.3) is highly convincing. It proves that UGR achieves low *Intra-Task Jitter* (12.31 $\times 10^{-4}$) for stability, while maintaining high *Inter-Task Jitter* (21.79 $\times 10^{-4}$) at task boundaries for agile adaptation. In contrast, Momentum-Merge shows zero separation (68.79 $\times 10^{-4}$ vs. 68.53 $\times 10^{-4}$), proving its trajectory is dominated by unconstrained flat-space noise. This decomposition is a brilliant empirical validation of the torque mechanism.

3. **Claim: UGR is highly computationally efficient and viable for production-grade serving.**
   * *Supported?* **Yes.** The timing benchmarks (Table 4) on an Intel Xeon CPU core show that UGR adds $<0.07$ ms of latency per query over stateless SABLE. Crucially, the fully Softmax-free variant is shown to be significantly faster (0.436 ms / 2295.3 QPS), demonstrating excellent practical viability.
