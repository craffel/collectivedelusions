# 4. Experimental Check

## Rigor of Experimental Design
As an empiricist reviewer, we find the empirical rigor of this paper to be exceptional and far exceeding the standard in modern machine learning publications:
* **Statistically Sound Sample Size:** Rather than relying on a single run or a standard 3-seed evaluation, the authors evaluate all methods across **10 independent, perfectly synchronized random seeds** in the synthetic Analytical Coordinate Sandbox (ICS), and across **5 independent seeds** on the real-world text classification task. All major results are reported as `mean ± standard deviation`.
* **Code-Level Baselining Hygiene:** The authors conducted a thorough code-level audit of previous stateful ensembling baselines. They discovered that prior implementations of the *Momentum-Merge (Advanced)* baseline had an initialization bug where the boundary prior at layer 3 was overwritten with the target vector of layer 4, artificially eliminating any transition shock and yielding suppressed jitter statistics. The authors corrected this "target-injection cheat" by enforcing a strict, scientifically hygienic uniform initialization ($1/K$) for all uncoupled models, raising the baseline's jitter and ensuring a fair comparison.
* **Isolating Confounding Factors:** The authors compare both the *Reset* (stand-alone serving) and *Coupled* (recurrent cross-query serving) configurations of all stateful baselines (ChemMerge and Momentum-Merge). This isolative setup allows them to scientifically decouple the performance gains of the curved geodesic flow geometry from the temporal coupling mechanism itself.

## Evaluation of Results and Evidence Supporting the Claims
The empirical results provide rock-solid support for the paper's central claims:

1. **Superior Accuracy Claim:**
   * *Evidence:* In the synthetic sandbox (Table 1), UGR achieves **75.08%** Joint Mean Accuracy, outperforming SABLE (74.74%), ChemMerge Reset (69.65%), and Momentum-Merge Advanced Reset (74.69%). On the real text stream (Table 2), UGR achieves **92.25%** Joint Mean Accuracy, outperforming Coupled Momentum-Merge by **+4.13%** and Coupled ChemMerge by **+21.60%** absolute.
2. **Superior Trajectory Stability Claim:**
   * *Evidence:* In Table 1, UGR reduces intra-query routing jitter ($L \ge 5$) to **19.51 $\times 10^{-4}$** (a **2.10$\times$ reduction** over ChemMerge). In Table 2, UGR slashes jitter to **3.68 $\times 10^{-4}$** (a **1.63$\times$ reduction** over Coupled Momentum-Merge).
3. **Resolving the Jitter Contradiction (Agility vs. Noise Jitter):**
   * *Evidence:* In Table 1, UGR's overall transition agility (Jitter $L \ge 4$) is higher than flat baselines under randomized streams. The authors resolve this contradiction by decomposing the layer-to-layer ensembling trajectory into *Intra-Task Jitter* (stability within identical tasks) and *Inter-Task Jitter* (agility during task switches). UGR exhibits a clean 1.8$\times$ stability-plasticity separation (Intra: **12.31 $\times 10^{-4}$**, Inter: **21.79 $\times 10^{-4}$**), whereas corrected Momentum-Merge shows virtually zero separation (Intra: **68.79 $\times 10^{-4}$**, Inter: **68.53 $\times 10^{-4}$**). Under a realistic block-structured stream, UGR's overall routing jitter naturally drops by 40% to **11.63 $\pm$ 1.39 $\times 10^{-4}$** while maintaining top-tier accuracy.
4. **Computational Efficiency Claim:**
   * *Evidence:* In Table 3, wall-clock latency benchmarks show that standard UGR adds only **0.07 ms** per query over the stateless baseline, achieving **2052.7 QPS** (and **2295.3 QPS** for the Softmax-Free target variant), significantly outperforming the ODE-based ChemMerge.

## Comprehensive Ablation Studies
The ablation studies are highly thorough and leave no stone unturned:
* **UGR (Born Target):** Evaluates the exact square-root target mapping $\mathbf{w}_t^{(l)} = \sqrt{\mathbf{e}_t^{(l)}}$ to eliminate quadratic sharpening distortion. It acts as a trajectory-smoothness maximizer, slashing routing jitter in NLP by **2.3$\times$ to 1.60 $\times 10^{-4}$** while achieving a robust **90.67%** accuracy.
* **UGR (Softmax-Free Target):** Replaces target Softmax with ReLU and $L_1$-normalization. It delivers unprecedented trajectory stability, slashing routing jitter in NLP to a pristine **1.50 $\times 10^{-4}$** (a **4.0$\times$ reduction** over Coupled Momentum-Merge).
* **UGR (Hybrid Reset):** Mitigates the single-layer boundary transition shock at layer 4 by wiping the state to uniform if the prior alignment falls below a threshold ($0.50$), slashing synthetic boundary shock by over **2.5$\times$** and improving NLP classification accuracy to **92.38%**.
* **Centroid Sample Efficiency:** Evaluates performance as a function of calibration samples (from 4 to 128), showing that UGR remains stable ($91.82\%$ accuracy with only 4 samples), proving high sample-efficiency.
* **Continuous Damping and Reset sweeps:** Conducts a continuous sweep of damping parameter $\lambda \in [0, 1]$ showing that low continuous damping ($\lambda = 0.10$) acts as a superior, soft alternative to the hard threshold reset.

## Empirical Limitations and Weaknesses (Empiricist Critique)
While the statistical rigor and ablation depth are exemplary, an empiricist must highlight several practical limitations in the evaluation:

1. **Scale of Real-World Evaluation:**
   * *Critique:* The "real-world" evaluation is conducted on the classic `20newsgroups` dataset. While this dataset provides a controlled text-routing stream, it is a relatively small classification task. The representations are simple TF-IDF vectors (max features D=1024), and the expert models are 2-layer MLPs. For a top-tier machine learning conference (ICML 2026), the lack of empirical evaluation on modern, deep pre-trained transformer backbones (e.g., RoBERTa, LLaMA) with actual Parameter-Efficient Fine-Tuning (PEFT) LoRA experts on standard multi-task benchmarks (MMLU, GLUE) is a notable limitation. The authors present a highly detailed "Real-World Serving Blueprint" (Appendix Section A.3), but they do not empirically evaluate this scale.
2. **Hyperparameter Domain Sensitivity:**
   * *Critique:* The optimal step size $\eta$ shifts heavily between the high-frequency synthetic sandbox (75% switching probability, optimal $\eta=0.80$) and the stable NLP block-structured stream (2% switching probability, optimal $\eta=0.10$). This indicates that UGR's performance is sensitive to the expected switching frequency of the stream. In many real-world production environments, the switching frequency is not known a priori, meaning practitioners must hand-tune $\eta$ or risk carrying over stale temporal priors.
3. **Centroid Quality and Representation Noise:**
   * *Critique:* Although the authors ablate the number of calibration samples (showing high sample efficiency), the expert centroids are assumed to be static. In practice, task representations are highly overlapping, and streams can exhibit continuous domain shift. While the authors derive and validate an online centroid adaptation rule starting from random initialization in the Appendix, the main evaluations rely on pre-computed centroids. Evaluating the framework's robustness under active semantic drift on real-world text would strengthen the empirical claims.
