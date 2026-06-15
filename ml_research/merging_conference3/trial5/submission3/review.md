# Peer Review: Robust Linear Routing (RLR)

**Overall Recommendation:** 4: Weak Accept (Technically solid paper with high clarity and a powerful, scientifically honest deconstruction of over-engineered trends using Occam's razor. However, there are significant logical inconsistencies and performance trade-offs that must be addressed to elevate the work to its full potential).

---

## 1. Summary of the Submission
This paper presents **Robust Linear Routing (RLR)**, a minimalist dynamic model merging framework designed to blend task-specific expert models on-the-fly. Guided by the principle of Occam's razor, the work critiques the recent trend of escalating architectural and conceptual complexity in dynamic model merging—most notably represented by *Quantum Wavefunction Superposition Merging (QWS-Merge; Vance et al., 2025)*. QWS-Merge introduces quantum-inspired metaphors (such as task eigenstates, phase projectors, and wave interference) to resolve a reported "catastrophic collapse" of classical linear routing on high-variance, out-of-distribution (OOD) tasks like SVHN.

The authors demonstrate that this reported collapse is not an inherent structural limitation of linear gating networks, but rather a standard, preventable overfitting and logit-variance issue arising from unregularized training on tiny calibration datasets. To prove this, the paper introduces RLR, which retains a classical 768-parameter linear routing layer, but stabilizes its optimization using standard techniques: $L_2$ weight decay (Frobenius norm regularization) and Softmax Temperature scaling. The gating network is calibrated in under a second on a tiny 64-sample calibration set using an unweighted, uniform multi-task cross-entropy loss.

Evaluated on a 4-task vision benchmark (MNIST, FashionMNIST, CIFAR-10, SVHN) using a ViT-Tiny backbone, the paper provides a definitive empirical deconstruction of the quantum-inspired paradigm, showing that simple classical routing—regularized or unregularized—is highly robust. In addition to direct comparisons with reported numbers, the authors include a local execution of the QWS-Merge baseline, a routing representation source layer ablation study, and a 2D hyperparameter sensitivity heatmap.

---

## 2. Key Strengths
1. **Compelling Adherence to Occam's Razor:** Deconstructing over-engineered, mathematically obfuscated deep learning trends (like quantum wavefunctions for model merging) using standard, well-understood classical baselines is of immense value to the community. This work acts as a crucial sanity check that redirects the field toward simplicity and transparency.
2. **Complete Alignment of Claims and Codebase:** The discrepancy in previous versions between the paper's mathematical formulation (unweighted uniform loss) and the implementation (task-balanced weights) has been **completely and successfully resolved**. Both `run_experiments.py` (lines 404-406) and `run_seed_sweep.py` (lines 211-213) now implement `task_weights = np.ones(len(tasks))`, fully realizing the unweighted calibration loss described in Section 3.3.
3. **Outstanding Intellectual Honesty and Self-Critique:** The authors display an exceptional level of transparency and academic rigor:
   * They honestly report and analyze when their regularized method (RLR) trades off a minor margin of peak homogeneous performance to secure robustness, explicitly acknowledging that RLR and the unregularized classical router are statistically indistinguishable in homogeneous mean performance.
   * In Section 4.4, they openly address the *heterogeneity collapse* of dynamic merging methods under mixed-task test streams, explaining why static methods like OFS-Tune remain superior in low-latency mixed settings. This provides exceptionally clear and realistic deployment guidelines for practitioners.
4. **Empirical Rigor via Multi-Seed Sweeps:** The multi-seed sweep over 5 random calibration seeds (Section 4.3) statistically proves that the classical Linear Router achieves an average SVHN accuracy of $91.20\% \pm 1.85\%$ and Joint Mean accuracy of $91.53\% \pm 0.41\%$. This completely debunks the $15.30\%$ collapse reported in prior work, showing it was merely an artifact of sub-optimal optimization rather than a structural failure.
5. **Extreme Parameter and Calibration Efficiency:** RLR requires only 772 parameters, optimizes in under a second on a single GPU for 100 steps on 64 calibration samples, introduces zero runtime overhead, and maintains an elegant, 100-line implementation.
6. **Exceptional Clarity and Presentation:** The paper is beautifully structured, professional, and grammatically flawless. The visualizations (`comparison_plot.png`, `heterogeneous_plot.png`, and Figure 3's sensitivity heatmaps) are highly polished, legible, and integrated seamlessly.

---

## 3. Crucial Weaknesses & Areas for Improvement (Key Critiques)

While the paper is technically solid and highly rigorous, we identify **three critical flaws and methodological gaps** that must be addressed:

### Critique 1: Severe Logical Inconsistency in the "Representation Warping" Hypothesis
The paper's core motivation and its empirical findings are in direct logical contradiction:
* **The Motivation:** In Section 3.2, the authors claim that deep representations are highly specialized and "task-warped," which triggers extreme logit variance and causes unregularized classical linear routing to collapse catastrophically. Thus, they argue that RLR's regularization and early-layer routing are required.
* **The Empirical Findings:** To evaluate this, the authors perform a systematic ablation in Table 4 (Section 4.5) by extracting features from Early (Patch Embed), Middle (Block 5), and Late (Block 11) layers. The results show that routing from Late blocks achieves the **highest Joint Mean accuracy of $95.41\%$** (vs $90.65\%$ for early layers) with **zero collapse**, even for the unregularized classical router.
* **The Contradiction:** If unregularized routing from Block 11 achieves "exceptional stability" and the highest performance ($95.41\%$), then:
  1. The "deep task-warped representation shift causing collapse" hypothesis is empirically disproven by their own ablation study.
  2. If unregularized routing from Block 11 works beautifully without any collapse, the entire motivation for introducing RLR's regularization (weight decay and temperature scaling) to stabilize deep routing is rendered moot.
  3. Why do the authors promote "Early" routing (first patch embedding) as their primary model configuration in Tables 1 and 3, which yields a much lower Joint Mean accuracy of $0.8953$ for RLR and $0.9078$ for the Linear Router? Presenting the sub-optimal Early routing as the default when their own ablation shows that Late-layer routing achieves $95.41\%$ is a major weakness that weakens the paper's overall results.

### Critique 2: Empirical Domination (RLR is Never the Best Choice)
A closer look at the empirical results reveals that the proposed RLR method is never the optimal or most rational choice under any evaluated serving conditions:
* **In Homogeneous Settings (Table 1):** RLR's Joint Mean is **$0.8953$** (on seed 42) and **$91.46\% \pm 0.42\%$** (across 5 seeds). Meanwhile, the unregularized classical Linear Router gets **$0.9078$** (on seed 42) and **$91.53\% \pm 0.41\%$** (across 5 seeds). So RLR underperforms or is statistically indistinguishable from the unregularized router, making the proposed regularizations empirically redundant or even slightly detrimental to peak performance.
* **In Heterogeneous Settings (Table 3):** Under mixed-task streams, dynamic routers experience heterogeneity collapse as batch size $B$ increases. RLR degrades to **$0.7795$** at $B=256$.
  Meanwhile, the supervised static baseline OFS-Tune maintains a robust **$0.8623$** accuracy ($+8.28\%$ absolute improvement over RLR). Even the local implementation of QWS-Merge gets **$0.8308$** at $B=256$ ($+5.13\%$ absolute improvement over RLR).
* **Summary:** Since RLR is dominated by the unregularized router in homogeneous settings, and strictly dominated by static methods / QWS-Merge in heterogeneous settings, its practical utility as a new parameter fusion method is highly limited.

### Critique 3: Scale of Empirical Validation
While the deconstruction of QWS-Merge is highly successful, the empirical validation is restricted to small-scale, 10-class vision datasets (MNIST, FashionMNIST, CIFAR-10, SVHN) using a tiny ViT backbone (5.7M parameters). Modern model merging is most actively used in Large Language Models (LLMs) with billions of parameters (e.g., LLaMA, Mistral). Although Section 5 provides a detailed conceptual discussion on LLM scaling pathways (Sequence-Level Routing, LoRA Experts, and Linear Mode Connectivity), the paper lacks any actual experiments on larger-scale architectures or language tasks, limiting its empirical generalizability.

---

## 4. Evaluation of Specific Dimensions

### Soundness: Fair (Falls short of Good due to Critique 1 and 2)
The underlying mathematical formulation of linear routing with temperature scaling and weight decay is sound. However, the severe logical contradiction between the "representation warping causing collapse" motivation and the Table 4 empirical results (which show deep unregularized routing works best without collapse) severely limits the conceptual soundness of the paper's claims.

### Presentation: Excellent
The paper is exceptionally clear, structured, and adheres flawlessly to conference style guidelines. The prose is engaging, professional, and grammatically flawless. The visualizations (including the newly added 2D hyperparameter heatmaps) are high-quality, informative, and mathematically consistent with the text.

### Significance: Good
The paper acts as an essential sanity check for the model merging community. By deconstructing over-engineered trends, it proves that standard classical regularizations/optimization choices are highly effective, steering the community back to transparent, simple, and reproducible deep learning practices. However, its significance as a *new method* is limited by RLR's performance domination (Critique 2).

### Originality: Good
While the individual components (weight decay, temperature scaling) are classic machine learning tools, their targeted application to post-hoc routing networks to dismantle convoluted, quantum-inspired frameworks is highly creative, original, and impactful.

---

## 5. Questions and Minor Suggestions for the Authors

1. **Reconciling routing layers:**
   Can you explain why you choose "Early" routing (first patch embedding) as your primary model configuration when your own ablation study shows that "Late" routing (Block 11) gets a massive $95.41\%$ Joint Mean accuracy with zero collapse? If you promoted Late routing as your default, it would vastly improve your results and align closer to the individual expert ceiling ($96.27\%$).
2. **Reconciling reporting inconsistencies:**
   In Section 4.4, you state that RLR's performance degrades to $78.23\%$ at $B=256$. However, in Table 3, RLR's accuracy at $B=256$ is listed as $0.7795$. Please reconcile this discrepancy.
3. **MoE Gating Contextualization:**
   In Section 2.2, please cite foundational sparse MoE gating regularization literature (e.g., load-balancing losses, entropy regularization) to properly position RLR as adapting these classic MoE gating concepts to the specific post-hoc merging setting.
