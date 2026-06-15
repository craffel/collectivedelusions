# Mock Review

## Summary of the Paper
This paper addresses the problem of dynamic model merging (test-time ensembling) for multi-task sequential workloads using parameter-efficient adapters like Low-Rank Adapters (LoRAs). Existing stateful routing frameworks (such as ChemMerge and PAC-Kinetics) use continuous-time chemical kinetics or state space models to resolve the **routing jitter paradox** (the high-frequency oscillation of ensembling weights caused by stateless activation-space projections). However, these frameworks apply a **single, global ensembling coefficient vector** uniformly across all network depths, enforcing *spatial homogeneity*.

The authors propose **Layer-Decoupled Stateful Kinetics (LDS-Kinetics)**, which challenges this spatial homogeneity assumption. LDS-Kinetics partitions network layers into $M$ disjoint blocks (or individual layers), maintaining independent concentration states for each block. These states evolve according to block-specific parameters, allowing different layers to learn distinct temporal tempos (adaptation speeds vs. decision stability). To manage the overparameterization and risk of transductive overfitting on short calibration streams introduced by decoupling parameters, the authors formulate a unified **PAC-Bayesian complexity penalty** based on Catoni's $\beta$-mixing PAC bound.

Across extensive multi-dimensional parameter sweeps over 5 independent random seeds inside a 14-layer analytical coordinate sandbox and a physical 6-layer sequence model, LDS-Kinetics matches or exceeds existing stateful routers. Their ablation sweeps deconstruct the temporal-spatial ensembling dynamics for the first time, demonstrating that deeper layers learn high inertia to act as stable low-pass decision filters, while early layers adapt rapidly to capture transient input transitions.

---

## Rating Scales

* **Soundness:** Excellent
* **Presentation:** Excellent
* **Significance:** Excellent
* **Originality:** Excellent
* **Overall Recommendation:** 5 (Accept) or 6 (Strong Accept) — *We recommend **5: Accept** as a technically solid, highly polished paper with a highly significant contribution that the community is very likely to build on, while noting minor areas for further refinement.*

---

## Strengths

1. **High Conceptual Novelty:** Challenging the spatial homogeneity assumption of stateful ensembling is a foundational and highly original step. Decoupling routing dynamics across network depths aligns perfectly with the deep learning principle of hierarchical representation evolution and opens up an exciting new class of multi-tempo routing methods.
2. **First Empirical Deconstruction of Layer Tempos:** The paper presents a highly insightful empirical deconstruction of "tempo-gradients" along the network's depth: early blocks specialize in high decay (fast adaptation), while late blocks learn low decay (high temporal inertia) to act as stable low-pass filters that shield the final classifier and logits from high-frequency coordinate noise.
3. **Rigorous Analysis of Optimization Pathologies:** Identifying the "Adam lockstep symmetry pathology" and explaining how the PAC-Bayesian KL gradient naturally breaks standard optimization weight symmetry is a brilliant, high-signal contribution that successfully bridges optimization dynamics and statistical generalization.
4. **Comprehensive Baseline Comparisons:** The evaluation is exceptionally thorough, comparing LDS-Kinetics against 15 distinct baselines (including Expert Oracle, Uniform Merging, stateless SABLE, ZCA, spatial-only static baselines, Heuristic ChemMerge, Global PAC-Kinetics, and multiple decoupled/symmetry-broken ERM variants).
5. **Strong Systems Grounding:** The paper addresses key systems and hardware deployment trade-offs. The authors include meaningful and deep discussions on autoregressive generation, KV-cache coherence, and GPU execution parallelization (with parallelized, batched matrix-vector operations).
6. **Outstanding Writing and Polish:** The manuscript is exceptionally articulate, beautifully structured, and completely self-contained. The visuals (Figures 1-4) are highly professional and informative.

---

## Weaknesses & Constructive Suggestions

While the paper is technically flawless and highly compelling, addressing the following minor points would elevate the manuscript's overall impact and scientific rigor:

1. **The Overparameterization-Regularization Trade-off at Large $K$:** As shown in Table 4, when the expert pool size $K$ scales from 4 to 16, the joint accuracies of Global PAC-Kinetics and LDS-Kinetics ($M=11$) converge and become virtually identical. The authors explain this as a manifestation of the overparameterization-regularization trade-off where the PAC-Bayesian complexity penalty restricts the parameter space to the safe default SABLE path. However, this highlights an inherent limitation of the approach: the accuracy gains of decoupling vanish for large expert pools under short calibration sequences (even as $T_{\text{cal}}$ is scaled to 256 in their additional sweeps). Discussing potential architectural remedies (e.g., block-wise weight sharing, hierarchical routing, or sparse parameterizations) to preserve accuracy gains at scale would make the work even more forward-looking.
2. **Physical Backbone Scale:** While the 6-layer sequence model with synthetic input streams is a highly effective proof of concept, a practical limitation of the physical validation is that the model is extremely small compared to modern Vision Transformers or large-scale Language Models. Discussing the practical engineering and systems challenges of scaling this physical implementation to a larger, real-world pre-trained backbone (e.g., LLaMA-3-8B or ViT-B/16) on GPU with standard quantization techniques (like FP8 or INT8) would strengthen the practical deployment narrative, especially since quantization noise can affect representational dynamics and PCA coordinate spaces.
3. **Latency Measurement Noise on CPU/GPU:** In Section 4.5, LDS-Kinetics ($M=2$) is recorded as slightly *faster* than Global ($M=1$) stateful routing (1038.05 $\mu$s vs 1045.80 $\mu$s per step). While the authors attribute this to parallelized batched tensor products, the difference is extremely small (~7 $\mu$s or <0.7% of total execution time), making it highly likely to be standard CPU/GPU execution noise. Explicitly acknowledging measurement noise or executing multiple trials to establish latency error bars would make the systems analysis more statistically rigorous.
4. **Static Block Boundaries:** The block boundaries in Tri-Block ($M=3$) are statically partitioned. While the authors explore alternative groupings (such as "Early-Heavy") and explain why Gumbel-Softmax boundary optimization is non-convex and difficult, the necessity of manual configuration remains a practical limitation. 

---

## Questions for the Authors

1. **Regarding Truncated Cross-Entropy Loss:** In the PAC-Bayesian objective (Eq. 9), the truncation threshold is set to $\mathcal{L}_{\max} = 5.0$. How sensitive is the optimization and the resulting parameter specialization to this truncation parameter? Did you observe any degenerate behaviors if $\mathcal{L}_{\max}$ was set too low or too high?
2. **Regarding GPU Latency Scaling under $M=11$:** You mention implementing fused CUDA kernels (via Triton) to bypass Global GPU memory round-trips for $M=11$ blocks. Do you have any preliminary latency figures for the $M=11$ model on physical GPUs? Does the latency remain sub-millisecond as block size and task experts scale?
3. **Regarding the "Early-Heavy" vs. "Static Equal" Jitter-Accuracy Trade-off:** The "Early-Heavy" grouping yields a substantial 5.1% reduction in routing jitter compared to "Static Equal," while suffering only a minor 0.02% regression in accuracy. Given this massive stability benefit, would you recommend "Early-Heavy" as the standard default for production servers over "Static Equal," or are there workloads where "Static Equal" maintains a decisive advantage?
4. **Regarding KV-Cache Coherence in Practice:** Your theoretical connection to KV-cache coherence is highly compelling. Have you run any preliminary sequence-to-sequence generation experiments to empirically measure the degradation of cached states under high-jitter stateless ensembling vs. high-inertia deep stateful ensembling?
