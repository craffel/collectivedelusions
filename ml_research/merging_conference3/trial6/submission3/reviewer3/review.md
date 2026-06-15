# Peer Review of "Deconstructing Capacity and Generalization in Dynamic Model Merging: The Block-wise Weight-Sharing Router"

## Review Summary
This paper presents a thorough, scientifically rigorous, and highly complete empirical deconstruction of dynamic model merging. The authors address a critical bottleneck in post-hoc weight-space ensembling: the overparameterization, optimization instability, and "coefficient ruggedness" associated with learning unshared, independent routing networks across sequential layers. To resolve this, they introduce the **Block-wise Weight-Sharing Router (BWS-Router)**, which groups layers into uniform block groups and shares routing parameters within each block. 

From an empirical perspective, this paper is exemplary. It is backed by an exceptionally large-scale grid sweep of over 1,280 experiment configurations across 5 independent random seeds ($42, 43, 44, 45, 46$), honestly reporting means and standard deviations. Crucially, the authors transition their evaluation from a simplified virtual-layer sandbox to a **physical sequential weight-space model-merging framework** on PyTorch multi-layer MLP experts, bridging a major gap in the model ensembling literature. They also provide detailed sensitivity analyses, modern model scaling estimates, a GPU-level compiler-friendly implementation recipe, and an empirical pilot profiling on an actual Vision Transformer backbone.

Because the paper is technically sound, methodologically appropriate, exceptionally well-written, and extremely thorough in its evaluations, I recommend a **6: Strong Accept**.

---

## Strengths
1. **Exceptional Empirical Rigor and Honesty:** 
   The evaluation is a gold standard for empirical machine learning research. The authors do not rely on single runs or selected configurations; instead, they report complete means and standard deviations across five independent seeds. Every hyperparameter (learning rates, regularization scales, scaling ceilings, bias initializations, PCA projection dimensions, and kernels) is systematically swept, mapped, and ablated.
2. **Realistic Physical Sequential Framework:** 
   The authors do not hide behind a stylized sandbox. Recognizing that virtual-layer ensembling sandboxes (which average coefficients across layers) do not model sequential deep transformations, they construct and evaluate a true physical sequential weight-space merging framework on PyTorch 3-layer MLP experts. This setup serves as a realistic, highly convincing testing ground for sequential feature propagation.
3. **Outstanding Parameter and Computational Compression:** 
   The paper proves that fine-grained layer-wise routing specialization is highly redundant. Sharing weights across layer blocks of size $M$ (such as $M=3$ or $M=12$) delivers a **66.7% to 91.7%** parameter and routing forward pass footprint reduction in the sandbox with absolutely zero loss in dynamic routing accuracy (retaining ~79.6% accuracy). Scaling calculations show that on CLIP and LLaMA-2-7B, BWS-Router yields a staggering **94.4% to 96.4%** reduction in parameters and routing passes.
4. **Principled Mathematical Modeling:** 
   The authors formalize Expected Ruggedness using a generalized mathematical model that incorporates depth-dependent variance scales and adjacent block correlations, providing a solid theoretical justification for block sharing.
5. **Effective Gating Bias Trick:** 
   The negative bias initialization ($B_{group} = -2.0$) is a simple yet brilliant mechanism to establish a sparse, "inhibitory default" state. It prevents catastrophic multi-task interference at initialization and solves the optimization sluggishness of Sigmoidal routing under lower learning rates.
6. **Detailed and Actionable Practitioner Guidance:** 
   The paper is written with a focus on practical applicability. It provides clear rules of thumb for choosing regularization scales, selecting gating activations (Softmax vs. Sigmoid) based on open/closed environments, and setting pre-projector PCA dimensions ($d$). The step-by-step ViT implementation recipe and the CPU latency profiling pilot (\texttt{vit\_tiny\_patch16\_224}) make this paper highly valuable for downstream practitioners.

---

## Weaknesses and Areas for Improvement
1. **Scale-up on Large Foundation Models:** 
   While the authors execute a practical CPU latency and parameter-footprint pilot demonstration on an actual Vision Transformer backbone (\texttt{vit\_tiny\_patch16\_224}) in the appendix, the actual downstream multi-task accuracy sweeps are conducted within the synthetic sandbox and physical MLP models. Evaluating downstream task generalization on massive pre-trained backbones (e.g., LLaMA or Stable Diffusion) remains an exciting future direction.
2. **High Sequential Propagation Variance:** 
   The seed-wise standard deviation under physical sequential mixed-batch streams is relatively high ($43.20 \pm 22.49\%$). Although the authors show that sequential smoothing regularization ($\mathcal{L}_{\text{smooth}}$) successfully stabilizes this standard deviation down to $13.41\%$ without sacrificing absolute performance, developing even more robust, feedback-driven, or self-tuning routing architectures to suppress sequential drift is a promising research pathway.
3. **Noisy Domain Ceiling:** 
   All methods struggle on SVHN (Task 3) in the physical weight-space setup due to high domain noise. Exploring advanced domain adaptation techniques or multi-layer non-linear expert classification heads to lift this baseline ceiling is highly recommended.

---

## Detailed Evaluation

### Soundness: Excellent
The paper's soundness is outstanding. The claims are fully supported by empirical data and rigorous mathematical modeling. The Proposed BWS-Router is methodologically appropriate, addressing overparameterization and sequential representation drift with block sharing. The mathematical formalization of Expected Ruggedness is complete and does not rely on overly simplified i.i.d. assumptions. Every empirical metric is reported with confidence intervals/standard deviations across multiple seeds, and the baselines are rigorously tuned and fairly compared.

### Presentation: Excellent
The submission is beautifully written, clear, and exceptionally well-structured. The narrative flow is cohesive, and the figures (such as the schematic in Figure 1 and the empirical plots in the appendix) are of publication quality. The paper contextualizes itself masterfully relative to prior static merging (TIES, task arithmetic, DARE) and dynamic routing (Routing Soups, L3-Router, QWS-Merge) literature, clearly spelling out its unique "delta."

### Significance: Excellent
The significance of the overall contribution is exceptionally high. Dynamic model merging has the potential to unlock post-hoc multi-task adaptation for large foundation models at zero additional inference-time cost. However, high-capacity unshared routing networks have been a major bottleneck due to computational and parameter overhead. BWS-Router's ability to slash parameters and routing passes by 91% to 96% with zero loss in performance is a massive practical win. The transition to a physical sequential framework and the detailed implementation guidelines will likely have a broad impact on the PEFT and model ensembling communities.

### Originality: Excellent
While weight-sharing is an established concept, applying it block-wise to parameter-space routing represents a highly original and elegant solution to unshared overparameterization. Combining block-sharing with unsupervised low-dimensional PCA projections and unit-sphere normalization is a clever and highly effective configuration. The formalization of Expected Ruggedness and the evaluation under a physical sequential weight-blending framework represent a significant conceptual leap from prior stylized sandbox simulations.

---

## Constructive Comments and Questions for the Authors
1. **Learnable Ceiling Exploration:** The learnable scaling ceiling ($\lambda_{max}$) sweep is highly fascinating, converging stably to $2.5712$ and improving Joint Mean Accuracy to 80.66%. Did you observe any weight destabilization or representation scale drift in the deeper layers when $\lambda_{max}$ climbed above 1.0, or does LayerNorm / unit sphere normalization completely filter this out?
2. **Data-Driven Block Grouping:** In Section 12 (Limitations/Future Work), you propose an elegant data-driven dynamic block grouping method based on gradient cosine similarity alignment. Have you run any preliminary pilots with this Ckmeans.1d.dp partitioning scheme, and did it yield a non-uniform coarse-to-fine block structure similar to your ViT pilot?
3. **Task-Specific ceilings on SVHN:** Given that all methods struggle on SVHN (Task 3) in the physical weight-space setup due to domain noise, have you explored whether learning task-specific ceilings ($\sigma_k$) helps the router dynamically boost noisy SVHN experts to lift the joint performance?

---

## Overall Recommendation

**Rating: 6: Strong Accept**

**Justification:** 
This is a technically flawless, exceptionally thorough, and beautifully written paper. The experimental evaluation is outstanding, satisfying the highest standards of empirical rigor by sweeping over 1,280 configurations across 5 independent random seeds with complete confidence intervals. The transition to a physical sequential weight-space model-merging framework on PyTorch experts bridges a major gap in the literature, converting what was previously a stylized sandbox simulation into a realistic sequential propagation environment. With its ability to slash routing parameter and computational footprints by over 91% (and up to 96% on large foundation models) with absolutely zero performance degradation, BWS-Router is a highly significant, elegant, and impactful contribution to parameter-efficient multi-task adaptation. I strongly recommend accepting this work.
