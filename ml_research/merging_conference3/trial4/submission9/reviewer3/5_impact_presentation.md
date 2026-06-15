# 5. Presentation, Impact, and Significance

## Major Strengths of the Paper
Despite its limitations, the paper possesses several distinct strengths:

1. **Exhaustive Sensitivity and Ablation Analyses:**
   * The paper includes detailed sensitivity sweeps for the coherence retention factor $\gamma$ (Table 4) and validation size scaling (Table 6). 
   * The inclusion of a 500-step optimization study comparing TLC-Tune against AdaMerging and ZipMerge provides a clear visual and empirical deconstruction of search-step convergence.

2. **Honest Discussion of Paradigm Trade-offs:**
   * Section 4.3 contains a highly transparent, detailed discussion of the physical and operational trade-offs between Multi-Task Fine-Tuning (MTL), Parameter-Efficient Fine-Tuning (PEFT/LoRA), and weight-space model merging. 
   * The authors do not hide the absolute accuracy drop, framing it as a necessary trade-off for zero-oracle, zero-overhead deployment on resource-constrained devices.

3. **Detailed Mathematical and Algorithmic Blueprints:**
   * The mathematical identity demonstrating that Soft-EPA is a convex combination of hard exclusivity and Task Arithmetic (Equation 12) is well-formulated.
   * The inclusion of Algorithm 1, detailing the step-by-step routing protocol for decoder-only Large Language Models (LLMs), is highly practical and provides a useful implementation guide for practitioners.

4. **Rigorous Standardization Discussion:**
   * The statistical analysis of the 13.8% scale override rate across 5.52M parameters provides a concrete, data-driven justification for the proposed scale-decoupled design.

---

## Major Areas for Improvement

1. **Lack of Foundation Model Scaling:**
   * The paper is restricted entirely to a 5.7M parameter ViT-Tiny. To be highly convincing, the authors must scale their experiments to modern generative models (e.g., Llama-3 or Mistral) or larger Vision-Language Models (e.g., CLIP), especially since they provide explicit blueprints for LLMs in Algorithm 1.

2. **Unfair Baseline Optimization Setup:**
   * The authors must evaluate SOTA layer-group-wise adaptation baselines (AdaMerging and ZipMerge) under their native, first-order gradient descent optimization pipelines rather than forcing them to use a zero-order (1+1)-ES search. Running these baselines in an optimizer mismatch environment creates a biased and unscientific comparison.

3. **Theoretical Grounding for Dynamic Coherence Scheduling (DCS):**
   * The quadratic schedule $\gamma(p) = \gamma_0 + (1 - \gamma_0) \cdot p^2$ is presented as an empirical heuristic. The paper would be significantly strengthened by providing a rigorous mathematical or probabilistic derivation for why the coherence factor should scale quadratically with sparsity $p$.

4. **Severe Accuracy Degradation under Sparse Regimes:**
   * The authors must address why EPM is heavily outperformed by DARE (by over 14.49% absolute accuracy) under 80% sparsity. The failure of coordinate-exclusive routing under extreme pruning needs a deeper, more rigorous analysis.

---

## Overall Presentation Quality
* **Structure and Narrative:** The paper is exceptionally well-structured, logical, and easy to follow. The mathematical notation is clean and consistent throughout.
* **Clarity of Graphics:** Figures 1 and 2 are highly polished, professional, and do an excellent job of illustrating the core concept and the optimization trajectories.
* **Tone and Verbosities:** The writing is highly polished, but it is excessively verbose and defensive in Section 4.3. The authors spend multiple pages preemptively defending their method against criticisms regarding optimizer mismatch, performance trade-offs, and toy datasets. While thorough, this defensiveness can distract from the core technical contribution.

---

## Potential Impact and Significance
* **Impact Level:** **Low to Moderate.**
* **Significance:** Weight-space model merging is an active and highly relevant field. However, the extreme performance degradation from $\sim$95% (expert ceiling) to $\sim$46% (merged joint mean) on basic classification benchmarks indicates that the proposed method has limited practical utility in its current form. 
* Additionally, without empirical validation on large-scale LLMs or VLMs, the paper remains a highly localized study on toy vision tasks. Unless EPM is shown to maintain high expert-level accuracies on modern foundation architectures, its impact on the broader machine learning community will remain minor.
