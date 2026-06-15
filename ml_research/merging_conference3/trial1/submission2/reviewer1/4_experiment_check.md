# 4. Experimental Evaluation and Empirical Support Analysis

## Evaluation of the Experimental Setup
The experimental setup is exceptionally well-conceived and rigorous:
- **Isolation of Weight Merging Dynamics:** By adopting a **Task-Incremental Continual Learning** setting with an oracle task ID, the authors select the task-specific classification head during evaluation. This prevents classifier-head interference and representation drift from confounding the analysis, allowing them to isolate and measure the precise performance of the merged backbone weights.
- **Architectural Selection:** Evaluating on Vision Transformers (ViT-Tiny, ViT-Base) is highly appropriate. Transformers are the dominant modern architecture, and understanding how weight merging dynamics scale on their self-attention projections and multi-layer perceptron blocks is of high interest.
- **Decoupled Grid Design:** Crossing 5 optimizers and 3 merging strategies yields 15 distinct configurations under two different mixing parameter regimes ($\lambda=0.0$ and $\lambda=0.2$). This structured multi-seed approach provides comprehensive coverage and prevents selective reporting.

## Analysis of Empirical Support for Claims

### 1. Claim: Optimizer-driven flatness is the foundational driver of merging success
* **Empirical Evidence:**
  - Under sequential parity ($\lambda=0.0$), transitioning from AdamW to standard global SAM under Task Arithmetic results in a massive **+9.87% absolute improvement** in average accuracy (68.31% vs. 58.44%) and improves Backward Transfer by **+10.08%** (from -40.06% to -29.98%).
  - Under active mixing ($\lambda=0.2$), moving from AdamW to standard global SAM yields a **+12.30% absolute improvement** under Task Arithmetic (73.83% vs. 61.53%) and a **+7.44% absolute improvement** under Isotropic SVD Merging (76.42% vs. 68.98%).
* **Verdict:** Highly supported. The performance gains from optimization-stage flatness are consistent and represent the single largest contribution to overall accuracy across all configurations.

### 2. Claim: SVD-based Isotropic Merging is boundary-condition sensitive and redundant under parity
* **Empirical Evidence:**
  - Under sequential fine-tuning parity ($\lambda = 0.0$), applying SVD-based isotropic merging consistently degrades average accuracy across all optimizers (dropping AdamW from 58.44% to 53.38%, and SAM from 68.31% to 61.33%).
  - Under active weight mixing ($\lambda = 0.2$), SVD-based isotropic merging acts as a highly effective regularizer, boosting AdamW's accuracy from 61.53% to 68.98% (+7.45%) and SAM's from 73.83% to 76.42% (+2.59%).
* **Verdict:** Fully supported. The results perfectly demonstrate that SVD's spectral interpolation is not a magic post-processing cure-all but is highly sensitive to the parameter mixing regime. It is mathematically redundant and distortive on un-mixed expert weights but highly effective when active parameter consolidation occurs.

### 3. Claim: SA-BCD contains an algebraic bug and is suboptimal/slow when corrected
* **Empirical Evidence:**
  - The literal published formula of SA-BCD completely fails, yielding random chance accuracy (~4.5%) across all merging strategies due to gradient-multiplication-induced divergence.
  - The corrected SA-BCD (Std Adam) under Task Arithmetic gets 62.94% ACC, which is suboptimal compared to standard globally perturbed SAM (68.31%).
  - SA-BCD (Std Adam) requires 279.9s of wall-clock training time—representing an **18.5% increase** compared to standard global SAM (236.1s), despite only updating 30% of parameters.
* **Verdict:** Fully supported. The authors mathematically and empirically prove the existence of the typo, and correctly demonstrate that the coordinate-restricted selection mechanism in SA-BCD is mathematically suboptimal and computationally inefficient on GPUs due to sorting and indexing.

### 4. Claim: SVD-based merging is redundant on flat low-rank manifolds (LoRA-SAM)
* **Empirical Evidence:**
  - Optimizing low-rank adapters with LoRA-SAM boosts Task Arithmetic merging performance to **74.12% ACC** (+14.78% over LoRA-AdamW's 59.34%).
  - Under LoRA-SAM, adding SVD-based isotropic merging only improves accuracy by a negligible **+0.73%** (74.85% vs. 74.12%), whereas under LoRA-AdamW, SVD improves performance by **+2.11%** (61.45% vs. 59.34%).
  - LoRA-SAM introduces <2.5% wall-clock overhead and <1.5% GPU VRAM overhead compared to standard LoRA-AdamW.
* **Verdict:** Fully supported. The low-rank constraints of LoRA naturally prevent high-dimensional spectrum collapse. When paired with optimization-stage flatness (LoRA-SAM), naive Task Arithmetic is highly effective and SVD-free merging is completely viable, introducing virtually zero training overhead.

## Identified Empirical Limitations and Suggestions
While the empirical results are compelling, we identify a few areas for minor constructive improvement:
1. **Single-Seed Scale Validation on ViT-Base:** The scale validation results on the 86M parameter ViT-Base backbone (Table 3) are excellent, but the authors explicitly acknowledge that they are based on a single seed due to the immense computational cost of full-parameter training. While acceptable for a deconstruction paper, reporting multi-seed averages (e.g., across 3 seeds) for ViT-Base in future revisions would strengthen the statistical rigor of the scale validation.
2. **Translation to Class-Incremental Continual Learning:** The paper focuses exclusively on Task-Incremental continual learning (oracle task ID is provided). In many real-world applications, an oracle task ID is not available at test-time, requiring a Class-Incremental setting where classifier-head interference is severe. It would be highly valuable for the authors to include a brief discussion on how optimizer-driven flatness (SAM) and SVD isotropic merging might translate to the more challenging Class-Incremental setting.
3. **Wider NLP Benchmarks:** The authors outline an excellent and feasible experimental design for NLP practitioners (using BERT-Base on GLUE tasks) to test cross-domain generalizability. Actually executing a subset of these NLP experiments would greatly strengthen the paper, although the theoretical SVD scaling analysis on wider matrices ($4096 \times 4096$) in Appendix B goes a long way in highlighting the computational bottlenecks in large-scale NLP.
