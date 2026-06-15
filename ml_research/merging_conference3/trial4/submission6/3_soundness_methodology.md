# Intermediate Review File 3: Soundness and Methodology of the Revised Paper

This file provides a rigorous analysis of the technical soundness, methodological strengths, and remaining weaknesses of the revised paper.

## 1. Major Methodological Improvement: Resolution of the Hyperparameter Confounder
The previous review cycle identified a severe hyperparameter tuning confounder: the authors had optimized the scaling coefficient $\lambda = 0.8$ for Tuned STA, while keeping the baselines (Task Arithmetic, TIES-Merging, DARE) restricted to a fixed, un-tuned $\lambda = 0.3$. 

In this revised version, the authors have successfully resolved this confounder by implementing a **fully symmetric hyperparameter sweep** over the scaling coefficient $\lambda \in [0.1, 1.0]$ for ALL baselines. The results are now reported at each method's individual optimal scaling coefficient $\lambda^*$:
- **Tuned Task Arithmetic (s=100%):** Peaks at $\lambda^* = 0.5$ (Avg = 88.64%).
- **Tuned DARE (s=20%):** Peaks at $\lambda^* = 0.4$ (Avg = 88.95%).
- **Tuned TIES-Merging (s=20%):** Peaks at $\lambda^* = 0.5$ (Avg = 90.16%).
- **Tuned STA (Ours, s=20%):** Peaks at $\lambda^* = 0.8$ (Avg = 90.53%).

By showing that Tuned STA still slightly outperforms Tuned TIES-Merging (+0.37% absolute) and Tuned DARE (+1.58% absolute) when all methods are fully tuned, the authors have successfully addressed the previous soundness flaw. This provides a much more rigorous and scientifically sound foundation for their minimalist hypothesis.

## 2. Remaining Soundness and Methodological Weaknesses

### Weakness 2.1: Marginal Performance Gain and Statistical Significance
While Tuned STA (90.53%) achieves the highest average accuracy in Table 1, its margin of improvement over Tuned TIES-Merging (90.16%) is only **0.37% absolute**. 
- Evaluated on a 16-batch validation split containing 2,048 samples per dataset, a difference of 0.37% corresponds to only **7 samples** out of 2,048. 
- For a pre-trained Vision Transformer, a difference of 7 samples is well within the margin of statistical error and random variance.
- Therefore, the claim that Tuned STA "substantially outperforms" or "outperforms" TIES-Merging is overstated. Scientifically, the results indicate that Tuned STA performs *comparably* to Tuned TIES-Merging, which still supports the minimalist hypothesis but refutes any claim of superior performance.

### Weakness 2.2: Mathematical Misnomer in "Isotropic Pruning"
Throughout the paper, the authors refer to their pruning process as **"isotropic layer-wise magnitude pruning."**
- In multidimensional space, **isotropic** means having physical properties that are identical in all directions (i.e., invariant under rotation).
- Magnitude-based pruning is coordinate-dependent and highly **anisotropic** because it selectively retains parameter updates with the largest absolute values. 
- Indeed, in Section 4.3, the authors explicitly explain that magnitude pruning suffers from **"tail-bias"** because it selectively retains the extreme tails of the update distribution, which distorts the parameter variance and causes "parameter explosion." 
- Calling a process "isotropic" when it is highly anisotropic and selectively distorts variance is a fundamental mathematical contradiction. The authors should correct their terminology, replacing "isotropic magnitude pruning" with "layer-wise uniform magnitude pruning" or simply "layer-wise magnitude pruning."

### Weakness 2.3: Unrealistic Independence Assumption in Collision Analysis
The authors calculate the probability of coordinate-wise parameter collision across tasks as $P = (s/100)^2$, assuming that the pruning masks $M_a$ and $M_b$ are independent.
- While the empirical mask overlap in their 4-task suite (3.1%–4.3%) aligns perfectly with the 4.0% theoretical bound at $s=20\%$, this is a consequence of their chosen toy datasets spanning highly diverse and unrelated domains (digits, apparel, natural images, street numbers).
- In real-world applications, fine-tuned experts derived from a shared pre-trained base model often update the same salient parameter regions (e.g., attention projections or early representation layers) to adapt to downstream tasks.
- For similar tasks (e.g., merging multiple sentiment analysis experts, or multiple translation experts fine-tuned on an LLM), the coordinate-wise mask overlap is expected to be much higher, leading to severe sign conflicts. The assumption of independence is unrealistic and represents a limitation of the paper's theoretical framework.
