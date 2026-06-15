# 4. Experimental Evaluation and Verification

## Critical Evaluation of the Experimental Setup
The experimental setup is exceptionally well-designed to isolate the causal drivers of model merging:
- **Vision Transformer Backbone:** The primary backbone is a ViT-Tiny (5M parameters) and the scale validation uses a ViT-Base (86M parameters) from PyTorch Image Models (`timm`). This is a highly modern and representative setup.
- **Dataset and Protocol:** The use of Split CIFAR-100 divided into 5 sequential, disjoint tasks represents a standard and rigorous benchmark for task-incremental continual learning.
- **Ablation Axes:** Evaluating both standard sequential parity ($\lambda=0.0$) and active weight mixing ($\lambda=0.2$) is crucial. It separates sequential adaptation from active parameter consolidation, which serves as a vital boundary-condition check.

## Evaluation of Baselines
The paper compares its configurations against an exceptionally rich and exhaustive set of baselines:
1. **Optimization Baselines:** AdamW, standard SAM, and three variants of SA-BCD (Literal, Std Adam, Adam GT).
2. **Merging Baselines:** Task Arithmetic, Isotropic SVD Merging, and Scalar Update Decay.
3. **Spectral Control Baselines:** Norm-Matching (which separates Frobenius norm scaling from singular value decomposition) and Scale-Calibrated (which avoids scale-dampening over time).
4. **Post-Hoc Weight Consolidation Baselines:** TIES-Merging and DARE.

This comprehensive set of baselines is rare in model merging literature and provides a highly complete picture of the landscape.

## Verification of Claims Against Empirical Results

The quantitative results presented in Table 1 (Scoreboard, $\lambda=0.0$), Table 2 (Active Mixing, $\lambda=0.2$), and Table 3 (PEFT results) fully and robustly support the authors' central claims:

### 1. "Optimizer-driven flatness is the foundational driver of merging success"
- **Table 1 ($\lambda=0.0$):** Under Task Arithmetic, moving from AdamW to SAM boosts average accuracy from **58.44%** to **68.31%** (a massive **+9.87%** absolute improvement) and reduces forgetting (BWT improves from **-40.06%** to **-29.98%**).
- **Table 2 ($\lambda=0.2$):** Under Task Arithmetic, SAM boosts accuracy from **61.53%** to **73.83%** (**+12.30%** absolute). Under Isotropic SVD Merging, SAM boosts accuracy from **68.98%** to **76.42%** (**+7.44%** absolute).
- **Table 3 (PEFT):** For LoRA, moving from LoRA-AdamW to LoRA-SAM under Task Arithmetic increases average accuracy from **59.34%** to **74.12%** (a massive **+14.78%** absolute improvement).
- *Verdict:* This claim is completely and consistently supported across all model scales, parameters (full-parameter vs. PEFT), and mixing regimes.

### 2. "The benefit of SVD isotropic merging is boundary-condition-sensitive"
- **Table 1 ($\lambda=0.0$):** Under standard sequential fine-tuning parity, where there is no active parameter mixing, SVD isotropic merging degrades average accuracy for AdamW (from **58.44%** to **53.38%**, a **-5.06%** drop) and for SAM (from **68.31%** to **61.33%**, a **-6.98%** drop).
- **Table 2 ($\lambda=0.2$):** Under active weight mixing, SVD isotropic merging acts as a helpful regularizer, boosting AdamW's accuracy from **61.53%** to **68.98%** (**+7.45%**) and SAM's accuracy from **73.83%** to **76.42%** (**+2.59%**).
- *Verdict:* This provides definitive proof that SVD isotropic merging is redundant and distortive under standard sequential parity, but becomes a helpful spectral regularizer under active mixing regimes.

### 3. "The proposed SA-BCD optimizer has a fatal algebraic bug and is suboptimal"
- **Table 1 ($\lambda=0.0$):** The literal published formula for SA-BCD yields an accuracy of only **~4.5%** (random chance on 20-class sub-tasks), confirming optimization divergence.
- Even the corrected variant, SA-BCD (Std Adam), yields **62.94%** under Task Arithmetic, which is **5.37%** worse than global SAM (**68.31%**).
- SA-BCD (Std Adam) requires **279.9s** of training time compared to standard SAM's **236.1s**, representing an **18.5% increase** in wall-clock time despite perturbing only 30% of the parameters.
- *Verdict:* This claim is thoroughly supported by both the mathematical divergence of the literal formula and the empirical suboptimality/computational inefficiency of the corrected variants.

### 4. "LoRA-SAM renders SVD isotropic merging redundant under PEFT"
- **Table 3 (PEFT, $\lambda=0.2$):** Under LoRA-SAM, Task Arithmetic (naive weight averaging) yields **74.12%** average accuracy, whereas Isotropic SVD Merging yields **74.85%**. The performance gap is a negligible **+0.73%**.
- *Verdict:* This proves that because low-rank adapters are already highly constrained, optimizing them for flatness using LoRA-SAM makes post-hoc SVD spectral adjustments completely redundant, allowing practitioners to deploy a zero-overhead, purely SVD-free merging pipeline.

### 5. "Pre-merging flatness synergizes with and enables modern merging baselines (TIES/DARE)"
- **Table 2 ($\lambda=0.2$):** Crossing AdamW with DARE yields **40.81%** average accuracy. Crossing SAM with DARE yields **57.70%** ACC (a massive **+16.89%** absolute improvement).
- *Verdict:* This strongly supports Proposition 3.1, confirming that training-stage sharpness minimization (SAM) is an essential prerequisite that makes model weights structurally robust to subsequent high-dimensional sparsification or pruning.
