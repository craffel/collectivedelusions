# Novelty Check

## Key Novel Aspects
1. **Systematic Auditing of Quantization-Merging Interactions:** While PEFT and model-merging literatures have flourished, they have largely operated in a full-precision abstraction. This work is the first to systematically audit merged low-rank adapters under downstream post-training quantization (PTQ) constraints, exposing the row-wise magnitude discrepancy between base model weights and fine-tuned updates that causes "Re-Quantization Silence."
2. **Identification and Deconstruction of the Quantization Granularity Bifurcation:** The paper demonstrates that "Re-Quantization Silence" is highly dependent on the quantization grid granularity (Per-Tensor vs. Per-Channel). It shows that per-channel grids natively preserve adapter updates, rendering naive merging virtually lossless and proving that the "silence" is highly localized to aggressive, sub-optimal per-tensor configurations.
3. **Deconstruction of the Double Quantization Format-Shift Noise:** The paper exposes a massive methodological blindspot in QLoRA merging: the transition from quantile-based non-linear NF4 (used in training) to linear INT8/INT4 (used in deployment) introduces huge representation errors (+16.5% to +29.4% Relative Frobenius error on ViT-Base) completely independent of any merging or adapter updates.
4. **Exposing the "Representation Scale Preservation Dilemma" in Data-Free Scaling:** The authors mathematically prove that simultaneous scale preservation of base and adapter representations is impossible in a unified merged model. Attempting to scale back the entire output by $1/\gamma^l$ to preserve activation scale collapses the pre-trained base representations by a factor of 10 to 100.
5. **Proposing the "Zero-Interference RQA Protocol":** This protocol offers a clean way to isolate downstream quantization noise from pre-existing weight-space task interference.

## Delta from Prior Work
- **PEFT and Model Merging (e.g., LoRA, Task Arithmetic, TIES-Merging, DARE):** These works evaluate exclusively in FP16/FP32. This paper extends their scope to practical deployment settings by introducing post-training quantization.
- **Quantization-Aware Model Merging (e.g., Q-Merge):** Q-Merge proposes complex, joint quantization-aware merging with STE optimization. This paper differs by showing that such complexity is largely unnecessary in practice because standard per-channel quantization (the industry standard) is already virtually lossless.
- **Test-Time Adaptation (e.g., AdaMerging, RegCalMerge):** These methods optimize blending coefficients in FP16/FP32. The authors evaluate them through the quantization operator (QA-ACS) using STE, but crucially deconstruct their fragility (unsupervised entropy collapse under severe noise) and show that basic regularization or supervision is required to stabilize them.

## Characterization of Novelty
From a conceptual and methodological standpoint, the novelty of this paper is **significant**. Instead of presenting a highly complex, over-engineered solution and claiming an artificial SOTA, the paper performs a thorough, critical deconstruction of the entire system. It reveals that:
1. The problem (Re-Quantization Silence) is actually highly localized to per-tensor grids, and is already solved by the simplest, standard deployment baseline (per-channel quantization) with zero extra engineering.
2. The proposed data-free mitigation (SAWS) has a fundamental mathematical limitation (the scale preservation dilemma) and works via selective boosting rather than true scale preservation.
3. The proposed optimization-based mitigation (QA-ACS) is fragile and prone to entropy collapse under high noise unless regularized.

Through a **Minimalist** lens, this critical and deconstructive novelty is exceptionally valuable. It steers the community away from overly complex, fragile, and unnecessary algorithms (such as complex scaling and optimization frameworks) and highlights that the simplest, standard baseline (per-channel post-training quantization) is already highly robust and sufficient in standard deployment scenarios.
