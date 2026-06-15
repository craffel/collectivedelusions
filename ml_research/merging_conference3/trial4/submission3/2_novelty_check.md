# 2. Novelty and Originality Check

## 2.1. Comparison with Existing Literature
The paper is positioned at the intersection of model merging, parameter-efficient fine-tuning (PEFT), and post-training quantization (PTQ). Its novelty lies in exposing and deconstructing the interactions between these three domains, which are typically studied in isolation.

1. **Model Merging (e.g., Task Arithmetic, TIES, DARE, AdaMerging):** 
   - *Prior Work:* Existing merging methods operate under the assumption of a high-precision (FP16/FP32) evaluation abstraction. They demonstrate impressive merging gains in weight-space but ignore the downstream quantization required for deployment.
   - *This Paper:* Exposes that naive re-quantization back to low-bit constraints silently obliterates the subtle low-rank updates in these merged models—a phenomenon named the "Re-Quantization Silence."

2. **Weight Scaling Mitigations (e.g., SmoothQuant):**
   - *Prior Work:* SmoothQuant applies activation-weight scaling to mitigate quantization noise. It applies mathematically exact inverse scaling to weights and activations, preserving the exact linear equivalence of the original model.
   - *This Paper:* Introduces Scale-Adaptive Weight Shifting (SAWS). Crucially, the authors prove that applying mathematically exact inverse scaling at inference is self-defeating for adapter merging because it scales down pre-trained base representations, collapsing performance. Instead, SAWS operates via *selective task-vector boosting*, purposefully shifting representation geometry to allow updates to survive quantization.

3. **Test-Time Adaptation (e.g., AdaMerging, RegCalMerge):**
   - *Prior Work:* These approaches optimize merging coefficients in high-precision, followed by post-hoc quantization (Post-Hoc Quantized AdaMerging), which overfits to the continuous space and degrades under discretization noise.
   - *This Paper:* Evaluates Quantization-Aware Adapter Coefficient Search (QA-ACS), which optimizes coefficients *through* the quantization operator using Straight-Through Estimators (STE).

4. **Quantization-Aware Model Fusion (e.g., Q-Merge, Cross-Schema Generalization, ZipMerge):**
   - *Prior Work:* Q-Merge optimizes merging weights directly; subsequent work exposed cross-schema generalization gaps and representation collapse under joint pruning.
   - *This Paper:* Focuses specifically on the silent erasure of PEFT adapters under uniform post-training quantizers. It deconstructs the spatial variance of task vectors to explain why standard per-channel grids are nearly lossless while per-tensor grids cause catastrophic collapse.

## 2.2. Novelty of the Audit and Deconstructions
The core originality of this work does not just come from the proposed mitigations, but from its **extraordinarily thorough methodological audits and self-critical analyses**. These contributions are highly original and raise the bar for deep learning evaluation:

- **The Representation Scale Preservation Dilemma:** Formulating and proving the mathematical contradiction of true scale preservation in merged models.
- **Double Quantization Noise Audit:** Pointing out and measuring the "format shift" reconstruction error (NF4 to INT4/INT8) as a major confounding variable in QLoRA merging audits, showing relative Frobenius error absolute increases of up to $+29.4\%$.
- **Cache-Fitting vs. DRAM-Latency Bifurcation:** Profiling CPU execution on a 128-core Xeon CPU to show that small toy models (which fit completely in cache) can hide the latency of separate adapters (co-existence), while multi-billion parameter LLMs are DRAM-bandwidth bound and require weight-space merging to bypass linear DRAM weight-loading penalties.
- **Zero-Interference Protocol and Individual Expert Auditing:** Explicitly decoupling quantization-induced erasure from pre-existing weight-space task interference. This control experiment proves that under standard per-channel grids, low performance is driven entirely by task interference, not quantization.

## 2.3. Originality Rating
**Rating: Excellent**  
While the individual techniques (weight scaling, STE coefficient search) have foundations in prior literature, their unique combination, adaptation, and rigorous deconstruction represent a highly original contribution. The paper successfully moves the field beyond simplistic "propose a new SOTA" papers and instead provides a foundational, highly transparent, deployment-aware audit of model fusion.
