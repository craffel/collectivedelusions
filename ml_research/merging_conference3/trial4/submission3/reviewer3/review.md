# Peer Review

## 1. Summary of the Paper
The paper investigates the intersection of post-training quantization (PTQ) and model merging (specifically, Low-Rank Adaptation (LoRA) and QLoRA). It addresses a potential methodological blindspot: evaluating model merging in full-precision (FP16/FP32) while ignoring the downstream post-training quantization mandatory for actual edge deployment. The central focus is the **"Re-Quantization Silence"**—the phenomenon where the subtle, task-specific, low-magnitude updates in merged low-rank adapters are rounded to zero during low-bit (e.g., 4-bit) post-hoc quantization, returning the merged model's performance to the level of the unadapted base model.

To systematically analyze and address this phenomenon, the authors propose:
1. **Multi-Axial Re-Quantization Auditing (RQA) Framework:** A systematic evaluation of model merging across different quantization granularities (per-tensor vs. per-channel), bit-widths (4-bit vs. 8-bit), and formats (symmetric vs. asymmetric).
2. **Two Mitigations:**
   - **Scale-Adaptive Weight Shifting (SAWS):** A data-free, closed-form weight-scaling method that boosts the magnitude of the adapter updates relative to the base weights before merging, to prevent them from being rounded to zero.
   - **Quantization-Aware Adapter Coefficient Search (QA-ACS):** An optimization-based method that tunes layer-wise merging coefficients directly through the quantization operator using the Straight-Through Estimator (STE) and prediction entropy minimization on a tiny calibration set of 16 unlabeled samples.
3. **In-Depth Deconstruction and Validation:** A self-critical mathematical and empirical analysis of both the proposed mitigations and the underlying hardware constraints, including:
   - Quantifying the confounding error introduced by **Double Quantization format shift** (transitioning from NF4 to INT4/INT8).
   - An individual expert auditing control experiment to decouple task interference from quantization noise.
   - A physical CPU latency profiling benchmark exploring the cache-fitting vs. DRAM-latency bifurcation for co-existence vs. merging.

---

## 2. Strengths and Weaknesses

### Strengths
- **Outstanding Scientific Honesty and Transparency:**
  Unlike many papers that obscure the failures or limits of their proposed methods, this work is exceptionally transparent. The authors critically deconstruct their own proposed methods: they mathematically prove why "mathematically exact" scale preservation in SAWS collapses base representations, and they expose the vulnerability of QA-ACS to entropy collapse under aggressive noise.
- **Methodological Rigor and Creative Control Experiments:**
  The paper is packed with high-quality, creative methodological analysis. The **Individual Expert Auditing** control experiment (Table 8) is an extremely elegant way to isolate quantization noise from pre-existing task interference, and the **Double Quantization Noise** study (Table 1) provides useful insights into format-shift representation errors.
- **Excellent Physical Profiling and Latency Analysis:**
  The physical CPU latency profiling (Appendix D.3) is highly insightful. It reveals the "Cache-Fitting vs. DRAM-Latency Bifurcation," providing a hardware-grounded justification for why weight-space merging is necessary for larger models despite co-existence being competitive on tiny, cache-fitting models.
- **Clarity and Mathematical Rigour:**
  The paper is highly articulate, structured, and mathematically formal. The mathematical derivations are clean, correct, and provide valuable theoretical insights.

### Weaknesses
- **Limited Conceptual Novelty of the Core Phenomenon:**
  The paper frames the "Re-Quantization Silence" as a widespread, catastrophic methodological blindspot. However, the authors' own discovery of the **"Quantization Granularity Bifurcation"** significantly deflates this claim. The empirical results show that under standard **per-channel** configurations (which are the industry standard for edge deployment in packages like AWQ and GPTQ), naive, unmitigated re-quantization is nearly lossless, dropping only $0.15\%$ to $0.30\%$ mean accuracy in 8-bit and $1.80\%$ in 4-bit. Catastrophic collapse is strictly localized to **per-tensor** configurations, which are rarely used in practice due to their known representation limits. Thus, the "Re-Quantization Silence" is a highly localized artifact of an aggressive, sub-optimal quantization configuration, rather than a universal barrier to model-merging deployment.
- **Incremental and Redundant Mitigations:**
  - **Scale-Adaptive Weight Shifting (SAWS):** Crucially, under the only regime where "silence" is catastrophic (per-tensor constraints), Global SAWS actually performs *worse* than doing nothing (Naive-RQ: $56.40\%$ vs. $56.75\%$). It only yields substantial gains under per-channel configurations, where naive re-quantization was already virtually lossless. This indicates that SAWS fails to solve the aggressive per-tensor silence it was designed for, and is largely redundant under standard per-channel configurations.
  - **Quantization-Aware Adapter Coefficient Search (QA-ACS):** This method applies the standard Straight-Through Estimator (STE) to optimize merging coefficients on a tiny calibration set using prediction entropy. STE and prediction entropy minimization (e.g., AdaMerging, Tent) are pre-existing techniques. The paper's own analysis reveals that QA-ACS is highly fragile under noise, suffering from **entropy collapse** (predicting a single incorrect class with high confidence) unless it is constrained by supervised labels (which defeats the "unsupervised" test-time adaptation pitch) or strict $L_2$ regularization.
- **Scale of the Primary Backbone:**
  The main experiments are restricted to a tiny Vision Transformer (`vit_tiny`, 5.7M parameters). While this enables rapid multi-axial profiling, quantization and merging dynamics on toy networks may not fully generalize to multi-billion parameter LLMs. While the authors discuss scaling dynamics theoretically in Appendix C, the lack of full multi-task auditing results on an LLM is a notable weakness.
- **High Pre-existing Task Interference Confounder:**
  The continuous, full-precision baseline (Naive FP16 Merge) is already severely degraded (averaging $66.65\%$ compared to the $93.85\%$ expert ceiling). This reflects severe weight-space conflicts from merging highly distinct image classification datasets (MNIST, FashionMNIST, CIFAR-10, SVHN) using simple task arithmetic. Studying downstream quantization on a model that is already severely degraded makes it difficult to isolate quantization-induced representation erasure from pre-existing task interference.

---

## 3. Soundness
**Rating: Excellent**

**Justification:**
The paper is technically flawless and mathematically highly rigorous. The mathematical descriptions of the uniform symmetric quantizer, the dynamic scale factor, the closed-form weight alignment factor $c^l$ for SAWS, and the STE-based Adam updates in QA-ACS are correct and complete. The experimental methodology is highly systematic, evaluating across four diverse datasets and four post-training quantization configurations. Crucially, the authors design and execute exceptional control experiments, such as the individual expert auditing protocol (Table 8) and the physical CPU latency profiling benchmark (Table 11), which provide profound empirical validation of their claims.

---

## 4. Presentation
**Rating: Excellent**

**Justification:**
The paper is beautifully written, articulate, and well-structured. The narrative is easy to follow, transitioning naturally from the mathematical formulation of merging and quantization to the mechanics of collapse, the proposed mitigations, and finally a critical deconstruction of their limits. The tables (Tables 1-11) and figures are clean, clear, and highly professional. The authors do an outstanding job of properly positioning their work relative to prior PEFT, model merging, and PTQ literature, and the appendices are exceptionally detailed, making the work highly reproducible.

---

## 5. Significance
**Rating: Fair**

**Justification:**
While the paper addresses an important and relevant problem (edge deployment of merged models), the significance of the contribution is limited. The discovery that "Re-Quantization Silence" is a per-tensor artifact rather than a general blocker under standard per-channel grids means that practitioners deploying merged models in standard edge settings do not experience catastrophic collapse. Standard per-channel or group-wise quantization natively avoids the silence issue, which makes complex mitigations like SAWS and QA-ACS largely redundant in practice. However, the paper's emphasis on deployment-aware evaluation and its rigorous control protocols will likely influence academic model-merging research to adopt post-training quantization as a standard benchmarking step.

---

## 6. Originality
**Rating: Fair**

**Justification:**
The conceptual originality of both the investigated phenomenon and the proposed solutions is relatively incremental. The core problem of "Re-Quantization Silence" is a localized artifact of sub-optimal per-tensor quantization rather than a widespread, paradigm-shifting barrier. The proposed SAWS represents a straightforward global scale adjustment based on Frobenius norms, and channel-wise SAWS uses row-specific Frobenius norms. QA-ACS applies standard Straight-Through Estimators (STE) to optimize coefficients on a tiny calibration set using prediction entropy. Therefore, the paper acts as a detailed diagnostic audit of a localized quantization artifact, rather than introducing a truly original, ambitious, or highly significant new methodological direction.

---

## 7. Overall Recommendation
**Rating: 3: Weak reject**

**Justification:**
Although the paper is technically flawless, exceptionally well-written, and remarkably thorough in its control experiments and hardware profiling, the conceptual novelty and significance of both the core phenomenon and the proposed mitigations are relatively incremental. 

The core thesis of the paper—the "Re-Quantization Silence"—is shown by the authors' own analysis to be virtually non-existent under standard per-channel configurations ($0.15\%$ to $1.8\%$ drop), which are the industry standard for edge deployment. Under the only regime where the silence is catastrophic (per-tensor grids), the proposed SAWS actually performs worse than doing nothing (Naive-RQ), and the proposed QA-ACS suffers from catastrophic entropy collapse unless it is constrained by supervised labels or strict regularization. 

Thus, the paper acts as an outstandingly detailed diagnostic audit of a localized quantization artifact (per-tensor quantization noise) rather than a big, bold conceptual leap that changes how the community thinks about model merging or quantization. To transition to a high-impact paper, the authors need to:
1. Shift the focus from localized per-tensor artifacts to challenging, large-scale settings (such as multi-billion parameter LLMs under group-wise or ultra-low-bit formats like 2-bit/3-bit).
2. Propose a truly robust, scale-aware mitigation that achieves substantial improvements where naive pipelines collapse, rather than being redundant or underperforming.
3. Apply the proposed "Zero-Interference RQA Protocol" to decouple pre-existing task interference from quantization noise in the main experiments.

---

## 8. Constructive Questions / Feedback for Authors
1. **Evaluation at Scale:** Why are the primary multi-task auditing and mitigation results restricted to a toy model like `vit_tiny` (5.7M parameters)? Given that model merging and QLoRA are almost exclusively deployed on multi-billion parameter LLMs (such as LLaMA or Mistral), evaluating the complete multi-task framework on at least a 1B parameter language model (e.g., Pythia-1B or LLaMA-1B) under group-wise/block-wise configurations is highly necessary to validate your scaling hypotheses.
2. **Mitigation Redundancy:** Since "Re-Quantization Silence" is shown to be nearly lossless under standard per-channel configurations (losing only $0.15\%$ to $1.80\%$ mean accuracy), and SAWS degrades performance under per-tensor grids, what is the practical utility of SAWS for real-world edge deployment where per-channel grids are standard?
3. **Task Interference Confounder:** Simple task arithmetic (Naive FP16 Merge) collapses the continuous model to $66.65\%$ compared to the $93.85\%$ expert ceiling due to massive task interference. Why did you not implement your proposed **"Zero-Interference RQA Protocol"** (e.g., domain-aligned tasks or multilingual translation adapters) directly in this study to isolate quantization-induced representation erasure from pre-existing representation conflicts? Doing so would yield a much cleaner, unconfounded characterization of re-quantization erasure.
4. **Group-Wise Quantization:** In Section 3.2.3, you theoretically analyze group-wise or block-wise quantization (e.g., block size 128) and row-wise SAWS scaling. Why did you not include empirical results for group-wise quantization and block-wise SAWS scaling in the paper, since group-wise format is highly prevalent in modern LLM compression packages (AWQ, GPTQ)?
5. **Supervised vs. Unsupervised QA-ACS:** Since prediction entropy minimization under high 4-bit noise is highly prone to entropy collapse, and your Supervised QA-ACS completely stabilizes the optimization (boosting mean accuracy to $60.05\%$), does this not suggest that unconstrained unsupervised test-time optimization is fundamentally unsuitable for low-bit model merging? Why is unconstrained unsupervised QA-ACS presented as a primary proposed mitigation when the supervised and regularized variants are mathematically and empirically far more stable?
