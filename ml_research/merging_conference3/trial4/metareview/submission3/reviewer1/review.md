# Peer Review

## Summary of the Paper
This paper investigates the downstream Post-Training Quantization (PTQ) behavior of merged parameter-efficient fine-tuning (PEFT) models—specifically QLoRA adapters merged into pre-trained Vision Transformer backbones. It exposes a widespread methodological blindspot in existing model-merging literature, which routinely reports results only in full precision (FP16/FP32). 

The paper mathematically deconstructs and audits the **"Re-Quantization Silence"** phenomenon, where low-bit quantization (such as 4-bit) of merged models can round task-specific adapter updates to zero due to a severe magnitude mismatch between base model weights and low-rank updates. To systematically evaluate this, the authors introduce a **Multi-Axial Re-Quantization Auditing (RQA)** framework across multiple quantization bit-widths (4-bit vs. 8-bit), formats (symmetric vs. asymmetric), and granularities (per-tensor vs. per-channel). They evaluate two proposed mitigations: **Scale-Adaptive Weight Shifting (SAWS)**, a data-free closed-form scaling method, and **Quantization-Aware Adapter Coefficient Search (QA-ACS)**, an optimization-based test-time adaptation (TTA) method. 

Crucially, the paper stands out for its exceptional self-critical honesty: it mathematically proves a **"Representation Scale Preservation Dilemma"** in SAWS (showing its success is driven by selective task-vector boosting rather than true scale preservation, which would collapse base model representations), exposes the fragility and risk of **unsupervised entropy collapse** in QA-ACS under high discretization noise, deconstructs **double quantization format-shift noise** as a key confounder, and conducts an **individual unmerged expert control experiment** to decouple pre-existing task interference in weight space from quantization-induced degradation.

---

## Strengths and Weaknesses

### Strengths
1. **Outstanding Intellectual Honesty and Transparency:** The authors deserve significant praise for their highly self-critical and deconstructive approach. Rather than hiding the limitations of their proposed mitigations to report an artificially inflated State-of-the-Art (SOTA), they mathematically and empirically lay bare the exact failure modes of both methods (the Scale Preservation Dilemma in SAWS and Entropy Collapse in QA-ACS). This level of transparency is rare and highly valuable to the ML community.
2. **Methodological Elegance in Decoupling Audits:** The design of the individual unmerged quantized expert control experiment (Table 7) is brilliant. By applying quantization directly to separate, unmerged experts, the authors successfully isolate the discretization noise from pre-existing weight-space task interference. This reveals that under standard per-channel configurations, post-training quantization does *not* erase adapter updates, proving that the low performance of merged quantized models is driven entirely by weight-space representation conflicts, not quantization erasure.
3. **Rigorous and Mathematical Grounding:** The paper is theoretically solid. The mathematical formulations of QLoRA merging, uniform symmetric/asymmetric quantization, double-quantization format-shift noise (including Table 1), and the closed-form derivations of SAWS scale factors are clear, precise, and rigorous.
4. **Insightful Quantization Granularity Bifurcation:** The multi-axial audit reveals a highly significant finding: the "Re-Quantization Silence" is not a universal catastrophe, but is highly localized to aggressive, sub-optimal per-tensor grids. Under industry-standard per-channel configurations, naive re-quantization is nearly lossless (dropping only 0.15% to 1.80% mean accuracy), rendering complex mitigations largely unnecessary.

### Weaknesses
1. **Scale and Model Generalization Limitations:** The primary empirical evaluation is conducted on a toy scale: a very small `vit_tiny` model (5.7M parameters) on simple classification datasets (MNIST, FashionMNIST, CIFAR-10, SVHN). While this choice is justified for rapid multi-axial profiling and the authors provide base-weight Frobenius error scaling on `vit_base` (86M parameters), it remains unverified whether these findings and mitigations translate to the multi-billion parameter LLMs (e.g., 7B+ parameter models) where QLoRA and model merging are most commonly deployed.
2. **Ineffectiveness of Proposed Mitigations Under Severe Constraints:** Under the aggressive 4-bit per-tensor configuration—the only setting where naive re-quantization actually collapses (dropping 10% accuracy)—both proposed mitigations fail to provide robust protection:
   - **SAWS** performs slightly *worse* than the unmitigated Naive-RQ baseline ($56.40\%$ vs. $56.75\%$), because a global layer-wise scaling multiplier cannot adapt to the non-uniform rounding boundaries of per-tensor grids.
   - **QA-ACS** suffers from unsupervised **entropy collapse**, dropping MNIST performance to $37.80\%$ (below Naive-RQ's $42.00\%$) because minimizing unconstrained prediction entropy under severe discretization noise drives the low-capacity network into degenerate, collapsed output distributions.
3. **Over-Engineering of the Optimization-Based Mitigation:** The proposed QA-ACS method introduces substantial algorithmic complexity—including test-time adaptation, continuous layer-wise coefficient search, straight-through estimator (STE) gradient approximations, tracking running second moments with Adam to filter gradient noise, and calibration set dependencies—only to prove highly fragile and prone to collapse unless further regularized. This highlights that complex, optimization-based test-time mitigations are often fragile and over-engineered compared to simple baseline solutions.

---

## Dimension Ratings

### Soundness: Excellent
The paper's technical claims are exceptionally well-supported, theoretically sound, and rigorously validated. The authors are incredibly honest about the strengths and weaknesses of their methods. The design of the individual expert control experiment is a masterclass in isolating confounding variables, making the soundness of the empirical methodology outstanding.

### Presentation: Excellent
The paper is beautifully written, exceptionally well-structured, and easy to follow. The mathematical notation is precise and self-contained, and Algorithm 1 provides a clear, reproducible implementation of the QA-ACS loop. The tables are clean and present a dense, multi-axial set of results in a highly legible format.

### Significance: Good
The paper addresses a highly important and realistic deployment problem, establishing a necessary deployment-aware evaluation standard for model merging. However, the significance is slightly limited by the small-scale backbone (`vit_tiny`) and toy datasets used in the main experiments. Nonetheless, the core insight—that standard per-channel quantization is already virtually lossless and natively preserves task updates—is highly significant as it simplifies deployment pipelines and prevents researchers from chasing unnecessary, complex mitigations.

### Originality: Excellent
The paper provides deep, novel insights into the interaction between post-training quantization and model merging. Exposing the quantization granularity bifurcation, deconstructing the double-quantization format shift noise, mathematically proving the Representation Scale Preservation Dilemma, and proposing the Zero-Interference RQA Protocol are all highly original, insightful, and valuable contributions to the literature.

---

## Overall Recommendation

**Rating: 5: Accept**

### Recommendation Justification
This is a highly rigorous, theoretically sound, and beautifully written paper that addresses a crucial and under-explored methodological blindspot in the PEFT and model-merging literature. 

The paper's greatest contribution is its **profound commitment to simplicity and rigorous deconstruction**. Instead of presenting an over-engineered, highly complex method and claiming an artificial SOTA, the authors perform a transparent and self-critical analysis of the entire system. They prove that:
1. The simplest, standard deployment baseline (per-channel post-training quantization) is already virtually lossless and natively preserves task-specific updates, meaning no added engineering complexity is required in standard scenarios.
2. The proposed closed-form scaling (SAWS) has a fundamental mathematical scale dilemma and works via selective boosting rather than true scale preservation.
3. The proposed optimization-based test-time adaptation (QA-ACS) is highly fragile and prone to entropy collapse under high noise, illustrating the limitations of over-engineered test-time optimization.
4. The primary bottleneck in these systems is actually pre-existing task representation conflict in weight space, rather than quantization noise.

By steering the community away from over-engineered, fragile test-time mitigations and redirecting focus toward solving weight-space task interference in its simplest form, this paper provides an exceptionally high-value service to the field. Although the experimental scale is currently limited to a toy-scale Vision Transformer, the methodological rigor and self-critical deconstruction make this paper an outstanding and highly complete contribution that fully deserves acceptance.
