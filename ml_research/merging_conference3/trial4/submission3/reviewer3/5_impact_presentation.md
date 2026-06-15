# 5. Impact and Presentation

## Major Strengths
1. **Outstanding Scientific Honesty and Self-Criticism:**
   Unlike many papers that obscure the failures or limits of their proposed methods, this work is exceptionally transparent. The authors critically deconstruct their own proposed methods: they mathematically prove why "mathematically exact" scale preservation in SAWS collapses base representations, and they expose the vulnerability of QA-ACS to entropy collapse under aggressive noise.
2. **Methodological Rigor and Creative Controls:**
   The paper is packed with high-quality, creative methodological analysis. The **Individual Expert Auditing** control experiment (Table 8) is a highly elegant way to isolate quantization noise from pre-existing task interference, and the **Double Quantization Noise** study (Table 1) provides useful insights into format-shift representation errors.
3. **Comprehensive and Detailed Appendices:**
   The supplementary materials are of exceptional quality. The authors include exhaustive sweeps over SAWS and QA-ACS hyperparameters, a CPU latency profiling benchmark on a 128-core Xeon processor, and a solid theoretical analysis of scaling dynamics to multi-billion parameter LLMs.
4. **Physical Latency Profiling:**
   The physical CPU latency profiling (Appendix D.3) is highly insightful. It reveals the "Cache-Fitting vs. DRAM-Latency Bifurcation," providing a hardware-grounded justification for why weight-space merging is necessary for larger models despite co-existence being competitive on tiny, cache-fitting models.

## Areas for Improvement
1. **Scale of the Primary Backbone:**
   The main experiments are restricted to a tiny Vision Transformer (`vit_tiny`, 5.7M parameters). While this enables rapid multi-axial profiling, quantization and merging dynamics on toy networks may not fully generalize to multi-billion parameter LLMs. While the authors discuss scaling dynamics theoretically in Appendix C, the lack of full multi-task auditing results on an LLM is a notable weakness.
2. **Extreme Task Interference Confounder:**
   The full-precision unquantized baseline (Naive FP16 Merge) is already severely degraded (averaging $66.65\%$ compared to the $93.85\%$ expert ceiling). This reflects severe weight-space conflicts from merging highly distinct image classification datasets (MNIST, FashionMNIST, CIFAR-10, SVHN) using simple task arithmetic. The authors propose a "Zero-Interference RQA Protocol" to resolve this in future work, but the paper would have been much more impactful if this protocol had been implemented directly in the study.
3. **Redundancy of Mitigations under Standard Configurations:**
   The authors discover that "Re-Quantization Silence" is nearly lossless under per-channel configurations (losing only $0.15\%$ to $0.30\%$ accuracy in 8-bit and $1.80\%$ in 4-bit). Since almost all modern edge deployment pipelines utilize per-channel or group-wise quantization (such as AWQ or GPTQ), this means the "catastrophe" of Re-Quantization Silence only occurs under per-tensor grids, which are obsolete/rarely used. Consequently, the proposed SAWS and QA-ACS methods represent relatively modest improvements under standard, real-world deployment settings.

## Overall Presentation Quality
The presentation quality is **Excellent**:
- **Writing Style:** Highly articulate, formal, and mathematically rigorous. The terminology used is precise and professional.
- **Organization:** The logical progression is cohesive, starting with a clear identification of the magnitude mismatch, detailing the proposed mitigations, and following up with extremely thorough empirical and theoretical analyses of their limitations.
- **Visuals and Tables:** The tables (Tables 1-11) are clean, well-formatted, and clearly documented. Figure 1 provides an intuitive overview of the multi-task performance across different configurations.

## Potential Impact and Significance
The potential impact of this paper is **Fair to Good**:
- **Significance for Practitioners:** Because the "Re-Quantization Silence" is shown to be a per-tensor artifact rather than a general blocker under standard per-channel grids, the practical impact of the proposed mitigations (SAWS and QA-ACS) is somewhat limited. Engineers deploying merged models in real-world pipelines can simply use standard per-channel or group-wise quantization and experience very little degradation, making complex mitigations unnecessary.
- **Significance for Academics:** The paper’s true impact lies in its **methodological contributions**. By advocating for a deployment-aware approach to model-merging evaluation and providing rigorous control protocols (such as individual expert auditing and double-quantization error tracking), this work will likely influence future model merging and PEFT research to adopt post-training quantization as a standard benchmark step.
