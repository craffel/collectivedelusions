# Impact & Presentation

## Major Strengths
1. **Outstanding Conceptual Transparency and Honesty:** Unlike typical SOTA-chasing papers, this work is exceptionally self-critical. It rigorously deconstructs and presents the fundamental mathematical and optimization limitations of its own proposed methods (SAWS and QA-ACS), detailing the "Representation Scale Preservation Dilemma" and "Entropy Collapse" under high discretization noise.
2. **Methodologically Elegant Decoupling Audits:** The design of the individual unmerged quantized expert control experiment (Table 7) is brilliant. It successfully decouples pre-existing weight-space task interference from post-training quantization noise, revealing that standard per-channel quantization does not erase task-specific adapter updates.
3. **Rigorous and Mathematical Formulation:** The paper provides exact formulations for QLoRA merging, uniform quantization, double-quantization noise, and the proposed mitigations, making it theoretically sound and academically rigorous.
4. **Comprehensive Multi-Axial Auditing:** Evaluating performance across four distinct quantization schemas (INT8/INT4, symmetric/asymmetric, per-tensor/per-channel) ensures a highly complete and robust characterization.

## Areas for Improvement
1. **Scale and Model Generalization:** The primary empirical evaluation is restricted to a small 5.7M parameter Vision Transformer (\texttt{vit\_tiny}) on toy classification datasets. To be highly convincing for modern AI deployment, the framework needs to be evaluated on multi-billion parameter Large Language Models (LLMs) under group-wise or block-wise quantization formats (such as AWQ or GPTQ).
2. **Ineffectiveness of Proposed Mitigations when Needed:**
   - Under per-tensor grids (the only configuration where Naive-RQ collapses), SAWS performs worse than Naive-RQ ($56.40\%$ vs. $56.75\%$), and QA-ACS suffers from unsupervised entropy collapse ($37.80\%$ on MNIST).
   - Under standard per-channel grids, Naive-RQ is already virtually lossless (losing only $0.15\%$ to $1.80\%$), making the proposed mitigations practically unnecessary.
   - Thus, the proposed mitigations do not provide a robust, practical solution under the severe constraints where they are actually required.

## Overall Presentation Quality
The presentation is **excellent**. The writing is clear, concise, and highly professional. The mathematical notations are precise, and the narrative flow transitions logically from the formulation of the problem and format-shift audits to experimental evaluations and deep empirical deconstructions. The tables are clean and well-structured, presenting data transparently.

## Potential Impact & Significance
The potential impact of this paper is **highly significant**. By exposing a major methodological blindspot (full-precision evaluation of merged models), the paper establishes a new deployment-aware standard for PEFT and model-merging research. 
Crucially, from a **Minimalist** perspective, the paper has a highly positive, grounding influence on the community:
- It shows that standard, simple per-channel post-training quantization (which is already the industry standard) is highly robust and natively preserves adapter updates.
- It prevents researchers from chasing overly complex, fragile, and unnecessary scaling or optimization-based mitigations.
- It redirects the field to focus on the actual, primary bottleneck of model merging: **weight-space task interference**, which accounts for the massive $27\%$ performance drop before any quantization is even applied.
- The proposed "Zero-Interference RQA Protocol" provides a concrete, clean path forward for isolating and benchmarking future quantization-merging interactions under pristine conditions.
