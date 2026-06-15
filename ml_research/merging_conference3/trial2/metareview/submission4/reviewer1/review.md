# Peer Review

## Summary of the Paper
The paper addresses parameter-space model merging, aiming to compose multiple task-specific expert models (fine-tuned from a common base model) into a single unified multi-task model without joint training. To overcome the latency and memory footprints of gradient-based adaptive model merging (like SyMerge or AdaMerging), the paper proposes **EdgeMerge**, a training-free, forward-only weight-space routing framework. 

EdgeMerge calculates localized, channel-wise merging coefficients in closed-form on a target "choke-point" bottleneck layer (specifically, the visual projection layer `model.visual.proj` in CLIP, representing $<0.5\%$ of parameters) using a tiny, unlabeled calibration dataset (e.g., 32 samples per task) in a single forward pass. This is achieved via three main stages: Forward-Only Activation Sampling (FOAS), Scale-Normalized Delta Activation Salience (SNDAS), and Channel-Wise Softmax Gating (CWSG). 

Additionally, to resolve a mathematical scaling discrepancy introduced by softmax normalization, the paper proposes **Decoupled Scale Routing (DSR)**, which decouples the scaling coefficient of the gated projection layer ($\lambda_{proj}$) from the static layers ($\lambda_{static}$). 

On an 8-task visual classification benchmark using CLIP ViT-B/32, EdgeMerge reduces calibration time to 11.95 seconds ($50\times$ speedup over SyMerge) and achieves a peak multi-task average accuracy of **69.58%** (outperforming standard Task Arithmetic's peak of 68.74%). It also demonstrates a broader, more stable hyperparameter scaling plateau than standard Task Arithmetic.

---

## Strengths and Weaknesses

### Strengths:
1. **Outstanding Clarity and Structure:** The paper is exceptionally well-written, mathematically rigorous, and highly organized. Figures and tables (especially the Pareto frontier in Figure 1 and the stability plot in Figure 3) are extremely polished and professional.
2. **Exceptional Transparency and Scientific Honesty:** The authors deserve massive commendation for their exemplary intellectual honesty. They explicitly present ablation studies that critically evaluate their own core mechanism and openly discuss a substantial 21.05% performance gap compared to server-grade, backpropagation-based methods (68.69% vs. 89.74%). Such transparency is rare and highly scientific.
3. **Rigorous and Honest Baselines:** The evaluation does not rely on a weak or unoptimized baseline. Standard Task Arithmetic is thoroughly optimized via a grid sweep to establish a strong empirical upper bound of 68.74% at $\lambda=0.20$. Comparisons are also drawn against several advanced static weight-alignment frameworks (Git Re-Basin, ZipIt!, TIES) and server-grade adaptive methods (SyMerge).
4. **Strong Practical Engineering Alignment:** The paper is deeply grounded in the real-world constraints of edge systems. Methodological trade-offs (such as sharing the base encoder's activations across all experts to save $8\times$ latency and memory) are clearly articulated and thoroughly justified both conceptually and empirically.
5. **Reproducibility:** The paper provides complete hyperparameters, precise dataset details, calibration settings, and a rigorous statistical standard error analysis ($SE_{avg} \approx 0.51\%$) to validate their subset evaluation, ensuring that the work is highly reproducible.

### Weaknesses:
1. **The Redundant Gating Paradox (Inert Core Novelty):** The primary conceptual contribution of this work is the activation-guided channel-wise softmax gating machinery (FOAS, SNDAS, CWSG), which is designed to resolve inter-task interference by routing individual weight channels based on task-specific saliency. However, the ablation studies (Section 4.3.4) show that:
   - **Full EdgeMerge (CWSG):** **69.58%** average accuracy.
   - **Layer-wise Gating (LWG):** **69.59%** average accuracy.
   - **Uniform Gating (Flat $\alpha_k = 1/K$):** **69.58%** average accuracy.
   
   This is a critical scientific and conceptual concern. It demonstrates that the entire elaborate calibration machinery (sampling activations over data, computing Frobenius norms, and routing channels via softmax) performs **exactly the same** as a completely uniform gating where every expert task vector in the projection layer is given a flat $1/K$ scale. The data-driven activation statistics are doing **no functional work**. If dynamic routing actually resolved inter-task weight conflicts, it should outperform uniform blending, which does not filter out conflicting channel updates. Because they are identical, the core proposed weight-routing paradigm is shown to be functionally inert.
2. **The Working Contribution is Highly Incremental:** Since the elaborate activation gating loop is redundant, the actual empirical benefits of the paper (the performance boost to 69.58% and the stabilization of the hyperparameter space) are driven **entirely** by **Decoupled Scale Routing (DSR)**. DSR is simply independent scale tuning of two different layers ($\lambda_{static}=0.25$ and $\lambda_{proj}=0.20$). While practical and useful, decoupling and tuning layer-specific scaling factors is a highly standard, well-known, and incremental parameter-tuning technique in the model merging literature. This significantly diminishes the conceptual originality and significance of the work.
3. **Standard Coupled EdgeMerge Underperforms:** Under standard, single-scale (coupled) configurations, EdgeMerge achieves a peak average accuracy of **68.69%**, which is actually *inferior* to the simple Task Arithmetic baseline (**68.74%**). This highlights that the core activation-gating algorithm, when not saved by the decoupled scale tuning (DSR), fails to outperform the simplest possible static weight average.
4. **Modality and Architecture Limitations:** Although the authors outline a highly promising and detailed mathematical translation of EdgeMerge to generative LLMs (e.g., SwiGLU FFNs) in Section 5, the entire empirical section is limited to Vision-Language CLIP ViT-B/32. Actually implementing and evaluating the method on larger backbones (ViT-L/14) and text-generative LLMs would be required to validate their broad generalizability claims.

---

## Soundness
* **Rating:** **Fair**
* **Justification:** Mathematically, the paper is highly rigorous and precise. The execution of the experiments, baseline optimization, and standard error bounding are mathematically sound. However, the paper falls short of the standard because its central conceptual claim—that activation-guided channel-wise gating resolves inter-task interference by dynamically routing channels to the most salient experts—is not supported by the authors' own empirical evidence. The ablation study refutes this claim, demonstrating that the activation-routing machinery is functionally inert and performs identically to a simple, static uniform composition of the same layers.

---

## Presentation
* **Rating:** **Excellent**
* **Justification:** The writing is outstandingly clear, elegant, and structured. The narrative flow is exceptionally easy to follow. Crucially, the level of intellectual transparency regarding the limitations, ablation failures, and performance-accuracy gaps relative to server-grade baselines is exemplary and represents a gold standard for scientific reporting.

---

## Significance
* **Rating:** **Fair**
* **Justification:** While the paper is highly practical and provides useful optimization heuristics for edge deployments, its overall scientific significance is modest. Because the core proposed weight-space routing paradigm is functionally redundant compared to uniform blending, the actual working contribution of the paper is an incremental layer-wise scaling adjustment (DSR). This is a useful hyperparameter-tuning guardrail for Task Arithmetic, but it does not represent a major conceptual leap that is likely to influence future machine learning research or change how the community thinks about model composition.

---

## Originality
* **Rating:** **Fair**
* **Justification:** The paper introduces an elegant and conceptually original idea of post-hoc activation-guided weight routing. However, because this beautiful concept is shown to have no functional advantage over static uniform composition, the working part of the framework is simply independent layer-wise scale tuning (DSR). Decoupling or optimizing layer-specific scales is highly standard in the model merging literature, meaning that the actual functioning novelty of the framework is highly incremental.

---

## Overall Recommendation
* **Rating:** **3: Weak reject**
* **Justification:** This paper has clear merits, including exceptional writing, baseline rigor, reproducibility, and outstanding scientific honesty. However, the weaknesses overall outweigh the merits. The core, highly-advertised conceptual contribution of the paper—forward-only activation-based channel routing—is shown by the authors' own ablation experiments to be functionally inert and redundant, achieving the exact same performance as static uniform blending. The actual working component of the framework is simply a highly standard, incremental layer-wise scaling adjustment (DSR). 

To be suitable for publication, the paper requires significant revision. The authors must find an experimental setup or model layer where activation-guided routing actually works and outperforms uniform blending, or they must completely reframe the paper around Decoupled Scale Routing, exploring more advanced decoupling schemes or theoretical analyses of layer scales rather than focusing on a redundant weight-routing mechanism.
