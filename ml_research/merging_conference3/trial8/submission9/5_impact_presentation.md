# 5. Presentation and Impact Check

This section evaluates the writing quality, structural clarity, positioning, and potential scientific and engineering impact of the paper.

### 1. Presentation, Structure, and Writing Quality

The paper is exceptionally well-written, clearly structured, and easy to follow. It maintains a compelling, systems-focused narrative (written from "The Pragmatist" perspective) that bridges the gap between machine learning theory and physical edge hardware constraints.

- **Logical Flow:** The narrative progresses logically from the practical limitations of offline calibration data, to the mathematical formulations of the two zero-shot ensembling paradigms (EER and EPL-OCA), followed by detailed empirical evaluations, a systems-level latency/energy profile, and a discussion of real-world deployment challenges.
- **Clear Contextualization:** The work is excellently positioned relative to prior/concurrent literature. Section 2 clearly differentiates the proposed paradigms from static parameter merging (e.g., AdaMerging), centroid-routing activation ensembling (SPS-ZCA, SABLE), and test-time adaptation (TTA), detailing the exact memory, latency, and operational advantages of a forward-pass-only, calibration-free approach.
- **Mathematical Clarity:** All equations—including Normalized Shannon Entropy (Eq. 4), online centroid updates (Eq. 10), activation divergence, and the FLOP complexity equations—are clearly stated, mathematically consistent, and use standard notation.
- **Professional Formatting & Hardware Spec:** The authors explicitly document that wall-clock CPU execution latency was profiled on a single core of an AMD EPYC 7763 CPU @ 2.45GHz (Section 4.4), enabling precise practitioner calibration of hardware overheads. Physical units ($\mu\text{J}$, $\text{pJ}$, $\text{W}$, $\text{GHz}$) are perfectly typeset in standard math-mode formatting. The tables (Tables 1--8) are highly structured, reporting means and standard deviations across 5 seeds.

#### Remaining Areas for Presentation Improvement:
1. **Highlight the "Calibration-Free" Hybrid Classification of CG-EER More Prominently:**
   - Although the authors honestly re-classify CG-EER as a hybrid semi-supervised method in Section 4.10, the abstract and introduction still present the paper's main focus as "zero-shot calibration-free." Because CG-EER is the only method that functions effectively on real embeddings (achieving 61.50% compared to 35.38% for pure EER), this hybrid nature should be more clearly and prominently declared from the outset.
2. **Expand on Real-world Scale Manifestation of the Representational Sparsity Paradox:**
   - Section 4.11 ("Empirical Limitations and Real-World Roadmap") is solid, but it would benefit from explaining how the "Representational Sparsity Paradox" would manifest on large-scale models (e.g., LLaMA-3 or ViT-B/16). Would the higher dimensionality of these models exacerbate class-orthogonality or help reduce spatial jitter due to larger representational capacity?

### 2. Scientific and Engineering Impact

The potential impact of this paper is substantial, particularly within the **systems-ML** and **edge computing** communities.

- **Practical Edge Relevance:** Deploying multiple specialized LoRA adapters on resource-constrained devices (mobile, IoT, wearables) is a massive industry trend. By eliminating the manual calibration bottleneck and the backward-pass memory footprint of TTA, this work provides a direct blueprint for fully autonomous, plug-and-play multi-task on-device serving.
- **Bypassing the Representational Sparsity Bottleneck:** The identification and thorough scientific analysis of the *Representational Sparsity Paradox* in centroid-routing is a highly valuable contribution. The demonstration that soft activation blending ($\tau=0.5$) acts as a spatial regularizer to mitigate centroid jitter will likely guide future researchers working on training-free activation ensembling.
- **Hardware-Aware Design:** The integration of physical CPU benchmarking, DRAM/SRAM memory bandwidth analysis, and edge energy estimations represents a significant step forward from typical pure-algorithmic ML papers, ensuring that the proposed methods are physically viable on low-power devices.
- **Actionable Open-Source Potential:** The inclusion of concrete engineering mitigations like **Amortized Pseudo-Labeling** and **Centroid-Gated Entropy Routing (CG-EER)** makes this work immediately actionable for developers building PEFT serving frameworks (such as vLLM or LoRAX) on edge nodes.
