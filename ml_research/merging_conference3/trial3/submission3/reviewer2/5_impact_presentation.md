# 5. Impact and Presentation Evaluation

This file lists the major strengths, areas for improvement, overall presentation quality, and potential impact/significance of the paper.

## Overall Presentation Quality
The overall presentation quality of this paper is **excellent**. It is exceptionally well-written, mathematically structured, and easy to follow. 
* **Strengths in Presentation:** The narrative flow is highly logical, starting from the edge deployment context, identifying a concrete threat (Noise-Entropy Collapse), proposing a clean theoretical solution (FlatMerge), and concluding with a transparent discussion of hardware and computational tradeoffs (Section 3.5 and Section 5.1). The inclusion of detailed latency, SRAM memory, and DRAM transfer bandwidth analysis represents high-quality writing.
* **Minor Presentation Weaknesses:**
  1. **Notational Disconnect:** The perturbation radius is written as $\sigma$ in the methodology but referred to as $\rho$ in the experiments.
  2. **Broken Citation:** The citation for the core baseline **PolyMerge** is undefined in the bibliography, resulting in broken/missing references in the manuscript.
  3. **Incomplete Method Details:** The exact learning rate ($\eta$) used for adaptation is missing.

## Major Strengths of the Paper
1. **Strong Motivation and Practical Focus:** The focus on resource-efficient, robust test-time model merging for edge devices is highly relevant and addresses a real-world engineering challenge.
2. **Elegant Conceptual Shortcut:** Applying sharpness-aware minimization (SAM) directly to the extremely compact 12-parameter coefficient space via Zeroth-Order optimization is a brilliant and computationally trivial alternative to standard high-dimensional weight-space SAM.
3. **True Zero-Activation Memory Caching:** Bypassing backpropagation entirely and keeping peak adaptation memory identical to standard forward inference is a massive advantage for SRAM-constrained edge accelerators.
4. **Honesty and Transparency:** The authors are exceptionally honest about the DRAM-to-SRAM weight-reconstruction bandwidth bottleneck (Section 3.5) and lay out very practical, asynchronous periodic adaptation strategies to mitigate it.

## Key Areas for Improvement
1. **Bridge the Simulation-to-Real Gap:** The paper's primary 12-layer Vision Transformer (ViT-B/32) results are based entirely on a simulated Rastrigin-like loss landscape sandbox. The authors should evaluate FlatMerge on actual, physical CLIP ViT-B/32 weights to prove that the simulation-to-real gap is indeed manageable and that FlatMerge works on actual high-capacity Vision Transformer parameters.
2. **Resolve the "Adaptation-Decline" Paradox:** On the physical 5-layer CNN model, the static Task Arithmetic baseline (uniform blending of 0.3) outperforms all adaptive methods, including ZO-FlatMerge, by a massive margin ($58.20\%$ vs $48.57\%$ on clean data; $40.67\%$ vs $29.20\%$ under moderate noise). The authors need to address this critical weakness. Why run a complex, computationally expensive on-device adaptation loop if a simple static uniform configuration performs $10\%$ better? The authors should find ways to improve FlatMerge's performance on real convolutional weights so that it actually beats the static baseline.
3. **Implement and Evaluate the Adaptive Perturbation Radius:** In Section 3.3, the authors proposed a highly interesting, novel feature—an **Adaptive Perturbation Radius** ($\sigma(X)$) that scales with batch entropy to handle non-stationary noise. This was completely omitted from the experiments. Implementing and evaluating this feature would dramatically enhance the paper's novelty and practical adaptability.
4. **Fix the Bibliography and Notation:** Add the missing `polymerge` bibliographic entry and unify the perturbation scale symbol ($\sigma$ vs $\rho$).

## Potential Impact and Significance
* **If current form:** The significance is **fair**. While the concept is beautiful, the primary evaluation is simulated, and the physical validation is on toy networks where adaptation actually degrades performance compared to doing nothing (static Task Arithmetic). This severely limits its immediate practical significance.
* **If improved (evaluating on real ViT weights and beating static baselines on physical models):** The significance would be **excellent/outstanding**. Making adaptive model merging robust to test-time noise under strict edge-hardware boundaries would represent a major milestone for on-the-fly multi-task learning.
