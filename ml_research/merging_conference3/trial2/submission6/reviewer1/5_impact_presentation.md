# 5. Impact and Presentation

This document evaluates the presentation quality, major strengths, areas for improvement, and potential impact/significance of the **Q-Merge** paper.

## Presentation Quality
The presentation quality of this submission is **outstanding**.
*   **Structure:** The narrative flows logically from the introduction of the deployment bottleneck to weight merging, PTQ, and the proposed Q-Merge framework.
*   **Clarity:** The writing is highly professional, concise, and academically rigorous. The mathematical notations are precise, and terms are clearly defined.
*   **Aesthetics:** The figure and tables are extremely detailed, self-contained, and provide a clear, high-signal visualization of the quantitative results.
*   **Exhaustive Appendix:** The appendix is exceptionally thorough, preempting and comprehensively addressing almost every common reviewer concern (such as activation quantization, scale factor discretization, backpropagation memory complexity, and calibration stream noise) with both rigorous theoretical formulations and empirical ablations.

## Major Strengths

1.  **Exemplary Scientific Rigor and Controls:** The paper stands out for its meticulous experimental design. The deconstruction of the optimizer confounding factor (separating the benefits of Adam GD from the quantization operator) is brilliant and represents a high standard of ML research.
2.  **High Practical Edge Utility:** Q-Merge is designed with real-world deployment constraints in mind. It converges in seconds on tiny calibration sets and introduces **zero inference-time latency or parameter overhead** since the blending coefficients are discarded after final weight quantization.
3.  **Comprehensive Systems-Level Analyses:** The authors provide detailed analyses of systems-level constraints, including fixed-point scale factor discretization sensitivity, activation quantization (W8A8/W4A4), peak memory consumption (comparing activation caching in STE vs. 1+1 ES), and dynamic stream task-balancing heuristics.
4.  **Excellent Performance:** 8-bit Q-Merge achieves near-lossless multitasking fusion ($74.30\%$ vs $74.38\%$ unquantized ceiling), while 4-bit Q-Merge with per-channel quantization successfully overcomes model collapse to achieve an outstanding $63.36\%$ average accuracy.

## Areas for Improvement

1.  **Toy-Scale Experimental Generalizability:** The primary weakness of the paper is the scale of the experiments. Evaluating exclusively on a **ViT-Tiny backbone (5.7M parameters)** on a classification benchmark represents a toy-scale setting. The machine learning community is heavily focused on multi-billion parameter Large Language Models (LLMs) and Vision-Language Models (VLMs) where model merging and PTQ are most commercially valuable. The paper would be significantly stronger if it included experiments on larger models (e.g., LLaMA-1B or CLIP-ViT-B) and generative benchmarks.
2.  **Validation under Extreme Parameter Drift:** The task experts are trained in a low-data regime (512 images), meaning parameter drift is low. In enterprise settings, experts are fully fine-tuned on massive datasets, causing them to diverge far from the base model. While the authors discuss this challenge theoretically in Appendix G, they provide no empirical evidence of how Q-Merge performs under high parameter drift.
3.  **Low Baseline Accuracy on SVHN:** Due to low-data training, the absolute performance on SVHN is low (e.g., $41.34\%$ for individual experts, $35.87\%$ for Q-Merge), which limits the overall strength of the classification claims.

## Potential Impact and Significance

*   **High Real-World Impact:** Model merging and post-training quantization are two of the most critical techniques for deploying models under strict edge constraints. By bridging this gap, Q-Merge has immediate practical significance for edge systems engineers.
*   **Methodological Influence:** Proving that first-order optimization via STE is highly stable and superior to zero-order mutation for low-dimensional test-time adaptation parameters could inspire researchers to apply STE to other non-differentiable test-time adaptation problems (e.g., discrete prompt tuning, token pruning, or routing optimization).
*   **Viability of Low-Bit Merging:** Correcting the assumption of "4-bit collapse" by demonstrating that per-channel quantization preserves mode connectivity, and that Q-Merge optimizes it further, is a highly valuable insight that can shift how the community approaches extreme compression in multi-task networks.
