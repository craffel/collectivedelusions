# Presentation and Impact Evaluation: Q-PolyMerge

## 1. Major Strengths
1. **Mathematical and Analytical Rigor:** The paper is written with high technical maturity. The mathematical formulations are complete, and the detailed systems-level derivations (activation cache derivation, fully-integerized activation execution formulas) are highly transparent and educational.
2. **Clear Problem Identification:** Identifying the "Overfitting-Optimizer Paradox" on tiny test-time calibration streams is a highly pragmatic and insightful contribution that directly addresses a real-world edge deployment bottleneck.
3. **Dual-Optimization Vision:** Providing both a first-order gradient pathway (STE) and a zero-order derivative-free pathway (1+1 ES) is a highly logical and complete design. It demonstrates a deep understanding of hardware memory limitations (gradient caching overhead) and attempts to solve them algorithmically.
4. **Comprehensive Systems Context:** The inclusion of orthogonal Chebyshev bases to handle Runge's phenomenon in deep architectures and the detailed hardware-specific operator execution strategies show excellent architectural foresight.

## 2. Areas for Improvement
To elevate the paper to a high-impact, tier-1 conference standard, the authors should address the following critical gaps:
1. **Scale Up to Foundation Models:** Evaluate Q-PolyMerge on modern foundation models (such as CLIP-ViT-B/16 or LLaMA-7B/70B) on challenging, large-scale multi-task benchmarks (such as VTAB, ImageNet-1K/C, MMLU, or GSM8k). A 5.7M parameter ViT on upsampled 28x28 grayscale MNIST images is too toy-scale to be empirically convincing.
2. **Incorporate the Simple Task-Wise Baseline:** Add an optimized **Task-Wise Q-Merge** (4 parameters, uniform scaling across layers) to the comparison. This is essential to prove that learning a continuous polynomial trajectory across layers is actually superior to learning a single scale per task.
3. **Conduct Physical Hardware-in-the-Loop Measurements:** Replace the "modeled" latency, energy, and theoretical SRAM tables with empirical measurements taken on actual physical edge silicon (e.g., ARM Cortex-M7 STM32H7 or RISC-V GAP8) to substantiate the "physical edge viability" claims.
4. **Address and Stabilize Zero-Order Volatility:** Explore and evaluate more stable derivative-free optimization techniques (such as CMA-ES, Multi-Candidate Population Search, or Historical Momentum Filtering) to address the high variance and fragility of 1+1 ES in low-bit landscapes (especially the SVHN failure in 4-bit).

## 3. Overall Presentation Quality
The presentation quality of the submission is **excellent**. 
- The paper is exceptionally well-structured, clear, and easy to follow.
- It properly positions itself relative to both classic model merging literature and concurrent 2025/2026 low-bit merging works (such as TVQ, E-PMQ, and 1bit-Merging).
- Figures (such as the average accuracy comparison and the qualitative learned trajectory visualization) are high-quality, clear, and directly support the text.

## 4. Potential Impact and Significance
The paper addresses a highly important and relevant problem: consolidating multiple specialized task-specific experts into a single, low-bit model that can adaptively align itself on low-power edge nodes.
- If the proposed continuous polynomial subspace constraint is shown to generalize successfully to large-scale LLMs and physical hardware deployments, it has **high potential impact** for edge AI practitioners, smart sensor networks, and autonomous robotics.
- However, in its current state, the empirical convincingness is severely bottlenecked by the toy-scale experiments, the high variance across seeds, the lack of physical hardware validation, and the omission of the simple task-wise baseline. Addressing these limitations is essential to unlock its full scientific and practical significance.
