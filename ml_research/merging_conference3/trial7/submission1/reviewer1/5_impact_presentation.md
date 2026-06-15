# Impact and Presentation Report (`5_impact_presentation.md`)

## 1. Major Strengths
Despite severe empirical and conceptual issues, the paper exhibits several strong qualities:
- **Outstanding Intellectual Honesty:** The authors are highly commended for Section 3.5, where they explicitly articulate the *Batch-Averaged Multi-Task Inference Paradox*. Instead of trying to sweep this massive logical flaw under the rug, they openly analyze why dynamic model merging under batch-averaging is either redundant (homogeneous batch) or equivalent to static merging (mixed batch). This level of scientific transparency is rare and exemplary.
- **Rigorous Systems-Level Analysis:** In Appendix 3, the authors present a comprehensive systems-level audit of memory-bandwidth and latency scaling in larger models (e.g., 7B parameter LLMs on H100 GPUs). This analysis mathematically demonstrates why full-parameter dynamic weight blending on the fly is a memory-bound bottleneck that doubles inference latency, thereby justifying why dynamic merging is only practical when restricted to PEFT modules (LoRA).
- **Thorough Robustness Checks:** The appendices contain robust empirical checks, including sensitivity to the projection dimension ($d$) and random projection seeds, confirming that their SVD Collinearity Ratio is statistically stable and robust to random projection variations.
- **Detailed Explanation of Failures:** The authors do not hide their failures; they openly discuss why their DeepMLP-12 results collapse to the level of random guessing and why the static OFS-Tune outperforms their dynamic router on TinyCNN-4.

## 2. Areas for Improvement
- **Abysmal Absolute Performance:** The authors must address the fact that their proposed system is functionally unusable. A 16% accuracy on Split-MNIST for MLP and a 52.5% accuracy on CNN under Cross-Domain task conflict represent complete model failures compared to the $>98\%$ Oracle ceiling.
- **Correction of Incorrect Gradient Mathematics:** The claim in Section 4.4 that the BSigmoid router "decouples gradient paths" and "avoids zero-sum competitive clashing" is mathematically incorrect. The post-gating sum-normalization re-introduces the competitive coupling at both the forward and backward gradient levels. The authors must correct this mathematical derivation.
- **Overstated Collinearity Refutation:** The authors' claim to "completely deconstruct" the rank-1 collapse is overstated. With reported collinearity ratios of $0.64$ to $0.74$ (where the absolute minimum for $K=2$ is $0.50$), the first singular value still heavily dominates. While it is not perfectly rank-1, the coefficients remain highly collinear.
- **Over-reliance on Toy Sandboxes:** The entire physical evaluation is restricted to Split-MNIST. The authors must scale their evaluation to full-scale, complex benchmarks (such as ImageNet, CIFAR-10, SVHN) with high-capacity models to demonstrate any real practical utility.
- **Lack of Open-Source Code:** No code repository, reproduction script, or environment configurations are provided. For a paper that critiques others' methodology, providing a fully reproducible open-source implementation of their own work is a critical necessity.

## 3. Overall Presentation Quality
The writing quality is **good**—the narrative is structured logically, and the mathematical and systems-level arguments are articulated in a sophisticated manner. The visual heatmaps (Figure 2) and scaling curves (Figure 4) are well-designed and easy to interpret. 

However, the framing is overly reactive, focusing excessively on debunking a single recent preprint (`[anonymous]`). The paper would benefit from a more balanced framing that focuses on the intrinsic challenges of weight-space merging, rather than an adversarial stance against a specific anonymous paper.

## 4. Potential Impact & Significance
The potential impact of this paper is unfortunately **extremely low**:
- **Practical Irrelevance:** Because the proposed dynamic layer-wise router is consistently outperformed by a simple static baseline (OFS-Tune) on CNNs, performs at the level of random guessing on MLPs, and has a massive performance gap compared to Oracle routing, researchers and practitioners have zero incentive to adopt this method.
- **Unresolved Fatal Paradox:** Since the authors do not solve the Batch-Averaged Multi-Task Inference Paradox (which they themselves identified), the entire class of dynamic model-merging methods they are analyzing remains a practical dead-end for heterogeneous batch serving.
- **Low Scientific Significance:** While the SVD diagnostic is a neat visualization tool, and the systems audit in the appendix is highly informative, these contributions are secondary and cannot rescue a paper whose core physical method is functionally non-viable.
