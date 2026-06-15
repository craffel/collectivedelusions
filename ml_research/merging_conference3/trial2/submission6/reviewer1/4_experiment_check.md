# 4. Experiment Check

This document provides a critical evaluation of the experimental setup, datasets, baselines, and empirical results of the **Q-Merge** paper to determine if the central claims are fully supported by evidence.

## Critical Evaluation of the Experimental Setup & Datasets

### 1. Strengths of the Setup:
*   **Multi-Seed Robustness:** All core experiments are evaluated across **three independent random seeds (42, 100, 2026)** with both means and standard deviations reported. This is a highly robust statistical setup that prevents cherry-picking and confirms the stability of the optimization.
*   **Data-Efficient Design:** The use of an extremely compact, disjoint calibration set (16 images per task, 64 in total) is a realistic simulation of edge-deployment scenarios where local data is highly scarce or restricted.
*   **Standard Benchmark:** Fusing MNIST, FashionMNIST, CIFAR-10, and SVHN classification tasks is a standard, well-established multi-task benchmark in the model-merging literature.

### 2. Limitations in Scale (The "Toy-Scale" Critique):
*   **Model Backbone Scale:** The experiments rely exclusively on a pre-trained **ViT-Tiny** backbone from the `timm` library. While ViT-Tiny (5.7M parameters) is an ideal candidate for low-resource edge deployment, it is very small by modern machine learning standards. In practice, model merging and PTQ are heavily utilized on multi-billion parameter models (e.g., LLaMA, Mistral) and high-capacity vision-language models (e.g., CLIP-ViT). The generalizability of these findings to large-scale generative models remains unproven in this paper.
*   **Low-Capacity Expert Training (Low Parameter Drift):** The downstream experts are fine-tuned on disjoint subsets of only **512 images per task** for 5 epochs. Because of this low-data training, the experts remain structurally and parameter-wise very close to the pre-trained base model (low parameter drift). 
*   **SVHN Performance Limitation:** Under this low-data fine-tuning regime, the unmerged SVHN expert achieves only **41.34%** accuracy, which is far from state-of-the-art. The uniform merged model drops to **32.42%**. Under 4-bit, individual experts drop further to **32.10%** and the merged model falls to **26.89%** (barely above the 10% random guess).
*   **Real-world Enterprise Contrast:** In real-world enterprise applications, expert models are often fully fine-tuned on massive datasets, resulting in significant parameter drift where weights diverge far from the base checkpoint and linear mode connectivity is more severely challenged. While the authors discuss this parameter-drift challenge in Appendix G, the actual empirical validation of Q-Merge under high parameter drift is missing.

## Critical Evaluation of Baselines

The baselines evaluated in this work are **exceptionally comprehensive and scientifically rigorous**:
1.  **Uniform FP16 & AdaMerging FP16 ceilings:** The authors include the true unquantized ceilings, which sets a clear baseline for how much performance is lost due to quantization.
2.  **Order of Operations (Q-then-M vs. M-then-Q):** The authors compare both merging discrete quantized expert weights (Q-then-M) and quantizing naive uniform merged weights (M-then-Q), covering both standard practitioner workflows.
3.  **Optimizer Confounding Factor Deconstruction (Crucial Scientific Control):** To avoid attributing the performance boost of Q-Merge (Adam GD) over AdaMerging (ES) solely to a "quantization regularization effect", the authors implemented a fully differentiable *AdaMerging (FP16 Optimized with Adam GD)* baseline. By isolating the optimizer, they proved that:
    *   Under 8-bit, the unquantized Adam ceiling ($74.38\%$) is slightly superior to 8-bit Q-Merge ($74.30\%$), showing a tiny $0.08\%$ representation loss.
    *   Under 4-bit, optimizing directly under the operator (Q-Merge Adam GD) achieves $63.36\%$, outperforming post-hoc quantized AdaMerging ($62.01\%$) by $1.35\%$.
    This represents a masterclass in experimental control.
4.  **Advanced PTQ (AdaRound):** Comparing Q-Merge against standalone AdaRound (a leading PTQ local reconstruction optimizer) demonstrates that Q-Merge's global coordinate alignment is conceptually superior to local rounding optimization, and showing that they are highly complementary when combined sequentially ($64.46\%$ accuracy).

## Do the Results Support the Claims?

The empirical results in Tables 1 and 2, alongside the ablation studies, provide **complete and compelling support** for the paper's three central claims:

### Claim 1: Overcoming the 8-Bit Quantization Gap
*   *Support:* Under 8-bit quantization, Q-Merge (Adam GD with STE) achieves **74.30%** average accuracy. This is nearly lossless compared to the unquantized Adam-optimized ceiling (**74.38%**), and it strictly outperforms the uniform unquantized FP16 baseline (**71.88%**) and standard unquantized AdaMerging ES (**73.21%**). The results fully validate that 8-bit low-bit model merging is ready for deployment.

### Claim 2: First-Order (STE) vs. Zero-Order (1+1 ES) Stability
*   *Support:* 
    *   Under 8-bit, Adam GD achieves $74.30\% \pm 0.38\%$ vs. 1+1 ES at $72.57\% \pm 1.06\%$ (over $2.7\times$ lower seed-to-seed variance).
    *   Under 4-bit, Adam GD achieves $63.36\% \pm 1.18\%$ vs. 1+1 ES at $57.83\% \pm 1.47\%$.
    *   Adam GD also converges faster (20 steps) than 1+1 ES (40 steps).
    The evidence is clear and statistically sound: gradient-based optimization using STE is vastly superior to derivative-free stochastic search in quantized weight spaces.

### Claim 3: Unlocking 4-Bit Model Merging
*   *Support:* Under 4-bit quantization, per-channel quantization successfully prevents weight-space collapse, allowing the naive baseline to achieve **56.66%** average accuracy. Combining per-channel quantization with Q-Merge (Adam GD with STE) further optimizes weight alignment, achieving **63.36%** average accuracy (outperforming naive by **6.70%** absolute). This empirically proves that extreme 4-bit model merging is viable.

## Summary Checklist
*   [x] Are the claims well supported? Yes, backed by multi-seed experiments.
*   [x] Are the methods appropriate? Yes, per-channel PTQ and STE are highly suited.
*   [x] Are the baselines exhaustive? Yes, including unquantized ceilings, naive pipelines, and advanced PTQ.
*   [ ] Is the scale sufficient? **No, the backbone (ViT-Tiny) and experts (512-image fine-tuning) are toy-scale.** This is the primary scientific limitation of the paper.
