# 4. Experimental Setup and Empirical Evaluation Check

## Experimental Setup and Dataset Choices
The experimental design is highly systematic, well-controlled, and rigorous:
*   **Backbone:** Vision Transformer (\texttt{vit\_tiny\_patch16\_224}) partitioned into $L=14$ layer groups. This is a highly appropriate and challenging backbone, as ViTs are notorious for sharp local minima and high quantization sensitivity, making it an excellent "stress-test" environment.
*   **Tasks:** MNIST, FashionMNIST, CIFAR-10, SVHN. This is a diverse 4-task classification suite covering simple digits, fashion items, natural objects, and street numbers, representing a substantial multi-task parameter fusion challenge.
*   **Scale and Tractability:** The authors restrict pre-training to 512 images per task. While this budget results in lower absolute multi-task accuracy (with individual unquantized experts around $64.28\%$), it is a highly practical and standard design choice to make extensive, multi-axial grid sweeps computationally tractable (spanning 5 SAM radii, 4 merging methods, 8-bit and 4-bit precision, and 3 independent random seeds, totaling hundreds of individual experiments).

## Baselines Under Evaluation
The paper evaluates against four highly relevant and comprehensive baselines:
1.  **SGD Q-Merge ($\rho=0.0$):** Standard SGD experts quantized post-merging, with coefficients optimized directly in quantized space via STE. This serves as the natural sharp-expert baseline.
2.  **NaiveUniform:** SAM-trained experts merged using static uniform coefficients ($\lambda^l_k=0.3$) followed by PTQ, which isolates the power of expert-level geometry alone.
3.  **AdaMerging-PostQ:** Coefficients optimized in full FP32 on SAM experts, then quantized post-hoc, isolating the impact of test-time optimization in floating-point versus quantized space.
4.  **Individual-Quantized:** Task-specific SAM experts evaluated independently under quantization without merging, providing a realistic upper bound on performance.

## Do the Results Support the Claims?
Yes, the experimental results in Tables 1 and 2, along with the detailed discussions, provide **overwhelming empirical support** for the paper's central claims:
1.  **Precision-Dependent Flatness-Robustness Synergy:** 
    *   *Claim:* SAM flatness is highly critical in low-precision (4-bit) regimes but yields negligible gains in standard (8-bit) regimes.
    *   *Support:* Table 1 shows 8-bit accuracies for SGD ($\rho=0.0$) and optimal SAM ($\rho=0.05$) are virtually identical ($44.63\%$ vs $44.62\%$). Table 2 shows a massive, statistically significant boost under 4-bit quantization, with FlatQ-Merge increasing from $23.00\%$ (SGD) to $30.44\%$ ($\rho=0.05$), a **+7.44\% absolute gain**.
2.  **Pre-Merging Geometry Dominates Adaptation:**
    *   *Claim:* Loss landscape flatness is a far more critical driver than test-time adaptation sophistication.
    *   *Support:* NaiveUniform on flat experts ($\rho=0.05$, no test-time adaptation) achieves **29.03%** accuracy in 4-bit, outperforming FlatQ-Merge on standard SGD experts ($\rho=0.0$, full test-time adaptation) which gets **23.00%** (a **+6.03% absolute accuracy gain**).
3.  **The Over-Perturbation Threshold:**
    *   *Claim:* Large SAM radii ($\rho \ge 0.1$) trigger representation convergence and performance collapse.
    *   *Support:* Performance drops at $\rho=0.1$ and completely collapses at $\rho=0.2$ ($\approx 11\%$ accuracy). Task-vector cosine similarity analysis shows a dramatic surge from $0.071$ (SGD) to $0.247$ ($\rho=0.2$), empirically confirming representation convergence.
4.  **Peak Memory Footprint Advantage:**
    *   *Claim:* Direct quantized adaptation minimizes peak device RAM compared to FP32 post-hoc adaptation.
    *   *Support:* For ViT-Tiny, AdaMerging-PostQ requires loading FP32 weights, requiring $22.8\text{MB}$ peak memory compared to FlatQ-Merge's $2.85\text{MB}$ (an $8\times$ reduction), verifying the systems-level edge utility.

## Exceptional Quality and Thoroughness of Ablations
The empirical validation is exceptionally robust, featuring high-signal, rigorous ablations that address every potential counter-argument:
1.  **Independent Clipping vs. Convex Softmax Combination:** Confirms independent clipping is far superior (+8.20% in 8-bit, +3.03% in 4-bit) and provides a deep, layer-wise coefficient stability analysis showing that test-time adaptation operates on a stable, narrow sub-pixel manifold without exploiting boundaries.
2.  **DARE Baseline Integration:** Shows FlatQ-Merge is fully orthogonal to and compatible with advanced sign-conflict and parameter-pruning merging methods like DARE, yielding a +5.96% absolute gain when combined.
3.  **Implicit Regularization Validation:** Directly compares FlatQ-Merge's 56-parameter coefficient optimization against a high-dimensional TENT-style adaptation of all 5.7M weights. The high-dimensional baseline completely collapses to random guessing ($13.28\%$), while FlatQ-Merge remains highly stable ($27.64\%$), proving the implicit regularizing power of the low-dimensional search space.
4.  **Alternative Flatness Pathways (SWA vs. SAM):** Reveals a brilliant geometric distinction: SWA (trajectory averaging) works well under moderate (8-bit) noise but fails completely under extreme (4-bit) noise (22.62% vs SAM's 30.44%), proving that SAM's coordinate-wise adversarial formulation is uniquely necessary for low-bit robustness.
5.  **Direct Weight-Space Curvature Measurement:** Actively measures Hessian trace proxy via random Gaussian parameter perturbations, demonstrating an **8x reduction in curvature** for SAM ($\rho=0.05$) experts. Curvature metrics correlate perfectly with 4-bit merging resilience (SGD: 0.1579 curvature / 23.00% accuracy; SAM $\rho=0.05$: 0.0197 curvature / 30.44% accuracy), closing the theoretical loop.

## Limitations
*   **Absolute Accuracy Level:** Restricted to 512 images per task to make sweeps tractable. While this reduces absolute accuracy compared to fully converged models on full datasets, the authors are highly transparent about this and provide a clear scaling recipe in the Limitations section.
*   **Small Backbone Scale:** ViT-Tiny is a small network. However, the authors' focus is on second-order optimization theory, which is fundamentally architecture-independent, and they outline scaling to LLMs as a high-potential future direction.
