# Impact and Presentation

## 1. Major Strengths
- **Conceptual Breakthrough**: Exposing the **Dynamic Routing Paradox** and the **Batch-Average Smoothing Confounder** are powerful, paradigm-clarifying insights that challenge the prevailing assumptions of test-time dynamic model merging. It exposes that many complex dynamic routing systems in the literature are overfitted, with their overfitting hidden by batch-average evaluation.
- **Exceptional Empirical Rigor**: Run across 10 independent random seeds. Exhaustive sweeps over subspace overlap $\rho \in [0.0, 1.0]$, projection dimension $d \in \{2, 4, 8, 16\}$, calibration data size $|D_{\text{cal}}| \in \{64, 128, 256, 512, 1024\}$, multi-layer routing depth (MLP), and Dynamic LoRA adapter rank $r \in \{2, 4, 8, 10, 12\}$.
- **Outstanding Intellectual Honesty**: The authors openly demonstrate that their proposed Task-Variance Regularization ($\mathcal{L}_{VR}$) loss is empirically redundant because the simple zero-initialized Softmax architectural prior does all the heavy lifting. This transparency is rare and highly valuable.
- **Systems-Level Depth**: Includes a detailed, rigorous analysis of systems-level and hardware bottlenecks (VRAM expansion, memory bandwidth bounds, reduced GPU arithmetic intensity) and proposes a mathematically elegant, systems-efficient solution (Dynamic LoRA parameter assembly) that completely eliminates VRAM footprint expansion.
- **Real-World Grounding**: Beautifully bridges the gap between synthetic sandboxes and real-world deep neural networks by validating the findings on actual MNIST + FashionMNIST experts with a shared CNN backbone.

## 2. Areas for Improvement (Constructive Suggestions)
- **Scale of Real-World Evaluation**: While the MNIST+FashionMNIST visual expert merging experiment and the Appendix A roadmap for CLIP ViT-B/16 are excellent, the main results remain centered on the 192-dimensional Analytical Coordinate Sandbox. Evaluating "Vectorization Collapse" and the "Dynamic Routing Paradox" directly on actual large-scale vision-language models (like full CLIP ViT-L/14 or LLaMA-70B) in the main text would elevate the paper's impact even further.
- **Functional Impact of Sequential Smoothness**: The proposed Sequential Smoothness Regularizer ($\mathcal{L}_{\text{smooth}}$) is shown to reduce sequential routing weight jitter by over 57.5% with zero accuracy degradation, which is excellent. However, because of the sandbox's layer-averaging simplification, routing jitter has no functional impact on final accuracy in this toy setup. Showing the direct functional benefit of this jitter reduction on accuracy in deep models where parameters are processed sequentially is a valuable direction for future work.

## 3. Overall Presentation Quality
- **Excellent**: The paper is beautiful, structured logically, and written in a clear, monospaced-friendly Markdown.
- Mathematical equations are highly precise and complete, down to numerical stabilizers ($\epsilon$) and details on population vs. sample variance.
- Tables are clean, well-formatted, and include standard deviations.
- The text is self-contained and offers a comprehensive guide for reproduction.

## 4. Potential Impact and Significance
This paper has **substantial potential impact** on the machine learning community:
- **Reshaping Research Directions**: It is likely to divert researchers away from designing overly complex, non-monotonic routing equations (like wave cosine activations) and towards focusing on simple architectural priors, proper initialization, and scaling calibration datasets.
- **Raising Evaluation Standards**: By exposing that batch-average evaluation masks overfitting, this paper serves as a vital methodological warning that will encourage future model merging papers to report results at $B=1$ under heterogeneous streams.
- **Promoting Static Uniform Merging**: By demonstrating that the well-regularized router only yields a marginal $+1.16\%$ gain over Uniform Merging, it highlights Uniform Merging as an exceptionally strong, cost-free default baseline, which will have massive practical utility for practitioners deploying models in resource-constrained environments.
