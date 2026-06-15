# 5. Impact, Presentation, and Actionable Areas for Improvement

This section outlines the paper's key strengths, critical areas for improvement, overall presentation quality, and potential real-world impact.

## Major Strengths

1. **Extreme Simplicity and Practical Efficiency**:
   * NETA is completely training-free, parameter-free, and data-free. It requires zero GPU/CPU forward or backward passes at test-time, zero training epochs, and zero calibration datasets.
   * This represents a massive practical advantage. It can be executed instantaneously (in less than a second) directly on model weights. This is ideal for resource-constrained edge devices or high-throughput production pipelines where optimization loops are impractical.

2. **Rigorous Engineering Design and Physical Intuition**:
   * The paper does not just present a basic mathematical normalization; it carefully handles the physical realities of deep neural networks:
     * **Layer-wise Scaling**: Formulated based on the physical intuition that early layers represent general abstractions and deep layers represent specialized tasks, preventing early-stream dominance.
     * **Composite Input Grouping (Group 0)**: Prevents numerical instability and early spatial distortions by jointly scaling positional embeddings, class embeddings, and early convolutions with the first block.
     * **Noise-Damping Stabilizer ($\beta$)**: Elegant soft-thresholding to prevent noise amplification in layers with near-zero expert updates.
     * **Scale Compensation ($\gamma^l$)**: A mathematically sound, closed-form factor that analytically counteracts directional norm contraction without requiring hyperparameter sweeps.

3. **High Conceptual Value (The Overfitting-Optimizer Paradox)**:
   * The paper exposes a highly valuable, practical critique of unsupervised test-time adaptation (TTA) weight merging. 
   * Demonstrating that minimizing joint prediction entropy over unlabeled data overfits to easy, low-entropy tasks and actively suppresses harder, high-entropy tasks is a major warning for practitioners who assume TTA is a "free lunch."

## Areas for Improvement

1. **Severely Limited Evaluation Scope**:
   * Evaluating only on four small, low-resolution toy datasets (MNIST, FashionMNIST, CIFAR-10, SVHN) on a single small backbone model (CLIP ViT-B/32) is a major weakness.
   * To establish NETA's practical generalizability, the authors should evaluate it on standard, high-resolution 8-dataset CLIP benchmarks (including ImageNet, Stanford Cars, SUN397, etc.) and scale it to larger ViT backbones or autoregressive Large Language Models (LLMs).

2. **Underwhelming Average Performance on Zero-Shot Settings**:
   * NETA ($\alpha=1.0$) drops the average multi-task accuracy from $87.76\%$ in standard Task Arithmetic to $87.17\%$ due to performance drops on SVHN.
   * Even with continuous $\alpha$-relaxation ($\alpha=0.5$, yielding $87.51\%$) or scale-compensation ($\gamma^l$, yielding $87.28\%$), NETA still fails to outperform standard Task Arithmetic in terms of average multi-task utility.
   * To make NETA practically compelling, the authors need to show scenarios where NETA's isotropic regularization delivers clear, average utility gains, rather than just acting as a "fairness regularizer" that compromises peak capabilities.

3. **Incomplete Demonstration of TTA Overfitting**:
   * The authors critique Layer-Wise AdaMerging as being "prone to local calibration-set overfitting" due to its 52 parameters. However, in Table 1, Layer-Wise AdaMerging achieves the highest test accuracies across the board (including $84.04\%$ on FashionMNIST), with an average of $90.89\%$.
   * To prove that Layer-Wise AdaMerging suffers from overfitting that degrades generalizability, the authors should design an experiment testing generalization on out-of-distribution test sets or test sets with different class proportions, comparing its performance decay against NETA's robust stability. Without this, a practitioner is highly likely to prefer the $+3.7\%$ performance boost of Layer-Wise AdaMerging despite its 52 parameters.

## Overall Presentation Quality
The presentation quality is **excellent**. 
* The narrative is extremely cohesive, logical, and easy to follow.
* The transition from standard Task Arithmetic's limitations, to the formalization of NETA, and finally to exposing the Overfitting-Optimizer Paradox is beautifully orchestrated.
* The mathematical rigor is high, the proofs are sound, and the physical/geometric motivations are meticulously explained.
* The authors are commendably honest and transparent about their experimental limitations (including CPU/Slurm queue limits and the sub-sampling of 1024 images) and theoretical trade-offs (such as directional norm contraction).

## Potential Impact and Significance
* **High Conceptual Significance**: The paper deepens the community's understanding of weight-space geometry in model merging, highlighting the physical and spatial role that layers play in downstream expert representations. Exposing the fragility of unsupervised entropy minimization will likely steer future research toward more robust test-time objectives.
* **Moderate Practical Significance**: For production pipelines with strict compute constraints (where GPUs are unavailable or test-time latency must be under a millisecond) or where calibration data is completely unavailable, NETA provides an incredibly simple, robust, and elegant zero-shot baseline. However, for server-side deployments where a small calibration set and minor test-time compute can be allocated, Layer-Wise AdaMerging's massive $+3.7\%$ performance advantage will limit NETA's direct adoption unless further improved to recover its performance gap on dominant tasks.
