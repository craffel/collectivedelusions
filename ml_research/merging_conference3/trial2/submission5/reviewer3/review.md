# Peer Review of Conference Submission

## Strengths and Weaknesses

### Strengths
1. **Practical Simplicity and Zero-Overhead Deployment**:
   The proposed method, Norm-Equalized Task Arithmetic (NETA), is completely training-free, parameter-free, and data-free. It can be executed instantaneously (in less than a second) directly on model weights. In production environments where collecting calibration data is challenging and running backpropagation at test-time is computationally prohibited or slow, a closed-form weight-space transformation represents a massive practical advantage.

2. **Rigorous Engineering Design and Physical Intuition**:
   The paper meticulously accounts for the physical and spatial realities of deep neural network architectures:
   * **Layer-wise Normalization**: Formulated based on the spatial intuition that early layers represent general abstractions while deep layers represent task-specific features, preventing high-norm updates from dominating early visual processing.
   * **Composite Input Grouping (Group 0)**: Jointly scales early positional embeddings, class embeddings, and early convolutions with the first block, preventing numerical instabilities and early spatial distortions.
   * **Noise-Damping Stabilizer ($\beta$)**: Uses soft-thresholding to prevent noise amplification in intermediate layers that undergo near-zero expert updates.
   * **Scale Compensation ($\gamma^l$)**: Proposes a mathematically sound, closed-form factor that analytically counteracts directional norm contraction without requiring grid searches over the global scaling factor.

3. **Valuable Conceptual Critique of Test-Time Adaptation**:
   The identification and analysis of the **Overfitting-Optimizer Paradox** is a significant strength. Exposing how unsupervised joint prediction entropy minimization on unlabeled data overfits to simple, low-entropy tasks and actively suppresses harder, high-entropy tasks provides a vital warning for practitioners who assume test-time weight optimization is a "free lunch."

### Weaknesses
1. **Limited Evaluation Scope (Toy Datasets & Small Model)**:
   The empirical evaluation is restricted to four standard, low-resolution toy datasets: **MNIST, FashionMNIST, CIFAR-10, and SVHN** on a single, small backbone model (**CLIP ViT-B/32**). For a powerful vision-language foundation model pre-trained on 400M image-text pairs, these datasets do not represent realistic, high-resolution downstream applications. Standard CLIP model merging benchmarks typically evaluate across an 8-dataset suite (including ImageNet, Stanford Cars, SUN397, etc.). The lack of evaluation on larger ViT backbones (ViT-L/14, ViT-H) or text-domain Large Language Models (LLMs) severely limits our confidence in NETA's scalability and generalizability to real-world industrial settings.

2. **Underwhelming Average Performance on Zero-Shot Settings**:
   Under the default zero-shot settings, NETA ($\alpha=1.0$) drops overall average multi-task accuracy from $87.76\%$ in standard Task Arithmetic to $87.17\%$ due to performance drops on SVHN. Even with continuous $\alpha$-relaxation ($\alpha=0.5$, yielding $87.51\%$) or scale-compensation ($\gamma^l$, yielding $87.28\%$), NETA still fails to outperform standard Task Arithmetic in terms of average multi-task utility. If a practitioner's primary goal is to maximize average system capability across all tasks, standard Task Arithmetic remains the superior choice.

3. **Performance Gap and Incomplete Critique of High-Dimensional Optimization**:
   The authors critique Layer-Wise AdaMerging as being "prone to local calibration-set overfitting." However, in Table 1, Layer-Wise AdaMerging achieves the highest test accuracies across the board (including $84.04\%$ on FashionMNIST), with a substantial average multi-task accuracy of $90.89\%$. This represents a massive **$+3.72\%$** absolute performance advantage over NETA. The paper lacks an experiment demonstrating that Layer-Wise AdaMerging's "overfitting" actually degrades generalization on out-of-distribution test sets or test sets with different class proportions. Without this empirical proof, a practitioner with access to 256 unlabeled calibration images and a GPU would easily favor the massive performance gains of Layer-Wise AdaMerging over NETA, despite the additional 52 optimization parameters.

---

## Soundness
**Rating**: Excellent

**Justification**:
The mathematical formulation of NETA is exceptionally clear, rigorous, and technically sound. The proofs verifying perfect magnitude isotropy and the preservation of cumulative individual norms are standard and mathematically correct. Furthermore, the authors display commendable scientific honesty by explicitly addressing *directional norm contraction* (Equation 10) and proposing a closed-form scale-compensation factor ($\gamma^l$) to analytically mitigate it. All engineering and physical heuristics (such as composite input grouping, layer-wise scaling, and noise-damping) are logical, well-reasoned, and appropriate.

---

## Presentation
**Rating**: Excellent

**Justification**:
The submission is beautifully written, extremely structured, and cohesive. The narrative is engaging and easy to follow, transitioning smoothly from task arithmetic's limitations to the formulation of NETA and the exposure of the Overfitting-Optimizer Paradox. The figures and tables are clear, well-labeled, and highly informative, with comprehensive details provided about the active visual parameters, Hugging Face hub checkpoints, and baseline training hyperparameters, which ensures outstanding reproducibility.

---

## Significance
**Rating**: Fair

**Justification**:
While NETA is conceptually valuable for understanding weight-space geometry and exposing TTA vulnerabilities, its practical significance is limited:
1. The evaluation is restricted to low-resolution academic toy datasets on a small, early-generation CLIP backbone. There is no proof of generalizability to realistic, high-resolution visual tasks or modern, large-scale architectures.
2. In zero-shot settings, NETA does not outperform standard Task Arithmetic or DARE in terms of overall multi-task average utility, acting primarily as a regularizer that sacrifices peak performance on dominant tasks for representation equity.
3. In settings where test-time optimization is feasible (with a small calibration set and minor GPU compute), Layer-Wise AdaMerging outperforms NETA by a massive $+3.72\%$ absolute margin. The lack of an experiment demonstrating actual generalization failures in Layer-Wise AdaMerging due to overfitting makes NETA a less compelling option for standard deployments.

---

## Originality
**Rating**: Good

**Justification**:
Applying layer-wise Frobenius normalization to model merging is simple and represents a moderately incremental extension of established weight and layer normalization concepts. However, the paper's main original contribution is the formal identification and naming of the **Overfitting-Optimizer Paradox**. Exposing why unsupervised joint prediction entropy minimization on unlabeled data overfits to easy, low-entropy tasks and suppresses complex, high-entropy tasks in low-dimensional spaces is a highly original and valuable conceptual insight that deepens our understanding of test-time weight adaptation.

---

## Overall Recommendation
**Rating**: 4: Weak Accept

**Justification**:
This is a technically solid, mathematically rigorous, and exceptionally well-written paper that introduces an elegant, training-free, and data-free model merging baseline. The proposed NETA framework successfully addresses task dominance analytically in the parameter space and introduces several clever, physically-motivated heuristics (such as composite input grouping, noise-damping, and scale compensation) that demonstrate a deep understanding of neural architectures. Additionally, the critique of unsupervised test-time adaptation (the Overfitting-Optimizer Paradox) is highly insightful.

However, the submission exhibits notable weaknesses that limit its immediate practical impact: the evaluation is narrow (limited to MNIST, FashionMNIST, CIFAR-10, SVHN on CLIP ViT-B/32), and NETA's overall average multi-task accuracy is lower than standard Task Arithmetic and significantly lower than Layer-Wise AdaMerging (which holds a $+3.72\%$ absolute advantage). The lack of empirical evidence demonstrating that Layer-Wise AdaMerging degrades generalization on out-of-distribution test sets further weakens the practical argument for NETA in settings where minor calibration is possible. 

Despite these limitations, the paper is technically flawless, highly reproducible, and provides valuable conceptual insights that the model merging community is likely to build upon. Therefore, a Weak Accept is recommended.
