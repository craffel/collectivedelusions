# Peer Review: Norm-Equalized Task Arithmetic (NETA)

## Summary of the Paper
The paper introduces **Norm-Equalized Task Arithmetic (NETA)**, a training-free, data-free, and parameter-free closed-form method for multi-task model merging. When independent task experts are fine-tuned from a shared pre-trained checkpoint, their parameter updates (task vectors) often possess highly disparate scales. Consequently, standard linear merging (Task Arithmetic) is susceptible to "task dominance," where high-magnitude tasks (such as SVHN) hijack the parameter space and degrade performance on other tasks (such as MNIST). 

NETA addresses this imbalance analytically in a single step. At each layer of the deep network, NETA computes the Frobenius norm of each task vector, calculates the average norm across all tasks, and scales each task vector so that its norm is perfectly equalized to this average. To enhance real-world deployment flexibility and stability, the authors propose several closed-form extensions:
1. **$\alpha$-Relaxed NETA**: A continuous interpolation framework allowing practitioners to smoothly balance representation fairness and peak task performance without any test-time training or backpropagation cost.
2. **Noise-Damping Stabilizer ($\beta$)**: A soft-thresholding denominator that prevents the unstable amplification of near-zero, noisy parameter updates in intermediate layers.
3. **Composite Visual Input Grouping**: A structural grouping that stabilizes norm calculation for extremely low-dimensional or frozen parameters in the early visual stream.
4. **Scale Compensation Factor ($\gamma^l$)**: A closed-form analytical factor designed to counteract the directional norm contraction of the merged update vector, restoring the cumulative update magnitude to that of standard Task Arithmetic.

Additionally, the paper uncovers a key limitation of current state-of-the-art Test-Time Adaptation (TTA) merging techniques (such as AdaMerging), which the authors term the **Overfitting-Optimizer Paradox**: when optimizing task or layer-wise coefficients using unsupervised joint prediction entropy minimization, the optimizer is naturally biased toward easy, low-entropy tasks and suppresses harder, high-entropy tasks, causing a catastrophic performance collapse on the more challenging tasks.

---

## Strengths and Weaknesses

### 1. Soundness
* **Rating**: **Good**
* **Justification**:
  The mathematical formulation of NETA is rigorous and highly reproducible. The derivation of the layer-wise Frobenius norm equalization, the soft-thresholding noise-damping stabilizer $\beta$, the continuous $\alpha$-relaxation, and the closed-form scale-compensation factor $\gamma^l$ are mathematically sound and physically grounded. The active parameter scope is clearly defined, and Algorithm 1 provides a clean, easily implemented pipeline.
  
  However, the methodology incorporates an ad-hoc, architecture-specific heuristic: the composite grouping of positional/class embeddings with the first Transformer block (`Group 0`) to prevent positional distortions. While physically justified, the paper lacks a systematic, automated method for detecting and grouping structurally distinct or low-dimensional layers, limiting its immediate generalizability to other architectures (e.g., LLMs or ConvNets) without custom manual engineering. Furthermore, the grid search shows that despite removing task-wise coefficients, the final merged weights remain highly sensitive to the global scaling factor $\lambda_0$.

### 2. Presentation
* **Rating**: **Excellent**
* **Justification**:
  The paper is exceptionally well-written, logical, and highly structured. The writing is direct and clear, and the terminology is precise. The authors deserve high praise for their scientific honesty: they do not hide NETA's slight drop in average accuracy or its performance deficit on the hardest task (SVHN), instead providing detailed geometric analyses (e.g., directional norm contraction) to explain these phenomena. The tables and figures are clean, and the "Omitted Baselines" and "Boundary Convergence" sections show commendable thoroughness.

### 3. Significance
* **Rating**: **Fair**
* **Justification**:
  From a practical deployment standpoint, the current significance is somewhat limited by the scale of the evaluation:
  * **Toy Datasets**: The empirical validation is conducted on MNIST, FashionMNIST, CIFAR-10, and SVHN. These are extremely small-scale, simplified visual classification datasets.
  * **Foundation Model Overkill**: Evaluating a CLIP ViT-B/32 backbone on 28x28 and 32x32 images does not represent real-world, high-resolution visual domain shifts, leaving a major question mark over NETA's scalability.
  * **Performance Deficit on Challenging Tasks**: On the hardest dataset (SVHN), NETA's performance drops significantly by **-3.12%** compared to standard Task Arithmetic (from 80.14% to 77.02%). In practical, high-stakes deployments, maintaining peak capability on the most difficult task is often a critical requirement that cannot be sacrificed for artificial "fairness" across simpler, already high-performing tasks. NETA's overall multi-task average accuracy (87.17%) also remains slightly below standard Task Arithmetic (87.76%) and DARE (87.78%).
  * **TTA Comparison**: While the authors' critique of Test-Time Adaptation is highly compelling, Layer-Wise AdaMerging still outperforms NETA by a massive margin (**90.89%** vs. **87.17%**). In actual industrial applications where accuracy is paramount, a $+3.72\%$ absolute gain would easily justify the complexity of optimizing 52 parameters over 20 epochs.
  
  Nonetheless, the exposure of the Overfitting-Optimizer Paradox is of high significance. It provides a crucial warning to the community regarding the instability and transductive overfitting risks of using joint prediction entropy minimization blindly under task-difficulty imbalances.

### 4. Originality
* **Rating**: **Good**
* **Justification**:
  Applying layer-by-layer Frobenius norm equalization to task vectors is a logical, elegant, and highly practical extension of weight-space model merging. While the underlying mathematical tool (vector normalization) is standard, the combination of this simple geometric normalization with practical, training-free mechanisms (like the analytical scale compensation factor $\gamma^l$) is highly creative and novel. The formalization and empirical analysis of the Overfitting-Optimizer Paradox also represent a highly valuable and original conceptual contribution.

---

## Overall Recommendation
* **Rating**: **4: Weak Accept**
* **Justification**:
  This is a technically solid, highly reproducible, and exceptionally well-written paper. Its greatest merit is its elegant simplicity: NETA provides a zero-shot, data-free, and closed-form weight-space transform that completely bypasses the computational overhead, hyperparameter tuning, and transductive overfitting risks of Test-Time Adaptation optimization loops. The identification and formalization of the Overfitting-Optimizer Paradox is an important, high-quality conceptual contribution that will benefit the community.
  
  However, the practical utility of the method is currently limited by its evaluation on toy-scale datasets, its performance degradation on the most difficult task (SVHN), and the manual, ad-hoc nature of the composite layer grouping heuristic. The paper is recommended for a Weak Accept, as its technical merits and conceptual contributions are clear, but its impact is currently constrained by these limitations.

---

## Questions and Constructive Feedback for the Authors

1. **Scalability of the Evaluation**:
   To demonstrate the real-world scalability and utility of NETA, have you considered evaluating it on standard model-merging benchmarks with larger, high-resolution datasets? Specifically, running NETA on the standard 8-dataset visual classification suite (which includes ImageNet-1K, Stanford Cars, RESISC45, etc.) or on modern Large Language Models (LLMs) would greatly strengthen your claims and make the work far more compelling to practitioners.

2. **Automating the Composite Layer Grouping**:
   The composite `Group 0` grouping (combining positional and class embeddings with the first Transformer block) is currently a manual, architecture-specific heuristic designed for CLIP's visual encoder. How do you propose automated detection and grouping of structurally distinct or low-dimensional layers for other deep architectures (e.g., LLMs or ConvNets) to make NETA truly plug-and-play?

3. **Anisotropic Scaling formulations**:
   Since early layers in deep networks capture general representations and deeper layers specialize, have you explored anisotropic scaling formulations? For example, enforcing strict isotropic norm equalization in shallow layers to preserve the visual stream's consistency, while allowing selective task dominance in deeper layers, could potentially mitigate the performance deficit on the hardest tasks (such as SVHN) while preserving multi-task fairness.

4. **Analysis of Scale Compensation ($\gamma^l$) on Larger Sets**:
   The closed-form scale-compensation factor $\gamma^l$ (Equation 13) is a highly elegant and training-free addition. However, in Table 2, NETA + $\gamma^l$ is evaluated across the 3 seeds, but it is not included in the main Table 1 comparison, nor is its scale sensitivity fully explored in the $\lambda_0$ grid search. Could you provide a comprehensive comparison showing how NETA + $\gamma^l$ scales across a wider range of global scaling factors $\lambda_0$ compared to standard Task Arithmetic?
