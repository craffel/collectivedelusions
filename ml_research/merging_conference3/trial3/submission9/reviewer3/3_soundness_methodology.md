# 3. Soundness and Methodology Evaluation

## Clarity of the Description
The methodology of the paper is exceptionally well-written, clear, and mathematically rigorous. Each component of the pipeline—sharpness-aware fine-tuning, layer-wise dynamic blending, straight-through post-training quantization, and joint entropy minimization—is formalized with clear equations and detailed conceptual explanations.
*   The notation is consistent and standard.
*   The structural details of the backbone (Vision Transformer with $L=14$ blocks, $K=4$ tasks) are clearly defined.
*   The execution flow is structured logically, moving from weight-space pre-training to dynamic blending, discrete quantization, and test-time optimization.

## Appropriateness of Methods
The methods employed are highly appropriate and standard in their respective domains, combined in a highly thoughtful and theoretically motivated way:
1.  **SAM for Flatness Control:** Since the paper's goal is to study the relationship between expert flatness and downstream merging under quantization, utilizing SAM (Sharpness-Aware Minimization) to sweep the perturbation radius $\rho$ is the gold-standard approach.
2.  **Per-Channel Symmetric PTQ:** Symmetric uniform PTQ is the industry standard for on-device deployment due to its simplicity and computational efficiency.
3.  **STE for Quantization-Aware Blending:** The Straight-Through Estimator (STE) is the most standard and empirically proven method to propagate gradients through non-differentiable rounding operators.
4.  **Joint Entropy Minimization:** Unsupervised prediction entropy minimization is highly appropriate for test-time adaptation on unlabeled calibration data, as it does not require access to ground-truth task labels and can adapt to domain shift.

## Mathematical Rigor and Correctness
The mathematical derivation bridging weight-space and coefficient-space flatness is **impeccable and highly elegant**.
*   The derivation of the coefficient-space Hessian $H^l_{\Lambda}$ as the projection of the weight-space Hessian $H^l_{\theta}$ via the task vector matrix $T^l$ ($H^l_{\Lambda} = (T^l)^T H^l_{\theta} T^l$) is mathematically correct and highly insightful.
*   The bounds on the maximum eigenvalue and trace of the coefficient-space Hessian ($\lambda_{\max}(H^l_{\Lambda}) \le \lambda_{\max}(H^l_{\theta}) \cdot \|T^l\|_2^2$) are correct and provide a formal, rigorous proof for why pre-training experts via SAM to minimize $\lambda_{\max}(H^l_{\theta})$ directly flattens the test-time adaptation landscape.

## Verification of Technical Design Choices
The authors address several critical technical design choices that are frequently overlooked in similar papers:
1.  **Independent Blending Bounds vs. Softmax Normalization:** Standard merging methods often enforce a sum-to-one constraint. The authors logically argue that a rigid convex combination restricts capacity and forces a zero-sum game, whereas independent bounds allow scaling task vectors independently. They empirically validate this choice, showing that independent clipping $[0, 1]$ dramatically outperforms Softmax combinations (by +8.20% in 8-bit and +3.03% in 4-bit).
2.  **Addressing STE Gradient Mismatch:** The authors acknowledge the gradient mismatch error of STE and explain why optimization is stable: (1) the low-dimensional parameter bottleneck (only 56 parameters to optimize) averages out individual approximation errors, and (2) a moderate Adam learning rate of $1 \times 10^{-3}$ prevents local oscillations.
3.  **Preventing Degenerate Class Collapse:** Entropy minimization is notoriously prone to class collapse. The authors hypothesize that their tight coefficient bottleneck (56 parameters) acts as an implicit structural regularizer, preventing the parameters from drifting into degenerate regimes. They empirically validate this by comparing with TENT-style high-dimensional adaptation (5.7M parameters), which completely collapses to random guessing ($\approx 13\%$), while FlatQ-Merge remains highly stable (27.64%).
4.  **Classification Heads:** They clarify that classification heads are kept in full FP32 and mapped individually, completely avoiding structural shape incongruence issues. This is a very sound engineering decision.

## Reproducibility
The paper is highly reproducible. The authors provide:
*   Exact details on datasets (MNIST, FashionMNIST, CIFAR-10, SVHN), data budgets (512 images per task), test sizes (1,000 images per task), and random seeds (42, 100, 2026).
*   Hyperparameters (Adam optimizer, 40 steps, learning rate $10^{-3}$, calibration batch size $N=64$).
*   A procedural step-by-step description in Algorithm 1.
*   Full equations for all operators (PTQ, STE, Entropy).

## Technical Flaws or Weaknesses
No major technical flaws were identified. The methodology is remarkably sound, thoroughly justified, and supported by rigorous theoretical and empirical arguments.
