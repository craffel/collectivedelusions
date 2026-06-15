# Peer Review

## Paper Summary
The paper investigates weight-space post-hoc model merging, specifically addressing the downstream post-training quantization (PTQ) robustness of test-time adaptive merging (TTA) frameworks. The authors identify a vulnerability termed **Quantization-Operator Overfitting**: unconstrained test-time coefficient optimization (e.g., AdaMerging) converges to extremely sharp local minima that yield high performance in FP32 but collapse under subsequent post-training quantization (PTQ).

To resolve this, the authors propose **CR-PolySACM (Clipping-Regularized Sharpness-Aware Subspace Model Merging)**, a unified framework that combines:
1. **Global Structural Subspace Constraints (PolyMerge):** Restricting layer-wise blending coefficients to a low-degree polynomial of network depth, compressing the optimization search space from 56 parameter variables to only 12.
2. **Local Landscape Flatness Optimization (CR-SACM):** Explicitly minimizing local loss sharpness in the blending coefficient space using a first-order minimax approximation.
3. **Clipping-Regularized Scale Balancing:** Identifying a fundamental **task-vector norm scale pathology** where unnormalized sharpness optimization is blind to highly sensitive, low-norm layers (such as final layer normalization). CR-SACM resolves this by clipping task-vector norms to a robust minimum floor ($\beta=0.10$), balancing scale sensitivity across layers without triggering gradient explosion.

The authors evaluate their method on a Vision Transformer (ViT-Tiny) backbone across four vision classification tasks (MNIST, FashionMNIST, CIFAR-10, SVHN) under six post-training quantization schemas (FP32 down to INT4).

---

## Strengths and Weaknesses

### Strengths
1. **Pioneering Theoretical Analysis:** The connection between local landscape flatness and post-training quantization (PTQ) robustness during the test-time adaptation (TTA) of model merging is highly insightful. The second-order Taylor expansion (Equation 11) decomposing quantization noise into in-subspace projected perturbation and out-of-subspace noise is elegant and theoretically satisfying.
2. **Discovery of the Task-Vector Scale Pathology:** The empirical discovery and analysis of the 50-fold scale discrepancy in task-vector norms (e.g., intermediate blocks vs. final layer normalization) is a major contribution. It provides a clear, high-signal explanation for why unconstrained sharpness-aware optimization has historically failed or collapsed in model merging.
3. **High-Efficiency Design:** The proposed Sharpness-Aware Coefficient Minimization (SACM) is highly practical. By using a scale-balanced, first-order minimax perturbation approximation, it requires only two forward-backward passes and completes in $1.56$ seconds ($52.8\times$ faster than exact Hessian trace calculation), making it viable for resource-constrained edge devices.
4. **Outstanding Transparency:** The authors are highly honest and transparent about their paper's limitations. They explicitly highlight the "expert-to-merge drop" (the massive accuracy gap between experts and merged models due to domain disconnect) and openly admit that the absolute 4-bit (INT4) accuracy of 19.07% is practically unusable for production systems. This scientific integrity is exemplary.
5. **Thorough Ablation Studies:** The paper includes excellent, rigorous ablation studies on the regularization strength $\gamma$, the clipping threshold $\beta$, the calibration stream size $N$ (in the Appendix), and robustness to class imbalance on the calibration stream (in the Appendix).

### Weaknesses
1. **Regularization Bias and Performance Degradation in Standard Precisions:** Under standard deployment regimes (FP32 and all four INT8 schemas), the proposed CR-PolySACM consistently degrades performance compared to the standard PolyMerge baseline:
   - **FP32:** $57.00\%$ (CR-PolySACM) vs. **$57.40\%$** (PolyMerge)
   - **INT8 Sym Tensor:** $56.62\%$ vs. **$57.62\%$**
   - **INT8 Sym Channel:** $57.23\%$ vs. **$58.15\%$**
   - **INT8 Asym Tensor:** $56.48\%$ vs. **$56.57\%$**
   - **INT8 Asym Channel:** $56.93\%$ vs. **$57.43\%$**
   
   The proposed method only outperforms PolyMerge under the extreme and noisy **INT4 Symmetric Channel** format ($19.07\%$ vs. $18.10\%$). This means that for standard precisions where models are actually deployed, the proposed flatness regularization acts as a detrimental bias rather than a benefit.
2. **Practical Non-Viability of the INT4 Results:** While the relative $+0.97\%$ improvement over PolyMerge in the INT4 regime is statistically significant, the absolute accuracy of **19.07%** is extremely low. On these 10-class datasets (MNIST, FashionMNIST, CIFAR-10, SVHN), random guessing on each yields $10.0\%$ accuracy. An accuracy of $19.07\%$ is barely functioning. Thus, the only regime where CR-PolySACM outperforms the baseline is a regime where the model is unusable anyway.
3. **Backbone and Dataset Scale Limitations:** 
   - **Backbone:** The framework is evaluated solely on a toy-sized **Vision Transformer (\texttt{vit\_tiny\_patch16\_224})** with only **5.7M parameters**. Post-training quantization and post-hoc model merging are most critical for large-scale models (e.g., ViT-Base with 86M parameters, ViT-Large, or 1B+ LLMs) where full-precision deployment is prohibitive. It is empirically unverified if the scale pathology behaves similarly in larger models, or if the clipping threshold $\beta=0.10$ generalizes.
   - **Datasets:** The evaluation relies on simple, low-resolution toy datasets (MNIST, FashionMNIST, CIFAR-10, SVHN) that are resized to 224x224. This represents a highly artificial multi-task setup.
4. **Missing Standard Baselines and Weak Static Comparison:**
   - **Missing Baselines:** The authors discuss **TIES-Merging** and **DARE** in the related work, but do not include them in the experimental comparison in Table 1. TIES-Merging is a highly standard static baseline that resolves parameter conflicts and should be included.
   - **"Strawman" Task Arithmetic:** The "Uniform Task Arithmetic" baseline is evaluated with a fixed coefficient of $\lambda_k^l = 0.25$ without any tuning. Standard Task Arithmetic baseline in the literature involves a grid-search for a single global scaling factor $\lambda \in [0, 1]$ that maximizes performance on a validation set. By not tuning this scale, the authors compare against a weak static baseline.
5. **Oracle Task-ID Assumption:** The multi-task evaluation protocol dynamically swaps task-specific classification heads. This assumes that a "task ID oracle" is available at test-time, which is a major assumption that is often not met in practical edge deployment.

---

## Detailed Ratings

### Soundness: Good
The theoretical formulation, the quadratic noise decomposition, and the clipping-regularized SACM derivation are mathematically sound, rigorous, and highly elegant. However, the soundness is limited by the small scale of the backbone model (ViT-Tiny), the simple toy datasets, and the fact that the proposed method degrades performance under standard, non-aggressive formats (FP32/INT8).

### Presentation: Excellent
The paper is exceptionally well-written, articulate, and highly structured. The figures are clean and do an excellent job of illustrating the concepts. The authors' transparency in highlighting the expert-to-merge drop and the absolute unviability of the INT4 results is exemplary and demonstrates great scientific integrity.

### Significance: Fair
The scientific and theoretical significance is high; the discovery and resolution of the task-vector norm scale pathology represents a highly valuable contribution to weight-space composition literature. However, the practical significance is currently fair because the method degrades performance in standard formats (FP32 and INT8) and only provides a relative benefit in an ultra-low precision format (INT4) where the absolute accuracy is too poor to be usable.

### Originality: Good
The paper offers a novel integration of test-time flatness optimization with subspace-constrained model merging. The mathematical analysis of the scale pathology and the subsequent clipping-regularized scale balancing (CR-SACM) are highly original, creative, and well-justified.

---

## Overall Recommendation

**Rating: 3: Weak reject**

**Justification:** 
The paper is exceptionally well-written, mathematically elegant, and introduces a highly original and valuable theoretical contribution (the task-vector norm scale pathology and its resolution via CR-SACM). However, from an empirical perspective, the evaluation is currently restricted to toy-scale models (ViT-Tiny with 5.7M parameters) and simple image datasets (MNIST, FashionMNIST, CIFAR-10, SVHN) in a highly artificial multi-task setup. Crucially, the proposed method (CR-PolySACM) degrades performance under standard deployment formats (FP32 and INT8) compared to the standard PolyMerge baseline, and only outperforms it under aggressive 4-bit quantization where the absolute performance (19.07%) is unusable in practice. Furthermore, standard baselines (like TIES-Merging, DARE, and a tuned Task Arithmetic) are missing. 

While the scientific value of the paper is clear, the empirical weaknesses—specifically the lack of scaling results, the degradation in standard precisions, and the missing baselines—must be addressed before the paper is ready for acceptance.

---

## Questions and Constructive Feedback for the Authors

1. **Performance Degradation under FP32/INT8:** Why does CR-PolySACM consistently degrade performance under standard formats (FP32 and INT8) compared to standard PolyMerge? Is there a way to dynamically adjust the regularization strength or scale of the perturbation ($\rho$) based on the target quantization schema so that the flatness regularization does not introduce a detrimental bias under high precisions?
2. **Backbone and Dataset Scaling:** Can you provide results on a larger, more realistic backbone (e.g., ViT-Base with 86M parameters) or on a standard multi-task/domain-shift benchmark (e.g., DomainNet or VTAB)? Showing that the task-vector norm scale pathology and the optimal clipping threshold $\beta$ generalize to larger models and more realistic transfer learning scenarios would significantly strengthen the paper's empirical foundation.
3. **Baselines Comparison:** Please include standard static model merging methods, such as **TIES-Merging** and **DARE**, in Table 1. Furthermore, please compare against a tuned Task Arithmetic baseline (where a single global scaling factor $\lambda \in [0, 1]$ is searched over a validation set) rather than a fixed uniform scale of $0.25$.
4. **Percentile-based Automated Blueprint:** In the Appendix, you propose an automated percentile-based blueprint to dynamically set $\beta$. Can you evaluate this automated blueprint across different backbones to verify if the 10th percentile choice is robust and automatically scales to deeper networks without manual tuning?
5. **Confidence Interval Reporting:** To ensure statistical significance, please include the standard deviations or confidence intervals directly in Table 1.
