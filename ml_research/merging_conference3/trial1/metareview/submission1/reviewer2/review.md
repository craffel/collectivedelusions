# Peer Review for "QP-Merge: Quantization-Preserving Task Vector Merging"

## Summary of the Paper
The paper addresses the post-training quantization (PTQ) of merged task-specific neural networks for low-bit edge deployments. Model merging combines several task-specific models (task vectors) without joint retraining but suffers from severe accuracy loss under standard low-bit quantization (such as INT4). This degradation is attributed to heavy-tailed weight outliers stretching the quantization scales, alongside activation range mismatches across different tasks. 

To resolve this, the authors propose **QP-Merge**, which integrates two techniques:
1. **Outlier-Residual Decoupling (ORD):** The top $\le 1\%$ extreme weight updates of task vectors are isolated into a sparse, high-precision (FP16) tensor, leaving a range-bounded dense base weight to be quantized homogeneously to INT4/INT8.
2. **Quantization-Error Aware Scale Calibration (QE-Calib):** A zero-labeled-data calibration step that jointly optimizes layer-wise diagonal weight scaling parameters ($D_l$) and task-blending coefficients ($\lambda$) on a tiny set of $M=128$ unlabeled samples by minimizing the mean-squared error (MSE) of the final output embeddings against the unquantized FP32 merged model.

The framework is evaluated using a pre-trained `ViT-B-32` on a dual-task digit classification benchmark (MNIST and SVHN) in INT4 and INT8 modes, showing strong accuracy recovery, robustness to out-of-distribution shifts, and robust calibration under data imbalance.

---

## Overall Assessment
The paper is exceptionally well-written, clear, and provides a highly practical exploration of a crucial deployment bottleneck (the merging-quantization gap). The authors' inclusion of a thorough, honest physical hardware and memory profile on modern GPU nodes is highly commendable and shows strong engineering integrity. 

However, from a rigorous empirical and methodological perspective, the paper exhibits critical weaknesses. The evaluation is restricted to low-complexity toy digit tasks (MNIST/SVHN) using a massive, overpowered model (`ViT-B-32`), which trivializes the merging and compression challenges. Furthermore, the paper's own ablation study reveals that the primary technical mechanism proposed—the hybrid dense-quantized + sparse-unquantized ORD execution path—is practically redundant, yielding a negligible $0.03\%$ average accuracy improvement over a standard, homogeneous INT4 model with simple scale calibration. Finally, the "calibration" methodology permanently alters the unquantized model's function mapping without mathematical equivalence, making the "scale calibration" framing technically misleading.

Therefore, while the paper represents a promising direction, it requires significant revisions, realistic multi-task evaluations, and a reconsideration of its architectural complexity before it is ready for publication.

---

## Strengths and Weaknesses

### Strengths
1. **High Relevance and Practical Value:** Co-designing post-training quantization and model merging is an important, high-impact research direction for making multi-task models serveable under strict real-world edge VRAM and latency constraints.
2. **Outstanding Presentation Quality:** The paper is exceptionally clear, logically organized, and beautifully structured. The mathematical formulations are complete, exact, and easy to follow.
3. **Professional Hardware and Memory Profiling:** The authors do not merely rely on theoretical FLOPS or parameters; they conduct a thorough physical VRAM footprint and wall-clock latency analysis on an NVIDIA Hopper GPU. They are highly transparent about PyTorch's high-level API launch overhead ($50\ \mu$s), displaying strong engineering integrity.
4. **Comprehensive Generalization and Robustness Evaluations:** The evaluations include extensive sweeps over calibration size $M$, out-of-distribution (OOD) corruptions (Gaussian noise and contrast shifts), and highly biased/imbalanced calibration data distributions. These validate that the scale optimization generalizes well.

### Weaknesses
1. **Weak and Non-Representative Empirical Benchmark:** 
   Evaluating exclusively on dual-task digit classification (**MNIST and SVHN**) using an **86M parameter ViT-B-32** is a major empirical limitation. MNIST and SVHN are solved, low-complexity toy benchmarks. Running an 86M parameter Transformer to classify digits is extreme overkill. Because the model is vastly overpowered, the fine-tuned task vectors are highly localized, making model merging and quantization artificially easy. Real-world model merging is evaluated on diverse 8-task image classification suites (including ImageNet, CIFAR-100, EuroSAT, RESISC45, etc.) or multi-task NLP benchmarks (GLUE, MMLU). The paper fails to demonstrate that QP-Merge generalizes to realistic multi-task challenges.
2. **Empirical Redundancy of the Proposed ORD Path:**
   The paper's core architectural contribution is Outlier-Residual Decoupling (ORD), which splits weights into a quantized dense path and a sparse FP16 path. However, Table 3 shows that removing ORD ("No ORD" ablation) only drops INT4 average accuracy by a minuscule **$0.03\%$** (from $94.52\%$ to $94.49\%$), while on MNIST it actually *improves* accuracy from $98.96\%$ to $99.08\%$. In INT8, the difference is only $0.01\%$. 
   Introducing a complex hybrid sparse matrix-multiply (SpMM) runtime, coordinate-list (COO/CSR) memory layouts, and separate kernel launch overheads for a $0.03\%$ accuracy gain—which is well within the $\pm 0.13\%$ margin of random noise—is structurally and practically unjustified. A standard homogeneous INT4 model with scale calibration is far simpler and just as effective.
3. **Methodological Misdirection in "Scale Calibration":**
   The QE-Calib step is framed as "activation scale calibration" (similar to SmoothQuant). However, because $D_l$ is applied permanently to the weight updates and is **not** accompanied by an inverse activation scaling ($D_l^{-1}$) during inference, it permanently distorts the unquantized model's mapping. This is not a mathematically equivalent transformation; it is literally gradient-based, unsupervised parameter fine-tuning on the calibration set (optimizing $d_{\text{in}}$ parameters per layer, plus merging weights $\lambda$). Calling it "scale calibration" is mathematically misleading.
4. **Asymmetrical Scaling without Justification:**
   In Eq. 15, the diagonal scaling $D_l$ is applied on the column side of the *task-vector updates* but **not** to the pre-trained base weights $W_{l, \text{base}}$. Since the base weights constitute the vast majority of the weight matrix, scaling only the updates modifies the relative magnitude and importance of the task-specific features relative to the base features, which lacks mathematical or conceptual justification.
5. **Lack of Comparative Merging Baselines:**
   The paper only quantizes simple Task Arithmetic. It fails to compare against standard, more advanced model merging methods (e.g., Ties-Merging or DARE) followed by standard quantization. It is possible that advanced merging heuristics naturally mitigate weight-range stretching, rendering hybrid dense-sparse decoupling entirely unnecessary.

---

## Detailed Evaluation Ratings

### Soundness: Fair
The mathematical description of the framework is precise, and the code hyperparameters are thoroughly documented. However, the soundness is rated as *fair* due to:
- Applying permanent diagonal scaling without inverse activation scaling, which breaks mathematical equivalence and functions as unsupervised parameter fine-tuning.
- Asymmetrical scaling of task updates without scaling base weights.
- The extreme empirical redundancy of the ORD path, where the core dense-sparse split provides only a statistically insignificant $0.03\%$ average accuracy benefit in INT4 while adding substantial serving complexity.

### Presentation: Excellent
The paper's writing style is exceptional. The narrative is engaging, the motivation is clear, and the engineering details in the hardware profiling are presented with outstanding clarity and professional transparency.

### Significance: Fair
While the target problem (deploying merged models under low-bit quantization) is of high significance, the paper's actualized significance is *fair* because:
- The evaluation is restricted to toy digit classification benchmarks (MNIST/SVHN), making it unclear if the method scales to real multi-task computer vision or NLP systems.
- The proposed hybrid ORD framework is practically redundant, as a standard homogeneous INT4 model with scale calibration achieves virtually identical accuracy without any sparse execution overhead.

### Originality: Good
Co-designing model merging and quantization is a novel and important perspective. The adaptation of dense-sparse weight partitioning to task-vector updates and the optimization of scale parameters using an end-to-end embedding loss are creative combinations of existing PTQ ideas.

---

## Questions and Constructive Feedback for the Authors

1. **Realistic Multi-Task Benchmarking:** To establish the practical significance of this work, the framework must be evaluated on more realistic, high-dimensional multi-task benchmarks. 
   - For computer vision, please evaluate on a standard 8-task image classification suite (e.g., including CIFAR-100, EuroSAT, RESISC45, and STL-10).
   - Alternatively, evaluate on a multi-task NLP model (e.g., merging and quantizing instruction-tuned models or LLaMA-scale adapters on standard benchmarks). Does QP-Merge remain lossless under these more complex, high-entropy representation spaces?
2. **Justification of the ORD Path:** In Table 3, the "No ORD" ablation (which runs QE-Calib but does not decouple outliers, using a standard homogeneous INT4 format) performs virtually identically to the Full QP-Merge model (average accuracy of $94.49\%$ vs. $94.52\%$). Given that ORD introduces CSR layouts, SpMM operations, and high-level API launch overheads, why should a machine learning practitioner implement ORD rather than simply running the QE-Calib scale optimization on a standard INT4 model? Please provide a compelling scenario or dataset where ORD provides a substantial, statistically significant improvement over "No ORD".
3. **Statistical Significance of Sweeps:** Please report the mean and standard deviation across multiple random seeds for the sensitivity sweeps in Table 4 ($\gamma$) and Table 5 ($M$). Given the standard deviation in Table 1 is $\pm 0.13\%$, are the tiny differences in these sweeps (such as $94.74\%$ vs. $94.71\%$) statistically significant, or are they within the noise level?
4. **Mathematical Equivalence and Terminology:** Since the scaling matrix $D_l$ is permanently applied to the weight updates without applying $D_l^{-1}$ to the activations at inference, the unquantized model's mapping is modified. 
   - Why is this technique framed as "activation scale calibration" rather than "unsupervised scale and merging weight fine-tuning"?
   - Why is $D_l$ applied only to the task updates and not to the base weights $W_{\text{base}}$? Please provide a theoretical or empirical justification for this asymmetry.
5. **Advanced Merging Baselines:** How does direct quantization of advanced merging methods (such as Ties-Merging or DARE, which inherently prune low-magnitude updates and resolve sign conflicts) compare to QP-Merge? It is important to demonstrate that QP-Merge outperforms standard quantization of these more advanced merging techniques.

---

## Overall Recommendation
**3: Weak Reject**

The paper has clear merits: it is exceptionally well-written, addresses a highly relevant deployment problem, and provides an outstanding physical hardware and latency profiling analysis. However, the critical weaknesses—specifically the extremely weak empirical evaluation on toy digit classification tasks (MNIST and SVHN) and the fact that the proposed core architectural technique (ORD) is shown to be practically redundant (adding massive execution complexity for a $0.03\%$ accuracy gain)—outweigh the paper's current strengths. 

To become a strong candidate for publication, the authors must significantly expand their empirical evaluation to realistic multi-task settings (such as 8-task classification or LLM-scale merging) and either demonstrate a scenario where the dense-sparse hybrid ORD path provides a substantial, statistically significant benefit or simplify their framework to a homogeneous quantized format utilizing their effective scale optimization technique.
