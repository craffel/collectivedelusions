# Conference Submission Peer Review

## Summary of the Paper
The paper addresses the challenge of **parameter interference** and subsequent **representation collapse** when merging multiple specialized task experts into a single multi-task model without joint retraining. To resolve this, the authors propose **Grassmannian Subspace Consensus Merging (GSC-Merge)**, a partial weight-space model merging framework:
1. **Targeted Merging**: It targets only the major linear projection layers inside the Transformer blocks (comprising >95% of block parameters) for weight-space merging. Non-target parameters (such as layer normalization, biases, and patch/positional embeddings) are kept task-specific and swapped task-conditionally at test-time to prevent statistic mismatch across disparate visual domains.
2. **Joint Multi-Task Update Matrix**: For each target linear layer, task-specific task vectors ($V_k = W_k - W_{base}$) are horizontally concatenated across all $K$ experts to construct a joint update matrix $\mathbf{M}^{(l)} \in \mathbb{R}^{d_{out} \times K \cdot d_{in}}$.
3. **SVD Projection onto Grassmannian**: Singular Value Decomposition (SVD) is performed on $\mathbf{M}^{(l)}$. The top $r = \lfloor \gamma \cdot d_{out} \rfloor$ left-singular vectors are used to construct an orthogonal projection operator $P^{(l)} = U_r^{(l)} (U_r^{(l)})^T$, representing an $r$-dimensional subspace on the Grassmannian manifold $\mathbf{Gr}(r, d_{out})$.
4. **Spectral Consensus Denoising**: The task vectors are projected onto this low-rank subspace, filtering out high-frequency task-specific noise (associated with tail singular values) while retaining the coherent shared consensus directions.
5. **Offline Few-Shot Validation Tuning (OFS-Tune)**: The final merged weight is a linear combination of projected task vectors, where the layer-wise blending coefficients $\alpha_k^{(l)}$ are optimized via backpropagation on a tiny validation calibration set (e.g., 16 samples per task) using the Adam optimizer.

The authors evaluate GSC-Merge on a ViT-Tiny backbone across four classification datasets (MNIST, FashionMNIST, CIFAR-10, SVHN). The empirical results show that GSC-Merge outperforms coordinate-wise heuristic baselines (TIES-Merging, Sparse Task Arithmetic) and acts as a robust spectral regularizer, dampening the split-sensitivity variance of few-shot tuning. Under truly task-agnostic settings, performance drops significantly for all models.

---

## Strengths and Weaknesses

### Strengths
1. **Mathematical Rigor**: The paper is exceptionally well-grounded in spectral theory and manifold geometry. The application of SVD to horizontally concatenated task updates and the use of the Eckart-Young-Mirsky Theorem to prove optimal low-rank reconstruction are highly elegant and mathematically sound.
2. **Exhaustive Baselines**: The authors compare their method against a wide range of standard model merging baselines (Uniform, Task Arithmetic, STA, TIES-Merging, and Unconstrained OFS-Tune) and perform grid sweeps on baseline thresholds to avoid under-tuning bias.
3. **Statistical Soundness**: The use of a 5-seed statistical evaluation over independent random validation splits ensures that the reported means and standard deviations are robust and scientifically reliable.
4. **Comprehensive Appendices**: The appendices provide excellent analytical depth, including CPU benchmarks for Randomized SVD, singular value decay/cumulative energy curves across transformer layers, and a study comparing left-singular (output-space) vs. right-singular (input-space) projection directions.
5. **Presentation Quality**: The overall presentation, structure, writing style, and formatting are outstanding. The mathematical notations are precise, the figures are professional, and the narrative flow is very easy to follow.

### Weaknesses
1. **The Task-Conditional Swapping Dilemma (Practical Contradiction)**: To prevent statistic mismatch across disparate visual domains, lightweight non-target parameters (classification heads, linear biases, layer normalization, and embeddings) are kept task-specific and swapped task-conditionally at test-time. This requires knowing the task ID at inference time. If the deployment environment is already capable of routing inputs and dynamically swapping parameters based on the task ID, there is **no practical reason** to perform weight merging and accept a massive performance penalty. Keeping separate PEFT/LoRA adapters is highly parameter-efficient (<1.5% of total parameters) and achieves 100% of individual expert performance (74.96% average) under the exact same task ID routing constraints, completely bypassing representation collapse. In contrast, GSC-Merge forces weight-space merging and loses **32.83% absolute accuracy** (from 74.96% down to 42.13%) while still requiring task-conditional swapping of normalization, biases, and embeddings.
2. **Truly Task-Agnostic Failure**: When task-conditional swapping is disabled (truly task-agnostic setting where norm and bias parameters are frozen at pre-trained values), GSC-Merge's performance collapses to **19.08%** (at $\gamma=0.3$) or **20.61%** (at $\gamma=0.5$). In a multi-task suite consisting of four 10-class image classification datasets, a random guess baseline is 10%. Achieving 19% or 20% accuracy means the model is functionally broken. From an industry deployment perspective, a model with such low accuracy is completely useless.
3. **Toy Backbone and Contrived Datasets**: The authors evaluate their method on the ViT-Tiny model, which has only **5.7M parameters** and a hidden dimension of 192. Modern workloads typically involve high-capacity vision models or large-scale language backbones. The datasets used are MNIST, FashionMNIST, CIFAR-10, and SVHN, which are low-resolution, toy vision datasets. Evaluating model merging strictly on these low-resolution toy datasets does not reflect real-world, industry-scale settings.
4. **Surprisingly Weak and Poorly Trained Experts**: The independent task-specific expert performance ceilings reported in the paper are remarkably low (CIFAR-10 Expert at 54.00%, SVHN Expert at 65.20%). A standard ViT-Tiny model fine-tuned on CIFAR-10 should easily achieve **85--90% accuracy**, and on SVHN it should easily exceed **95% accuracy**. The extremely low accuracies indicate that the expert models are severely under-optimized (likely due to the highly restricted training budget of only 2 epochs). Utilizing poorly trained, weak experts as the starting point for model merging compromises the generalizability and scientific validity of the entire experimental analysis.
5. **Comparison to the Unconstrained Baseline**: GSC-Merge does not actually outperform unconstrained tuning in terms of mean accuracy (e.g., Unconstrained OFS-Tune achieves $44.08\%$, whereas GSC-Merge achieves $42.13\%$ at $\gamma=0.3$ and $43.88\%$ at $\gamma=0.5$). The only benefit is a slight reduction in standard deviation (variance across splits). For a practitioner, trading off precious mean performance (which is already extremely low) to reduce split-sensitivity variance is rarely a compelling trade-off.
6. **Lack of Accuracy Validation for Randomized SVD**: Although the authors discuss and provide CPU benchmarks for Randomized SVD in Appendix A (showing significant speedup), the paper **does not provide any downstream task evaluation** demonstrating that Randomized SVD does not degrade the merged model's accuracy compared to exact SVD. For practitioners, this remains an unverified gap.

---

## Soundness
**Rating**: **Good**

**Justification**:
The paper is technically sound in terms of mathematical formulations, derivations, and proofs. The projection operator on the Grassmannian manifold is well-defined, and the proof of the optimal low-rank approximation via the Eckart-Young-Mirsky Theorem is correct and rigorous. The empirical validation utilizes a 5-seed statistical design, which is highly appreciated.
However, there are significant practical soundness and methodological limitations:
1. The reliance on task-conditional parameter swapping of normalization, biases, embeddings, and heads is methodologically contradictory. If the system is capable of task-conditional parameter routing at inference time, utilizing separate parameter-efficient (PEFT/LoRA) adapters is far more practical and avoids the massive performance drop (74.96% vs 42.13%).
2. The claims regarding the resolution of the "Overfitting-Optimizer Paradox" are overextended. Proposition 3.2 shows that the projection is a non-strict contraction on the Frobenius norm. While this bounds the norm of the updates, it does not theoretically prevent overfitting. Empirically, GSC-Merge shows a classic bias-variance trade-off (reducing variance but also slightly reducing mean accuracy compared to unconstrained tuning) rather than a complete resolution of the paradox.

---

## Presentation
**Rating**: **Excellent**

**Justification**:
The presentation of the paper is outstanding. It is clearly written, well-structured, and highly polished. The narrative flows logically from the introduction of parameter interference to the spectral consensus formulation, followed by rigorous proofs, experiments, and comprehensive discussions in the appendices. The mathematical notations are precise, the tables are clear, and the figure visualizations are professional. The inclusion of extensive details in the appendices (including randomized SVD, singular value decay, input vs. output projections, and alternative optimizers) demonstrates exceptional attention to detail.

---

## Significance
**Rating**: **Fair**

**Justification**:
While the paper is mathematically elegant and provides an interesting theoretical perspective bridging model merging and Grassmannian geometry, its practical significance and real-world utility are highly restricted:
1. **Catastrophic Performance Degradation**: A drop in accuracy from the expert ceiling of 74.96% to 42.13% (task-conditional) and 20.61% (task-agnostic) is unacceptable for real-world deployment.
2. **Deployment Contradiction**: If task ID routing is available, practitioners would use PEFT/LoRA adapters, which achieve 100% expert performance with negligible parameter overhead, making the merged model obsolete. If task ID routing is not available, GSC-Merge collapses to near-random guessing (20%), making it unusable.
3. **Toy Scale**: Evaluating strictly on ViT-Tiny (5.7M parameters) and toy low-resolution vision datasets (MNIST, CIFAR-10) limits the generalizability and significance of the findings for modern, high-capacity industrial workloads (such as LLMs or large visual backbones).

---

## Originality
**Rating**: **Good**

**Justification**:
The work presents a solid and clean mathematical progression. Formulating weight-space merging as finding a shared consensus subspace using SVD on horizontally concatenated task vectors, and projecting onto the left-singular vectors of the Grassmannian manifold, represents a nice geometric alternative to coordinate-wise heuristics (such as sign voting or magnitude thresholding).
However, the originality is incremental to moderate. Utilizing SVD and low-rank projections of model weights, activations, or updates is a standard technique in machine learning (e.g., in matrix factorization, adapter compression, or concurrent adapter merging works like MADE-IT or GAM). The integration of this spectral projection with standard layer-wise coefficient tuning (OFS-Tune) is straightforward.

---

## Overall Recommendation
**Rating**: **3: Weak Reject**

**Justification**:
This paper has clear merits, particularly its exceptionally rigorous mathematical formulation, beautiful geometric grounding in Grassmannian manifolds, and highly professional presentation. The SVD-based projection operator is elegant and provably optimal under the Frobenius norm.
However, the weaknesses currently outweigh these merits from a practical and methodological perspective:
1. **Performance and Practical Contradiction**: Under the task-conditional setting, there is a catastrophic performance gap between GSC-Merge ($42.13\%$) and the expert ceiling ($74.96\%$). If task-conditional parameter swapping is enabled at inference time, utilizing separate parameter-efficient (PEFT/LoRA) adapters is far more practical as it achieves 100% expert performance with negligible overhead, rendering weight merging obsolete.
2. **Task-Agnostic Collapse**: Under the truly task-agnostic setting, GSC-Merge's performance collapses to **20.61%**, which is barely above random guessing (10%) and practically unusable.
3. **Toy Scale and Weak Experts**: The evaluation is restricted to a toy ViT-Tiny backbone and toy low-resolution datasets (MNIST/CIFAR-10). Furthermore, the expert checkpoints are surprisingly weak (CIFAR-10 Expert at only 54.00%), which undermines the reliability and generalizability of the experimental results.
4. **No Practical Advantage over Unconstrained Tuning**: GSC-Merge slightly underperforms unconstrained tuning in terms of mean accuracy, offering only a slight reduction in variance across calibration splits.

To be suitable for publication, the paper requires significant revisions: scaling the evaluation to high-capacity models and realistic datasets, properly optimizing the expert baselines, showing downstream accuracy evaluations of Randomized SVD, and addressing the practical routing contradictions.

---

## Constructive Feedback and Questions for the Authors
1. **Scale Up to Realistic Benchmarks**: Please evaluate GSC-Merge on realistic, high-capacity models (e.g., ViT-Base/Large on ImageNet-1K, or Large Language Models like LLaMA-7B on NLP benchmarks as discussed in Appendix D). Evaluating strictly on ViT-Tiny and MNIST/CIFAR-10 limits the significance and generalizability of your findings.
2. **Address the Practical Routing Contradiction**: Can you provide a compelling deployment scenario where a practitioner would choose GSC-Merge (with a 32.8% absolute performance drop) over keeping separate PEFT/LoRA adapters, given that both setups require task-conditional parameter swapping at inference time? If task routing is active, keeping separate adapters is extremely parameter-efficient and preserves 100% of individual expert performance.
3. **Optimize the Expert Checkpoints**: The independent expert ceilings (CIFAR-10 at 54.00% and SVHN at 65.20%) are remarkably low for a ViT-Tiny model. Please train your expert models to standard convergence ceilings (e.g., >85% on CIFAR-10) before performing merging. Poorly trained experts weaken the scientific validity of the experimental comparisons.
4. **Downstream Validation of Randomized SVD**: In Appendix A, you show significant CPU speedups for Randomized SVD. However, you do not provide downstream task evaluations. Please run experiments comparing the joint mean accuracy of GSC-Merge using exact SVD vs. Randomized SVD to verify that the randomized approximation does not degrade performance.
5. **Analyze the Bias-Variance Trade-off**: Since GSC-Merge slightly degrades mean performance compared to unconstrained OFS-Tune (while reducing variance across random splits), can you explore adaptive rank strategies or regularization penalties that could recover the peak mean performance of unconstrained tuning while retaining the stability of the Grassmannian projection?
