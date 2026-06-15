# Intermediate Evaluation: Summary of the Paper

## Main Topic and Goal
The paper addresses the challenge of **parameter interference** and **representation collapse** in weight-space model merging. The primary objective is to consolidate multiple task-specific expert neural networks (fine-tuned from a shared pre-trained base) into a single multi-task model without performing full joint retraining, while preserving individual task performance.

## Proposed Approach: GSC-Merge
The authors propose **Grassmannian Subspace Consensus Merging (GSC-Merge)**, a partial weight-space model merging framework:
1. **Targeted Merging**: It targets only the major linear projection layers inside the Transformer blocks (comprising >95% of block parameters) for weight-space merging. Non-target parameters (such as layer normalization, biases, and patch/positional embeddings) are kept task-specific and swapped task-conditionally at test-time to prevent statistic mismatch across highly disparate visual domains.
2. **Joint Multi-Task Update Matrix**: For each target linear layer, task-specific task vectors (the difference between fine-tuned and pre-trained weights, $V_k = W_k - W_{base}$) are horizontally concatenated across all $K$ experts to construct a joint update matrix $\mathbf{M}^{(l)} \in \mathbb{R}^{d_{out} \times K \cdot d_{in}}$.
3. **SVD Projection onto Grassmannian**: Singular Value Decomposition (SVD) is performed on $\mathbf{M}^{(l)}$. The top $r = \lfloor \gamma \cdot d_{out} \rfloor$ left-singular vectors are used to construct an orthogonal projection operator $P^{(l)} = U_r^{(l)} (U_r^{(l)})^T$, which represents a point on the Grassmannian manifold $\mathbf{Gr}(r, d_{out})$.
4. **Spectral Consensus Denoising**: The task vectors are projected onto this low-rank subspace, which mathematically filters out high-frequency task-specific noise (associated with tail singular values) while retaining the coherent shared consensus directions (associated with the leading singular values).
5. **Offline Few-Shot Validation Tuning (OFS-Tune)**: The final merged weight is a linear combination of projected task vectors, where the layer-wise blending coefficients $\alpha_k^{(l)}$ are optimized via backpropagation on a tiny validation calibration set (e.g., 16 samples per task) using the Adam optimizer.

## Key Findings
* **Naive Merging Collapse**: Simple linear averaging (Uniform Merging) collapses performance (down to near-random guessing, 11.16% joint mean accuracy).
* **GSC-Merge Superiority**: GSC-Merge with $\gamma=0.3$ and $0.5$ consistently outperforms coordinate-wise heuristic merging baselines such as TIES-Merging and Sparse Task Arithmetic (STA).
* **Implicit Spectral Regularization**: Direct, unconstrained OFS-Tune is highly sensitive to the validation calibration split and suffers from high variance (the *Overfitting-Optimizer Paradox*). Projecting updates onto the low-rank Grassmannian subspace prior to tuning restricts the optimization space and acts as a robust spectral regularizer, significantly reducing split-sensitivity variance across validation runs.
* **Task-Agnostic Performance Drop**: When task-conditional parameter swapping is disabled (truly task-agnostic setting), all methods experience a severe drop in performance. GSC-Merge with $\gamma=0.5$ achieves the highest task-agnostic performance ($20.61 \pm 4.80\%$), matching unconstrained tuning while significantly restricting the active search parameter space.

## Explicitly Claimed Contributions (with Evidence)
1. **Mathematically Principled Framework**: Formulates model merging as finding a shared consensus subspace using SVD. **Evidence**: Full mathematical derivation of the joint update matrix and projection operators (Section 3).
2. **Provable Optimal Low-Rank Approximation**: Invokes the Eckart-Young-Mirsky Theorem to prove that the projection minimizes representation distortion under the Frobenius norm. **Evidence**: Theorem 3.1 and its mathematical proof (Section 3.3).
3. **Resolution of the Overfitting-Optimizer Paradox**: Shows that the projection acts as a non-strict contraction in both spectral and Frobenius norms. **Evidence**: Proposition 3.2 and its proof (Section 3.6), and empirical results showing reduced standard deviation (from $\pm 4.31\%$ in unconstrained tuning to $\pm 2.76\%$ in GSC-Merge with $\gamma=0.3$) across 5 random validation splits (Table 5.1).
4. **Rigorous Empirical Evaluation**: Compares GSC-Merge against several baselines using a ViT-Tiny backbone across four conflicting classification datasets (MNIST, FashionMNIST, CIFAR-10, SVHN). **Evidence**: 5-seed statistical analysis reported in Table 5.1 (task-conditional swapping) and Table 5.2 (truly task-agnostic).
