# Summary of the Paper

## Main Topic and Approach
This paper introduces **Grassmannian Subspace Consensus Merging (GSC-Merge)**, a mathematically principled, partial weight-space model merging framework designed to consolidate multiple fine-tuned expert models into a single multi-task network. 

To address the limitations of existing coordinate-wise heuristic merging methods (like TIES-Merging or Sparse Task Arithmetic), the authors propose targeting the major linear projection layers inside Transformer blocks (comprising over 95% of block parameters) while keeping lightweight normalization, bias, and embedding parameters task-specific (swapped task-conditionally at inference time). 

The core of the approach is as follows:
1. At each targeted linear layer, the task vectors (parameter updates from the pre-trained base model) across all $K$ expert models are horizontally concatenated to form a **joint multi-task update matrix** $\mathbf{M}^{(l)}$.
2. Singular Value Decomposition (SVD) is performed on $\mathbf{M}^{(l)}$ to extract the principal directions of output parameter variation.
3. A low-rank basis matrix $U_r^{(l)}$ is constructed using the top $r$ left-singular vectors (corresponding to a point on the Grassmannian manifold $\mathbf{Gr}(r, d_{out})$), which forms the orthogonal projection matrix $P^{(l)} = U_r^{(l)} (U_r^{(l)})^T$.
4. The task updates are projected onto this low-rank consensus subspace: $\tilde{V}_k^{(l)} = P^{(l)} V_k^{(l)}$.
5. The final merged weight is a linear combination of these denoised task vectors: $W_{merged}^{(l)} = W_{base}^{(l)} + \sum_{k=1}^K \alpha_k^{(l)} \tilde{V}_k^{(l)}$, where the layer-wise blending coefficients $\alpha_k^{(l)}$ are optimized using **Offline Few-Shot Validation Tuning (OFS-Tune)** on a tiny validation set (16 samples per task, 64 total).

## Key Findings
- **Catastrophic Interference of Non-Tuned Methods:** Direct parameter interpolation methods (Uniform Merging, Task Arithmetic, Sparse Task Arithmetic, and TIES-Merging) achieve extremely poor joint mean performance (ranging from 11.16% to 32.35% under task-conditional swapping, and 8.34% to 18.80% under task-agnostic settings) on highly conflicting visual classification tasks (MNIST, FashionMNIST, CIFAR-10, SVHN).
- **Competitiveness of Subspace Projection:** GSC-Merge (with rank $\gamma = 0.5$) achieves a joint mean accuracy of 43.88% (task-conditional) and 20.61% (task-agnostic), representing a substantial improvement over static, coordinate-wise baselines.
- **Competitiveness with Unconstrained Tuning:** GSC-Merge is highly competitive with unconstrained OFS-Tune (which obtains 44.08% task-conditional and 20.86% task-agnostic) while restricting the parameter search space to a low-dimensional consensus manifold.

## Explicitly Claimed Contributions and Evidence
1. **GSC-Merge Framework:** Grounding weight-space model merging in spectral theory and Grassmannian projection. *Evidence:* Full mathematical formulation in Section 3 and empirical evaluations in Section 4.
2. **Mathematical Optimality Guarantee:** Proving that the SVD-based projection yields the mathematically optimal low-rank reconstruction of multi-task updates under the Frobenius norm. *Evidence:* Proof using the Eckart-Young-Mirsky Theorem in Section 3.3.
3. **Resolution of the Overfitting-Optimizer Paradox:** Showing that Grassmannian projection serves as an elegant spectral regularizer that restricts active degrees of freedom and prevents optimization collapse. *Evidence:* Proposition and proof in Section 3.6, with discussion of split-sensitivity variance reduction in Section 4.3.
4. **Rigorous Statistical Evaluation:** Evaluating ViT-Tiny across 4 conflicting image tasks over 5 independent random validation calibration splits under both task-conditional and truly task-agnostic settings. *Evidence:* Table 1, Table 2, and Figure 1 in Section 4.
