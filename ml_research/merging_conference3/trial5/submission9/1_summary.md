# Summary of the Paper

## Core Details
- **Title:** Grassmannian Subspace Consensus Merging: A Spectral Filter for Multi-Task Parameter Alignment
- **Author:** David Vance (Princeton University)
- **Problem Addressed:** Weight-space model merging is computationally efficient for multi-task model consolidation, but suffers from parameter interference and representation collapse when merging models trained on highly disparate or conflicting tasks. Prior methods rely on coordinate-wise heuristics (e.g., TIES-Merging, Sparse Task Arithmetic) that treat weight parameters independently, discarding structural correlations and lacking mathematical guarantees.
- **Proposed Solution:** **Grassmannian Subspace Consensus Merging (GSC-Merge)**. GSC-Merge targets major linear projection layers (comprising over 95% of Transformer block parameters) while keeping lightweight normalization, biases, and embedding parameters task-specific. It horizontally concatenates multi-task updates into a joint update matrix, performs Singular Value Decomposition (SVD), and constructs an optimal low-rank projection operator onto a shared Grassmannian manifold $\mathbf{Gr}(r, d_{out})$.
- **Key Theoretical Contributions:**
  1. **Optimal Low-Rank Approximation:** Leverages the Eckart-Young-Mirsky Theorem to prove that the SVD-based projection operator provides the mathematically optimal low-rank approximation of joint task updates under the Frobenius norm, minimizing representation distortion.
  2. **Overfitting-Optimizer Paradox Resolution:** Identifies that unconstrained optimization of layer-wise merging coefficients on small validation sets (OFS-Tune) suffers from transductive overfitting to validation noise. Proves that GSC-Merge acts as a spectral regularizer (producing a non-strict contraction of updates in both spectral and Frobenius norms) to restrict the parameter search space and prevent optimization collapse.
- **Key Empirical Contributions:**
  1. Evaluates a Vision Transformer (ViT-Tiny) backbone across four highly conflicting datasets: MNIST, FashionMNIST, CIFAR-10, and SVHN.
  2. Conducts a rigorous 5-seed statistical analysis to assess validation split sensitivity.
  3. Demonstrates that GSC-Merge consistently outperforms coordinate-wise baselines (Uniform, STA, TIES-Merging, Task Arithmetic) and achieves competitive accuracy to unconstrained tuning but with significantly reduced variance across validation splits (e.g., $\pm 2.76\%$ vs $\pm 4.31\%$).
  4. Evaluates a "truly task-agnostic" setting (where non-target parameters are fixed to pre-trained base values rather than swapped), showing GSC-Merge remains robust and matches unconstrained performance.
  5. Includes extensive appendices addressing SVD scalability (empirical benchmarks on LLaMA-7B sizes with Randomized SVD), cumulative energy analyses (showing over 90% update energy is captured at $\gamma = 0.3$), projection directions (showing output-space projection dramatically outperforms input-space or bilateral projection), and future extensions (NLP feasibility and GSC-Route routing framework).
