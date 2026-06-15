# 3. Soundness and Methodology Evaluation

## Clarity of the Description
The description of the mathematical framework is exceptionally clear, detailed, and structured. 
* **Pipeline Flowchart:** Figure 1 provides a very clear, color-coded schematic of the RIMO model merging pipeline, transitioning from standard parameter spaces (light gray) to the Lie Group manifold (light blue), and then to the tangent Lie algebra space (light red), and back.
* **Algorithmic Formalization:** Algorithm 1 provides a complete pseudocode overview of RIMO, RIMO-Pruned, and alternative decomposition pipelines, making the mathematical steps very accessible.
* **Notation:** The paper uses standard, rigorous mathematical notation (e.g., Lie algebra $\mathfrak{so}(d)$, orthogonal group $\mathrm{O}(d)$, SVD, real Schur decomposition, Cayley maps) with consistent variable definitions.

## Appropriateness of Methods
* **Orthogonal Procrustes Decoupling:** This is a highly appropriate and elegant way to extract the rotational component ($R_k$) of standard Euclidean neural network parameters relative to a base model ($W_0$), leaving the linear residuals ($\rho_k$) to capture coordinate distortions.
* **Cayley Transforms:** Using the inverse and forward Cayley transforms to map between the Lie Group $\mathrm{O}(d)$ and Lie algebra $\mathfrak{so}(d)$ is mathematically well-justified and avoids the expensive matrix logarithm and exponential maps.
* **Magnitude-Corrected Aggregation:** Direct averaging of tangent vectors can lead to rotational magnitude decay. The inclusion of a Frobenius norm-based scalar correction factor ($c$) is a highly appropriate heuristic to preserve task-specific rotational energies.
* **Symmetry-Preserving Alternatives:** Instead of just reporting SVD's failure, the authors appropriately introduce and evaluate:
  1. **Real Schur Decomposition:** A standard matrix decomposition that directly extracts conjugate pairs of eigenvalues of real skew-symmetric matrices into $2\times2$ skew-symmetric blocks, preserving the Lie algebra structure by design and avoiding the projection step.
  2. **Complex Hermitian Solver:** Mapping $Q$ to a complex Hermitian matrix ($i Q$) to perform complex Hermitian eigen-decomposition, which maintains symmetry natively and allows batched, GPU-accelerated operations.
* **Rank-Preserving Spectral Pruning:** Pruning instead of smoothing is an appropriate, stable proxy that avoids inflating zero-valued singular dimensions.

## Mathematical and Technical Soundness (including Theorems)
The paper is exceptionally sound and rigorous. The authors do not merely report experimental success; they identify a major mathematical pitfall and formalize it through two key theorems:
* **Kernel Distortion Theorem (Theorem 3.2):** This theorem is mathematically correct and highly insightful. Standard SVD solvers (such as QR or divide-and-conquer in libraries like PyTorch) are unaware of the underlying skew-symmetry of the input matrix. Consequently, they introduce arbitrary non-symmetric gauge transformations (an orthogonal matrix $P \in \mathrm{O}(m)$) in the degenerate $m$-dimensional null space. When the singular values in this null space are inflated to a non-zero value $\hat{\sigma} > 0$ and projected via $\mathcal{P}(M) = \frac{1}{2}(M - M^T)$, this non-symmetric gauge injects massive, spurious skew-symmetric noise.
* **Spectrum Distortion Theorem (Theorem 3.3):** This theorem is also mathematically sound. It proves that non-uniform spectral modifications in $\mathfrak{so}(d)$ (such as isotropic smoothing) violate the fundamental algebraic relation $R \Sigma = -\Sigma R^T$ (where $R = U^T V$). As a result, the subsequent projection operator $\mathcal{P}$ required to restore skew-symmetry distorts the spectrum, shifting actual singular values away from their targets.
* **Curvature and Rotational Noise Propagation:** The authors show that because the forward Cayley map $R = (I+Q)(I-Q)^{-1}$ is highly non-linear, small tangent-space perturbations in inactive dimensions are mapped to large rotation angles $\theta_i = 2 \arctan(\hat{\sigma}_i)$ in the Lie group. This is technically robust and beautifully explains why RIMO collapses to random guess performance when $t > 1.0$.

## Potential Technical Flaws / Limitations
1. **Block-Diagonal Partitioning Constraints:** For non-square rectangular weights, the authors partition the weights into square $b \times b$ blocks. They argue that this introduces zero coordinate clipping or boundary distortions. While this is algebraically true (each block is independently orthogonal), it artificially restricts the orthogonal degrees of freedom to localized coordinate compartments. A global layer-wide rotation is not possible under this scheme. The authors should discuss this representational capacity trade-off.
2. **Computational Complexity of SVD and Schur Decomposition:** Standard SVD and real Schur decompositions are $O(d^3)$ operations. While the complex Hermitian solver addresses this on GPU, the $O(d^3)$ bottleneck remains a challenge for very large layers (e.g., $d=4096$).
3. **Soft Orthogonal Regularization Limits:** The paper notes that soft training-time orthogonal constraints do not place parameters perfectly on $\mathrm{O}(d)$, generating residual components $\rho_k$ that introduce coordinate warp under the Cayley map. This limit is inherent to soft regularization and is a well-known limitation of penalty methods in constrained optimization.

## Reproducibility
The reproducibility of this work is **excellent**. 
* Detailed hyperparameter settings ($t$, $\rho_{\text{scale}}$, keep-ratio, optimization learning rates) are provided for both standard and orthogonal training regimes.
* The exact steps in the algorithms are fully specified in the main text.
* Execution environments and latencies are documented on specific hardware (Intel Xeon CPU and NVIDIA H100 GPU), providing realistic reference benchmarks.
