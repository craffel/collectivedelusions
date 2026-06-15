# 3. Soundness and Methodology Evaluation

## Clarity of the Description
The description of the mathematical framework is exceptionally clear, precise, and rigorous. The authors do a commendable job laying out the step-by-step pipeline of **RIMO** (Figure 1 and Algorithm 1) and providing formal definitions for all operations, including the Procrustes decoupling, tangent space projection, magnitude-corrected aggregation, and retraction back to the manifold. The proofs of Proposition 3.1, Theorem 3.2, and Theorem 3.3 in Appendix A are detailed and easy to follow.

## Appropriateness of Methods
- **Orthogonal Procrustes Decoupling:** Using SVD to solve the Orthogonal Procrustes problem is a mathematically sound way to extract the "closest" rotational component $R_k \in \mathrm{O}(d)$ from an unconstrained weight matrix. However, the residual $\rho_k = W_k - W_0 R_k$ is a standard Euclidean displacement. If this residual is large, the majority of the "merging" is actually happening via standard linear addition of these residuals ($\rho_{merged}$), making the "manifold merging" aspect a minor correction rather than the primary mechanism.
- **Spectral Balancing in $\mathfrak{so}(d)$:** Attempting to translate SAIM's Euclidean spectral balancing to the tangent Lie algebra of a curved manifold is a natural and highly interesting research direction.
- **Symmetry-Preserving Alternatives:** Proposing real Schur decomposition and complex Hermitian eigensolvers to perform projection-free spectral operations shows high mathematical maturity and is highly appropriate for maintaining algebraic closure without heuristic projection steps.

---

## Potential Technical Flaws and Logical Gaps (Skeptical Critic Critique)

### 1. The "SVD Projection Distortion" is a Red Herring
The authors present the **Kernel Distortion Theorem** (Theorem 3.2) and the **Spectrum Distortion Theorem** (Theorem 3.3) as the key theoretical explanations for the "spectral balancing pitfall in Lie algebra spaces" under standard SVD solvers:
- They argue that standard SVD solvers introduce non-symmetric coordinate gauges in the null space, which inject non-zero skew-symmetric noise during projection, and that non-uniform singular value modifications inevitably distort the spectrum under projection.
- However, their own empirical results in Tables 1 and 2 show that **RIMO-Schur-Balanced** and **RIMO-Complex-Balanced** (which are mathematically guaranteed to completely avoid SVD, require no projection step, have zero projection error, and exhibit zero kernel/spectral distortion) **suffer from the exact same catastrophic performance collapse** (collapsing to $\sim 12\%$ and $\sim 20\%$ average accuracy, matching RIMO's collapse).
- This is a major logical gap. If the projection-free, mathematically perfect Schur-based spectral balancing collapses just as severely as SVD-based balancing, then *SVD projection distortion and coordinate gauge issues are not the actual cause of the performance collapse*.
- The true culprit is simply the **non-linear coordinate inflation under the Cayley map** (Theoretical Insight 2), where inflating any small or zero singular values in the tangent space translates to spurious, non-zero rotations across thousands of inactive high-dimensional planes, scrambling representation alignment.
- Thus, while Theorems 3.2 and 3.3 are mathematically correct, they are decoupled from the physical bottleneck of the model collapse. Framing them as the primary cause of the pitfall is technically misleading and represents a severe over-complication of a simple geometric noise propagation phenomenon.

### 2. Omission of the Capacity Bottleneck of Orthogonal Constraints
The entire framework of Riemannian model merging relies on the "Orthogonality Condition":
- The authors show that without orthogonal regularization, Procrustes residuals are huge and collapse performance. Therefore, they argue that models *must* be trained with soft orthogonal regularization ($\lambda_{ortho}=2.0$) or hard manifold constraints (projected Riemannian SGD).
- However, forcing weight matrices to be orthogonal is a extremely restrictive bottleneck that severely limits the representational capacity and expressiveness of neural networks. 
- While this capacity reduction is hidden in simple toy benchmarks like Split-MNIST (where an MLP of $d=256$ has massive redundant capacity), it is a well-known failure mode in large-scale machine learning: strictly orthogonal models suffer from severe optimization difficulties and significant performance degradation on complex datasets (e.g., ImageNet, WikiText).
- The authors completely fail to discuss or evaluate this fundamental trade-off. They assume orthogonal training is always feasible and harmless, which is a major oversimplification of modern deep learning practices.

### 3. SVD Mean Singular Value Scaling Flaw (Eq. 8)
In Section 3.4, the authors define the target isotropic strength as the arithmetic average of all $d$ singular values (Eq. 8):
$$\bar{\sigma} = \frac{1}{d} \sum_{i=1}^d \sigma_i$$
- For a low-rank task update of rank $k \ll d$, the sum is taken over $d$ values where $d-k$ values are exactly $0$.
- In large networks (e.g., $d=4096$), if $k=8$, then $\bar{\sigma}$ is heavily scaled down by a factor of $4096$, making the target isotropic strength $\bar{\sigma}$ extremely close to zero.
- Consequently, the "isotropic balancing" under Eq. 9 actually *shrinks* the active task singular values by a massive factor (since they are pulled towards $\bar{\sigma} \approx 0$).
- Shrinking the active singular values of the task-specific update towards zero is equivalent to **catastrophic forgetting or erasing the task updates entirely**!
- This is an obvious, major methodological flaw: by defining the mean over the entire dimension $d$ (including the massive kernel), the formula automatically nullifies the task-specific representation in high dimensions, rather than balancing it. If they had averaged only over the active $k$ dimensions, the active components would not have been erased, and the inactive components would have remained zero (avoiding noise propagation). This is a simple but critical mathematical flaw in their spectral balancing design.

---

## Reproducibility
The reproducibility of this paper is rated as **Excellent**. 
- The training hyperparameters, model architectures (3-layer MLP and custom 2-head ViT), dataset splits (Split-MNIST), and optimizer details are explicitly described in the text and Appendix.
- The algorithm (Algorithm 1) is fully detailed with exact mathematical formulas.
- The use of standard, lightweight benchmarks (Split-MNIST) ensures that any researcher can easily reproduce and verify the reported results within a few minutes on a single CPU/GPU.
