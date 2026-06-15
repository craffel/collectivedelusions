# Peer Review of the Conference Submission

## Summary of the Paper
The paper presents a theoretical and empirical study of the limits of representational isotropy and spectral operations within non-linear manifolds, specifically focusing on the orthogonal group $\mathrm{O}(d)$ and its tangent Lie algebra $\mathfrak{so}(d)$. The authors propose a geometric model merging framework called **RIMO** (Riemannian Isometry-respecting Manifold Operations) which projects fine-tuned expert models to $\mathfrak{so}(d)$ using Orthogonal Procrustes decoupling and the inverse Cayley transform. 

In this tangent space, they attempt to translate Euclidean-based spectral balancing (such as SAIM) to restore isotropy. They identify a "spectral balancing pitfall": inflating smaller singular values to force isotropy in the tangent space catastrophically collapses the model's accuracy (dropping performance to $\sim 13\%$). They prove two theorems (the *Kernel Distortion Theorem* and the *Spectrum Distortion Theorem*) to show that standard numerical SVD solvers introduce coordinate gauge mismatches that distort the skew-symmetric projection required for algebraic closure. Under the forward Cayley map, inflating these singular values maps to spurious rotations across inactive high-dimensional planes, destroying feature representations. 

To mitigate this, they propose **RIMO-Pruned** (rank-preserving spectral pruning) which zeroes out inactive singular values, and introduce alternative projection-free, symmetry-preserving decompositions (real Schur and complex Hermitian). They evaluate their framework on Split-MNIST using an MLP and a down-scaled Vision Transformer.

---

## Detailed Evaluation: Strengths and Weaknesses

### Originality (Rating: Good)
- **Strengths:** 
  - The paper is the first to attempt to translate spectral balancing and representational isotropy techniques from Euclidean spaces to curved Riemannian manifolds. 
  - The discovery of the tangent-space spectral balancing pitfall and the formalization of its noise propagation are original contributions that provide valuable insights into why standard Euclidean operations do not easily generalize to non-linear geometries.
- **Weaknesses:**
  - The proposed successful method, **RIMO-Pruned**, is conceptually highly incremental. Rather than developing a new geometric isotropic operator, it relies on standard low-rank spectral truncation, which is a well-known technique in deep learning.

### Soundness (Rating: Fair)
- **Strengths:**
  - The mathematical derivations are technically detailed, and the proofs in Appendix A for Proposition 3.1, Theorem 3.2, and Theorem 3.3 are mathematically correct.
  - The authors are highly rigorous in validating their assumptions across multiple seeds, block-diagonal sensitivity sweeps, and architectures.
- **Weaknesses (Critical Flaws & Logical Gaps):**
  - **The SVD Projection Theorems are a Logical Red Herring:** The authors dedicate significant theoretical weight to proving the *Kernel Distortion Theorem* (Theorem 3.2) and the *Spectrum Distortion Theorem* (Theorem 3.3) to explain why standard SVD is structurally incompatible with Lie algebras. However, their own empirical results in Tables 1 and 2 completely disprove the importance of these theorems. When they evaluate **RIMO-Schur-Balanced** (which utilizes real Schur decomposition to guarantee perfect skew-symmetry and completely bypasses SVD projection, exhibiting zero projection or kernel distortion), the model **still collapses catastrophically to $12.36\%$ (soft-regularized) and $16.50\%$ (hard constraints)**. This demonstrates that SVD projection and coordinate gauge errors are *not* the driver of the performance collapse. The collapse is driven purely by the non-linear Cayley map noise propagation under singular value inflation. Therefore, framing Theorems 3.2 and 3.3 as the primary explanations for the pitfall is logically inconsistent and mathematically misleading.
  - **SVD Mean Singular Value Scaling Formula Flaw (Eq. 8):** The target isotropic strength is defined as the arithmetic average of all $d$ singular values (Eq. 8): $\bar{\sigma} = \frac{1}{d} \sum_{i=1}^d \sigma_i$. For an update with an active rank of $k \ll d$ in a high-dimensional space (e.g., $d=4096$), the sum is heavily dominated by the $d-k$ zero singular values in the kernel. This forces the mean $\bar{\sigma}$ to scale down close to zero. Under Eq. 9, the active task-specific singular values are interpolated towards $\bar{\sigma}$, which effectively **shrinks them close to zero, erasing the task representation entirely**. This is a critical methodological flaw: the formula itself causes catastrophic forgetting of the active task updates in high dimensions, rather than balancing them.
  - **Omission of the Capacity Bottleneck of Orthogonal Constraints:** For the Procrustes decoupling to succeed, the models must be trained with soft orthogonal regularization ($\lambda_{ortho}=2.0$) or hard manifold constraints. Forcing weight matrices to lie on or near $\mathrm{O}(d)$ is an extremely restrictive constraint that severely limits a network's capacity and expressiveness. While this capacity reduction is masked in toy MNIST benchmarks, it is a well-known bottleneck in larger networks. The authors completely ignore this fundamental optimization trade-off.

### Significance (Rating: Fair)
- **Strengths:**
  - The theoretical and diagnostic insights regarding Cayley mapping noise propagation and the limitations of tangent space linear operators are highly valuable to a specialized audience in geometric deep learning and Lie group optimization.
- **Weaknesses:**
  - **Consistently Beaten by Simple Baselines:** The proposed framework fails to demonstrate any practical advantage over standard flat-space **Task Arithmetic (TA)**:
    - Under standard training, TA achieves **$91.11\%$** vs. RIMO-Pruned's **$90.47\%$**.
    - Under orthogonal training, TA achieves **$94.00\%$** vs. RIMO-Pruned's **$91.49\%$** (a $2.51\%$ gap), and advanced baselines like DARE ($93.89\%$) and TIES ($93.34\%$) also easily outperform it.
    - Given that Task Arithmetic is training-free, computationally cheap ($O(1)$), hyperparameter-light, and achieves significantly higher accuracy, there is zero practical motivation for practitioners to adopt the complex, $O(d^3)$ SVD-based RIMO-Pruned pipeline.
  - **Toy Setup and Lack of Large-Scale Generalizability:** The primary evaluation is restricted to Split-MNIST on a 3-layer MLP ($d=256$). The "Vision Transformer" validation is performed on a custom, extremely down-scaled toy model ($d=32$, 2 heads, sequence size 16, fewer than 50K parameters), which is also trained on Split-MNIST. There is no validation on standard, standard-sized models (e.g., standard ViT, ResNet) or standard multi-task benchmarks (e.g., GLUE, VTAB) to prove that the findings generalize to modern deep learning scales.

### Presentation (Rating: Good)
- **Strengths:**
  - The paper is highly structured, and the narrative flow is exceptionally clear. Figure 1 provides an excellent visual overview of the pipeline, and the mathematical notation is standard and precise.
- **Weaknesses:**
  - **Misleading and Unfair Latency Comparison:** In Section 4.5, the authors report a latency of $7.66$ ms for their Complex Hermitian solver, claiming a "spectacular acceleration" of 8.1x over SVD and 12.2x over Schur. However, they compare a parallel PyTorch implementation running on an **NVIDIA H100 Tensor Core GPU** against sequential SVD and Schur implementations running on a **single-threaded Intel Xeon CPU**. Comparing a massively parallel GPU accelerator to a single-threaded CPU is a highly unfair and misleading way to claim algorithmic speedup. The latency comparison must be performed on identical hardware.
  - **Strawman Comparison with AdaMerging:** The authors critique AdaMerging by evaluating it on a highly biased disjoint setup (feeding a calibration batch from a single task), showing that it overfits and collapses. Since this directly violates AdaMerging's core assumption of a representative calibration batch, this is an unfair strawman comparison.

---

## Detailed Questions and Actionable Feedback for Authors

1. **Address the Logical Inconsistency of Theorems 3.2 and 3.3:** Since the projection-free **RIMO-Schur-Balanced** method suffers from the exact same catastrophic collapse as standard SVD-based RIMO, please explain how you justify presenting Theorems 3.2 and 3.3 as the primary explanations for the pitfall. If projection error is not the cause of the collapse, aren't these theorems a mathematical side-show? Please revise the paper to clarify that Cayley mapping noise propagation is the dominant factor, and downgrade the SVD-specific theorems to secondary numerical issues.
2. **Correct the Mean Singular Value Scaling Formula (Eq. 8):** In Eq. 8, the average is taken over all $d$ dimensions. Why not average *only* over the active singular values of the task updates (e.g., the top $k$ dimensions where $\sigma_i > \epsilon$)? This would prevent the target isotropic strength from scaling down to zero in high dimensions and stop the formula from erasing the active task updates. Please provide results where the mean is computed exclusively over the active subspace.
3. **Run Large-Scale, Standard Experiments:** To prove that these findings are relevant to modern deep learning, please evaluate your framework on standard, large-scale architectures. For example, merge standard ViT-B/16 encoders on Vision benchmarks, or merge LLaMA-7B adapters (LoRA) on language benchmarks.
4. **Discuss the Capacity Trade-off:** Please provide a discussion and empirical analysis of the single-task performance gap between unconstrained experts and orthogonally regularized experts. How much capacity is lost when forcing the network parameters to remain near the orthogonal manifold?
5. **Provide a Fair Latency Benchmark:** Please report SVD, Schur, and Complex Hermitian solver execution times under identical hardware conditions (e.g., all parallelized on the NVIDIA H100 GPU) to demonstrate a fair algorithmic comparison.

---

## Section Ratings

- **Soundness:** **Fair** (due to the major logical gap between SVD-projection theories and empirical collapse, and the mathematical scaling flaw in Eq. 8).
- **Presentation:** **Good** (narrative is clear and structured, but suffers from an unfair CPU-vs-GPU latency comparison and a strawman evaluation of AdaMerging).
- **Significance:** **Fair** (the negative results and theoretical noise derivations are of moderate academic interest to geometric deep learning, but the practical significance is extremely low since Task Arithmetic consistently outperforms the proposed method).
- **Originality:** **Good** (first spectral investigation on Riemannian manifolds, though the successful mitigation method is highly incremental).

---

## Overall Recommendation

**Rating: 3: Weak Reject**

### Justification:
This submission has clear merits, including high mathematical rigor, detailed proofs, and an honest diagnostic of a profound negative result (the tangent-space spectral balancing pitfall). However, the weaknesses currently outweigh the merits. Specifically, the paper exhibits a major logical gap where its primary theoretical explanations (Theorems 3.2 and 3.3) are empirically shown to be decoupled from the performance collapse. Furthermore, there is a methodological flaw in the mean singular value formula (Eq. 8) that erases active task representations in high dimensions. Empirically, the evaluation is restricted to toy benchmarks on tiny models, and the proposed successful method (RIMO-Pruned) is consistently and significantly outperformed by simple, flat Euclidean Task Arithmetic, which is training-free and computationally cheaper. 

Before this paper can be recommended for acceptance, the authors must resolve these logical and mathematical inconsistencies, provide a fair latency benchmark on identical hardware, and demonstrate that their geometric framework has actual generalizability and practical utility on standard, large-scale models.
