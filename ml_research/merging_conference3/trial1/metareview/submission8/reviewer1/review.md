# Peer Review Report

## 1. Summary of the Paper
This paper presents a formal mathematical and empirical analysis of model merging on non-linear parameter manifolds—specifically, the orthogonal group $\mathrm{O}(d)$ and its associated Lie algebra tangent space $\mathfrak{so}(d)$. Building on the OrthoMerge framework, the authors investigate whether Euclidean spectral-balancing techniques (like SAIM) can be translated to curved manifold structures. 

The paper uncovers two main phenomena:
1. **The Orthogonality Condition**: Manifold-level merging is highly sensitive to parameter non-orthogonality. Merging models trained with soft orthogonal regularization ($\lambda_{ortho} = 2.0$) reaches $84.55\%$ accuracy, whereas standard unconstrained models collapse to $42.07\%$. Naive post-hoc SVD projection of standard models onto $\mathrm{O}(d)$ is highly destructive, collapsing performance to $15.00\%$.
2. **The Tangent Space Spectral Pitfall**: Modifying the singular spectrum in the Lie algebra tangent space $\mathfrak{so}(d)$ via SVD-based isotropic balancing ($t > 1.0$) catastrophically scrambles representations, dropping accuracy to $13.66\%$ (MLP) and $18.44\%$ (Vision Transformer). The authors formalize this via the **Kernel Distortion Theorem** and **Spectrum Distortion Theorem**, proving that standard SVD solvers introduce non-symmetric coordinate gauges in multi-dimensional null spaces and active subspaces, which distorts the spectrum under skew-symmetric projection and maps to massive, high-dimensional rotational noise under the Cayley map.

The authors propose three mitigations to bypass this SVD-induced coordinate gauge issue:
* **Symmetry-Preserving Real Schur Decomposition** (RIMO-Schur)
* **GPU-Compatible Complex Hermitian Solver** (RIMO-Complex)
* **Rank-Preserving Spectral Pruning** (RIMO-Pruned)

The proposed methods are evaluated on Split-MNIST and Split-CIFAR-10 using MLPs and a custom micro-Vision Transformer.

---

## 2. Strengths and Weaknesses

### Strengths
1. **Exceptional Mathematical Rigor**: The paper is mathematically elegant and rigorous. The derivations and formal proofs for the *Kernel Distortion Theorem* (Theorem 3.2), *Spectrum Distortion Theorem* (Theorem 3.3), and *Active Subspace Gauge Distortion* (Theorem 11.1) are sound, precise, and highly educational. They provide deep insights into why standard numerical SVD is fundamentally incompatible with skew-symmetric Lie algebra structures.
2. **Exemplary Presentation and Structure**: The paper is beautifully written, clearly structured, and easy to follow. Figure 1 provides an excellent conceptual flowchart of the pipeline, and the mathematical notation is consistent throughout.
3. **Scientific Candor and Transparency**: The authors are highly commendable for their honesty. Rather than hiding or minimizing the failure of their initial hypothesis (that spectral balancing would work in tangent space), they analyze the catastrophic performance collapse ($13.66\%$) in detail and use it to develop rigorous negative results. They also transparently discuss the persistent performance gap between their manifold methods and simple Euclidean Task Arithmetic.

### Weaknesses
1. **Severe Over-Engineering with No Practical Payoff**: 
   The proposed pipeline is a textbook example of excessive architectural and mathematical complexity. It requires extracting rotations via Orthogonal Procrustes decoupling (SVD), mapping to Lie algebra via inverse Cayley, performing a second SVD, Schur, or complex Hermitian decomposition, modifying the spectrum, applying skew-symmetric projection, mapping back via forward Cayley, and finally linearly averaging high-norm residuals.
   
   Despite this heavy machinery, **simple flat-space Euclidean Task Arithmetic consistently and substantially outperforms the proposed method across all experimental regimes**:
   * Under *Standard Training* (Table 1), simple **Task Arithmetic ($\lambda = 0.3$)** achieves **91.11%** average accuracy, while the proposed **RIMO-Pruned** achieves **90.47%**.
   * Under *Orthogonal Regularization* (Table 2), simple **Task Arithmetic ($\lambda = 1.0$)** achieves **94.00%** average accuracy, while the proposed **RIMO-Pruned** achieves **91.49%** (a **2.51%** absolute gap favoring the simple baseline).
   * Even in the *Multi-Task Scaling limit ($N=5$)* (Table 5) where the authors argue Euclidean averaging decays, **Task Arithmetic ($\lambda = 0.4$)** achieves **91.48%** average accuracy, whereas **RIMO-Pruned** achieves **91.46%**.
   
   Introducing highly complex, computationally heavy matrix decompositions to achieve *lower* performance than a single line of flat-space linear addition (`W_merged = W_0 + lambda * (W_1 - W_0 + W_2 - W_0)`) is fundamentally unjustified.

2. **Self-Induced Obstacles**:
   The core theoretical contribution of the paper—analyzing and mitigating the "tangent space spectral balancing pitfall"—is a solution to a problem that only exists because of the authors' highly complex formulation. A simpler, more elegant approach that avoids non-linear manifolds entirely remains superior and completely immune to these coordinate distortions, gauge errors, and projection noise. The paper spends significant effort proposing complex patches (real Schur decomposition, complex Hermitian solvers, rank pruning) to salvage a problem that is self-inflicted by the chosen geometric framework.

3. **Invasive and Rigid Training Constraints**:
   For the proposed manifold merging to function, weight parameters must lie extremely close to the orthogonal manifold during fine-tuning. This requires training the models with specialized soft orthogonal regularization ($\lambda_{ortho}=2.0$) or highly sensitive hard Riemannian constraints on Stiefel manifolds. This severely restricts representational capacity, complicates optimization, and prevents the method from being applied post-hoc to standard, unconstrained models (which the authors show collapses performance).

4. **Toy-Scale Empirical Evaluation**:
   The empirical validation is restricted to extremely small-scale setups: a 3-layer MLP on Split-MNIST and Split-CIFAR-10 with hidden dimension $d=256$, and a custom micro-Vision Transformer (ViT) with an embedding dimension of $d=32$ and sequence length of 16. There are no experiments on standard, modern-scale models (e.g., standard ViT-B or standard LLMs) where model merging is a vital, practical technique.

---

## 3. Detailed Ratings

### Soundness: Good
The mathematical soundness of the paper is excellent; the theorems are correct, and the proofs in the appendix are thorough. However, the empirical soundness is only fair because the central claim—that geometric manifold merging is a superior or necessary paradigm—is directly contradicted by the authors' own experimental results, which consistently show simple flat-space Task Arithmetic outperforming their highly complex pipeline. 

### Presentation: Excellent
The paper is exceptionally well-written. The introduction clearly motivates the work, the methodology is structured logically, and the limitations and failures are analyzed with remarkable clarity and honesty. 

### Significance: Poor
The practical significance of this work is very low. Practitioners seeking to merge models are highly unlikely to adopt a method that requires specialized training-time constraints, introduces multiple computationally expensive matrix decompositions (SVD/Schur/Complex Eigen-decomposition), and ultimately yields inferior accuracy compared to a simple, single-line Euclidean average.

### Originality: Good
The theoretical insights regarding the incompatibility of standard SVD solvers with skew-symmetric Lie algebras due to coordinate gauge choices are highly original and mathematically elegant. However, the proposed successful merging method, **RIMO-Pruned**, is a highly incremental modification of OrthoMerge that simply discards small singular values (rank pruning) rather than balancing them, essentially returning to a low-rank Euclidean-like projection.

---

## 4. Overall Recommendation

**Score: 3 (Weak Reject)**

### Justification of Recommendation
This paper possesses outstanding mathematical beauty and rigorous theoretical derivations. The analysis of SVD-induced coordinate gauge distortions in Lie algebra tangent spaces (Theorem 3.2 and 3.3) is elegant and represents a high-quality contribution to geometric deep learning theory. 

However, from a practical machine learning perspective, the paper fails to justify its immense complexity. It introduces a mountain of algorithmic and optimization overhead (SVDs, Cayley transforms, Schur/Complex decompositions, soft/hard training-time constraints) only to achieve performance that is consistently and substantially worse than simple, parameter-free Euclidean linear addition (Task Arithmetic). The "spectral balancing pitfall" is a self-inflicted obstacle, and the proposed mitigations are highly complex solutions to a problem that does not exist in standard model merging. Combined with the toy-scale empirical evaluation, the practical utility of this framework is extremely limited. 

Therefore, while the theoretical findings are interesting, the weaknesses currently outweigh the practical merits. The paper would be significantly stronger if the authors could demonstrate a clear, practical scenario on a standard, large-scale model where their complex manifold framework actually outperforms simple Euclidean Task Arithmetic.
