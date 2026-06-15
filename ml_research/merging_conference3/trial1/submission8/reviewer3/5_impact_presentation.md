# 5. Presentation, Strengths, Areas for Improvement, and Impact

## Major Strengths
1. **Mathematical Rigor:** The paper exhibits high mathematical maturity, offering rigorous derivations and formal proofs for Proposition 3.1 (algebraic closure of the inverse Cayley transform), Theorem 3.2 (Kernel Distortion Theorem), and Theorem 3.3 (Spectrum Distortion Theorem).
2. **Honesty and Self-Critique regarding Negative Results:** The authors are commendable for presenting highly detailed and honest negative results. Rather than hiding the catastrophic collapse of RIMO ($t > 1.0$), they highlight it, analyze it deeply, and provide a systematic diagnostic of the "tangent-space spectral pitfall."
3. **Exploration of Alternative Mathematical Pathways:** The paper does not stop at identifying the failure; it proactively explores and implements advanced mathematical alternatives, including real Schur decomposition (for perfect skew-symmetry and projection-free modifications) and complex Hermitian eigensolvers (for parallelizable, GPU-compatible execution).
4. **Thorough Appendix & Robustness Checks:** The appendix is extremely comprehensive, providing multi-seed robustness checks, block-diagonal sensitivity sweeps, rectangular weight boundary analyses, and generalizations to other Lie groups (Unitary group, Stiefel/Grassmann manifolds, hyperbolic spaces).

---

## Major Areas for Improvement (Critical Reviewer Lens)

### 1. Resolve the Logical Decoupling of Theorems 3.2 & 3.3 from the Performance Collapse
- **Issue:** The authors present Theorems 3.2 and 3.3 as the primary explanations for the "spectral balancing pitfall." However, their own results show that **RIMO-Schur-Balanced** and **RIMO-Complex-Balanced** (which are mathematically projection-free and completely immune to the distortions described in Theorems 3.2 and 3.3) **also suffer from the exact same performance collapse**.
- **Correction:** The authors must restructure their narrative. They should explicitly clarify that Theorems 3.2 and 3.3 address secondary *numerical and algebraic consistency* issues of SVD under projection, while the primary physical cause of the collapse is simply the **non-linear Cayley map noise propagation** (which is common to all methods). Currently, the elaborate SVD-specific proofs act as a theoretical distraction from the actual geometric bottleneck.

### 2. Generalize beyond Toy Benchmarks and Toy Models
- **Issue:** The entire empirical evaluation is restricted to toy benchmarks (Split-MNIST, Split-CIFAR-10) using toy architectures (a small MLP of $d=256$ and an extremely down-scaled ViT with $d=32$ and sequence size 16).
- **Correction:** To demonstrate the actual utility and generalizability of the findings, the authors must evaluate on standard, large-scale benchmarks. For example, they should show how RIMO-Pruned performs when merging standard ViT-B/16 models fine-tuned via LoRA on vision benchmarks, or standard language models on multi-task text benchmarks.

### 3. Discuss the Capacity and Expressiveness Bottleneck of Orthogonal Regularization
- **Issue:** The proposed framework depends on training experts with soft orthogonal regularization ($\lambda_{ortho}=2.0$) or hard manifold constraints to keep Procrustes residuals small. Forcing weight matrices to be orthogonal reduces network capacity and hampers optimization.
- **Correction:** The authors must explicitly discuss and evaluate this fundamental trade-off. They need to show if and how much single-task performance degrades when training with orthogonal regularization on complex tasks, and whether this degradation is worth the ability to perform manifold-level merging.

### 4. Overcome the Performance Gap with Task Arithmetic (TA)
- **Issue:** In all successful configurations, the simplest, training-free Euclidean Task Arithmetic consistently outperforms the proposed RIMO-Pruned framework (e.g., achieving $94.00\%$ vs. $91.49\%$ in the orthogonal regime). 
- **Correction:** The authors must identify settings where geometric manifold merging actually *outperforms* Euclidean Task Arithmetic. While they show a multi-expert scaling decay in Figure 5, they must demonstrate this on actual complex tasks with $N \ge 5$ experts, rather than relying on Split-MNIST with highly synthetic splits.

### 5. Fair Latency Comparisons
- **Issue:** Comparing single-threaded CPU sequential SVD and Schur to parallel PyTorch SVD on an NVIDIA H100 GPU is highly misleading and unfair.
- **Correction:** The latency benchmarks must compare parallel batched SVD on GPU with the complex Hermitian eigensolver on GPU under identical hardware constraints to prove genuine algorithmic speedup.

---

## Overall Presentation Quality
The overall presentation quality of this paper is **Excellent**.
- The narrative flow is exceptionally clear, logical, and structured.
- The pipeline schematic (Figure 1) is high quality, and the tables/plots are professionally formatted.
- The mathematical notation is precise and standard. 
- However, the authors must be careful not to oversell the "spectacular" performance of their methods given that they are consistently beaten by simple, baseline Euclidean Task Arithmetic.

---

## Potential Impact and Significance
- **Theoretical Significance: High.** For researchers in geometric deep learning, Lie group optimization, and manifold alignment, the theoretical insights regarding tangent-space spectral modifications and Cayley mapping noise propagation are highly valuable and deep. It establishes clear boundary conditions for translating Euclidean techniques to manifolds.
- **Practical Impact: Low/Negligible.** For the broader machine learning community and practitioners, the practical utility of the proposed method (RIMO-Pruned) is extremely low. It requires restrictive training-time constraints, introduces expensive $O(d^3)$ computations, and still achieves inferior multi-task accuracy compared to standard Task Arithmetic. Unless the authors can close the performance gap on large-scale models, the broader impact of this framework will remain highly specialized.
