# 3. Soundness and Methodology Evaluation

This section evaluates the technical soundness, appropriateness of the methodology, clarity of descriptions, potential technical flaws, and reproducibility of the submission.

---

## 1. Clarity of Description
The paper is exceptionally well-written and structured. The mathematical formulations of the system model, CGHR Pathways, and MBH partitioning are highly clear and rigorous:
- **CGHR Pathways:** Equation 1 to 5 provide clear definitions of the parametric gating and parameter-free cosine similarity projections.
- **Micro-Batch Homogenization:** Section 3.5 clearly defines the partitioning mechanics, and Appendix C (Algorithmic Flow of MBH) provides an exhaustive step-by-step trace of its execution.
- **Theoretical Appendices:** The proof of the extreme value normalization factor in Appendix A and the UNC-PFSR Equivalence Theorem in Appendix F are mathematically rigorous and easy to follow.

---

## 2. Appropriateness of Methods
The proposed solutions, CGHR and MBH, are highly appropriate for the identified failure modes:
- Using a zero-shot projection-based fallback (PFSR) to stabilize a data-scarce parametric router is a sound engineering approach to mitigate transductive overfitting.
- Isolating heterogeneous batches into task-homogeneous micro-batches (MBH) directly addresses the root cause of "heterogeneity collapse" (which is the representation-smoothing effect of mixing task gradients/logits).
- The inclusion of systems-level optimizations such as *Homogeneity Bypass*, *Fusion Weight Caching*, and *Segmented-BGEMM Triton kernels* shows a sophisticated, deployment-aware understanding of ML serving engines.

---

## 3. Potential Technical Flaws and Critical Limitations

While the methodology is sound, a critical analysis reveals several major technical limitations and structural shortcuts that restrict the generalizability of the findings:

### A. Severe Structural Asymmetry in the Sandbox Geometry
As transparently discussed in Section 3.6, there is a fundamental structural asymmetry in how the input features are handled for Pathway A versus Pathway B in the main sandbox evaluations:
- **Pathway A (Parametric Gating):** Takes the global $D$-dimensional feature vector $z_b \in \mathbb{R}^{192}$ as input and must learn to map features to task outputs while filtering out high-variance Gaussian noise in the other $K-1$ non-active block coordinates. Under scarce calibration data ($N \le 64$), this high-dimensional noise leads to overfitting, artificially degrading its performance (e.g., $54.80\%$ on CIFAR-10 in Table 1).
- **Pathway B (PFSR):** Takes local, block-specific representations $z_{k, b} \in \mathbb{R}^{48}$ as input. This means the parameter-free router is pre-provided with privileged architectural knowledge regarding the exact coordinate boundaries of each task.

This structural asymmetry heavily biases the main comparative experiments (Table 1) in favor of the parameter-free fallback (PFSR). If the parametric router were given the same localized coordinates, or if PFSR had to operate on the global unpartitioned vector, the baseline results would look completely different (as verified in Appendix F, standard Global PFSR without calibration collapses to $30.00\%$ accuracy under coordinate noise). While the authors propose Inference-Time block-wise Unit-Norm Calibration (IT-UNC) to resolve this, the main results in the body of the paper still rely on this privileged coordinate boundary assumption.

### B. Reliance on an Highly Idealized, Disjoint Coordinate Space
The "Isolating Coordinate Sandbox" assumes that task representations reside in completely disjoint, non-overlapping coordinate blocks. This represents an idealized setup where expert representation coordinates are orthogonal and completely decoupled.
- **Real-World Mismatch:** In actual deep neural networks (such as pre-trained Transformers), features are highly overlapping, non-orthogonal, and shared across tasks. In this scenario, standard coordinate-slicing breaks down.
- **SVD Projection Limitation:** To address this, the authors propose an elegant mathematical extension using SVD subspace projection operators (Appendix H). However, their empirical validation (Table 6) is restricted to another synthetic over-parameterized space with random orthonormal bases. The effectiveness of this projection-based gating in actual pre-trained Transformers and LoRA adapters remains qualitatively outlined (Section 9.2) but empirically unverified.

### C. Artificial Baseline Reductions in the Sandbox Setup
As the authors clarify in Section 4.1:
- Advanced static model merging methods (such as Task Arithmetic, TIES-Merging, and DARE) mathematically reduce exactly to Uniform Merging in this sandbox.
- Because different task experts reside in disjoint blocks of coordinate dimensions (non-overlapping weights), there are no conflicting parameter deltas across different experts. Consequently, sign-agreement checks (TIES-Merging) or random weight dropping (DARE) do not prune or modify any overlapping coordinate conflicts.

This coordinate-disjoint setup trivializes weight-space interference, which is the primary challenge static merging methods are designed to solve. By evaluating only in this environment, the authors cannot demonstrate whether their dynamic routing framework is superior to advanced static merging methods on a more realistic, overlapping weight space.

### D. Overconfidence and Gating Calibration Risk
Parametric routers are prone to overconfident incorrect predictions, especially on out-of-distribution (OOD) data. If the parametric router outputs an overconfident incorrect label, CGHR's confidence score will exceed the threshold ($\mathcal{C} \ge \gamma_{\text{conf}}$), bypassing the robust PFSR fallback and executing suboptimal expert merging. While the authors propose post-hoc temperature scaling, Platt scaling, and conformal prediction guarantees in Appendix I, these mitigations are discussed qualitatively but lack empirical evaluation under realistic overconfident regimes.

---

## 4. Reproducibility
The reproducibility of the synthetic sandbox experiments is exceptionally high:
- **Hyperparameter Disclosure:** Table 2 in Appendix B list precise experimental parameters (optimizer, learning rate, weight decay, epochs, dimensions, noise scales).
- **Algorithmic Flow:** Appendix C outlines the MBH execution flow step-by-step.
- **Statistical Rigor:** All experiments are averaged across 5 independent seeds, reporting both mean and standard deviations.

However, the reliance on a custom, synthetic simulation environment rather than standard, public multi-task benchmarks (such as GLUE or DomainNet) means that reproducing the *systems-level benefits* and *routing performance* on real-world deep learning pipelines requires a significant engineering effort from future researchers to implement the SVD scaling roadmap from scratch.
