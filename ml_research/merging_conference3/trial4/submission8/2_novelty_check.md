# Novelty Evaluation: CR-PolySACM

We evaluate the novelty and academic originality of the paper across three core dimensions: conceptual framing, theoretical formulation, and methodological design.

---

## 1. Conceptual Novelty: Quantization-Operator Overfitting
The concept of **Quantization-Operator Overfitting** is highly original and represents a significant conceptual advance in the understanding of test-time adaptive model merging (TTA). 

While prior TTA literature focuses on *transductive overfitting* (i.e., overfitting to local calibration stream statistics), this paper is the first to expose that unconstrained test-time coefficient optimization converges to extremely sharp local minima in continuous weight-space. These sharp minima yield high performance in FP32, but their representations are exceptionally fragile under downstream post-training quantization (PTQ) rounding noise.

By establishing a direct link between the loss landscape geometry of test-time adaptive merging and post-deployment hardware constraints, the paper shifts the paradigm of model merging from purely continuous, unconstrained optimization to hardware-aware, robust composition.

---

## 2. Theoretical Novelty: Task-Vector Norm Scale Pathology
The paper introduces a mathematically rigorous and empirically validated **task-vector norm scale pathology**. While sharpness-aware minimization (SAM) and Hessian trace regularization are well-established in standard neural network training, directly applying them to weight-space blending coefficients introduces a fatal, previously unrecognized scale-blindness.

The authors show that because weight-space perturbations are scaled directly by task-vector norms, a massive 50-fold discrepancy in layer-wise task-vector norms (e.g., intermediate blocks vs. final layer normalization) renders standard unnormalized flatness regularizers completely blind to low-norm layers. Furthermore, they mathematically prove that a naïve, scale-invariant normalization (dividing perturbations by task-vector norms) triggers a massive scale multiplier (>2500x) that causes immediate gradient explosion.

This theoretical insight successfully explains why previous attempts at unconstrained sharpness-aware model merging (such as unnormalized HessMerge) failed or degraded performance, turning a potential empirical bottleneck into a high-signal, rigorous mathematical contribution.

---

## 3. Methodological Novelty: Clipping-Regularized SACM and CR-PolySACM
The proposed **Clipping-Regularized Sharpness-Aware Minimization (CR-SACM)** is a simple, elegant, and highly effective methodological contribution. By clipping task-vector norms to a robust minimum threshold ($\beta = 0.10$), CR-SACM resolves the task-vector norm scale pathology without triggering singular gradient explosion. This allows the optimizer to successfully flatten low-norm layers under severe quantization rounding noise while maintaining numerical stability.

Furthermore, the integration of CR-SACM with a low-degree polynomial subspace constraint (**CR-PolySACM**) represents a novel, hybrid design. This framework achieves a highly effective "division of labor":
- **Global Structural Regularization:** Restricting the search space to a 12-dimensional polynomial manifold prevents overfitting to the calibration stream and shields against out-of-subspace noise.
- **Local Flatness Optimization:** Minimizing local loss sharpness within this well-conditioned manifold bounds the in-subspace sensitivity.

---

## 4. Distinction from Prior Work
The paper clearly distinguishes itself from and builds upon closely related works:
- **AdaMerging (Yang et al., 2023):** AdaMerging is unregularized and suffers from both transductive overfitting and catastrophic quantization collapse. CR-PolySACM introduces both subspace constraints and flatness regularization.
- **PolyMerge:** PolyMerge restricts the coefficient search space but does not optimize local flatness. CR-PolySACM introduces explicit sharpness-aware optimization within the polynomial manifold, achieving a +0.97% breakthrough under INT4 quantization.
- **HessMerge (Standard Curvature Minimization):** Standard unconstrained curvature minimization fails or degrades performance due to scale-blindness. The authors' upgraded HessMerge baseline with CR-SACM resolves this and consistently outperforms AdaMerging across all target precisions.
- **Static Merging (Task Arithmetic, TIES-Merging):** These methods use uniform, hand-tuned coefficients. CR-PolySACM dynamically optimizes coefficients at test-time while guaranteeing quantization robustness.
- **Extended Subspace Formulations (Appendix A.3):** The authors show that depth-dependent polynomial subspaces outperform other subspace parameterizations such as Random Projections (-1.90% drop) and Fourier-based Discrete Cosine Transform (DCT) subspaces, justifying the structural selection of PolyMerge.
