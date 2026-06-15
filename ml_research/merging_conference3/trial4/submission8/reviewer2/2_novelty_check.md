# Evaluation Component 2: Novelty Check

## Key Novel Aspects
1. **Test-Time Flatness-Quantization Connection in Model Merging:** Prior works connecting local landscape flatness to post-training quantization (PTQ) robustness (e.g., SAM) focus on the training phase of individual neural networks. This paper is the first to establish and analyze this geometric relationship for *test-time adaptive model merging* in the blending coefficient space.
2. **Identification of the Task-Vector Norm Scale Pathology:** The paper uncovers a physical scale discrepancy in deep neural networks where task-vector norms vary by over $50\times$ between layer groups (e.g., intermediate blocks vs. final layer normalization). It rigorously demonstrates how this discrepancy causes unnormalized coefficient-space flatness regularizers to suffer from "scale-blindness" (ignoring low-norm layers) while unmitigated normalization causes gradient explosion.
3. **Clipping-Regularized Scale Balancing (CR-SACM):** The introduction of a mathematically robust clipping threshold ($\beta$) to balance the scale of adversarial perturbations across layer groups is highly practical and clever, solving both scale-blindness and gradient explosion simultaneously.
4. **Second-Order Loss Decomposition in Subspace:** Equation 11 provides an elegant second-order Taylor expansion that decomposes the quantization-induced loss gap into:
   - First-order out-of-subspace noise
   - Second-order in-subspace sensitivity (which test-time coefficient optimization can actually minimize)
   - Second-order out-of-subspace noise

## Delta from Prior Work
- **From Unregularized TTA (AdaMerging):** AdaMerging and its variants optimize layer-wise coefficients purely on unsupervised entropy, ignoring local flatness and resulting in sharp minima that are highly fragile under weight-space noise.
- **From Parameter-Smoothing TTA (RegCalMerge / Elastic Spatial Regularization):** RegCalMerge applies a Total Variation penalty over adjacent layer coefficients to smooth transitions. While this mitigates some overfitting, it does not explicitly minimize landscape flatness or address the norm scale discrepancies across layers.
- **From Subspace-Constrained TTA (PolyMerge):** PolyMerge restricts coefficients to a depth-dependent polynomial subspace to prevent overfitting. CR-PolySACM builds directly on top of PolyMerge, demonstrating that subspace constraints alone are insufficient under aggressive quantization noise, and introducing local flatness optimization *within* that polynomial manifold.
- **From Second-Order Flatness regularizers (HessMerge):** HessMerge-Exact requires calculating the exact Hessian trace, necessitating $O(L \times K)$ double-backward passes (56 passes in this setup) which is computationally prohibitive for edge devices ($82.35$ seconds for $40$ steps). The proposed SACM formulation reduces this to a first-order minimax approximation requiring only two forward-backward passes ($1.56$ seconds for $40$ steps, a $52.8\times$ speedup).

## Characterization of Novelty
The novelty of this paper is **significant**. While the core building blocks—polynomial subspace parameterization (PolyMerge) and sharpness-aware minimization (SAM)—exist in separate contexts, their integration and the subsequent identification of the task-vector norm scale pathology represent a highly original contribution. The paper goes beyond a simple "A + B" combination by providing a rigorous mathematical foundation (the quadratic noise decomposition) and a novel, stable perturbation mechanism (CR-SACM) that addresses a physical, structural property of deep neural networks.
