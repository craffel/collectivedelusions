# Novelty and Delta Analysis

## Key Novel Aspects and the 'Delta' from Prior Work
The core novelty of this work lies in identifying, mathematically characterizing, and resolving the intersection of **test-time adaptive model merging** and **downstream post-training quantization (PTQ)**.

1. **Quantization-Operator Overfitting in Model Merging:**
   - *Prior Work:* Standard adaptive model merging methods (such as AdaMerging) optimize layer-wise blending coefficients solely in the continuous floating-point space, completely ignoring downstream deployment constraints.
   - *Delta:* This paper is the first to identify that unconstrained test-time adaptation on extremely small calibration streams (e.g., $N=64$) overfits to the local batch statistics, driving the model's weights into sharp, fragile local minima that catastrophically collapse under the rounding noise of downstream quantization.

2. **The Task-Vector Norm Scale Pathology:**
   - *Prior Work:* Standard Sharpness-Aware Minimization (SAM) and its variants apply uniform perturbations in parameter space. In model merging, baseline methods like HessMerge might apply uniform perturbations in the blending coefficient space.
   - *Delta:* The authors discover that applying uniform perturbations in the coefficient space leads to a major physical scale mismatch in weight space. Because layer-wise task-vector norms vary by over $50\times$, a uniform coefficient perturbation scales the weight-space perturbation of the final layer normalization by $1800\times$ less than the intermediate blocks. This "scale-blindness" is a newly identified pathology that renders standard sharpness optimization completely ineffective in model merging.

3. **Clipping-Regularized Sharpness-Aware Minimization (CR-SACM):**
   - *Prior Work:* Scale-invariant optimizers typically normalize gradients by their norms, but this causes singular division and division-by-zero or gradient explosion when norms approach zero.
   - *Delta:* The paper introduces CR-SACM, an elegant, mathematically stable scale-invariant regularizer that clips the normalization factor to a robust minimum threshold $\beta = 0.10$. This provides a stable, uniform weight-space perturbation across all layers, resolving scale-blindness without triggering singular gradient explosion in low-norm layers like Layer 13.

4. **The Integrated Subspace-Flatness Paradigm (CR-PolySACM):**
   - *Prior Work:* PolyMerge utilizes a continuous polynomial depth constraint to reduce optimization variables. However, it operates purely in the continuous space and has no sharpness-aware mechanism.
   - *Delta:* CR-PolySACM is a highly original synthesis that uses PolyMerge's global subspace constraint to shield the model from out-of-subspace quantization noise ($\delta_{\perp}$), while using CR-SACM to explicitly flatten the local landscape in-subspace, providing complete protection.

---

## Characterization of Novelty
The novelty of this paper is **significant and highly principled**. 
Rather than proposing a pure heuristic or simple empirical combination of existing techniques, the paper presents a deep, mathematically rigorous framework:
- It derives the exact relationship between weight-space quantization noise, low-dimensional polynomial coefficient projection, and the multi-task loss gap via second-order Taylor expansion (Eq. 12).
- It mathematically derives why unconstrained sharpness regularizers suffer from scale-blindness (the $(V_k^l)^2$ dependency) and provides a closed-form, stable solution (CR-SACM).
- This level of theoretical grounding, combined with empirical validation, represents a substantial advance over the existing model merging literature, which is heavily empirical and often lacks a formal understanding of loss landscapes.
