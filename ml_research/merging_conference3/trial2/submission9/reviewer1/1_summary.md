# Evaluation Part 1: Summary of the Paper

## Main Topic and Overview
The paper presents a critical, deconstructive audit of test-time adaptive model merging, focusing on the trade-offs between parameter frugality and multi-task capability. It investigates the extreme lower bound of parameter-space blending, where the trainable parameters are reduced to just $K$ global task-specific scalars (one for each of the $K$ fine-tuned expert models). To isolate these architectural boundaries, the authors introduce **Barycentric Proximity-Anchored Merging (BPAM)**, a simple, low-parameter framework. 

## Proposed Method
BPAM utilizes:
1. **Convex Barycentric Simplex Projection:** Restricts the merging coefficients to a convex barycentric simplex to mathematically bound the weight norms and prevent activation scale distortion without needing deep normalizing flows.
2. **Mean-Field Proximity Regularization:** A closed-form $\ell_2$ penalty that pulls task coefficients toward a uniform barycentric centroid to stabilize optimization and prevent transductive overfitting.
3. **Teacher-Guided Adaptation:** Optimizes the coefficients using joint KL-divergence between the merged model predictions and individual expert teacher predictions on unlabeled target streams.

The authors evaluate BPAM under three spatial/head configurations:
- **BPAM-Restricted:** Merging is restricted to a single projection layer (`model.visual.proj`) with frozen classification heads (8 parameters total).
- **BPAM-Static:** Whole-model merging across the image encoder with frozen classification heads (8 parameters total).
- **BPAM-Full:** Whole-model merging with concurrent classification head adaptation (8 task scalars + 388K classifier parameters).

## Explicitly Claimed Contributions and Evidence
1. **Critical Deconstructive Audit:** The paper maps the boundaries where layer-wise degrees of freedom (like in AdaMerging) or high-capacity mapping models (like FoldMerge/SyMerge) become essential for weight blending. 
   - *Evidence:* Table 1 shows that BPAM-Static (69.21% average accuracy) lags behind AdaMerging (83.17%) and FoldMerge (83.56%) under frozen heads by a wide margin ($\sim$14%).
2. **Convex Simplex Formulation & Mean-Field Proximity Penalty:** A scale-preserving formulation and geometric regularizer to stabilize test-time adaptation.
   - *Evidence:* Table 3 and Table 4 show how the proximity penalty protects against extreme low-data instability (5 samples per class).
3. **The 0-Weight Performance Mystery Resolution:** Explanation of how the model achieves high accuracy on tasks whose expert weight coefficients converged to exactly zero (SVHN/MNIST) via representation similarity.
   - *Evidence:* Centered Kernel Alignment (CKA) similarity scores are reported (e.g., MNIST CKA is 0.5000 vs 0.3754 for the base model).
4. **Asymmetric Co-adaptation Schedule Capability:** Addressing the scale imbalance between the 8 weight scalars and 388K head parameters using separate learning rates.
   - *Evidence:* Included as a codebase feature and discussed in Section 4.4.
