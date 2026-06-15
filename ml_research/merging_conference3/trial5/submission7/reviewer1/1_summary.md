# 1. Summary of the Paper

## Main Topic
The paper addresses the challenge of **test-time model merging (parameter-space model fusion)**, where multiple specialized, task-specific expert models (fine-tuned from a shared pre-trained base model) are combined into a single unified network at inference time without requiring the original training datasets. 

Specifically, the paper focuses on the **Overfitting-Optimizer Paradox** in active test-time adaptation (TTA) frameworks like AdaMerging. In these frameworks, the merging coefficients are optimized on the fly by minimizing the Shannon entropy of predictions on a small, unlabeled test-time calibration stream. Because this optimization is unconstrained and high-dimensional (e.g., layer-wise coefficients), it frequently leads to transductive overfitting to the local stream's noise, causing representational decay and joint multi-task performance collapse.

## Proposed Approach: Pruned Gradient Merging (PG-Merge)
To resolve this paradox under the guiding principle of Occam's razor, the authors propose **Pruned Gradient Merging (PG-Merge)**. Rather than introducing complex auxiliary loss terms, spatial regularizers, or rigid geometric trajectory constraints, PG-Merge restricts the optimization degrees of freedom during test-time adaptation using a simple, dynamic, and non-parametric sparse gradient mask:
1. **Gradient Computation:** Calculate the raw gradients of the test-time entropy loss with respect to all merging coefficients $\alpha_{k, l}$ on a local batch.
2. **Sparsity Masking:** Flatten the gradient vector, sort by absolute magnitude, and identify the threshold value representing the top-$p\%$ most sensitive coefficients (e.g., $p = 5\%$ or $15\%$). Create a binary mask that is $1$ for coordinates exceeding this threshold and $0$ otherwise.
3. **Pruned Update:** Update only the selected top-$p\%$ coefficients using a standard optimizer (e.g., Adam or SGD).
4. **Post-Update Parameter Projection:** To prevent advanced optimizers like Adam from updating masked coefficients via their historical momentum buffers, apply a strict projection step that explicitly resets all unselected coefficients to their pre-update values.
5. **Alternative SGD Formulation:** Discuss and advocate pairing PG-Merge with standard Stochastic Gradient Descent (SGD) without momentum, which naturally keeps unselected coefficients frozen without needing the post-update projection or maintaining running momentum states.

## Key Findings
- **Severe Parameter Interference:** Under a highly compact Vision Transformer backbone (`vit_tiny`), combining specialized experts statically using Uniform Merging (Task Arithmetic) drops the average joint accuracy to $62.16\%$ compared to the expert average ceiling of $78.08\%$.
- **Active Overfitting (The Paradox):** Unconstrained Online AdaMerging worsens performance further to $61.08\%$ ($1.08\%$ lower than static uniform averaging) due to transductive overfitting on the local $64$-sample calibration set.
- **Subspace Rigidity Failure:** PolyMerge, which restricts coefficients to a low-degree quadratic trajectory, fails catastrophically on MNIST ($13.48\%$), resulting in a joint average accuracy of only $46.97\%$.
- **Sparsity Sweet Spot:** Restricting active adaptation to an extremely sparse subset ($p=0.05$, updating only $\approx 3$ out of $56$ parameters) acts as an effective low-pass filter. PG-Merge with $p=0.05$ achieves a state-of-the-art joint mean of **$62.70\%$**, outperforming both unconstrained AdaMerging and the complex SOTA regularizer RegCalMerge ($62.35\%$).
- **Optimization Trajectory Dynamics:** Over 100 adaptation steps, PG-Merge demonstrates a stable reduction in entropy while consistently improving joint test accuracy, unlike AdaMerging, where entropy minimization correlates directly with joint accuracy decay.

## Explicitly Claimed Contributions (with Evidence)
1. **Exposes Redundant Complexity:** Argues and empirically demonstrates that complex, hyperparameter-heavy spatial regularizers (like RegCalMerge) and geometric restrictions (like PolyMerge) are unnecessary to solve the Overfitting-Optimizer Paradox (supported by comparative results in Table 1).
2. **Introduces PG-Merge:** Proposes a simple, training-free, non-parametric sparse gradient mask combined with strict post-update parameter projection (mathematically formulated in Section 3.4).
3. **Exhaustive Empirical Evaluation:** Evaluates a compact ViT backbone on 4 diverse vision datasets (MNIST, FashionMNIST, CIFAR-10, SVHN), showing that PG-Merge matches or exceeds SOTA regularizers with zero optimization or hyperparameter bloat (Table 1).
4. **Characterization of Sparsity:** Identifies a highly stable, task-agnostic "sparsity sweet spot" ($p=0.05$) via a comprehensive ablation study over $p \in \{0.05, 0.15, 0.30, 0.50, 0.75, 1.00\}$ (Table 2).
5. **Trajectory & Stability Analysis:** Analyzes prediction entropy and accuracy trajectories over 100 steps (Figure 3), along with an active mask stability analysis in Appendix B showing that the dynamic mask stabilizes into a consistent layer subset over time.
