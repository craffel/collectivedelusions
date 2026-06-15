# Soundness and Methodology Check

## 1. Technical Soundness
The mathematical formulation of PG-Merge is clear, precise, and logically sound. 
*   **The Overfitting-Optimizer Paradox:** The explanation of how unconstrained optimization over high-dimensional layer-wise coefficients (e.g., AdaMerging) leads to transductive overfitting is highly intuitive. In low-sample, unlabeled regimes, minimizing prediction entropy can lead to confidence collapse (where the model predicts a single class with 100% confidence), resulting in representational decay. PG-Merge directly targets this by restricting the active update coordinates.
*   **The Strict Parameter Freezing / Post-Update Projection:** The paper's inclusion of Equation 15 (strict post-update parameter projection) is technically rigorous. In advanced optimizers like Adam, historical momentum buffers can cause non-zero updates even when the current gradient is zero. The projection step successfully overwrites these drifting parameters, ensuring that $(100-p)\%$ of the parameters are kept mathematically frozen.

## 2. Potential Technical Flaws / Nuances

### A. Optimizer Momentum Buffer Decay (State Mismatch)
While the post-update projection step (Equation 15) successfully keeps unselected parameters frozen in the weight space, it introduces an internal state mismatch within the Adam optimizer:
*   **The Issue:** The Adam optimizer's running moment vectors ($m$ and $v$) are still updated with zero gradients for the masked coordinates at each step.
*   **The Consequence:** Over multiple adaptation steps, the momentum of frozen parameters will progressively decay to zero. When a parameter eventually becomes active again (due to a high gradient magnitude), its update step will be severely dampened because its momentum buffer has been depleted.
*   **Constructive Suggestion:** A more theoretically sound approach would be to freeze the optimizer state update entirely for the masked coordinates, rather than feeding zero gradients and then overwriting the weights. Alternatively, using standard SGD without momentum would completely bypass this issue and make the method even simpler, removing the need for the projection step altogether.

### B. Gradient Noise on Tiny Batches
The sparse gradient mask $M_{k, l}^{(t)}$ is computed dynamically at each step based on gradients from a small calibration/test batch ($B=64$).
*   **The Issue:** Gradients on low-sample regimes are highly noisy and exhibit high variance. 
*   **Mask Instability:** The set of selected coefficients may fluctuate wildly between steps. While the paper describes this as "allowing the network to naturally select different active routing paths," it remains unanalyzed. Is this chaotic switching beneficial, or would a smoother, temporally-consistent mask be more robust? Analyzing mask stability would be of great value.

## 3. Reproducibility
The methodology is exceptionally reproducible. The equations are complete, self-contained, and clear. Since the method does not require complex training procedures, a practitioner can easily implement PG-Merge in a few lines of code (e.g., sorting absolute values of `alpha.grad`, constructing a binary mask, and projecting the parameters post-update).
