# Soundness and Methodology Evaluation

## Clarity of Description
The manuscript is exceptionally well-written, with highly structured, clear, and mathematically rigorous descriptions of its equations, router architectures, and spectral diagnostic metrics. Hyperparameters, dataset splits, and optimization pipelines are described in sufficient detail to ensure high reproducibility.

## Appropriateness of Methods & Technical Flaws
While the paper presents a highly polished and mathematically elegant narrative, a deep, skeptical analysis reveals several **critical methodological and conceptual flaws**:

### 1. The Mathematical Flaw of the SVD Collinearity Ratio on Batch-Averaged Matrix $A$
The authors define the Batch-Averaged Layer-wise Coefficient Matrix $A \in \mathbb{R}^{L \times K}$ by averaging sample-specific coefficients over the entire test batch:
$$A_{l, k} = \frac{1}{B} \sum_{b=1}^B \lambda_{l, k}(x_b)$$
They then perform SVD on $A$ to compute $\rho_{collinear}$. However, under a balanced, heterogeneous test stream containing an equal mixture of all task samples, **any well-calibrated, symmetric dynamic router must mathematically converge to a constant uniform matrix as the batch size $B \to \infty$**:
$$\lim_{B \to \infty} A_{l, k} = \frac{1}{K} \quad \forall l, k$$
Because a constant matrix has exactly rank 1, **its SVD Collinearity Ratio $\rho_{collinear}$ must mathematically converge to 1.0 (perfect collapse), regardless of how dynamic, sample-specific, and non-collinear the routing actually is for individual samples!**
* **The Paradox:** A perfect, 100%-accurate sample-specific router on a balanced mixed batch will yield a collinearity ratio of exactly 1.0. Conversely, a lower collinearity ratio ($\rho_{collinear} < 1.0$) can only be obtained if the batch average is non-uniform across layers. This occurs if and only if the router has a *systematic, input-independent layer-wise bias* (e.g., early layers always route to Expert 0, deep layers always route to Expert 1, regardless of the input).
* **The Conclusion:** Therefore, the SVD Collinearity Ratio on the batch-averaged matrix is mathematically incapable of measuring dynamic routing capacity. A lower collinearity ratio actually indicates a *static layer-wise bias* rather than true dynamic, sample-specific specialization. This completely invalidates the authors' core spectral diagnostic method.

### 2. The "Batch-Averaged Multi-Task Inference Paradox" Undermines the Paradigm
In Section 3.5, the authors identify a devastating systems-level paradox: dynamic weight-space model merging on mixed batches collapses back to a static uniform compromise due to batch averaging. On homogeneous batches, it is logically redundant because one can simply route directly to the single-task expert (Oracle) and get 99% accuracy with zero representational damage.
* **The Critic's View:** If the authors' own analysis proves that dynamic model merging is either redundant (on homogeneous batches) or degraded to static merging (on mixed batches), the entire research paradigm of dynamic full-parameter model merging is functionally dead on arrival. 
* **The Proposed Remedies:** The authors propose restricting merging to low-rank PEFT/LoRA modules and running sample-specific passes (LR-SFP). However, if we keep LoRA adapters separate and run sample-specific low-rank updates (e.g., using frameworks like S-LoRA or Punica), we run separate adapter paths in parallel. **No weight-space merging is performed.** Thus, the ultimate solution to the paradox actually makes "model merging" entirely obsolete, replacing it with multi-adapter execution.

### 3. Exposing the "Routing Noise Hypothesis" on Destroyed Representations
On the DeepMLP-12 backbone under the Cross-Domain suite, the merged model achieves only **16.15% accuracy**, which is barely above the **12.5% random guessing threshold** for the 8-class subset. The authors honest-to-a-fault term this the "Random Guessing Barrier" and declare that "full-parameter linear interpolation of deep, fully connected layers under multi-task conflict is fundamentally a failed paradigm."
* **The Critic's View:** If the merged model is completely non-functional (scoring ~16% accuracy, close to random guessing), then the learned routing coefficients are essentially routing over a destroyed, chaotic parameter space. Is a spectral analysis of a garbage, non-functional model's routing weights actually meaningful? The router is being trained on a scarce 128-sample calibration split to minimize classification loss on a model whose weights have been blended, destroying its internal coordinate projections. The optimizer is merely learning arbitrary, high-frequency noisy adjustments to squeeze a tiny 3.65% bump above random guessing. Therefore, drawing major architectural conclusions about "depth-specialized routing policies" emerging as "a semantic necessity" is highly suspect when the underlying model is fundamentally broken.

## Reproducibility
The methodology is highly reproducible. The authors provide full details of the expert training parameters, calibration steps, and model architectures. However, reproducibility of a flawed or toy sandbox does not compensate for its underlying lack of technical soundness.
