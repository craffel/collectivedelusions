# 3. Soundness and Methodology Review

## 3.1 Mathematical Soundness and Theoretical Foundations
The paper displays an exceptionally high level of mathematical rigor. The key theoretical claims are clearly stated and rigorously derived:

1. **Hessian Projection Derivation**: The derivation showing that the coefficient-space Hessian $H_{\Lambda}$ is the projection of the weight-space Hessian $H_{\theta}$ onto the subspace spanned by the task vectors $T$:
   $$H^l_{\Lambda} = (T^l)^T H^l_{\theta} T^l$$
   is elegant and correct. The subsequent bounding of the maximum eigenvalue and trace of $H^l_{\Lambda}$ via the maximum eigenvalue of $H^l_{\theta}$ and the task vector norms is mathematically sound and provides a solid physical basis for the observed empirical results.
   
2. **Quantized Landscape Characterization**: Rather than ignoring the mathematical difficulties introduced by the non-differentiable `round` operator, the authors explicitly characterize the quantized loss landscape as piecewise constant with step-like jump discontinuities. They correctly note that a standard continuous Taylor expansion does not strictly hold everywhere, and instead interpret the expected entropy change under Gaussian perturbations as a measure of discretization noise tolerance and local plateau width. This is a highly mature and mathematically honest treatment of quantization.

3. **Straight-Through Estimator (STE) Analysis**: The authors provide a detailed discussion of the gradient mismatch error introduced by the STE and explain why optimization of the blending coefficients remains highly stable (low-dimensional parameter bottleneck of only 56 parameters, combined with learning rate tuning).

## 3.2 Methodological Implementation
The methodology is exceptionally clean and well-structured:
- **Vision Transformer Backbone**: Partitioning the ViT-Tiny into $L=14$ layer groups (Embedding, 12 Encoder Blocks, Final LayerNorm) and optimizing a layer-by-layer $14 \times 4$ coefficient matrix is a highly effective parameterization that balances representation capacity and optimization efficiency.
- **Classification Heads Treatment**: Keeping task-specific classification heads in full FP32 and routing them separately is a crucial methodological detail. It avoids shape mismatches (since MNIST has 10 classes and CIFAR-10 has 10 but different dimensions, and they are visual classification tasks) and ensures parameter fusion is restricted to the shared representation backbone.
- **Independent Clipping bounds**: The decision to use independent clipping bounds $[0, 1]$ rather than a normalized convex Softmax combination is well-reasoned and empirically validated, allowing different layers to scale task vectors independently to balance representation density.

## 3.3 Potential Concerns and Clarifications
While the methodology is sound, there are a few minor areas that could benefit from further clarification:
1. **Calibration Batch Size Sensitivity**: The authors state that calibration batch sizes $N \in \{16, 32, 64, 128\}$ yield equivalent performance. While this is encouraging for edge deployment, a more detailed discussion of how calibration data distribution shifts (e.g., unbalanced classes or out-of-distribution inputs) might affect the joint entropy objective would be valuable.
2. **Hessian Trace Proxy Assumptions**: The direct measurement of weight-space flatness using isotropic Gaussian perturbations as a proxy for the Hessian trace (Section 4.9) assumes that the loss function is locally quadratic. While this is standard in the flatness literature, the paper could briefly mention that at larger perturbation scales, higher-order terms may influence the empirical loss increase.

## 3.4 Soundness Rating
- **Rating**: **Excellent**
- **Justification**: The paper is mathematically rigorous, methodologically thorough, and shows deep intellectual honesty in addressing the non-differentiable nature of the quantized parameter space. The derivations are correct, the pipeline is clearly structured, and potential optimization hazards (such as STE gradient mismatch, class/task collapse, and parameter scale explosion) are explicitly addressed and resolved.
