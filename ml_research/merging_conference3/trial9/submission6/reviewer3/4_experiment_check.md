# Experimental Evaluation Check

## Evaluation of the Experimental Setup
The experimental evaluation is performed entirely within a simulated **Analytical Coordinate Sandbox** ($L=14$, $D=192$, $d=8$, $K=4$). While this is a helpful environment for isolating coordinate propagation, analyzing eigenvalue decay, and testing manifold properties, it possesses **very low ecological validity**:
1. **Low Dimensionality and Capacity:** A dimensionality of $D=192$ and rank $d=8$ is extremely small. Modern representation spaces (e.g., in LLaMA-3 or large ViTs) have dimensions $D=4096$ or $D=8192$ and are highly dense.
2. **Synthetic Data:** The data and task expert bases are artificially generated. There are no actual natural language, computer vision, or reinforcement learning tasks evaluated.

## Absence of Real-World Datasets and Models
The paper completely lacks validation on real-world benchmark datasets (such as GLUE, GSM8K, or ImageNet) and actual physical model weights (such as LLaMA-3, Mistral-7B, or RoBERTa). 
Although the authors outline a detailed "integration roadmap" and "future work" in the conclusion (including Hugging Face PEFT integration, token-level serving, and hardware profiling), a submission to a top-tier machine learning conference is expected to **already have physical validation**. Evaluating solely on a synthetic, low-dimensional coordinate simulator is insufficient to prove that the proposed method is viable or effective for real-world deep learning workloads.

## Critical Gaps in Baselines
The paper compares C-Lie-MM against 11 baselines, yet it completely omits the most relevant, direct, and simple baseline: **Linear Blend + QR/SVD Orthonormalization**. 
- In this simple baseline, the expert bases are blended linearly ($V_{\text{flat}} = \sum \alpha_k V_k$), and the result is orthonormalized using standard QR decomposition or SVD ($V_{\text{orth}} = \text{orth}(V_{\text{flat}})$) during the forward pass.
- This baseline achieves the exact same geometric guarantees as C-Lie-MM (symmetric, idempotent, rank-$d$ projection operator, zero deviation from projection manifold) and completely prevents coordinate collapse.
- It is fully differentiable, requires no reference point $Y_0$, no offline pre-computation of logarithms, and no complex sign-tracking wrappers.
- Without a direct comparison to this straightforward, elegant linear-algebraic alternative, it is impossible to determine whether C-Lie-MM's heavy differential-geometric machinery (Grassmannian geodesics, logarithmic/exponential maps, tangent space projections) actually provides any performance benefits, or if it is simply a redundant over-complication of basic orthonormalization.

## Cushioned Residual Settings and Practical Necessity
In Section 4.3, the authors conduct an ablation study introducing residual connections and LayerNorm to the sandbox (Equation 25). 
- In this realistic, "cushioned" environment, traditional **Uniform Merging does not collapse**, but instead achieves **$51.90\% \pm 5.22\%$** accuracy (compared to the $25.00\%$ complete collapse in the unbuffered setting).
- SABLE (a flat baseline) achieves **$64.60\% \pm 1.80\%$**, which is quite high.
- While C-Lie-MM still achieves the highest performance ($72.30\% \pm 4.71\%$), this ablation directly proves that **catastrophic coordinate collapse is largely a product of the unbuffered, artificial feedforward setup** of the sandbox.
- In actual modern deep networks (which are heavily equipped with residual identity paths and LayerNorm), raw coordinate collapse is already mitigated by standard architectural elements. This significantly weakens the practical justification for introducing a highly complex, curved-manifold ensembling method.

## Complexity Spiraling: The Triton Kernel and Chebyshev Approximations
The authors highlight a custom **fused Triton GPU kernel** and a **6th-order Chebyshev polynomial approximation** designed to bypass online SVD during serving, dropping GPU latency from $0.51$ ms to $0.11$ ms.
From a minimalist's perspective, this is a clear sign of **complexity spiraling**:
1. The authors introduce a highly complex geometric ensembling method (C-Lie-MM).
2. They discover that C-Lie-MM is computationally heavy and suffers from high online latency because it requires sample-wise SVD operations on GPUs.
3. To fix this latency, they introduce *another* layer of high engineering complexity: a custom-written, low-level Triton GPU kernel that performs on-chip, register-level Chebyshev matrix polynomial expansions to approximate the exponential map.

This spiraling of complexity could be completely avoided by employing a simpler, more elegant method (like the "Linear Blend + QR" baseline) which is highly parallelizable and computationally cheap in standard PyTorch, without requiring custom low-level GPU programming or complex polynomial approximations.
