# Soundness and Methodology Evaluation: HyperMerge

## Clarity of the Description
The mathematical exposition in the paper is exceptionally clear and structured. The transitions between the Poincaré Ball ($\mathbb{D}_c^D$) and the Beltrami-Klein model ($\mathbb{K}_c^D$) are well-articulated, and the definitions of Möbius primitives, exponential/logarithmic mappings, and Einstein midpoints are mathematically precise.

## Appropriateness of Methods & Technical Flaws

Despite the clear writing, there are several critical methodological concerns and potential technical flaws:

1. **The Projection Distortion Contradiction:**
   - The authors analyze the projection distortion $\delta = \|\mathbf{z} - h\|_2$ and prove that under a moderate curvature $c = 0.1$ and low-rank updates ($E \ll 1$), the cubic dependence ensures that the distortion is negligible ($\delta \approx 10^{-4}$).
   - This proof presents a major contradiction: if the projection distortion is negligible ($\delta \approx 10^{-4}$), it means that the exponential map $\exp_{\mathbf{0}}^c(h)$ is almost perfectly linear ($\mathbf{z} \approx h$).
   - If the mapping is virtually linear, then the entire hyperbolic representation is almost identical to the flat Euclidean representation. Consequently, the non-linear ensembling in hyperbolic space (BKSB) is functionally equivalent to a simple Euclidean weighted average.
   - This raises a fundamental question: why introduce conformal factors, Klein mappings, Lorentz factors, and Einstein midpoints if the actual operations are operating in a regime where they are virtually linear? The added complexity provides no functional benefit, which is empirically confirmed by SABLE outperforming HyperMerge.

2. **Extreme Computational Complexity and Overhead:**
   - HyperMerge requires projecting activations to the Poincaré Ball, mapping them to Klein coordinates, computing Lorentz factors, computing weighted Einstein midpoints, mapping back to Poincaré, and then mapping back to Euclidean space.
   - Performing this multi-step non-linear transformation (involving tanh, artanh, square roots, divisions, and coordinate-wise operations) at multiple attention layers for every single query introduces a massive computational overhead.
   - For edge and smart-device deployments (which the paper explicitly targets in Section 1), this added latency and floating-point complexity completely defeats the purpose of "parameter-free" and "zero-latency" ensembling.

3. **Ad-Hoc Bounding Radius Clamping:**
   - HyperMerge relies on strict bounding radius clamping ($\min(1, \frac{1/\sqrt{c} - \epsilon}{\|\mathbf{x}\|_2})\mathbf{x}$) to prevent NaNs and numerical overflow near the boundary of the Poincaré Ball.
   - This clamping is an ad-hoc safety mechanism that artificially truncates activations. This truncation can lead to information loss and unpredictably alter expert representation scales, undermining the "mathematically optimal" claims of the Klein-space midpoint.

4. **Shallow Routing Fragility:**
   - Routing and OOD detection are performed entirely on Layer 0 embedding representations. As the authors admit in Section 4.4, in real-world settings with covariate shift, noise, or complex tasks, Layer 0 routing is highly fragile as it captures low-level features rather than robust semantic abstractions.

## Reproducibility
The authors provide a complete PyTorch implementation of the `HyperMergeModule` in Appendix B. This code is self-contained and appears correct, which supports reproducibility. However, the lack of actual experiments on physical foundation models (rather than a synthetic sandbox) makes reproducing real-world efficacy impossible.
