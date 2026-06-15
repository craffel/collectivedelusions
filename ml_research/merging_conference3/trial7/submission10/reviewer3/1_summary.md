# Paper Summary: SPS-ZCA

## Main Topic and Objective
The paper addresses the challenge of serving multiple task-specific expert adapters (specifically Low-Rank Adaptations, or LoRA) from a shared pre-trained backbone model on resource-constrained edge devices. The main goal is to handle highly heterogeneous, mixed-task streaming workloads simultaneously without:
1. **Heterogeneity Collapse:** The degradation of accuracy when expert weights are statically merged (e.g., via Task Arithmetic or TIES-Merging) to compromise across conflicting tasks.
2. **Multi-Pass Latency Penalty:** The severe linear latency overhead introduced by prior dynamic systems like Micro-Batch Homogenization (MBH), which partition incoming batches on-the-fly and run up to $K$ sequential forward passes of the heavy base model backbone.

## Proposed Approach
The authors propose **SPS-ZCA** (Single-Pass Sample-Wise Routing with Zero-Shot Centroid Alignment), a completely training-free dynamic model-merging framework. Its core architectural pipeline consists of:
1. **Zero-Shot Centroid Pre-computation:** During an offline calibration phase, a robust task centroid $\mu_k^{(3)}$ is computed for each task expert from a tiny calibration split ($\mathcal{C}_k$ with 64 samples) in the shared early representation space (Layer 3) of the pre-trained backbone.
2. **Zero-Shot Centroid Alignment (ZCA) Routing:** During online inference, the shared, adapter-free early-stage layers (Layers 1–3) are executed, and the cosine similarity between the input's Layer 3 representation $h_b^{(3)}$ and the pre-computed task centroids $\mu_k^{(3)}$ is used as a coordinate vector. These coordinates are converted into sample-wise routing coefficients $\alpha_{k, b}$ via a temperature-scaled Softmax.
3. **Single-Pass Activation-Space Dynamic Blending (SPS):** Instead of split-batch sequential dispatch, SPS executes the base model backbone and its lightweight adapters in a single parallel pass, blending the output activations of the expert adapters layer-wise on-the-fly using the sample-wise coefficients $\alpha_{k, b}$.
4. **Robustness and Calibration Enhancements:**
   - **Unit-Norm Calibration (UNC):** Leverages cosine similarity to project representations and centroids onto a unit hypersphere, making the routing coordinates scale-invariant and neutralizing cross-expert scale imbalances.
   - **Intra-Task Dispersion Calibration (IDC):** Divides the similarity coordinates by the expected in-distribution similarity scale $s_k$ to neutralize asymmetric manifold dispersion.
   - **Coordinate GMM OOD Rejection:** Fits a diagonal Gaussian Mixture Model over calibration coordinates in the low-dimensional coordinate space $\mathbb{R}^K$ to filter out out-of-distribution (OOD) queries prior to adapter blending.

## Key Findings and Empirical Results
- **Expert Ceiling Recovery:** Under homogeneous and heterogeneous streaming configurations ($B=256$) over a 4-task suite (MNIST, F-MNIST, CIFAR-10, SVHN), SPS-ZCA achieves a Joint Mean accuracy of **79.80%**, recovering **100.0%** of the isolated Expert Ceiling and outperforming prior non-parametric SOTA methods (PFSR+MBH) by **+3.66%** absolute accuracy with zero trainable parameters.
- **Physical Wall-Clock Speedup:** Under small batch scales ($B=16$), the Vectorized Scatter-Gather (SPS-VSG) implementation achieves a verified **1.17$\times$ wall-clock speedup** out of the box in standard, uncompiled PyTorch (16.63 ms vs. MBH's 19.42 ms) on CPU.
- **Projected Analytical Speedup:** Under high batch scales ($B=256$), the authors honestly characterize the "serving gap" caused by framework overhead and sequential compute scaling. They project that a hardware-software co-designed compiler-fused loop layout can slash projected analytical costs by **3.90$\times$** (from 776.4 ms to 199.0 ms) on heterogeneous streams, preserving a flat execution profile.
- **Robustness metrics:** UNC restores accuracy from artificial scale-imbalance degradation, IDC neutralizes manifold variance to maintain balanced routing, and the coordinate GMM density estimator achieves a **95.2%** true positive OOD rejection rate at a **4.3%** false positive rate.

## Explicitly Claimed Contributions (with Evidence)
1. **Single-Pass Activation-Space Dynamic Blending (SPS):** Formulation of layer-wise activation blending that scales at constant $O(1)$ backbone execution latency with respect to task heterogeneity. Evidence is provided through mathematical formulation (Equation 5) and latency sweeps (Table 2).
2. **Zero-Shot Centroid Alignment (ZCA):** Bypassing noisy classification heads and resolving the temporal routing paradox by routing at Layer 3 of the backbone. Supported by Fisher Separability Criterion (FSC) profiling (FSC is 47.50 at Layer 3 vs. 0.18 at Layer 0) and zero task-routing errors over real image datasets.
3. **Calibration & Robustness Operators:** UNC and IDC normalizations to handle representation norm mismatches and asymmetric manifold dispersion, validated through targeted ablation sweeps (Ablations C and D).
4. **Coordinate-Space GMM Estimator:** High-precision OOD detection operating in the low-dimensional coordinate space, verified via ROC curves (Figure 3) and threshold sweeps (Table 3).
5. **Characterization of the "Serving Gap":** Honest profiling of physical PyTorch execution overheads versus idealized analytical FLOP limits, accompanied by co-designed fused compiled loops as an actionable roadmap for practitioners.
