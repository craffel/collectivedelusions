# 3. Soundness and Methodology Evaluation

## Clarity of the Description
The description of the methodology is exceptionally clear, rigorous, and logically structured. The paper provides complete mathematical formulations for:
- The block-partitioned parameter space and depth-dependent merging (Equation 1).
- The empirical base curvature estimation using FIM diagonal traces (Equation 2).
- The Shannon entropy optimization objective (Equation 3).
- The curvature-weighted spatial Total Variation (RCR-TV) and absolute coordinate anchoring penalties (Equations 4 and 5).
- The complete online test-time adaptation algorithm (Algorithm 1).
- The Gradient Norm Balancing (GNB) initialization and its scale-invariant re-parameterization guarantees (Equations 8-11).

Additionally, the authors provide a complete, standalone PyTorch recipe in the appendix (Section A.3) demonstrating how to compute base curvatures offline for a ViT and perform joint dual-regularized TTA online, which greatly enhances the clarity and readability of the proposed pipeline.

## Appropriateness of Methods
- **Diagonal FIM Trace Proxy**: Approximating the high-dimensional, anisotropic Fisher Information Matrix $G(\theta)$ with its diagonal trace $c_l$ for each layer is a computationally optimal choice. It reduces the metric tensor storage from $O(D^2)$ to $O(L)$, making second-order geometric optimization feasible for deep neural networks on edge devices.
- **Static Metric Approximation ($G(\theta_t) \approx G(\theta_0)$)**: Under small-step test-time adaptation, evaluating the metric tensor once offline at the pre-trained base model $\theta_0$ is a standard and mathematically sound local coordinate approximation (analogous to normal coordinates or exponential maps). 
- **Absolute Coordinate Anchoring**: The inclusion of $\gamma \|\boldsymbol{\lambda} - \boldsymbol{\lambda}_0\|_2^2$ is not just an empirical trick; the authors provide an elegant Taylor error bound (Equation 15) proving that absolute anchoring strictly bounds physical parameter drift, which in turn guarantees that the static metric approximation error remains bounded throughout adaptation.
- **Gradient Norm Balancing (GNB)**: Designing an unsupervised scale-invariant coordinate re-parameterization using maximum-entropy spectral perturbations ($\boldsymbol{\lambda}_{\text{pert}}$) is mathematically principled and highly appropriate for online deployment where validation labels are unavailable.

## Potential Technical Flaws and Methodological Concerns

An empirical reviewer must raise several critical methodological concerns regarding the assumptions and approximations made in the paper:

1. **Extreme Coarseness of Layer-wise Scalar Approximations**: Grouping millions of parameters in an entire transformer block into a single scalar curvature value $c_l$ is an extremely coarse-grained approximation. Within a single layer, attention projections ($\mathbf{W}_q, \mathbf{W}_k, \mathbf{W}_v, \mathbf{W}_o$) and feed-forward networks ($\mathbf{W}_{\text{gate}}, \mathbf{W}_{\text{up}}, \mathbf{W}_{\text{down}}$) perform fundamentally different functions and exhibit highly non-homogeneous sensitivities. While the authors discuss a "Tensor-wise" extension in Section 3.3, they show in Table 3 that the coarser layer-wise scheme actually performs *better* due to lower optimization variance. While this is an interesting finding, a single scalar representing an entire block still glosses over critical intra-layer parameter sensitivity variations.
2. **Calibration Dataset Size and Representation**: The authors use a tiny calibration set of $|D_{\text{cal}}| = 64$ samples to estimate base curvatures. For high-dimensional neural networks, a sample size of 64 is highly prone to sample variance. If the calibration set is slightly biased or noisy, the resulting normalized curvatures $\bar{c}_l$ could be highly distorted, leading to sub-optimal spatial barriers. The paper lacks a sensitivity analysis showing how robust the normalized curvatures are to the choice and size of the calibration dataset.
3. **Universality of the Simulator's Structural Assumptions**: The Coupled Model II Landscape simulator makes highly specific handcrafted assumptions about layer sensitivities: early layers ($l \le 3$) and late layers ($l \ge 10$) are highly sensitive, while middle layers ($4 \le l \le 9$) are robust. While this aligns with common deep learning intuition, it is a handcrafted assumption. The paper lacks evidence showing whether this specific "U-shaped" sensitivity profile is universally true for all deep architectures (e.g., modern MoE LLMs, deep CNNs, or multi-modal models).
4. **Online Curvature Re-estimation in OOD Streams**: For long-term non-stationary streams, the authors propose online curvature re-estimation (Triggered RCR-Merge) using a tiny batch of 16 samples. Computing weight gradients online on an incoming stream requires backpropagation, which re-introduces the exact computational and memory latencies that RCR-Merge is designed to avoid. Furthermore, a batch size of 16 is extremely noisy, and taking gradients of a shifting, unlabelled transductive stream could yield highly unstable and degenerate Fisher trace estimates.

## Reproducibility
The reproducibility of the paper is **excellent**. 
- The mathematical formulations are complete and self-contained.
- The simulator specifications (layer sensitivities, covariance coupling, noise offsets, optimization rates, and seeds) are detailed precisely in Section 4.1.
- The authors provide a complete, turnkey PyTorch/Hugging Face code recipe in the appendix (Section A.3) covering curvature estimation, joint TTA losses, GNB initialization, and optimization loops.
- All baseline comparisons and evaluations are described clearly, making it straightforward for an expert reader to reproduce both the synthetic simulations and the real-world pilot studies.
