# Soundness and Methodology Evaluation

This paper exhibits an exceptionally high standard of technical soundness and methodological rigor. Every theoretical claim is supported by either a direct mathematical derivation, a highly controlled empirical stress test, or both.

### 1. Theoretical Soundness and Correctness of Proofs
The authors provide clean, unambiguous, and correct mathematical derivations for the key structural behaviors they identify:
- **Layer-Averaging Collapse Proof**: The derivation in Equations 11-13 mathematically proves that when layer-wise routing coefficients are averaged across layers ($L$) and batches ($B$) at deployment, the 14-layer linear routing network mathematically collapses to a single global linear router. This is a crucial, rigorous proof that exposes architectural redundancy in prior works.
- **Gradient Damping Derivation**: The chain-rule analysis in Equation 14 correctly derives the $1/L$ gradient damping factor, providing a sound mathematical explanation for why training the over-parameterized multi-layer router reduces gradient variance and seed instability on tiny calibration splits.
- **Logit Ensembling Equivalence**: The derivation in Equation 8 correctly shows that on a frozen feature space, parameter-level head merging is mathematically equivalent to output-level logit ensembling. It also clearly derives in Equation 9 why this equivalence diverges non-linearly in deep networks, demonstrating excellent mathematical precision.
- **Heterogeneity Collapse Derivation**: Equation 17 mathematically demonstrates how batch averaging of unconstrained positive/negative coefficients on heterogeneous streams causes cancellation towards zero, neutralizing routing capability.

### 2. Empirical Methodology and Experimental Controls
The experimental design is exemplary, combining a controlled, synthetic representation-space sandbox with extensive sweeps and a real-world Vision Transformer validation:
- **Scientific Isolation**: Using a 14-layer representation-space sandbox is a well-justified methodological tool to isolate routing optimization from confounding physical factors like weight permutations.
- **Statistical Rigor**: All experiments (main results, complexity sweeps, leakage sweeps, stream audits) are conducted across **5 independent random seeds** with both means and standard deviations reported, ensuring statistical significance.
- **Robustness Sweeps**: Rather than reporting a single favorable setting, the authors evaluate TSAR under:
  - **Sensitivity Sweeps**: $\lambda_{anchor} \in [0, 1.0]$.
  - **Sample Complexity Sweeps**: $B_{cal} \in \{16, 32, 64, 128\}$.
  - **Subspace Leakage Sweeps**: Overlap factor $\eta \in [0.0, 0.4]$, representing heavily overlapping manifolds.
  - **Deployment Stream Audits**: Homogeneous vs. heterogeneous streaming at different batch sizes.
- **Physical Weight-Space Validation**: To bridge the gap to real deep networks, the authors fine-tune and merge classification heads of a real pre-trained Vision Transformer (`vit_tiny_patch16_224` from `timm`) across 4 visual tasks.

### 3. Scientific Transparency and Honesty (Exemplary Academic Integrity)
The paper is remarkably honest about its constraints and limitations, which is highly refreshing:
- **Classification Head Merging Limitation**: The authors explicitly state that their physical ViT experiment is limited to classification head merging over a frozen backbone, which is mathematically equivalent to logit ensembling. They openly state that merging internal, non-linear attention and MLP layers remains an open challenge.
- **SVHN Expert Performance Ceiling**: They explain that the SVHN expert ceiling is deliberately set to 19.28% via high simulated noise to act as an adverse stress test for the optimizer. They mathematically argue why this does not compromise generalizability.
- **PCGrad Complexity**: They acknowledge that PCGrad scales as $O(K)$, creating a bottleneck for massive multi-task systems, and provide 3 concrete mitigations (such as Stochastic Task Sampling) in the appendix, validating them on a 20-task setup.

### Weaknesses:
- **Uncentered PCA Projection and L2 Normalization Approximation**: In Section 3.1, the authors state that the PCA projection matrix $P$ is computed on mean-centered calibration features, but the forward projection is applied to raw, uncentered features $z(x)_b$ under $L_2$ normalization (Equation 1). They claim that any global feature translation is scaled down by the norm divisor, and residual uncentered offsets are absorbed by the downstream linear bias parameters $B_{l,k}$. However, mathematically, because the translation offset $\mu P$ resides inside the norm divisor ($\|z(x)_b P\|_2$), the divisor scale factor varies across samples depending on $z(x)_b$. This introduces a sample-dependent non-linear distortion rather than a constant spatial translation, meaning it cannot be perfectly absorbed by the static linear biases. While this approximation is likely harmless in practice when task centroids are highly separated, it represents a minor theoretical oversight in the methodology.
- **SVHN Anchor Instability Under High Noise**: The SVHN expert ceiling is set to 19.28% via an exceptionally high simulated noise level ($\sigma_{\text{SVHN}} = 0.95$). While this acts as a great stress test, computing a stable task-space feature anchor $\bar{\psi}_{\text{SVHN}}$ from only 16 calibration samples ($B_{cal} = 64$ divided across 4 tasks) under such extreme noise introduces immense sampling variance. Mathematically, the standard error of the mean under $\sigma = 0.95$ and $n = 16$ is $0.2375$, which is highly significant on a unit sphere of dimension $d=4$. This suggests that the SVHN task anchor itself is highly corrupted by sampling noise, a limitation that is not addressed in the methodology.
- **Systematic Sacrifice of Hard Tasks under PCGrad**: In Table 1, comparing `L3-Linear + TSAR (Ours)` to `TSAR + PCGrad (Ours)` reveals that while PCGrad boosts F-MNIST and CIFAR-10, it causes SVHN performance to drop from **15.52%** to **13.36%** (a -2.16% absolute degradation). This exposes a fundamental and unaddressed trade-off in PCGrad-based multi-task optimization: because SVHN is extremely noisy, its gradients conflict frequently with other tasks. PCGrad resolves these conflicts by projecting out or suppressing SVHN gradients to protect the simpler tasks, systematically sacrificing the performance of the hardest task to maximize the joint multi-task mean.
- **No Internal Layer Fusion on ViT**: The paper does not demonstrate weight merging of internal transformer layers (self-attention, MLPs) on real Vision Transformers. While the authors explicitly qualify this and discuss it as a key open direction, it remains a physical evaluation limitation.
- **Synthetic Feature Space**: The sandbox uses Gaussian feature spaces which are simpler than real-world distributions. However, the subspace leakage sweep ($\eta \in [0.0, 0.4]$) and the real ViT experiments heavily mitigate this by proving that the findings carry over to overlapping and real-world features.

### Soundness Rating: Excellent
The paper sets a benchmark for scientific transparency and analytical rigor in model-merging literature. The derivations are correct, the empirical sweeps are exhaustive, and the limitations are disclosed with exemplary honesty.
