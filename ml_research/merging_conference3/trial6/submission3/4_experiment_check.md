# 4. Experimental Evaluation Check

## Quality and Comprehensiveness of Experiments
The experimental evaluation is exceptionally thorough, comprehensive, and complete. It covers multiple dimensions of sensitivity, scalability, and deployment robustness:

1.  **Massive Hyperparameter Sweeps:** Over 1,280 experiment configurations are swept across 5 independent random seeds ($42, 43, 44, 45, 46$), ensuring a high level of statistical significance.
2.  **Extensive Baseline Comparisons:** The paper benchmarks BWS-Router against a comprehensive set of baselines:
    *   *Expert Ceilings* (upper performance limit)
    *   *Static Uniform Merging* (baseline linear interpolation)
    *   *Global Linear Routing* (Unreg and Reg)
    *   *Quantum Wavefunction Superposition* (QWS-Merge)
    *   *L3-Router (Unshared Layer-wise)* with Linear, Tanh, and Softmax activations (both Unreg and Reg)
3.  **Comprehensive Sensitivity Sweeps:** The appendix features highly targeted sensitivity checks across every major architectural and optimization hyperparameter:
    *   *Block-sharing Group Size $M \in \{1, 2, 3, 4, 6, 12\}$* (Table 2)
    *   *Gating Activation Function* under multiple learning rates (Table 3)
    *   *Deployment Streams* (Homogeneous $B=1$, Homogeneous $B=256$, and Heterogeneous $B=256$) (Table 4)
    *   *PCA Subspace Dimension $d \in \{2, 3, 4, 6, 8, 12, 16\}$* under both Sandbox and Physical setups (Table 7)
    *   *Unsupervised Projector Kernels* (Linear PCA, RBF, Cosine, Polynomial) (Table 8)
    *   *Gating Bias Initialization $B_{group} \in \{-2.0, -1.0, 0.0, 1.0, 2.0\}$* under multiple learning rates (Table 6)
    *   *Calibration Sample Complexity* scaling from 16 to 1024 samples (Table 5)
    *   *Task Scaling Ceiling $\lambda_{max} \in \{0.1, 0.2, 0.3, 0.4, 0.5\}$* (Table 1)
    *   *Expert Task Count $K \in \{4, 6, 8, 10\}$* (Table 11)
4.  **Physical Sequential Merging Verification:** Crucially, the authors validate their findings in a physical sequential weight-space model-merging setup on 3-layer MLP experts. This eliminates sandbox-specific virtual averaging and tests BWS-Router under cascading representation propagation.
5.  **Modern Architecture Profiling (ViT Pilot):** The authors perform a PyTorch-level pilot demonstration on a real Vision Transformer backbone (\texttt{vit\_tiny\_patch16\_224}) using the \texttt{timm} library, reporting actual wall-clock forward pass latency and dynamic weight blending overhead on host CPU.
6.  **Beautiful High-Resolution Plots:** Multiple plots are included visualizing main performance comparisons, block size sweeps, deployment stream heterogeneity robustness, and empirical regularization impact.

## Analysis of Key Experimental Findings
*   *Static Uniform Collapse:* The experiments prove that static average blending completely collapses under weight-space task conflicts, achieving only **23.56%** (sandbox) and **17.88%** (physical), whereas BWS-Router recovers performance near the Expert Ceilings (**79.57%** sandbox and **45.26%** physical).
*   *Stable Generalization and Compression:* Scaling the block size $M$ to $12$ (fully shared global routing) matches or slightly exceeds unshared L3 accuracy while achieving an extraordinary **91.7% parameter reduction** (20 params vs 240 params).
*   *Physical Merging Stabilization:* In the physical setup, block-wise sharing ($M=3$) drastically stabilizes adjacent gating coefficients, outperforming unshared routing ($M=1$) by **+10.93% absolute accuracy** on mixed heterogeneous streams.
*   *Optimization Sluggishness Resolved:* Setting gating bias $B_{group} = -2.0$ increases Joint Mean accuracy by **+17.25% absolute** under standard learning rates ($\eta = 10^{-2}$), showing negative bias establishes a sparse default state that simplifies optimization.
*   *Uncovering Projector Subspace Trade-offs:* The PCA dimension sweep reveals a fascinating trade-off: the sandbox has a low-dimensional sweet spot at $d \approx K$ (filtering out high-dimensional feature noise), whereas the deep physical setup benefits monotonically from scaling $d$ up to 16, as larger dimensions preserve crucial activations across sequential layers to prevent compounding feature distortion.
*   *Sequential Smoothing vs. Residual Links:* In physical sequential weight-blending, runtime residual links stabilize seed-wise variance but severely degrade mean performance by forcing coefficients towards a task-agnostic static average. In contrast, sequential smoothing regularization ($\lambda_{\text{smooth}} = 10^{-2}$) reduces standard deviation from **21.28%** to **13.41%** while fully preserving routing capacity and dynamic accuracy ceilings.
*   *Inference Latency Profile:* The ViT pilot shows that coarse-to-fine block sharing (grouping the first 8 generic shallow blocks into a single block) reduces dynamic blending overhead from 190.01 ms to 110.65 ms, representing a **17.2% overall latency reduction**.

## Areas for Improvement / Minor Limitations
*   *GPU Latency Profiling:* The ViT pilot profiling is executed on host CPU. Profiling the latency of physical weight blending on GPU (utilizing CUDA streams and batched, vectorized operations) would be highly valuable, as GPU execution is the standard production environment for modern backbones.
*   *Downstream Adaptation Data-Scarcity Bounds:* The sample complexity sweep is evaluated down to 16 total calibration samples. It would be valuable to explore the absolute data-scarcity limits, such as zero-shot dynamic model merging (using zero calibration data, where the routing parameters are computed purely based on unsupervised projection alignment without any gradient training).
