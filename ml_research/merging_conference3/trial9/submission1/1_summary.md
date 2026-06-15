# 1. Summary of the Paper

This paper introduces **PAC-Bayesian Smooth Trajectory Merging (PAC-STM)**, a novel, theoretically grounded paradigm for calibrating layer-wise ensembling parameters across deep neural networks.

## Key Contributions
1. **The Trajectory Prior & Posterior Framework:** Models layer-wise log-temperatures as a continuous depth-wise autoregressive trajectory (a Gaussian random walk), encoding an elegant depth-wise ensembling smoothness inductive bias.
2. **Closed-Form Trajectory KL Divergence:** Derives and proves a closed-form analytical expression for the Kullback-Leibler (KL) divergence between a Markovian trajectory posterior $Q$ and prior $P$.
3. **Unit-Norm PCA Subspace Projection (UN-PCA-SEP):** A robust coordinate extraction technique that eliminates representation scale variance and suppresses out-of-distribution heteroscedastic noise by projecting onto task-specific principal component bases.
4. **Generalization Guarantees:** Leverages PAC-Bayesian complexity bounds (McAllester, Catoni, Alquier) to derive a deterministic trajectory optimization objective, providing a formal generalization bound for activation-space ensembling.
5. **High-Fidelity Real-World & Sandbox Validation:** 
   - Proves PAC-STM's absolute immunity to heterogeneity and vectorization collapse across mixed-task batch serving.
   - Includes real-world evaluation on pre-trained Vision Transformers (\texttt{ViT-B/16}) fine-tuned with active LoRA adapters on MNIST and CIFAR-10.
   - Evaluates a non-linear Kernel PCA extension (UN-KPCA-SEP) and a parameterized contrastive projection head under severe representational distortion, illustrating a clear trade-off between separation accuracy and serving latency.
   - Evaluates a non-sequential, residual-aware prior topology that mirrors modern skip-connection architectures (such as ResNets and Transformers).

## Summary of Results
- Under ultra-low calibration regimes ($N = 16$), PAC-STM successfully mitigates transductive overfitting compared to standard unregularized Empirical Risk Minimization (ERM), achieving an absolute accuracy gain of **+2.05%** under heterogeneous batching streams on orthogonal manifolds (with highly significant pairwise improvements $p < 0.008$).
- PAC-STM achieves perfect immunity to Heterogeneity and Vectorization collapse. While weight-space merging techniques (QWS-Merge, Linear Router, PFSR) collapse catastrophics to near-random performance ($\approx 40\%$) under mixed heterogeneous batches ($B=16$), PAC-STM preserves individual activation pathways and maintains performance ($\ge 72\%$).
- On pre-trained \texttt{ViT-B/16}, PAC-STM achieves a beautifully continuous layer-wise log-temperature trajectory with a smoothness value of **0.109547**, almost 3 times lower/smoother than unregularized ERM (0.275478), without sacrificing classification performance.
- Under severe representation non-linearity, uncentered UN-KPCA-SEP successfully untangles non-linear manifold structures, outperforming linear PCA by **+6.63%** in accuracy. A trained contrastive projection head achieves comparable accuracy to linear PCA while delivering a **22.24x speedup** over Kernel PCA ($0.000558$ ms vs $0.012406$ ms per sample).
- Under overlapping manifolds, the Skip-Aware (residual-skip) prior topology achieves **65.70%** joint accuracy, outperforming the sequential prior by **+1.05%**, while reducing trajectory roughness by **3.33%** (smoothness $0.001594$ vs $0.001649$).
