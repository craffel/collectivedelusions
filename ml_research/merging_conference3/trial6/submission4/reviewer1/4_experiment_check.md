# 4. Experiment Check

## Critical Evaluation of the Experimental Setup
The paper utilizes a highly structured, hybrid experimental strategy:
1. **Primary Evaluation on a Representation Sandbox:** The main experiments are executed on a simulated 14-layer representation-space sandbox ($D=192$, $L=14$) modeling four tasks: MNIST, FashionMNIST, CIFAR-10, and SVHN. The features are generated synthetically using task-specific isotropic Gaussians centered around orthogonal prototypes.
   - *Strengths:* Excellent variable isolation. It decouples the routing dynamics from weight permutation conflicts and attention mechanics, allowing a direct, noise-free study of the router's calibration gradients.
   - *Weaknesses:* Real-world deep representation manifolds are highly non-linear, correlated, and possess complex topological structures, which are not perfectly captured by orthogonal Gaussian clusters.
2. **Subspace Leakage Sweep (Overlapping Manifolds):** The authors address the limitation of synthetic orthogonality in Section 4.7 by sweeping a leakage parameter $\eta \in [0.0, 0.4]$, creating heavily overlapping task manifolds. This is a very strong and rigorous experimental addition that successfully simulates real-world representation correlations.
3. **Physical Validation on pre-trained ViT:** In Section 15 and Appendix F, the authors fine-tune and merge classification heads of a physical Vision Transformer (\texttt{vit\_tiny\_patch16\_224}) on structured geometric patterns, and further validate on raw natural images from MNIST and CIFAR-10 (Section 15.1).
   - *Strengths:* Bridges the gap between representation space and real weight space, proving that TSAR generalizes to real-world pre-trained backbone embeddings.
   - *Weaknesses:* The physical merging is restricted to the linear classification heads, which is mathematically identical to output-level logit ensembling. It does not validate true weight merging within deep, non-linear intermediate layers (attention maps, MLPs).

## Datasets and Task Ceilings
The selection of tasks (MNIST, FashionMNIST, CIFAR-10, SVHN) represents a diverse multi-task suite spanning simple digit recognition, clothing classification, and natural image classification.
- **SVHN Adverse Noise Control:** Setting the SVHN expert ceiling to a low 19.28% is a unique and clever methodological control that stress-tests the optimizer under severe noise. The authors' follow-up evaluation in Section 17 under realistic SVHN expert performance (90.40%) confirms that the performance advantages of TSAR are structurally identical under low-noise, high-accuracy regimes.

## Evaluation of Baselines
The baselining in this paper is exceptionally comprehensive, leaving no obvious gaps:
- **Static Baselines:** Static Uniform Merging, Static Logit Ensembling, and AdaMerging.
- **Unconstrained/Unregularized Routers:** Global Linear Router and L3-Linear (Unregularized).
- **Alternative Regularization:** Standard $L_2$ weight decay (with systematic strength sweeps in Section 12).
- **State-of-the-Art (SOTA):** QWS-Merge (quantum wave-superposition ensembler).
- **Training-Free Baseline:** Centroid Router (Section 4.2), proving that gradient-based calibration under TSAR is statistically superior to directly setting weights to centroids.
- **Standard MoE Baselines:** Raw Softmax MoE and Raw Top-1 MoE (Section 16), proving that TSAR's low-dimensional projection is 97.4% more parameter-efficient and far more robust to noise than standard MoE gating layers operating on raw high-dimensional features.

## Do the Results Support the Claims?
Yes, the empirical findings are highly robust, consistent across 5 seeds, and fully support the authors' claims:
1. **Low-Data Overfitting Claim:** Supported by the catastrophic failure of the unconstrained Global Linear Router (Joint Mean 23.20%) vs. anchored routers under $B_{cal}=64$.
2. **TSAR Dominance Claim:** Supported by Table 1, where TSAR + PCGrad achieves the new SOTA Joint Mean accuracy of 57.06%, outperforming Static Uniform by +5.20% and QWS-Merge by a spectacular +17.18%.
3. **PCGrad Necessity and Scaling Anomaly Claim:** Supported by Table 3, which documents the collapse of Standard TSAR at $B_{cal}=128$ (Joint Mean dropping to 47.70%) due to hard-task gradient dominance, and shows that PCGrad successfully resolves it (stabilizing accuracy at 49.86%).
4. **Heterogeneity Collapse and Streaming Claim:** Supported by Table 4, showing unconstrained TSAR collapsing to 43.10% under mixed-task streams, and proving that non-negative activations (scaled Sigmoid) successfully bypass this collapse, achieving a stable 50.80% Joint Mean accuracy with zero runtime overhead.
5. **Representational Redundancy Claim:** Supported by Section 4.3, proving that a single-layer global router ($L=1$, 20 parameters) performs almost identically to the 14-layer router (53.98% vs 54.10% with standard TSAR), validating that the over-parameterized multi-layer routing collapse is mathematically and empirically real.
6. **Scalability Claims:** Supported by Section 14, demonstrating that Task Grouping and Stochastic Sampling achieve massive speedups (up to 5.1$\times$) on a massive 20-task setup, and showing that the single-layer router ($L=1$) completely bypasses the PCGrad complexity bottleneck.

In conclusion, the experimental section is outstandingly thorough, robust, and well-designed. However, the sheer volume of sweeps, ablations, and fine-tuning (while impressive) reinforces the Novelty Seeker's critique that the paper is focused on optimizing and engineering a relatively simple, straightforward regularizer (spatial anchoring) on top of an existing router framework, rather than introducing a major conceptual breakthrough.
