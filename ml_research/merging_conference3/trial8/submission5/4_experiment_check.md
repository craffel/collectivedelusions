# 4. Experimental Design and Results

## Evaluation Methodology and Rigor
The experimental evaluation in this paper is exceptionally thorough, rigorous, and carefully designed. It is structured into two main components that comprehensively address both the theoretical properties and the practical viability of the proposed method:

1. **Overlapping Subspace Sandbox (PyTorch Simulation):**
   - **Rigor:** Evaluated across 5 independent random seeds (Seeds 10 to 14), with mean and standard deviation reported for all tasks.
   - **Overlapping Subspace Layout:** Each task occupies a 96-dimensional subspace with a 64-dimensional overlap with neighboring tasks. This simulates realistic representational manifolds with substantial semantic overlap and ambiguity, resolving prior criticisms of trivial orthogonal subspaces.
   - **Stress Testing:** The SVHN expert manifold is deliberately configured as a specialized high-noise stress-test (adding an intrinsic noise scale of $1.20$, resulting in a 19.68% expert ceiling). This evaluates whether each router can successfully manage heavily degraded task domains without letting representation noise corrupt clean, high-accuracy domains (such as MNIST with a 100% expert ceiling).
   - **Comprehensive Deployment Regimes:** Methods are benchmarked across Homogeneous batches, Heterogeneous batches ($B=256$), and true batch-independent Heterogeneous vectorized streams ($B=1$).
   - **Ablation Studies:** Includes exhaustive sweeps for temperature $\tau \in [0.0001, 0.5]$, OOD rejection threshold $\gamma_{\text{OOD}} \in [0.00, 0.45]$, routing boundary layer $l_{\text{route}} \in \{0, 1, 2, 4, 6, 8, 10\}$ (validating the *Early-Layer Routing Compromise*), non-linear residual activations (using GeLU), and highly optimized low-noise expert regimes.
   - **Adaptive Thresholding Validation:** Benchmarks Adaptive Task-Specific Thresholding against global thresholds on a mixed stream of in-distribution and OOD noise queries, tracking False Acceptance Rates (FAR) and In-Distribution Accuracies.

2. **Real-World Empirical Validation (Pre-trained ViT Backbone):**
   - **Bridging the Sim-to-Real Gap:** Validated on actual real-world images from MNIST, Fashion-MNIST, CIFAR-10, and SVHN using an ImageNet pre-trained `vit_tiny_patch16_224` backbone from the `timm` library.
   - **Color Routing Paradox Validation:** Empirically demonstrates that pure Layer 0 routing acts as a color/texture router, limiting Joint Mean accuracy to 57.81%.
   - **Early-Layer Routing Compromise Success:** Shifting the routing boundary slightly deeper to Layer 1 or Layer 2 achieves **91.80%** and **95.31%** routing accuracy respectively, outperforming a trained, parameterized pre-backbone CNN router (**91.02%**) with zero trainable parameters.
   - **End-to-End LoRA Ensembling:** Validates actual LoRA ensembling on real images across all four tasks (MNIST, Fashion-MNIST, CIFAR-10, SVHN) with LoRA adapters inserted into **all 12 transformer blocks** of the backbone.
   - **ELFT Validation:** Compares standard training against ELFT training (freezing blocks 0 and 1 during fine-tuning), demonstrating that alignment yields near-perfect ceiling recovery.
   - **Latency Analysis:** Measures CPU execution latency. PEAR L2 adds a sequential latency delay of **20.78%** ($6.26$ ms) on ViT-Tiny and only **17.59%** ($36.09$ ms) on ViT-Base, but introduces virtually zero extra computational FLOPs, as intermediate activations are fully cached and reused for the remaining blocks. Furthermore, Section 4.8.4 shows how relative sequential delay scales down to $<5\%$ on larger models (e.g. ViT-Large), demonstrating high practical edge scalability.

## Baselines and Comparisons
The paper benchmarks PEAR against five appropriate and modern baselines:
- **Expert Ceiling (Oracle):** Direct routing of queries to isolated, specialized models.
- **Static Uniform Merging:** Parameter-level averaging of expert adapters.
- **Linear Router (Reg):** L2-regularized parameterized classical linear router.
- **PFSR + MBH SOTA (Sample-Wise PFSR):** Non-parametric classification-head routing (evaluated without scheduling to reflect real-time latency limits).
- **SABLE SOTA:** Non-parametric activation-space ensembling with Mid-Layer Routing (freezing blocks 1-10, late adaptation).

## Key Empirical Findings and Supported Claims
- **Claim: PEAR outperforms SABLE SOTA by capturing early-layer features.** Supported. SABLE achieves 55.30% Joint Mean accuracy across all stream configurations. PEAR's 100% layer adaptability captures crucial task-specific features in early blocks, achieving **59.34%** (+4.04% absolute improvement) in the overlapping sandbox, **60.38%** (+3.12% absolute improvement) under non-linear propagation, and **96.10%** (+1.74% absolute improvement) under the highly optimized regime. On real-world end-to-end LoRA ensembling, PEAR outperforms SABLE SOTA by a massive **+15.24%** ($55.08\%$ vs $39.84\%$).
- **Claim: PEAR completely eliminates Vectorization Collapse under $B=1$.** Supported. Under $B=1$ vectorized streams, the Linear Router collapses to **52.36%** due to overfitting on low-data splits. PEAR maintains its robust **59.34%** accuracy across all configurations.
- **Claim: Shifting the routing boundary deeper resolves the Global Average Color Routing Paradox.** Supported. Table 5 shows PEAR L2 routing accuracy on actual images reaches **95.31%**, whereas PEAR L0 is limited to **57.81%** due to low-level color-based representational overlap.
- **Claim: ELFT eliminates training-serving boundary representational mismatch.** Supported. Table 7 shows PEAR + ELFT achieves **53.52%** Joint Mean accuracy on real images, recovering an outstanding **85.10%** of its corresponding Expert Ceiling (compared to 82.46% in the unaligned standard setup).
- **Claim: Adaptive Task-Specific Thresholding resolves the security-selectivity trade-off.** Supported. Table 4 demonstrates that Adaptive Thresholding achieves a tight False Acceptance Rate of **5.47%** on MNIST and **17.19%** on SVHN, while fully preserving SVHN's in-distribution accuracy at **13.60%** (where a secure global threshold collapses it to **10.00%**).

## Overall Experiments Rating
The experimental evaluation is **excellent**. The inclusion of both a rigorous synthetic sandbox (with overlapping task subspaces, non-linear activations, and high-performance regimes) and actual real-world ImageNet pre-trained ViT experiments on real images is exceptionally complete. Every major claim is thoroughly supported by empirical data, and the systems-level measurements are extremely valuable.
