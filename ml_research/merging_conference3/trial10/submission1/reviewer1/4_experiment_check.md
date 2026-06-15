# 4. Experimental Evaluation Check

## Evaluation of Experimental Setup and Datasets
The experimental evaluation is designed with exceptional rigor, combining a controlled synthetic sandbox environment and a real-world physical validation setup:

1. **Analytical Coordinate Sandbox (ICS):**
   - **Dimensions:** 14-layer transformer model ($D = 192$, active layers 4 to 14, extraction at anchor Layer 3).
   - **Manifolds:** 3 distinct representation manifolds:
     - *Orthogonal Manifolds ($V=0$):* Complete orthogonality with continuous depth rotation (up to $0.60$ radians).
     - *Overlapping Manifolds ($V=12$):* Shared 12-dimensional subspace, modeling highly correlated task representations.
     - *Composite Task Manifolds:* Expert target shifts abruptly at Layer 9, modeling multi-task composition over depth.
   - **Noise and Biases:** Systematically applied task noise standard deviations ($\sigma = [0.05, 0.15, 0.40, 1.20]$) and task biases ($b = [0.0, 0.0, -0.90, -2.30]$) to represent realistic serving-time conditions.
   - **Workload Streams:** Tests both *Homogeneous Streams* (stable task) and *Heterogeneous Streams* (frequent task switches) to evaluate serving lag.

2. **Physical Validation (ResNet-18 on ImageNet-1K):**
   - **Model:** Pre-trained ResNet-18 model from `torchvision.models`.
   - **Task Taxonomy:** $K = 4$ highly diverse visual categories (Canines, Vehicles, Birds, Household/Furniture) spanning **exactly 40 distinct ImageNet-1K classes** (10 classes per task).
   - **Images:** Natural images programmatically downloaded from GitHub, processed with standard ImageNet transforms.
   - **Augmentations:** Evaluated over **exactly 200 query samples** using standard dynamic test-time data augmentations (resized cropping, horizontal flips, rotation, color jitter) to simulate realistic serving-time representation variance and noise.
   - **Modulation:** Extract block-wise activation signatures during calibration, applying channel-wise modulation ensembling at each residual block with **mean-preserving normalization** to prevent batch normalization scale collapse.

---

## Strength and Tuning of Baselines
The authors evaluate against an extensive suite of **seven state-of-the-art baselines**, covering all major routing strategies:
1. **Uniform Merging:** Simple baseline blending all expert weights evenly.
2. **Static/Anchor Routers (SABLE-Static, SPS-ZCA-Static):** Freeze weights based on the anchor Layer 3 representation, copying them across depth.
3. **Dynamic Routers (SABLE-Dynamic, SPS-ZCA-Dynamic):** Compute routing weights layer-by-layer independently using intermediate features.
4. **Stateless Spatial Smoothers (SABLE-CausalFilter, SABLE-Gaussian):**
   - *CausalFilter:* Applies an on-the-fly causal Exponential Moving Average ($\beta = 0.50$) across depth.
   - *Gaussian:* Runs a first pass, applies a symmetric 1D Gaussian smoothing filter ($\sigma_g = 1.0$, kernel $5$) post-hoc, and propagates activations in a second pass.
5. **Stateful Kinetics Models (ChemMerge, Momentum-Merge, PAC-Kinetics, Stateful ERM):** Maintain a temporal carrying-state across adjacent sequence samples to smooth spatial jitter at the expense of temporal inertia.

This set of baselines is exceptionally strong. It explicitly resolves the temporal "Stateful Reset" bug in previous comparisons, evaluating the stateful models under a scientifically consistent protocol. This ensures a fair and rigorous comparison of temporal lag and spatial smoothing.

---

## Comprehensiveness of Ablation Studies
The ablation studies are highly comprehensive and address several key dimensions:
1. **Truncated Backward Horizon ($H$) Sweep:** The authors sweep $H \in \{1, 2, 3, 4, 6, 8, 11\}$ under both Orthogonal and highly challenging Composite Heterogeneous workloads. The results (Table 4) confirm the Dobrushin contraction convergence, showing that a small horizon ($H=4$) is statistically indistinguishable from the exact full-depth ($H=11$) pass.
2. **Extrapolation Methods:** The authors evaluate three variants of the single-pass controller: QPathMerge (default constant future potentials), QPathMerge-LinearExtrap, and QPathMerge-RollingExtrap. This reveals the "spatial lag" collapse of rolling averages (collapsing to 91.42% accuracy) and the superior tracking of linear projection (99.67% accuracy).
3. **Temperature Sensitivity ($\tau$):** The paper details the sensitivity of the temperature parameter $\tau$, demonstrating that very low temperatures lead to hard-routing and high jitter, while very high temperatures flatten the distribution and lose task specialization.
4. **Calibration Sample Complexity:** The authors sweep the number of calibration samples (from 1 to 4 images), showing that the scale-invariant cosine similarity is highly sample-efficient and robust to calibration-time distribution shifts.
5. **Experts Registry Size ($K$):** The authors sweep $K \in \{4, 8, 16, 32, 64\}$, measuring theoretical FLOPs and empirical CPU latency to prove that the single-pass controller scales linearly and consumes negligible overhead.

---

## Statistical Soundness
The evaluation is statistically sound and adheres to high standards of empirical rigor:
- **Random Seeds:** Accuracies and metrics are reported as mean $\pm$ standard deviation across **5 independent random seeds** for the Coordinate Sandbox evaluations, and across **3 random seeds** for the physical ResNet-18 ImageNet-1K evaluation.
- **Multiple Metrics:** The authors evaluate and report three distinct metrics: Joint Serving Accuracy, Layer Jitter (spatial smoothness across depth), and Seq Jitter (temporal stability across sequence samples). This ensures that the accuracy-stability trade-off is quantified from both spatial and temporal perspectives.

---

## Support for Claims (Are Claims Backed by Data?)
The empirical results provide **rock-solid support** for the paper's key claims:

1. **Jitter Reduction:** In Table 1, under Orthogonal Heterogeneous streams, QPathMerge ($H=4$) slashes Layer Jitter to **0.003292**—a **$3.2\times$ reduction** compared to SABLE-Dynamic (0.010551) and SABLE-CausalFilter (0.010469). This is replicated on physical manifolds (Table 5), where QPathMerge reduces physical Layer Jitter to **0.078116** (a $3.23\times$ reduction compared to SABLE-Dynamic's 0.252176).
2. **Elimination of Hysteresis/Lag:** Under Heterogeneous streams, the stateful models experience a catastrophic drop in accuracy due to temporal lag (e.g., ChemMerge drops from 98.10% to 86.50% in Table 1; Momentum-Merge drops to 78.59%). QPathMerge maintains near-perfect accuracy (97.47%), matching or exceeding SABLE's stateless agility with zero temporal hysteresis.
3. **Zero Computational Overhead:** Table 6 sweeps $K$ up to 64, proving that QPathMerge-Single requires less than 67.5k FLOPs and 161 $\mu$s per layer step. System-level latency profiling on ResNet-18 confirms that QPathMerge adds only **$1.35$ ms** of total end-to-end inference latency ($5.35\%$ overhead).
4. **The Spatial Smoothing Trade-off:** Under the Composite Task manifold (Table 3), SABLE-Dynamic achieves 99.65% accuracy at the cost of extreme spatial jitter (0.204736). QPathMerge ($H=4$) introduces a spatial filter that slightly smooths the task transition boundary, resulting in a minor accuracy drop to 98.73%. Linear extrapolation (\texttt{LinearExtrap}) successfully breaks this drag to achieve **99.67% accuracy** (outperforming both), while rolling extrapolation collapses to 91.42%. These discoveries are fully supported by the quantitative results.
