# 4. Experimental Evaluation Check

## Experimental Setup and Scenarios
The experimental setup is exceptionally thorough and designed with deep scientific rigor:
1. **Model Backbone and Registries:** The authors use a 12-layer Vision Transformer (ViT-Tiny with 5.7M parameters) with a multi-task expert registry consisting of $K=4$ task experts: MNIST, Fashion-MNIST, CIFAR-10, and SVHN. The adapters are inserted into Layers 4 to 12 with a rank of $r=8$.
2. **Streaming Scenarios:** Frameworks are evaluated under both **Homogeneous Batching** (inputs grouped by domain, $B=256$) and **Heterogeneous Batching** (randomly interleaved inputs across domains, $B=256$), which mimics realistic dynamic serving workloads.
3. **Calibration Split:** Zero-Shot Centroid Alignment (ZCA-IDC) and Quantization-Aware Scale Calibration (QASC) utilize a highly compact split of only 64 samples per task ($|D_{\text{cal}}| = 64$).
4. **OOD Splits:** Conducted by querying the system with 1,000 SVHN images while treating MNIST, Fashion-MNIST, and CIFAR-10 as in-distribution tasks, providing a realistic test of the safety shield.

## Baseline Comparisons
The paper compares CG-Q-SPS against an extensive set of 8 baselines covering:
1. Standard performance ceilings (Expert Ceiling).
2. Static parameter-space merging (Uniform Merging FP32 and Quantized Uniform Merging INT4).
3. Traditional input-dependent gating (Linear Router with regularization).
4. SOTA dynamic serving systems (PFSR + MBH SOTA).
5. Unquantized activation ensembling (SPS-ZCA FP32).
6. Post-training quantization without calibration (Q-SPS INT4 with Round-To-Nearest).

This wide array of baselines represents a highly complete and challenging evaluation of the proposed framework.

## Analysis of Key Quantitative Results
The experimental results strongly support the authors' claims:
1. **Immunization Against Collapse:** Under heterogeneous streams, standard linear routers suffer from severe "heterogeneity collapse" (degrading to 42.95% Joint Mean). Statically averaging unquantized weights (Uniform Merging) collapses to 42.95%, and 4-bit static merging (Quantized Uniform Merging INT4) completely collapses to a near-random **30.70%** Joint Mean due to rounding errors. In contrast, CG-Q-SPS remains completely robust, maintaining a high **79.40% Joint Mean** under both homogeneous and heterogeneous streams.
2. **Impact of QASC Calibration:** Standard uncalibrated post-training quantization (Q-SPS INT4, RTN Baseline) degrades the Joint Mean to 78.44% due to discretization rounding noise. Our proposed QASC calibration, which optimizes clipping bounds via decoupled sequential MSE minimization, successfully reclaims the discretization loss to preserve **79.40% Joint Mean** (recovering **99.5%** of the unquantized FP32 ceiling of 79.80%).
3. **Lossless Gating Optimization:** CG-Q-SPS (with gating threshold $\theta = 0.01$ and temperature $\tau = 0.001$) yields the exact same accuracy profile as standard Q-SPS with QASC (79.40%). This empirically proves that conditional gating is completely lossless.
4. **Massive Memory Savings:** Quantizing expert adapters to 4-bit precision slashes their combined footprint from 2.76 MB to **0.345 MB** (a massive **87.5% savings**), allowing the expert adapters to fit natively inside highly restricted caches or microcontrollers with SRAM $< 512$ KB.
5. **Projected High-Throughput Serving:** Under highly interleaved streams, PFSR + MBH SOTA requires executing the heavy base model backbone sequentially $G=4$ times, resulting in **749.8 ms** cumulative latency. CG-Q-SPS processes the mixed batch in a single pass with fast integer arithmetic, consuming only **189.1 ms** cumulative latency—delivering a projected **3.97$\times$ physical speedup**.
6. **Robust OOD Rejection:** The coordinate-space diagonal GMM safety shield filters out high-dimensional visual variance early and achieves an outstanding **AUC of 0.98** (95.2% TPR at 4.3% FPR), significantly outperforming uncalibrated global cosine similarity thresholds (AUC = 0.72), early-stage Mahalanobis distance (AUC = 0.84), and late penultimate-layer Energy-based detection (AUC = 0.87).
7. **Task Manifold Entanglement and Redundancy of Orthogonalization:** The authors sweep representation entanglement $\epsilon \in [0.0, 0.8]$.
   * Standard nearest-centroid: routing accuracy degrades to 92.80% and flicker rate escalates to 13.25% at $\epsilon = 0.8$.
   * Gram-Schmidt CCO: asymmetrical template distortion collapses late expert paths, leading to a degraded routing accuracy of 92.70% and a high flicker rate of 13.86%.
   * Löwdin SMD: symmetric orthonormalization treats all experts symmetrically, maintaining high routing accuracy (94.40%) and low flicker (10.74%).
   * Raw ZCA-IDC (Ours): remains the most robust overall, preserving a Routing Accuracy of **94.70%** and a Flicker Rate of **10.34%**. This demonstrates that explicit orthogonalization is mathematically redundant and even detrimental under noise due to "noise spillover" along joint projection directions, providing an extremely practical and honest systems insight.
8. **Temporal Transition Lag Trade-off:** Over sequential $B=1$ streams, higher smoothing coefficients $\gamma$ in the EWMA routing filter successfully suppress routing flicker to under 0.8%, but introduce systematic transition lag (e.g., 2.67 steps at $\gamma = 0.80$), degrading transition-phase accuracy on interleaved streams from 79.40% to 78.62%. This quantitative sweep illustrates a classic systems trade-off.
9. **Expert Registry Scalability:** Sweeping expert registry size $K \in [4, 24]$ shows that the diagonal Coordinate GMM maintains a strong AUC of 0.88, while CG-Q-SPS dynamic execution-gating scales active adapter compute loads as $1/K$ (consuming less than 5% active load at $K = 24$), demonstrating exceptional scalability.
10. **Empirical Multi-Layer and Compounded Validation on Real Weights:** To ground their simulation, the authors perform empirical validation on a physical pre-trained Vision Transformer (\texttt{vit\_tiny\_patch16\_224} from `timm`) across Block 5, 9, and 12 MLP layers under 4-bit weight / 8-bit activation precision using 256 CIFAR-10 images.
    * Proposed QASC Dynamic Scaling dramatically slashes relative reconstruction MSE from 6.68% (uncalibrated RTN) to only **2.80%** (specifically 2.8015%) and restores output cosine similarity to **0.9861**.
    * Proposed QASC Static Scaling (using pre-calculated offline intermediate scales) achieves an identical relative reconstruction MSE of **2.80%** (specifically 2.8028%) and output cosine similarity of **0.9861**.
    * Multi-layer compounding simulation over all 12 blocks confirms that QASC Static Scaling successfully prevents error propagation, achieving the highest top-1 class prediction agreement of **84.38%** and slashing compounded relative logit MSE from 1.93% (RTN) to **1.20%**. This proves that pre-calculating intermediate activation scales offline captures the active representational boundaries without localized outlier noise, stabilizing the multi-layer trajectory while enabling branchless execution.
