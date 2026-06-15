# Paper Summary

## Main Topic and Approach
The paper introduces **SPS-ZCA** (Single-Pass Sample-Wise Routing with Zero-Shot Centroid Alignment), a training-free dynamic model-merging framework designed to serve multiple task-specific Low-Rank Adaptation (LoRA) experts from a shared pre-trained base model on resource-constrained edge CPUs. 

The core approach addresses two main bottlenecks of existing dynamic model-merging methods:
1. **The Latency Bottleneck (Sequential Backbone Execution):** Prior state-of-the-art methods like Micro-Batch Homogenization (MBH) partition incoming heterogeneous streams on-the-fly and run them sequentially, requiring up to $K$ (number of tasks) forward passes of the heavy base model. SPS-ZCA proposes **Single-Pass Activation-Space Dynamic Blending (SPS)**, which executes the shared base model and its expert adapters in a single, parallel forward pass, blending activations on-the-fly sample-wise.
2. **The Temporal Routing Paradox and Out-of-Distribution (OOD) Noise:** Prior routers depend on late-stage penultimate representation features, which requires executing the backbone twice. Furthermore, projecting inputs against noisy classification heads collapses on OOD tasks. SPS-ZCA proposes **Zero-Shot Centroid Alignment (ZCA)**, which routes inputs using task centroids pre-computed from a tiny calibration split in the pre-trained backbone's early representation space (Layer 3), bypassing noisy classification heads and resolving the temporal routing paradox.

To enhance robustness, the authors also introduce **Unit-Norm Calibration (UNC)** to handle representation scale imbalances, **Intra-Task Dispersion Calibration (IDC)** to normalize asymmetric task manifold variance, and a diagonal **Gaussian Mixture Model (GMM) Coordinate Density Estimator** for upfront OOD query rejection.

## Key Findings
- **Expert Ceiling Recovery:** SPS-ZCA recovers 100.0% of the expert ceiling accuracy (Joint Mean of 79.80% in simulation, and 76.14% in real PyTorch profiling) with zero trainable parameters, outperforming prior non-parametric SOTA methods by +3.66% absolute accuracy in simulation.
- **Physical Wall-Clock Speedup:** At low batch scales ($B=16$), the Vectorized Scatter-Gather implementation (SPS-VSG) achieves a verified physical 1.17$\times$ wall-clock speedup out of the box in uncompiled PyTorch over MBH.
- **Projected Analytical Speedup:** Under high batch heterogeneity, the proposed compiler-fused loop is projected to achieve a 3.90$\times$ analytical speedup (reducing latency from 776.4 ms to 199.0 ms).
- **OOD Rejection:** The coordinate GMM estimator achieves a 95.2% true positive rate for OOD task rejection at a low 4.3% false positive rate.
- **Feature Separability:** Physical profiling reveals that semantic abstraction occurs early, showing a massive jump in the Fisher Separability Criterion (FSC) from 0.1840 at Layer 0 to 47.4955 at Layer 3.

## Explicitly Claimed Contributions and Supporting Evidence
1. **Single-Pass Activation-Space Dynamic Blending (SPS):** Stated to eliminate the sequential execution bottleneck of MBH, maintaining $O(1)$ backbone execution time. Evidenced by analytical memory and compute profiling, and physical PyTorch CPU execution times showing a speedup at small batch sizes ($B=16$).
2. **Zero-Shot Centroid Alignment (ZCA):** Stated to resolve the routing paradox and classification head dependency. Evidenced by representation-separability profiling indicating high Fisher Separability at Layer 3 ($\text{FSC} = 47.50$), achieving 100.0% routing accuracy on physical vision/text models.
3. **Unit-Norm Calibration (UNC) and Intra-Task Dispersion Calibration (IDC):** Stated to neutralize scale imbalances and asymmetric manifold variances. Evidenced by ablation studies showing UNC restores Joint Mean from 79.22% to 79.80% under scale imbalances, and IDC restores balanced routing under asymmetric dispersions (from 95.40% misrouting down to 47.00% near-random chance).
4. **Coordinate GMM for OOD Rejection:** Stated to shield experts from OOD queries. Evidenced by ROC curve analyses showing 95.2% TPR at 4.3% FPR on SVHN OOD data, and sensitivity sweeps on the log-likelihood threshold $\eta$.
5. **Autoregressive Text Modality Generalizability:** Stated to extend to text models with KV cache sharing. Evidenced by GPT-2 experiments over Legal, Medical, and Code domains showing 98.50% routing accuracy and preserving perplexity (12.18 vs. 12.15 expert ceiling).
