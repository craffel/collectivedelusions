# Summary of the Paper

## Main Topic and Approach
This paper addresses the challenge of serving multiple concurrent Parameter-Efficient Fine-Tuning (PEFT) experts (specifically Low-Rank Adaptation, or LoRA, adapters) on resource-constrained edge CPUs and microcontrollers. Traditional weight-merging approaches suffer from "heterogeneity collapse" when handling interleaved, multi-task streams. Meanwhile, systems-level micro-batch partitioning (such as PFSR + MBH) avoids interference but requires sequential sub-batch passes of the base model, leading to latency that scales linearly with the number of tasks.

To resolve these bottlenecks, the paper builds on the recently proposed Single-Pass Activation-Space Dynamic Blending (SPS-ZCA) framework and introduces **Q-SPS** and its gated variant **CG-Q-SPS**. The core approach consists of:
1. **Low-Bitwidth Symmetric Quantization (Q-SPS):** Quantizing the expert LoRA adapter weights to 4-bit or 8-bit integers, while dynamically quantizing activations to 8-bit integers, executing the entire low-rank addition chain in pure integer arithmetic.
2. **Quantization-Aware Scale Calibration (QASC):** A training-free calibration protocol that dynamically computes task-specific scaling bounds over a 64-sample calibration split by sequentially decoupling the Mean Squared Error (MSE) minimization of the down-projection and up-projection scales.
3. **Conditional Expert Gating (CG-Q-SPS):** Bypassing/skips the execution of expert adapter pathways whose routing coefficients fall below a threshold ($\theta = 0.01$) under a sharp temperature setting ($\tau = 0.001$), reducing active adapter compute overheads.
4. **Intra-Task Dispersion Calibration (IDC):** An enhancement to Zero-Shot Centroid Alignment (ZCA) routing that scales cosine similarities by the expected in-distribution cosine similarity scale to equalize coordinate scales across tasks.
5. **Coordinate GMM Safety Shield:** Fitting a diagonal Gaussian Mixture Model (GMM) on the low-dimensional routing coordinates for out-of-distribution (OOD) task detection and rejection.
6. **Temporal-Aware Routing Hysteresis:** An EWMA-based coordinate smoothing filter to stabilize cache residency and prevent routing flicker under sequential $B=1$ streaming.

## Key Findings
- **Immunization Against Collapse:** CG-Q-SPS preserves a high simulated joint mean accuracy of 79.40% on heterogeneous streams under 4-bit quantization, recovering 99.5% of the unquantized float ceiling (79.80%), whereas quantized uniform merging suffers complete structural collapse (30.70%).
- **Calibrated Outperforming of PTQ:** QASC calibration outperforms uncalibrated Round-To-Nearest (RTN) quantization by +0.96% absolute joint accuracy at 4-bit precision.
- **Latency and Memory Footprint Reductions:** Quantizing LoRA experts to 4-bit slashes their footprint by 87.5% (from 2.76 MB to 0.345 MB). CG-Q-SPS is projected to achieve a 3.97$\times$ speedup over the sequential PFSR+MBH baseline on heterogeneous streams.
- **Energy Efficiency:** CG-Q-SPS (INT4) slashes dynamic serving energy to 0.46 J per batch, representing a 56.2% energy savings over sequential micro-batching.
- **High-Precision OOD Detection:** The low-dimensional diagonal Coordinate GMM safety shield achieves an AUC of 0.98 (95.2% True Positive Rate at a 4.3% False Positive Rate), outperforming raw feature-space OOD baselines.
- **Redundancy of Orthogonalization:** Analytical and empirical sweeps show that explicit centroid orthogonalization methods (Gram-Schmidt CCO or Löwdin SMD) are mathematically redundant and can propagate noise under inference, making the unorthogonalized ZCA-IDC baseline more robust under task entanglement.

## Explicitly Claimed Contributions (with Evidence)
1. **Transparent Analytical Simulation Study (ICS):** The authors establish a hardware-calibrated analytical simulation paradigm to isolate systems-level variables (Section 1, Section 4.1).
2. **Immunization Against Collapse at Low-Bit Precision:** CG-Q-SPS (INT4) recovers 99.5% of the FP32 accuracy ceiling and prevents the parameter-space collapse of static merging (Table 1).
3. **Resolution of Routing-Blending Contradiction:** Introducing conditional gating ($\theta=0.01$) to skip executing inactive low-rank expert pathways, leading to computational savings with zero loss in accuracy (Section 3.4, Table 1).
4. **Massive Expert Memory Footprint Savings:** Slashing LoRA expert memory by 87.5% (from 2.76 MB to 0.345 MB) via 4-bit quantization (Section 4.4, Table 2).
5. **High-Throughput Single-Pass Serving:** Delivering a projected 3.97$\times$ speedup over sequential sub-batching baselines on heterogeneous streams (Table 2).
6. **Robust GMM OOD Rejection:** The coordinate-space GMM achieves 95.2% TPR at 4.3% FPR (AUC = 0.98) (Section 4.5, Figure 4).
7. **Rigorous Evaluation of Orthogonalization Redundancy:** Showing that explicit orthogonalization is mathematically redundant and that the simpler, unorthogonalized ZCA-IDC baseline remains the most robust under task entanglement (Section 3.3, Section 4.6, Table 3).
8. **Empirical Validation and Scalability Sweeps:** Providing validation on real pre-trained Vision Transformer weights (Section 4.7), scaling sweeps to LLM-scale weights with activation outliers (Section 4.8), and physical CPU benchmarking (Section 4.9).
