# 2. Novelty Check and Delta from Prior Work

## Positioning relative to Prior Work

### 1. Parameter-Space Model Merging
- **Prior Work:** Heuristics such as Task Arithmetic, Ties-Merging, DARE, RegMean, SyMerge, and OrthoMerge operate strictly in high-precision floating-point formats (FP16 or FP32). They assume that the merged model can be served in full precision.
- **The Delta:** QP-Merge is one of the first frameworks to explicitly acknowledge that merged models must be quantized for real-world deployments. It shows that standard model merging heuristics break down under low-bit quantization (such as INT4), and proposes a co-designed merging-quantization system.

### 2. Post-Training Quantization (PTQ)
- **Prior Work:** Standard PTQ techniques (e.g., AdaRound, BRECQ, SmoothQuant, AWQ, GPTQ) are designed for single-task, single-model scenarios. They assume a smooth, typical pre-trained or fine-tuned weight distribution.
- **The Delta:** Standard PTQ techniques fail to handle the unique, heavy-tailed, scale-mismatched distributions that arise from merging multiple task-specific vectors. QP-Merge specifically tackles these multi-task joint distributions.

### 3. Outlier Decoupling (ORD) vs. Hybrid Quantization
- **Prior Work:** SqueezeLLM and SpQR decouple extreme weight outliers in LLMs into high-precision sparse matrices while keeping the dense remainder in low-bit formats. LLM.int8() isolates activation outliers dynamically during inference.
- **The Delta:** QP-Merge adapts this concept to the *model merging* domain by extracting outliers specifically from the *task-vector updates* ($\Delta W_t = W_t - W_{\text{base}}$) rather than the base weights or activation channels. Masking and routing outlier updates allows the base weights and the dense updates to be quantized homogeneously while preserving the sparse, highly unique task updates in high precision.

### 4. Calibration (QE-Calib) vs. Activation Scaling
- **Prior Work:** SmoothQuant scales weights and activations channel-wise with mathematically equivalent scaling factors ($D_l$ and $D_l^{-1}$) to mitigate activation outliers, maintaining exact unquantized equivalence.
- **The Delta:** In a multi-task merging scenario, different tasks have conflicting and mismatched activation ranges, meaning no single scaling matrix $D_l$ can scale activations for all tasks equivalently. QP-Merge abandons strict mathematical equivalence and instead applies diagonal scaling $D_l$ permanently to the weight updates *without* scaling activations at inference. It compensates for this structural change by using an end-to-end embedding MSE loss to optimize the scales on a tiny calibration set of unlabeled target domain samples.

## Characterization of Novelty
The novelty of this paper can be characterized as **incremental to moderate**:
1. **Conceptual Novelty (Moderate):** Co-designing post-training quantization and model merging is a very practical and highly relevant direction. Addressing the "merging-quantization gap" is a major step forward for edge AI.
2. **Technical Novelty (Incremental):** The two primary techniques are adaptations of existing methods from single-model PTQ literature:
   - *Outlier-Residual Decoupling (ORD)* is a direct adaptation of the dense-sparse decomposition seen in SqueezeLLM and SpQR, applied specifically to task-vector updates.
   - *Quantization-Error Aware Scale Calibration (QE-Calib)* is a variation of weight-scaling search (similar to AWQ or SmoothQuant-style search) optimized end-to-end, but without the corresponding inverse activation scaling (which changes the unquantized function mapping).
3. **Synergy (Moderate):** The combination of these two techniques to address the distinct bottlenecks (heavy-tailed weight updates and conflicting activation ranges) is logical, and the empirical results show they work well together, particularly under aggressive low-bit constraints (INT4).
