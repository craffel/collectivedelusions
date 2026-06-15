# 3. Soundness and Methodology

## Clarity of Description
The description of the 2D-STEM framework and its sub-components is **excellent** and highly transparent:
- **Problem Setup:** Clearly defines the dynamic multi-task expert serving scenario, the ensembling equation, and the SABLE nearest-centroid routing baseline.
- **Mathematical Formulations:** All equations (2D bilinear recurrence, Softmax similarity, stream-level similarity, Power-Law gating, Coordinate-Prior boundary condition) are explicitly written out with clearly defined variables.
- **Structural Connections:** Deeply connects the 2D-STEM recurrence to classical signal processing, discrete-time 2D autoregressive models of order 1 (AR(1)), first-order 2D digital filters, and 2D Kalman filtering.
- **Appendix Code:** Appendix A provides a clean, self-contained PyTorch implementation of the 2D-STEM routing module. This includes coordinate prior boundary conditions and power-law ATG, making the operational details highly concrete.

## Appropriateness of Methods
The proposed methods are mathematically elegant, physically intuitive, and highly appropriate:
- **Recursive Filtering:** Using a first-order 2D digital filter is a highly appropriate and lightweight way to low-pass filter high-frequency layer-wise (depth) and sample-wise (temporal) noise without adding parameterization or optimization overhead.
- **Simplex Preservation Proof:** The inductive proof of Theorem 1 is mathematically correct and provides a robust foundation for avoiding runtime projection/re-normalization overhead, which is a major system bottleneck.
- **Power-Law Gating (ATG-PL):** Proposing a power-law exponent ($\gamma$) to squash the transition similarity is a mathematically sound and effective way to resolve the upward bias of cosine similarities in non-negative coordinate spaces.
- **Coordinate-Prior Boundary Condition:** The formulation is highly appropriate, resolving the physical momentum cancellation of the Raw-Weight Boundary while avoiding the regularization/accuracy drag of the Uniformly-Buffered Boundary.
- **Sparsified Gating (Top-$k$ Coordinate Masking):** Highly appropriate for scaling, mathematically guaranteeing that transition similarity collapses to zero ($Sim_{\text{switch}}^{\text{sparse}} = 0.0$) in high-dimensional expert spaces.
- **MLP Coordinate Mapper:** Appropriate and highly practical for extremely fine-grained tasks, training in under 3 seconds on the tiny calibration split to guarantee semantic separability.

## Potential Technical Flaws or Limitations
1. **Trade-off on Maximum-Frequency Switch Streams:** On purely chaotic heterogeneous streams where tasks switch at every single step, carrying sequence-level temporal memory is mathematically counterproductive. The non-zero coordinate similarities under manifold overlap force a small residual temporal history to propagate, causing a small accuracy drop ($0.32\%$ on Orthogonal and $1.97\%$ on Overlapping) compared to stateless SABLE. The authors are highly transparent about this trade-off, noting that real-world edge streams (e.g., multi-turn chatbot conversations, sensor streams) are block-stable, making this a minor penalty in exchange for massive noise filtering.
2. **Isotropic Representation Assumption:** Nearest-centroid cosine routing assumes that the representation space remains relatively isotropic and that centroids remain representative under covariate shifts. The authors mitigate this by:
   - Showing high robustness to calibration set size ($N_{\text{cal}} = 5$ samples), where 2D-STEM's low-pass filter buffers against centroid deviations.
   - Proposing an explicit OOD Fallback Policy (Uniform fallback or Temporal State Bypass) when the maximum cosine similarity collapses below an OOD threshold $\delta_{\text{OOD}}$.
3. **Simulation-Focused Evaluation:** While the authors conduct a rigorous simulation study (ACS) and validate CLS representation trajectories on a physical pre-trained Vision Transformer (`vit_tiny`), they do not measure downstream task classification accuracies or GPU/DRAM memory transfer statistics on an end-to-end LoRA-blended edge application. However, the authors:
   - Outline a clear physical deployment and hardware compilation roadmap (ONNX, TensorRT, vLLM/Punica).
   - Profile the execution latency, proving that 2D-STEM executes in 1,436.20 $\mu$s per step, achieving a **49.5% reduction in serving-time latency** compared to the ChemMerge Dynamic ODE baseline.
   - Formulate concrete future work for end-to-end evaluations on a ViT-Base backbone.

## Reproducibility
The reproducibility of this paper is **excellent**:
- Every hyperparameter (e.g., $\beta_{\text{depth}} = 0.4$, $\beta_{\text{temp}, 0} = 0.4$, $\gamma = 3$, temperature $\tau = 0.10$, calibration size $N_{\text{cal}} = 64$) is explicitly specified.
- Grid sweeps are provided for the momentum coefficients ($\beta_{\text{depth}}, \beta_{\text{temp}, 0}$) and the sharpening exponent ($\gamma$).
- The sandbox environment (ACS) parameters (representation dimension $D=192$, layers $L=14$, adapted layers $4$ to $14$, experts $K=4$, LoRA rank $r=8$) are clearly detailed.
- The complete PyTorch implementation in Listing 1 provides all necessary code to reproduce the 2D-STEM router.
