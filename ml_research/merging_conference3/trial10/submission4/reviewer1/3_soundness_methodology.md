# Evaluation Stage 3: Soundness and Methodology

## Clarity of the Description
The methodology of QA-Merge is described with exceptional clarity and detail:
- The symmetric uniform quantization operator is mathematically formalized.
- The rationales for QCC, STE Gating, EF-Smooth (with PI-SPA), and AEF are well-articulated.
- Algorithms for the complete low-precision loop (Algorithm 1) and Hamilton simplex projection (Algorithm 2) are provided.
- Appendix sections extensively document microarchitectural Estimates, PyTorch integration details, and hyperparameter tables.

## Appropriateness of Methods
The proposed techniques are highly appropriate and logically target the specific causes of quantization collapse:
- **QCC** addresses centroid shift/overlap by quantizing final averages and utilizing scale-invariant cosine similarity.
- **STE Gating** resolves the zero-derivative vanishing gradient problem of the rounding operator during parameter tuning.
- **EF-Smooth** addresses weight rounding noise via DSP-style error diffusion.
- **AEF** solves the "Small-Step Quantization Bottleneck" by residually accumulating activation rounding errors across layers.

## Mathematical Rigor and Potential Flaws
- **Theorem 3.1 & Proof:** The proof of the telescoping bounded representational error of AEF is mathematically sound. It successfully demonstrates that under AEF, the cumulative quantization error is bounded relative to the "quantized-pull accumulated trajectory" (the trajectory defined by intermediate quantized states and quantized pulls).
- **Subtle Conceptual Detail (Remark 1):** The authors commendably include Remark 1, acknowledging that the "ideal accumulated trajectory" in Theorem 3.1 uses pull vectors calculated from already-quantized states, rather than the true continuous unquantized Float32 trajectory. Since intermediate quantized states affect subsequent pull calculations, the trajectory can mathematically diverge from the true Float32 model's path. While the authors state that this divergence is empirically small and benign, the theoretical bound is technically local to the quantized trajectory rather than global to the original unquantized model. This is an important distinction that should be noted.

## Reproducibility
The reproducibility of the work is exceptionally high:
- Complete hyperparameter details are compiled in Table 5 (Appendix K).
- A complete, self-contained, runnable PyTorch script (`submission/toy_qamerge_lora.py`) is provided, demonstrating QCC, PI-SPA, and AEF on a 3-layer dynamic LoRA-mixture layer.
- Another Python script (`submission/sweep_smoothquant.py`) reproducing the SmoothQuant sensitivity sweep is provided.
- The algorithms are clearly pseudocoded in the appendix.
