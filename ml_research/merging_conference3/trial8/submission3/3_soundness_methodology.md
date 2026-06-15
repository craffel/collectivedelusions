# 3. Soundness and Methodology Check

We evaluate the mathematical, theoretical, and logical soundness of **Scale-Aligned Quantized Activation Blending (SA-QAB)**.

## Mathematical & Theoretical Rigor

The proposed framework is highly rigorous and provides a detailed mathematical formulation that aligns perfectly with low-level, integer-only hardware execution. We highlight several exceptionally sound mathematical derivations:

### A. Heterogeneous Quantization with STE
- The uniform symmetric quantization operator $Q_b(X)$ is standard and mathematically correct.
- The integration of the **Straight-Through Estimator (STE)** in simulating quantization during training/fine-tuning (in the forward pass of `SandboxViT` in `run_experiments.py`) is standard and ensures correct gradient flow during Quantization-Aware Fine-Tuning (QAT).

### B. Quantization Scale Recovery (QSR) Formulation
- The scale-alignment factors $\beta_k^{(l)}$ are formulated as:
  $$\beta_k^{(l)} = \frac{\mathbb{E}_{s \in \mathcal{C}_k} \left[ \| \text{Adapter\_FP}_k(h_s^{(l-1)}) \|_2 \right]}{\mathbb{E}_{s \in \mathcal{C}_k} \left[ \| \text{Adapter\_Quant}_k(h_s^{(l-1)}) \|_2 \right]}$$
- This formulation is theoretically sound. It models the systematic "scale contraction" that occurs when adapters are compressed to low-bit integers, providing a training-free scaling correction factor to align activation magnitudes.

### C. Accumulator Overflow Immunity (Appendix B.6)
- One of the most impressive sections of the paper is the **Accumulator Overflow Analysis**. The authors calculate the worst-case accumulated values for INT4-weight/INT8-activation and INT8-weight/INT8-activation matrix multiplications.
- For a 192-channel dimension ($D=192$):
  - **INT4 Base Model Accumulation:** Bounded by $192 \times 127 \times 7 = 170,688 \approx 1.7 \times 10^5$.
  - **INT8 Adapter Model Accumulation:** Bounded by $8 \times 127 \times 127 = 129,032 \approx 1.3 \times 10^5$.
- This is compared against a standard 32-bit hardware register's capacity ($2^{31}-1 \approx 2.15 \times 10^9$). The safety margins of **12,500x** and **16,600x** mathematically guarantee that no intermediate register overflow can ever occur during physical edge deployment. This is extremely rigorous.

### D. ZCA Pre-whitening for Diagonal GMMs (Appendix B.7)
- Standard diagonal GMMs ignore cross-channel correlations ($O(D)$ complexity). Full-covariance GMMs capture them but require $O(D^3)$ determinants or $O(D^2)$ matrix math, which collapses edge hardware execution.
- The paper proposes to pre-compute a symmetric whitening matrix $W_{\text{zca}} = \Sigma_{\text{calib}}^{-1/2}$ offline, and fuse it directly into the preceding base weight matrix: $W_{\text{base}}^{(3)'} = W_{\text{base}}^{(3)} \cdot W_{\text{zca}}$.
- Since the whitening projection is linear, this de-correlates the channels on-device with **zero runtime latency and zero memory overhead**, making the diagonal GMM's axis-aligned assumption mathematically exact. This is elegant, correct, and highly innovative.

---

## Gaps, Weaknesses & Areas for Improvement

While the methodology is solid, we identify some minor areas where the clarity or rigor could be improved:

### A. Centroid Extraction Ambiguity
- In Section 3.2, the authors define task centroids as:
  $$\mu_k^{(3)} = \frac{1}{|\mathcal{C}_k|} \sum_{s \in \mathcal{C}_k} h_s^{(3)}$$
- However, they do not explicitly specify in the main body whether $h_s^{(3)}$ is extracted from the clean full-precision (FP16) backbone or the quantized (INT4) backbone.
- *Resolution from Code:* In `run_experiments.py`, the code computes both `centroids_layer3_fp` (FP16) and `centroids_layer3_quant` (INT4). At inference, `get_quantized_zca_coefficients` uses `centroids_layer3_quant[k]`.
- *Actionable Feedback:* The authors should clarify this choice in Section 3.2, explaining that using quantized-space centroids during calibration ensures alignment with the test-time quantized features, absorbing early-stage quantization noise. (Note: The paper has proactively added a clarification in Section 3.3 under the title *Transparent Disclosure on Calibration Inputs and Ambiguity Resolution* explaining this design decision.)

### B. GroupNorm/LayerNorm Pre-normalization Assumption
- In Section 3.2, the authors mention that:
  > "if layers are separated by a GroupNorm or LayerNorm, the feature vectors are inherently pre-normalized to unit variance, which allows practitioners to completely omit the runtime feature norm calculation..."
- While this is true for standard Transformer layers (which have LayerNorm before block outputs), the features at Layer 3 in a ViT-Tiny might not be fully normalized if there's an additive residual connection or if LayerNorm is applied *before* the attention blocks (Pre-LN).
- The authors should briefly clarify if the Coordinate Sandbox's features are normalized or if they rely on the fast fixed-point square-root approximation for unit-sphere projection.

## Summary of Soundness
The paper exhibits **exceptional soundness**. The mathematical equations are precise, well-reasoned, and align perfectly with real-world low-level integer execution constraints. The detailed analyses of accumulator overflow and ZCA pre-whitening elevate the theoretical rigor of the methodology far beyond standard TinyML papers.
