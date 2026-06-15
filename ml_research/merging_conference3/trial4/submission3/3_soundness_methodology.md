# 3. Soundness and Methodology Check

## 3.1. Mathematical Soundness and Theoretical Grounding
The mathematical foundations of this paper are exceptionally rigorous, elegant, and tightly integrated into the core narrative.

1. **Re-Quantization Silence Mechanics:** The mathematical description of the rounding threshold ($|\frac{\Delta W}{s}| \ll 0.5$) clearly and simply demonstrates why low-rank updates are zeroed out by post-hoc uniform quantization.
2. **SAWS Closed-Form Derivation:**
   - The adaptation norm ratio $\gamma^l$ is well-formulated.
   - The weight alignment factor $c^l$ is derived by minimizing the squared Frobenius norm distance between the original unquantized merged weight tensor and the scaled, re-quantized weight tensor:
     $$c^l = \arg\min_c \| W^l_{\text{merged}}(\Lambda) - c \tilde{W}^l_{\text{final, saws}}(\Lambda) \|^2_F$$
     The authors show that expanding this quadratic objective and taking the partial derivative yields:
     $$c^l = \frac{\langle W^l_{\text{merged}}(\Lambda), \tilde{W}^l_{\text{final, saws}}(\Lambda) \rangle}{\|\tilde{W}^l_{\text{final, saws}}(\Lambda)\|^2_F}$$
     This derivation is mathematically precise, elegant, and highly practical.
3. **Representation Scale Preservation Dilemma:** The proof that dividing the output activation by the boosting factor $\gamma^l$ scales down the pre-trained base representations by $10\times$ to $100\times$ is mathematically solid. It clearly exposes a fundamental representational trade-off in merged PEFT adapters.
4. **Global vs. Channel-Wise Scaling:** The deconstruction of how channel-wise scaling row-by-row warps the representation geometry of task vectors, and why global layer-wise scaling is necessary under per-tensor constraints, is highly sophisticated and mathematically sound.
5. **Straight-Through Estimator (STE) Gradient Mismatch:** The analysis of STE gradient noise under aggressive 4-bit constraints provides an excellent explanation of why standard SGD stalls while the Adam optimizer's adaptive second-moment tracking successfully acts as a temporal noise filter.
6. **Double Quantization Noise & Format Shift:** The relative Frobenius reconstruction error analysis across layers clearly demonstrates how the shift from QLoRA's native NF4 format to deployment INT formats introduces substantial base-weight representation error, independent of any adapter weights.

## 3.2. Evaluation of Research Methodology and Baselines
The research methodology is exemplary, aligning perfectly with the rigorous standards of "The Methodologist" persona:

- **Multi-Axial Scope:** Evaluating across 4 distinct quantization configurations (INT8 Symmetric Per-Channel, INT4 Symmetric Per-Channel, INT4 Asymmetric Per-Channel, INT4 Symmetric Per-Tensor) provides a comprehensive picture of how quantization parameters affect merged models.
- **Robust Baselines:** The paper evaluates all critical baselines: unmerged high-precision experts (performance ceiling), naive FP16 merging (ceiling of unquantized merging), naive re-quantization, decoupled "Quantize-then-Merge" (co-existence strawman), and post-hoc quantized AdaMerging (isolating the necessity of quantization-aware optimization).
- **Ablation Studies (Appendix):**
  - Section A.4 presents a thorough sensitivity sweep of the SAWS scaling constant $\alpha$.
  - Section A.5 implements and evaluates three supervised and regularized variants of QA-ACS (Supervised, Regularized, and Supervised + Regularized), demonstrating how proper constraints stabilize test-time optimization.
  - Section B (Ablation of Global vs. Channel-wise SAWS Scaling) empirically validates the representation geometry warp hypothesis across all 4 quantization schemas.
- **DRAM Bandwidth Scaling & Cache-Fitting:** The physical CPU latency profiling and the deconstruction of cache-fitting on a 128-core Xeon CPU is a brilliant piece of hardware-level analysis that exposes why sequential co-existence works on small networks but collapses under DRAM bottlenecks on large LLMs.

## 3.3. Soundness Rating
**Rating: Excellent**  
The paper's claims are backed by rigorous mathematical proofs, elegant derivations, and highly thorough empirical validations. All potential confounding variables (e.g., task interference, format-shift noise, hardware cache effects) are systematically isolated and addressed, establishing a gold-standard methodological design.
