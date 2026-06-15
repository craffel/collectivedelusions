# 5. Presentation and Impact Evaluation

## Presentation Quality

The paper is exceptionally well-written, clearly structured, and easy to follow. It adheres strictly to academic standards and satisfies all LaTeX layout requirements.

Key strengths in presentation include:
- **Clear Structure and Narrative Flow:** The introduction clearly articulates the practical, real-world deployment challenges of multi-task systems under low-bit quantization on resource-constrained devices. It moves logically into related work, detailed methodology, and systematic experiments.
- **Rigorous Mathematical Notation:** The methodology contains precise equations for the uniform quantization operator, quantized cosine similarity, temperature softmax routing, pre-computed scale recovery factors ($\beta_k^{(l)}$), GMM density estimation, and dyadic fixed-point scaling.
- **High-Quality Tables and Figures:** Tables and figures are placed logically and have professional, descriptive captions.
  - **Table 1** outlines serving trade-offs on edge devices.
  - **Table 2** provides a detailed accuracy sweep across Homogeneous and Heterogeneous evaluation streams.
  - **Table 3** details the physical profiling emulation on an STM32H7.
  - **Figures 1 and 2** visually demonstrate the performance curves and batch size heterogeneity sweeps.
  - **Appendix Tables/Figures** contain thorough sweeps over OOD thresholds, fallback policies, routing layer sensitivity, and microcontroller hardware profiling.
- **Scientific Honesty and Transparency:** The authors are highly transparent and honest about the limitations and methodology assumptions of their work:
  - Disclosing the synthetic nature of the Coordinate Sandbox and the use of coordinate-isolated features, followed by a rigorous task overlap sweep ($\Omega \in [0.00, 1.00]$).
  - Disclosing that the calibration dataset extracts activations from clean, full-precision streams to avoid compounding noise.
  - Disclosing that the microcontroller profiling is conducted via cycle-accurate emulation rather than direct on-board hardware execution.
  - Explicitly detailing the assembly-level compilation challenges (mixed-precision GEMM, bit-packing, cache locality).

---

## Potential Practical and Scientific Impact (Pragmatist Perspective)

The paper has substantial potential for both practical systems deployment (TinyML) and deep scientific research:

### A. Dynamic Serving and Flash Storage Scaling
On low-power edge microcontrollers with strict non-volatile flash limits, storing separate full-precision expert models or parallel ensembles is physically impossible.
- SA-QAB introduces a highly efficient storage scaling formula: $M(K) = 252.0 + K \times 27.2$\,KB.
- On a standard 2MB (2048\,KB) Flash target (e.g., STM32H753XI), SA-QAB can support up to **66 concurrent task experts** in a single dynamic serving registry.
- On a 1MB Flash target (e.g., STM32F7), it supports up to **28 concurrent experts**.
- This enables massive multi-expert scaling on low-cost edge chips, whereas full-precision ensembling would exceed these limits at $K \ge 2$.

### B. On-Device SRAM and Compute Footprint
By executing near-sparse routing, SA-QAB runs with a constant $O(1)$ expert compute footprint and requires only **360.75 KB** of active SRAM. This sits comfortably within the 1MB SRAM limit of microcontrollers, leaving plenty of headroom ($>60\%$) for other system tasks, whereas parallel ensembling (1224.75 KB) would immediately crash the board.

### C. The "Rejection Accuracy Boost" Discovery
One of the most scientifically profound and fascinating contributions of the paper is the empirical discovery surrounding GMM rejection fallback dynamics:
- Bypassing or averaging out expert adapters for atypical, low-likelihood samples under GMM filtering actually **improves** overall joint classification accuracy (raising it from **77.50%** to **78.00%** at $\eta = -245.0$).
- The paper provides a deep, representational explanation: atypical samples are prone to routing failures under heavy quantization noise. When routed to an incorrect specialized expert, the mismatched adapter acts as an *adversarial perturbation*, causing active representational corruption and collapsing classification accuracy to **29.6%**.
- By bypassing these adapters under Standard Fallback or averaging them out under Soft Fallback, the system shields atypical activations from specialized representational corruption, restoring accuracy.
- This reverses the traditional assumption of a "rejection penalty" and establishes the low-power diagonal GMM filter as not just an OOD shield, but an **in-distribution confidence gate** that dynamically protects representation safety under severe low-bit noise. This is exactly the kind of robust, noise-resilient systems design that a pragmatist values.
