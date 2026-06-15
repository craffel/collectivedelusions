# 5. Impact and Presentation Evaluation

## Major Strengths
1. **Elegant Mathematical Formulation:** The projection of layer-wise merging coefficients onto a low-degree continuous polynomial subspace of layer depth is a mathematically simple and elegant solution to regularize test-time adaptation. The normalization of layer depth guarantees scale invariance across depths.
2. **Deep Systems-Level Analysis:** The theoretical peak SRAM footprint analysis is exceptionally thorough. The paper provides a mathematically exact derivation of the **158.40 MB** activation cache required for backpropagation on a ViT-Tiny backbone, clearly explaining why first-order gradient descent is physically unviable on microcontrollers.
3. **Rigorous Physical Hardware Modeling:** The inclusion of modeled on-device hardware latency and energy profiles for popular edge processors (ARM Cortex-M7 STM32H7 and RISC-V GAP8) is highly commendable. It grounds the systems-level claims on actual physical edge silicon.
4. **Generalization Formulation in Appendix:** The discussion in the appendix of condition numbers of the monomial Vandermonde matrix and the generalization to Chebyshev orthogonal polynomial bases is rigorous and provides a clear scaling pathway for very deep architectures.
5. **Excellent Visualization:** The visualization of the learned coefficient profiles (`results/coefficient_profile.png`) clearly demonstrates that the polynomial constraint successfully recovers smooth, physically stable trajectories compared to the wild, jagged trajectories of unconstrained merging.

---

## Major Areas for Improvement

### 1. Resolve the "Utility Dilemma"
The most critical weakness of the paper is that the proposed method is not the optimal choice in any practical deployment scenario:
- **Offline (Server-side):** Standard **AdaMerging (Adam) + post-hoc quantization** outperforms Q-PolyMerge (Adam) by **+2.51% (8-bit)** and **+1.33% (4-bit)**.
- **On-Device (Edge):** Bypassing adaptation and deploying a **naive unadapted M-then-Q model** is superior or statistically equivalent to the edge-viable **Q-PolyMerge (ES)**, achieving **55.11% vs. 51.03% (8-bit)** and **42.92% vs. 43.05% (4-bit)** while requiring zero on-device computational overhead.
The authors must explain this discrepancy. Is there an alternative zero-order optimization strategy or a different loss formulation that allows Q-PolyMerge (ES) to significantly outperform the naive unadapted baseline?

### 2. Validate on Standard Training Protocols
The current evaluation is restricted to a downscaled protocol where experts are trained on only 512 images, resulting in highly undertrained, low-performing models. The authors must validate Q-PolyMerge on standard, fully trained expert models to ensure that the observed behaviors and the "Overfitting-Optimizer Paradox" are not artifacts of the low-data training regime.

### 3. Implement and Validate Fully-Integerized Activations
While the authors provide a "Blueprint for Fully-Integerized Activation and Operator Execution" in Appendix B.5, their experiments assume floating-point activations. Since low-cost microcontrollers lacking FPUs must emulate floating-point operations in software (incurring severe latency and energy penalties), the practical on-device viability of Q-PolyMerge is highly limited without fully quantized activations. The authors should implement and evaluate their proposed W8A8/W4A8 integer arithmetic pipeline to substantiate their edge-efficiency claims.

---

## Overall Presentation Quality
The presentation quality is **excellent**. The paper is clearly written, exceptionally well-structured, and easy to follow. The mathematical notation is precise and consistent throughout the text. The tables and figures are well-formatted and informative. The authors are commendable for their thoroughness, particularly in providing detailed mathematical derivations, systems-level analyses, and alternative basis formulations in the appendix.

---

## Potential Impact and Significance
The potential impact of the paper is **moderate**. 
- On one hand, the core idea of restricting layer-wise parameters to continuous polynomial trajectories across layer depth is a powerful, general, and valuable software prior. It has potential applications beyond model merging, such as in parameter-efficient fine-tuning (PEFT), federated learning, and dynamic network routing.
- On the other hand, the current empirical results fail to demonstrate any practical benefit for the only edge-viable pathway (zero-order ES) over a simple unadapted model. Until this utility gap is resolved, the practical impact on physical edge deployments will remain limited.
If the authors can successfully bridge this zero-order search gap (e.g., through more advanced gradient-free search or better loss proxies), Q-PolyMerge could become a foundational framework for low-bit model merging on resource-constrained hardware.
