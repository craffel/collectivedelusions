# Peer Review: Q-PolyMerge

## 1. Summary of the Paper
The paper proposes **Q-PolyMerge**, a parameter-efficient framework designed to enable quantization-aware, multi-task model merging on resource-constrained edge devices. Multi-task model merging (e.g., Task Arithmetic) combines specialized expert checkpoints into a single unified network, but edge devices strictly require post-training quantization (PTQ) to low bits (such as INT8 or INT4) to meet physical hardware limits. Traditional test-time adaptation (TTA) methods optimize layer-wise merging coefficients on-device via a small calibration stream. However, the authors argue that this unconstrained high-dimensional optimization suffers from the **Overfitting-Optimizer Paradox**—fitting transductive statistical noise on tiny data streams and learning jagged, physically nonsensical coefficient trajectories.

To resolve this, Q-PolyMerge projects the layer-wise coefficients onto a low-degree continuous polynomial subspace of normalized layer depth. Rather than optimizing independent layer-wise parameters, the optimizer adjusts a small set of polynomial coefficients of degree $d$ (typically $d=2$). This reduces the parameter search space by over 78% (from 56 to 12 parameters) and acts as a smooth continuous regularizer. Q-PolyMerge supports first-order optimization via the Straight-Through Estimator (STE) and zero-order optimization via a 1+1 Evolution Strategy (1+1 ES). Evaluating on a Vision Transformer (ViT-Tiny) across four image benchmarks (MNIST, FashionMNIST, CIFAR-10, SVHN) under 8-bit and 4-bit PTQ, the authors show that Q-PolyMerge stabilizes adaptation, reduces variance across random seeds, and achieves smooth learned coefficient profiles.

---

## 2. Strengths and Weaknesses

### Strengths
1. **Elegant Formulation:** Constraining layer-wise merging coefficients to a continuous, low-degree polynomial trajectory across layer depth is an elegant and simple structural prior. Normalizing the layer depth to $[0, 1]$ ensures scale invariance and numerical stability across models of varying depths.
2. **Exhaustive Systems-Level Analysis:** The theoretical peak SRAM footprint analysis in the appendix is exceptionally thorough. The paper provides an exact mathematical derivation of the **158.40 MB** activation cache required for backpropagation on a ViT-Tiny backbone, clearly highlighting why first-order gradient descent is physically unviable on edge microcontrollers.
3. **Rigorous Hardware Modeling:** The authors ground their systems-level claims on actual physical silicon by providing modeled on-device latency and energy profiles for popular edge processors (ARM Cortex-M7 STM32H7 and RISC-V GAP8).
4. **Generalization Formulation:** The appendix includes a rigorous analysis of the Vandermonde matrix condition numbers and introduces Chebyshev orthogonal polynomial bases to prevent Runge's boundary oscillations, providing a clear scaling pathway for very deep architectures.
5. **Excellent Visualization:** The visualization of the learned coefficient profiles clearly demonstrates that the polynomial constraint successfully recovers smooth, physically stable quadratic curves compared to the wild, jagged trajectories of unconstrained merging.

### Weaknesses
1. **The "Utility Paradox" of On-Device Zero-Order Adaptation:** The authors build a massive argument around on-device viability and how Q-PolyMerge makes adaptation feasible on microcontrollers. They argue that first-order backpropagation is physically unviable on-device, leaving zero-order search (1+1 ES) as the only physically viable on-device pathway. However, a rigorous empirical comparison reveals a critical contradiction:
   - **Under 8-Bit PTQ:** The naive, unadapted **Merge-then-Quantize (M-then-Q)** baseline (which requires **zero** test-time computation and **zero** on-device SRAM) achieves **55.11 $\pm$ 0.22%** average accuracy. The proposed edge-viable **Q-PolyMerge (ES)** achieves only **51.03 $\pm$ 4.35%** average accuracy. *Performing zero-order adaptation actively degrades accuracy by -4.08% compared to doing nothing.*
   - **Under 4-Bit PTQ:** The unadapted **M-then-Q** baseline achieves **42.92 $\pm$ 2.06%** average accuracy, while the proposed **Q-PolyMerge (ES)** achieves **43.05 $\pm$ 1.90%**. *The on-device adaptation provides a negligible and statistically insignificant +0.13% improvement, which is completely within the standard deviation.*
   
   If the only edge-viable optimization pathway is either actively harmful or useless compared to deploying a simple unadapted model, the core claim that Q-PolyMerge provides a viable on-device adaptation pipeline is severely undermined.

2. **Offline Baseline Anomaly (AdaMerging Outperforms Direct Quantization-Aware Merging):** 
   A comparison of the first-order results reveals that **AdaMerging (Adam)**—which represents optimizing coefficients on full-precision models (server-side) and subsequently quantizing the merged weights—strictly outperforms direct quantization-aware optimization:
   - **Under 8-Bit PTQ:** AdaMerging (Adam) achieves **62.27 $\pm$ 0.43%**, strictly outperforming Q-PolyMerge (Adam) at **59.76 $\pm$ 1.22%** (+2.51% absolute improvement).
   - **Under 4-Bit PTQ:** AdaMerging (Adam) achieves **50.20 $\pm$ 2.21%**, strictly outperforming Q-PolyMerge (Adam) at **48.87 $\pm$ 1.42%** (+1.33% absolute improvement).
   
   This shows that navigating the non-smooth, flat plateaus and step-cliffs of the rounded weight landscape via the Straight-Through Estimator actually harms optimization compared to optimizing in a smooth unquantized space and then quantizing. Thus, for any offline (server-side) scenario, Q-PolyMerge is not the optimal choice.

3. **Downscaled Experimental Protocol and Undertrained Experts:**
   To evaluate their method, the authors use a downscaled training pipeline where individual experts are fine-tuned on only **512 images** per dataset. This results in extremely undertrained, low-performing experts:
   - In Table 4 (Appendix), the full-precision individual experts achieve an average accuracy of only **79.20 $\pm$ 0.85%** (MNIST: 84.12%, FashionMNIST: 78.85%, CIFAR-10: 82.48%, SVHN: 71.37%). In standard literature, a ViT-Tiny fine-tuned on these benchmarks easily achieves over **98% (MNIST)** and **90%+ (CIFAR-10)**.
   - Undertrained models have very different weight distributions, gradient dynamics, and representation connectivity than fully converged, sharp minima. It is highly questionable whether the observed empirical trends and the "Overfitting-Optimizer Paradox" generalize to standard, fully trained, high-accuracy experts. The paper lacks validation on standard, full-scale training setups to confirm that the findings are not an artifact of the severely downscaled training pipeline.

4. **Floating-Point Activation Assumption on Microcontrollers:**
   The proposed "integer-weight edge pipeline" keeps activations and layer normalizations in floating-point format. On ultra-low-power microcontrollers lacking hardware FPUs, executing activations in floating-point format requires software emulation, which incurs severe latency and energy penalties. While the authors provide a "Blueprint for Fully-Integerized Activation and Operator Execution" in Appendix B.5, this is purely theoretical and is not implemented or validated in their experiments.

---

## 3. Detailed Dimension Ratings

### Soundness: Fair
The mathematical formulation of the continuous polynomial subspace and the first-order gradient derivation under STE are technically sound and elegant. However, the soundness is rated as **fair** due to the critical **utility paradox** of the on-device zero-order pathway: the only physically viable edge-adaptation method (zero-order ES) performs significantly worse than (in 8-bit) or statistically equivalent to (in 4-bit) the simple unadapted model (M-then-Q). The proposed edge adaptation has no practical advantage, making the on-device edge viability claims purely academic.

### Presentation: Excellent
The paper is exceptionally well-written, clearly structured, and easy to follow. The mathematical notation is precise and consistent. The figures and tables are informative and highly professional. The authors are commendable for their thoroughness, particularly in providing detailed mathematical derivations, systems-level analyses, and modeled hardware profiling metrics in the appendix.

### Significance: Fair
The significance is rated as **fair** due to two main limitations:
1. **The Utility Dilemma:** There is no deployment scenario presented where Q-PolyMerge is the optimal or even a beneficial choice over existing, simpler baselines. If offline (server-side), standard AdaMerging followed by quantization is superior. If on-device (edge), deploying a naive unadapted M-then-Q model is superior or equivalent, with zero computational overhead.
2. **Lack of Scale:** The evaluation is restricted to extremely undertrained experts on a downscaled training pipeline, raising concerns about the generalizability of the findings to fully converged, high-accuracy models.

### Originality: Good
The idea of restricting layer-wise merging coefficients to a low-degree continuous polynomial trajectory over normalized layer depth is a neat and relatively novel contribution to the model merging literature. It incorporates a valuable inductive bias that adjacent layers share functional similarity, and serves as a powerful regularizer to mitigate overfitting under data scarcity.

---

## 4. Overall Recommendation

**Rating: 3: Weak Reject**

### Justification
Q-PolyMerge presents an elegant, mathematically sound, and well-motivated framework for regularizing layer-wise merging coefficients via a continuous polynomial subspace of layer depth. The systems-level SRAM derivations, modeled hardware profiles, and qualitative coefficient visualizations are of high quality and provide excellent insights into on-device constraints.

However, from a rigorous empirical perspective, the paper in its current form has major weaknesses that outweigh its merits:
1. **No Practical Benefit for the Edge-Viable Pathway:** The only physically viable edge adaptation method (zero-order ES) fails to outperform the simple unadapted model (M-then-Q), actively degrading accuracy under 8-bit quantization and barely matching it under 4-bit quantization.
2. **Offline Baseline Anomaly:** Offline gradient descent in full precision followed by post-hoc quantization (AdaMerging Adam) strictly outperforms the direct quantization-aware Q-PolyMerge.
3. **Downscaled Training:** The experts are extremely undertrained (trained on only 512 images), leaving it unclear if the findings generalize to standard, high-accuracy experts.

Because there is no practical scenario where the proposed method is beneficial over existing, simpler baselines, the core claim of delivering a viable, high-performance on-device adaptation pipeline is not empirically supported. I encourage the authors to bridge this zero-order search gap (e.g., through more advanced gradient-free search or better loss proxies) and validate their method on standard, full-scale training setups to make the framework practically significant.
