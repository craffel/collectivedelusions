# Experimental Setup and Empirical Check

## 1. Experimental Design and Real-World Compatibility

The experimental evaluation in the paper is exceptionally rigorous, comprehensive, and well-designed, successfully bridging the gap between simulation and hardware:

1. **High-Fidelity Representation Simulation (Coordinate Sandbox):**
   * The experiments are conducted in the **Analytical Coordinate Sandbox (ICS)** environment (14 layers, $D=192$ dimensions) with 4 visual task signatures (MNIST, Fashion-MNIST, CIFAR-10, SVHN).
   * While the sandbox operates on synthetic vectors to isolate representation-space dynamics, the paper provides a **mathematical proof of generalization (Theorem 3.1)** and a **standalone PyTorch demonstrator (`toy_qamerge_lora.py`)** implementing a physical, multi-expert dynamic LoRA-mixture layer. This dual-layered verification ensures that the conclusions are mathematically generalizable to real-world neural network manifolds.
2. **Exhaustive Comparison across 18 Configurations:**
   * The authors compare QA-Merge against 6 major baseline configurations across Float32 and Quantized settings in both small-sample ($N_{\text{cal}} = 64$) and large-sample ($N_{\text{cal}} = 4000$) regimes.
   * Testing with a simulated weak SVHN expert (calibrated to **22.80%** accuracy) acts as an excellent representational distractor, demonstrating that the scale-invariant cosine similarity gating and STE-based optimization successfully isolate noisy pathways under quantization.
3. **Thorough Microarchitectural and On-Device Validation:**
   * Crucially, the authors deploy compiled integer coordinate propagation kernels on a physical **ARM Cortex-M7 (STM32H753XI) microcontroller** running at 480 MHz. 
   * They report a **5.2x latency speedup** (0.18 ms vs 0.95 ms) and **42% power reduction** (to 18 mW), providing empirical proof of hardware compatibility.
   * They address Amdahl's Law transparently: although the ensembling loop is small, running it natively in integer space avoids dynamic format conversion overheads (INT8 $\leftrightarrow$ FP32), which take up to 30% of MCU runtime. By enabling ensembling natively in the integer domain, QA-Merge eliminates these formatting overheads, creating a unified, end-to-end integer pipeline.

---

## 2. Analysis of Specific Results and Claims

* **Claim of "Quantization Collapse":**
  * **Evaluation:** Standard ensembling methods (SABLE, ChemMerge, Momentum-Merge) collapse directly to Uniform Merging under naive INT8/INT4 quantization. For instance, SABLE collapses from 68.8% to 66.1% at $\rho=0.0$ ($N_{\text{cal}}=64$). This collapse is thoroughly explained by rounding noise erasing directional boundaries.
* **Claim of "Complete Recovery via QA-Merge":**
  * **Evaluation:** QA-Merge recovers almost **100% of the continuous Float32 ceilings**. For example, SABLE (QA-Merge) achieves **85.40%** accuracy at $\rho=0.5$, matching the unquantized ceiling of 85.40% and outperforming SABLE (Naive) by 0.40% absolute. Under large-sample calibration ($N_{\text{cal}}=4000$), Momentum-Merge (QA-Merge) at $\rho=0.5$ achieves **90.50%** accuracy, matching its continuous ceiling and outperforming Uniform (Quant) by 2.90% absolute.
* **Empirical Trajectory Divergence:**
  * **Evaluation:** To verify Remark 3.2, the authors measure the divergence of the quantized trajectory from the true continuous path. The average $\ell_2$ distance at Layer 14 is **0.0413**, which is well below the threshold required to trigger downstream decision shifts, empirically confirming that feedback-driven divergence is benign.
* **SmoothQuant Sensitivity Sweep:**
  * **Evaluation:** Sweeping the migration parameter $\alpha \in [0.0, 1.0]$ under simulated outlier conditions demonstrates a classic U-curve for Logit MSE and a corresponding peak for Match Rate. Setting $\alpha \in [0.1, 0.3]$ balances activation and weight grids, achieving a peak Decision Match Rate of **97.80%** (at $\alpha=0.3$) and validating the proposed Dynamic Outlier-Aware Activation Scaling.
* **Trajectory Jitter and EF-Smooth Decay Factor $\beta$ Sweep:**
  * **Evaluation:** The sensitivity sweep over the decay factor $\beta \in [0.0, 1.0]$ reveals a beautiful systems-level trade-off: $\beta = 1.0$ recovers 100% of the continuous accuracy but has higher jitter, while $\beta = 0.0$ has zero jitter but collapses to naive accuracy. An intermediate value like $\beta = 0.8$ recovers 99.5% of the ceiling while reducing trajectory jitter by over **40%**, providing a highly tunable systems-level damping factor.

---

## 3. Recommended Areas for Empirical Clarification

While the experimental design and execution are outstanding, we suggest two areas for further clarification and refinement:

### A. Tuning of the EF-Smooth Decay Factor $\beta$ across Calibration Regimes
* **Critique:** In Table 3 (Hyperparameter Configuration Table), the decay factor $\beta$ is configured as $\beta = 1.0$ (perfect error feedback) in the small-sample regime ($N_{\text{cal}}=64$), but is set to $\beta = 0.0$ (no error feedback, fallback to naive quantization) in the large-sample regime ($N_{\text{cal}}=4000$). 
* **Impact:** If SABLE (QA-Merge) in the large-sample regime (Table 2) uses $\beta = 0.0$, then EF-Smooth is completely disabled, and the perfect performance recovery is driven solely by QCC and AEF. 
* **Question/Recommendation:** The authors should explain why $\beta = 0.0$ is optimal for the large-sample regime, while $\beta = 1.0$ is optimal for the small-sample regime. If EF-Smooth is a key trajectory-stabilization mechanism, why does its benefit diminish or disappear when the offline calibration sample size is larger?

### B. Impact of Discretization Chatter on Downstream Autoregressive Decoding
* **Critique:** The paper discusses how EF-Smooth and AEF convert low-frequency bias into high-frequency noise (trajectory jitter or discretization chatter), and notes that while this is benign for fixed visual task classification, it may propagate across self-attention KV caches and disrupt autoregressive decoding in LLMs.
* **Impact:** This is an exceptionally high-signal discussion. However, the current evaluation does not empirically track how this discretization chatter scales over very long autoregressive sequence lengths (e.g., sequence length > 1000 tokens).
* **Recommendation:** The authors should briefly expand their systems-pragmatic discussion to suggest how a low-pass post-routing smoothing filter or a second-order delta-sigma error diffusion loop can be configured to guarantee long-horizon logit stability in generative models.
