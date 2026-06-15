# 1. Summary of the Paper

## Main Topic and Motivation
The paper addresses a critical deployment bottleneck for multi-task learning: serving multiple specialized expert models on resource-constrained edge hardware. While weight-space **model merging** (e.g., Task Arithmetic) fuses specialized expert models into a single multi-task network with zero inference-overhead, practical deployment also requires **Post-Training Quantization (PTQ)** to INT8 or INT4 formats to meet strict storage and memory bandwidth budgets. 

Currently, weight merging and PTQ conflict in two ways:
1. **Merge-then-Quantize (M-then-Q):** Merging full-precision expert models followed by post-hoc quantization degrades multi-task accuracy due to rounding noise scrambling delicate multi-task decision boundaries.
2. **Quantize-then-Merge (Q-then-M):** Quantizing individual experts first and then merging their weights breaks linear mode connectivity, resulting in a non-functional model because their discrete quantization grids cannot be algebraically aligned.

To resolve this conflict, the paper proposes **Quantization-Aware Model Merging (Q-Merge)**.

---

## Proposed Approach: Q-Merge
Q-Merge is a lightweight, calibration-free test-time adaptation framework that optimizes layer-wise task-merging coefficients $\Lambda = \{\lambda^l_k\} \in [0, 1]^{L \times K}$ directly under the non-differentiable quantization operator. 

### Key Components:
1. **Layer-Wise Weight Parameterization:** Blending weights for layer $l$ are computed as:
   $$\theta^l_{\text{merged}}(\Lambda) = \theta^l_{\text{base}} + \sum_{k=1}^K \lambda^l_k (\theta^l_k - \theta^l_{\text{base}})$$
2. **Per-Channel Symmetric Uniform PTQ:** Standard per-channel quantization is applied to weight matrices to protect against outlier weights. Scale factors are dynamically computed based on the continuous merged weights, allowing gradients to flow through scale-factor calculations.
3. **Test-Time Joint Entropy Minimization:** Q-Merge optimizes $\Lambda$ by minimizing the Shannon entropy of predictions over a very compact, unlabeled calibration stream (typically 16 images per task, 64 total) to force confident, clear multi-task predictions.
4. **Optimization Paradigms:**
   - **Zero-Order (1+1 ES):** A derivative-free black-box mutation strategy with dynamic step size.
   - **First-Order (Adam GD with STE):** Uses the Straight-Through Estimator (STE) to approximate the rounding gradient as an identity function, enabling gradient-based backpropagation to update the continuous coefficients $\Lambda$.

---

## Key Findings and Evidence
* **Overcoming the 8-Bit Quantization Gap:** With a ViT-Tiny backbone on four vision benchmarks (MNIST, FashionMNIST, CIFAR-10, SVHN), 8-bit Q-Merge with Adam GD + STE achieves an average multi-task accuracy of **74.30%**, remarkably outperforming the unquantized uniform FP16 baseline (**71.88%**) and standard unquantized AdaMerging (**73.21%**), while recovering $99.9\%$ of the unquantized Adam-optimized ceiling (**74.38%**).
* **First-Order (STE) vs. Zero-Order (1+1 ES) Superiority:** Adam GD with STE consistently outperforms the derivative-free 1+1 ES in both 8-bit (74.30% vs. 72.57%) and 4-bit (63.36% vs. 57.83%) configurations, while exhibiting over $2.7\times$ lower seed-to-seed standard deviation (0.38% vs. 1.06% in 8-bit).
* **Unlocking 4-Bit Model Merging:** While naive post-merge quantization collapses to random guess levels under per-tensor INT4 quantization, adopting standard **per-channel weight quantization** preserves mode connectivity. Q-Merge (Adam GD + STE) further aligns the representations to achieve **63.36%** average accuracy under 4-bit PTQ, outperforming the naive M-then-Q baseline (**56.66%**) by **6.70%** absolute and post-hoc quantized AdaMerging (**62.01%**) by **1.35%** absolute.
* **Pragmatic Utility & Low Overhead:**
  - **Zero Inference Overhead:** Once optimized, the continuous merging coefficients are discarded, and the final integer weights are locked and deployed.
  - **Wall-Clock Speed:** Optimization takes only **2.43 seconds** (Adam GD) / **4.88 seconds** (1+1 ES) on an 8-core CPU, and less than **0.08 seconds** on an NVIDIA A100 GPU.
  - **Data Efficiency:** High performance is stable even with only 8 calibration images per task.
  - **Fully Integer Weight Pipeline:** Fully quantizing task classification heads post-hoc to 8-bit introduces virtually no performance degradation ($<0.01\%$).
  - **Stream Noise Robustness:** A proposed *Confidence-Based FIFO Stratification* heuristic successfully buffers non-stationary or highly skewed incoming test-time data to maintain balanced optimization.

---

## Explicitly Claimed Contributions
1. **Introduction of Q-Merge:** The first test-time adaptation framework to formulate and solve model merging directly under the post-training quantization operator.
2. **Exposing First-Order STE Feasibility:** Showing that first-order gradient flow through the non-differentiable quantization operator using STE is highly stable, converges faster, and achieves significantly lower seed-to-seed variance than zero-order evolutionary mutation (1+1 ES).
3. **Viability of 4-Bit Model Merging:** Remedying the "4-bit collapse" by combining standard per-channel quantization with quantization-aware optimization (STE), achieving high multi-task performance under aggressive compression.
4. **Decomposing Merging vs. Quantization Penalties:** Presenting a clean quantitative breakdown of performance losses attributable to weight-space multi-task interference versus quantization noise.
5. **Advanced PTQ Integration:** Showing that Q-Merge acts as a global coordinate-alignment step that outpaces and is highly complementary to localized post-hoc rounding optimizers like AdaRound.
6. **Systems-Level Edge Validation:** Providing practical decision guidelines for engineers, wall-clock latency analyses, calibration set size sensitivity studies, and on-device task-stream balancing heuristics with empirical validation.
