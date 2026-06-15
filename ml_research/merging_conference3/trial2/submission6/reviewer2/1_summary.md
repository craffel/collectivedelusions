# Evaluation Step 1: Summary of the Paper

## Main Topic
The paper addresses the challenge of deploying multi-task fused neural networks under extreme memory and storage constraints. Specifically, it explores the intersection of weight-space **model merging** and **Post-Training Quantization (PTQ)**. While model merging (e.g., Task Arithmetic) fuses task-specific expert weights into a single joint model to avoid serving multiple separate models, subsequent low-bit quantization (such as 8-bit or 4-bit) introduces severe noise that degrades accuracy. Conversely, merging pre-quantized experts fails due to alignment loss (discrete grids cannot be algebraically aligned). To bridge this gap, the paper proposes a test-time optimization framework to align model parameters under quantization constraints.

## Proposed Approach
The authors propose **Quantization-Aware Model Merging (Q-Merge)**. Q-Merge optimizes continuous, layer-wise merging coefficients ($\Lambda \in [0, 1]^{L \times K}$ where $L$ is the number of layers and $K$ is the number of tasks) directly under the non-differentiable rounding and clipping operators of standard symmetric uniform post-training quantization. 

To adapt the model without labeled target data at test-time, Q-Merge minimizes the **joint prediction entropy** over an extremely compact, unlabeled calibration stream (consisting of only 16 images per task, 64 images total). 

The paper formalizes and compares two distinct optimization paradigms:
1. **Zero-Order Optimization (1+1 ES):** A derivative-free black-box mutation strategy that stochastically updates merging coefficients using Rechenberg's 1/5th success rule, treating the quantized model as a black-box oracle.
2. **First-Order Optimization (Adam GD with STE):** Gradient-based optimization utilizing the **Straight-Through Estimator (STE)** to propagate gradients of the joint entropy loss through the non-differentiable rounding operator back to the continuous blending coefficients. This formulation explicitly propagates gradients through both the direct weight coordinates and the dynamically computed channel-wise scale factors.

At inference time, the optimized coefficients $\Lambda^*$ are locked and the model is compiled as a static, low-bit integer model, resulting in **zero inference latency or parameter overhead**.

## Key Findings
- **Overcoming the 8-Bit Quantization Gap:** Naive post-merge quantization (Merge-then-Quantize) degrades performance to 71.71%. Q-Merge using Adam GD with STE actively recovers this loss and achieves **74.30%** average accuracy, which remarkably surpasses both the unquantized uniform FP16 baseline (71.88%) and standard AdaMerging with 1+1 ES (73.21%), while recovering 99.9% of the true unquantized Adam-optimized ceiling (74.38%).
- **First-Order vs. Zero-Order Stability:** First-order optimization via STE is highly superior to zero-order random-walk mutation (1+1 ES). It converges faster, yields higher final performance, and reduces seed-to-seed variance by over **2.7x** (standard deviation of 0.38% vs 1.06% for 8-bit).
- **Unlocking 4-Bit Model Merging:** While naive per-tensor 4-bit quantization causes catastrophic model collapse, standard per-channel (channel-wise) weight quantization preserves linear mode connectivity. Under per-channel W4A16, Q-Merge with STE achieves **63.36%** average accuracy, outperforming the naive post-merge baseline (56.66%) by **6.70%** absolute and post-hoc quantized AdaMerging by 1.35% absolute.
- **Extreme Parameter and Memory Efficiency:** Q-Merge compresses the search space tremendously (e.g., from 5.7M parameters to only 56 coefficients for ViT-Tiny, a $1.02 \times 10^5 \times$ reduction, and up to $5.23 \times 10^7 \times$ for LLaMA-7B). Optimization takes only 2.43 seconds on an 8-core CPU or 80 milliseconds on an NVIDIA A100 GPU.
- **Robustness and System Utility:** The approach is highly robust to calibration data scarcity (stable performance with only 8 images per task), scale factor precision constraints (lossless performance even with 16-bit fixed-point scales), and non-stationary imbalanced streams (where the proposed "Confidence-Based FIFO Stratification" heuristic ensures stable adaptation).

## Explicitly Claimed Contributions (with Evidence)
1. **Differentiable Quantized Merging:** Establishing the first framework to formulate and solve model merging directly under the quantization operator using backpropagation through the dynamic scale factor calculations (Equation 15).
2. **First-Order vs. Zero-Order Comparison:** A rigorous empirical study exposing the superior stability, accuracy, and convergence of STE-based gradient descent over black-box mutation (Tables 1 and 2).
3. **Correcting the 4-Bit Catastrophe:** Demonstrating that 4-bit model merging is viable when combined with per-channel weight quantization, establishing an absolute design mandate for low-bit merging (Table 2 and Figure 1).
4. **Complementary Advanced PTQ Integration:** Showing that Q-Merge acts as a global coordinate-alignment step that is highly complementary to subsequent local rounding optimization like AdaRound (Table results show 64.46% accuracy when combined).
5. **Practical Systems-Level Analyses:** Providing comprehensive validation of dynamic activation quantization (W8A8/W4A4), scale discretization sensitivity, and online stream task-balancing heuristics.
