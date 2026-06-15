# 3. Soundness and Methodology

## Clarity of the Description
The methodology is exceptionally well-written, mathematically rigorous, and highly detailed. The authors do not hide behind high-level generalities; instead, they provide explicit formulations for:
- Symmetric uniform quantization operators ($Q_b(X)$).
- Dynamic routing coefficients via quantized cosine similarity and temperature Softmax.
- Scale recovery factors ($\beta_k^{(l)}$) and activation scale propagation across INT4/INT8 paths.
- The diagonal GMM formulation and complexity reduction.

The paper is easy to follow and uses precise terminology. The inclusion of hardware-specific details (e.g., fixed-point arithmetic, fast bit-shifting) is highly commendable.

## Appropriateness of Methods
The choice of methods is highly appropriate for the targeted TinyML and edge constraint context:
- **Decoupled Quantization (DHQ)** is logical: the shared base model represents the vast majority of the weights (and thus memory/bandwidth consumption), so compressing it to INT4 yields major gains. The experts are small, so keeping them in INT8 preserves task representations at minimal memory cost.
- **Early routing at Layer 3** avoids running the backbone and all experts in parallel up to the end, enabling sparse execution. This is critical for keeping compute to $O(1)$ and active SRAM footprints low.
- **Pre-computed Scale Recovery (QSR)** is highly pragmatic, providing a training-free way to align representational magnitudes.
- **Diagonal GMM** restricts the covariance matrix to bypass $O(D^3)$ inversion, which is a perfect engineering simplification for low-compute microcontrollers.

## Potential Technical Flaws & Areas of Skepticism (Practitioner's Lens)

While the methodology is solid, there are several key practical considerations and potential limitations that warrant a critical look:

1. **The Hidden Overhead of Dynamic Expert Paging (Flash-to-SRAM):**
   The authors argue that SA-QAB enjoys a true $O(1)$ expert compute footprint and low active SRAM footprint because hard-argmax executes only a single active expert path. However, they also boast that up to 66 concurrent experts can be stored in the 2MB Flash of the STM32H7. 
   - *The Catch:* If 66 adapters are stored in Flash ($66 \times 27.2\text{ KB} = 1795.2\text{ KB}$), they cannot all reside in the 1MB SRAM simultaneously alongside the 252 KB base model. 
   - *The Flaw:* Therefore, when the Q-ZCA routing layer dynamically changes the active expert on a sample-by-sample basis, the system must page/load the selected adapter weights from Flash into SRAM on-the-fly. The paper does not analyze or discuss this **Flash-to-SRAM weight-copy latency overhead**. In real embedded systems, copying 27.2 KB of weights from non-volatile Flash to SRAM on a microcontroller can take several milliseconds, which would completely obliterate the emulated 0.03 ms routing overhead and the 1.16x PyTorch speedup. This is a critical systems limitation for real-world deployment.

2. **The "Training-Free" vs. QAT Trade-off:**
   The paper pitches SA-QAB as a "training-free, forward-only framework." However, the results show that under pure post-training quantization (PTQ), SA-QAB achieves **50.00%** accuracy (or **70.10%** with activation-aware scaling). To recover the full **77.50%** accuracy, they must run a **5-epoch Quantization-Aware Fine-Tuning (QAT)** phase. 
   - While a 5-epoch adapter fine-tuning is indeed lightweight, it violates the "training-free" pitch. The authors are highly transparent about this, but as a practitioner, we must note that achieving the highest reported accuracy requires a GPU-bound training loop, making on-device local adaptation harder if no training capability exists on the edge.

3. **Routing at Layer 3 Sensitivity:**
   Placing routing at Layer 3 relies on the assumption that task-specific features are sufficiently separable very early in the network. While this works well for MNIST, Fashion-MNIST, CIFAR-10, and SVHN, it might fail on more complex real-world tasks where fine-grained class boundaries are only resolved in deeper layers. If the early-stage routing fails, the system will route inputs to the wrong expert, leading to catastrophic misclassifications.

## Reproducibility
Reproducibility appears to be **excellent**. The authors provide precise equations for every step, detailed architectural dimensions ($D=192$, 14 layers, rank-8 LoRA), calibration dataset sizes (64 samples), and exact hardware emulation specifications. The presence of LaTeX source files and the logical structure of the paper suggest that a skilled systems developer could reproduce the results with high fidelity.
