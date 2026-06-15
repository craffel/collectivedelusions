# Soundness and Methodology Check: Scale-Aligned Quantized Activation Blending (SA-QAB)

## 1. Clarity of Description
The mathematical and architectural description of SA-QAB is generally very clear. The authors provide explicit definitions for the quantization operator, the dynamic routing equations (Q-ZCA), the scaling recovery factors (QSR), and the GMM density estimation. The tables and diagrams in the main text and appendices are well-structured and help the reader trace the execution pipeline.

## 2. Appropriateness of Methods
- **Decoupled Quantization (DHQ):** Squeezing the base model to INT4 and keeping experts in INT8 is an appropriate, practical design pattern. It aligns with the fact that base model weights represent the majority of the parameter footprint, while adapters contain task-specific details that are sensitive to quantization noise.
- **Quantized Zero-Shot Centroid Alignment (Q-ZCA):** Using early-stage cosine similarity routing is computationally lighter than running a full late-stage classifier or ensembling all experts, which is appropriate for low-power edge microcontrollers.
- **Quantization Scale Recovery (QSR):** Pre-computing activation scales over a calibration set is a lightweight, training-free way to address representation scale contraction.

## 3. Potential Technical Flaws & Skepticism

### A. The "Training-Free" Claim vs. QAT Necessity
The authors repeatedly market SA-QAB as a "training-free, forward-pass-only framework" (Abstract, Intro, Methodology, Conclusion). However, the quantitative results in Section 4.3 reveal a major contradiction:
- Under pure, training-free Post-Training Quantization (PTQ), SA-QAB achieves only **50.00% joint accuracy** (a massive 34.90% drop from the FP16 ceiling of 84.90%).
- To achieve the reported **77.50% joint accuracy**, the authors must introduce **Quantization-Aware Fine-Tuning (QAT)** for the expert adapters and classification heads.
- A 5-epoch QAT phase means the method is **no longer training-free**. For users expecting a plug-and-play merging solution, requiring a custom QAT pipeline on the target adapters adds substantial computational complexity and training infrastructure overhead.

### B. The ZCA Pre-whitening Fusion Fallacy
In Appendix C.7, the authors propose a systems-level mitigation to bypass the diagonal covariance assumption of the GMM: **Zero-phase Component Analysis (ZCA) Pre-whitening**. They claim:
1. They compute a static whitening matrix $W_{\text{zca}} = \Sigma_{\text{calib}}^{-1/2} \in \mathbb{R}^{D \times D}$.
2. At test-time, the activations are whitened: $\tilde{h}_b^{(3)} = h_b^{(3)} \cdot W_{\text{zca}}$.
3. Crucially, they claim: *"the matrix $W_{\text{zca}}$ can be fused directly into the weight matrix of the preceding early backbone block ($W_{\text{base}}^{(3)'} = W_{\text{base}}^{(3)} \cdot W_{\text{zca}}$). Consequently, the pre-whitening is executed on-device with zero runtime latency and zero memory overhead."*

**This contains a major technical and mathematical flaw:**
- $h_b^{(3)}$ is the activation output of Block 3.
- If $W_{\text{zca}}$ is fused into the weights of Block 3 ($W_{\text{base}}^{(3)'} = W_{\text{base}}^{(3)} \cdot W_{\text{zca}}$), then the output of Block 3 is indeed the permanently whitened representation $\tilde{h}_b^{(3)}$.
- While this whitened representation is ideal for the diagonal GMM and Q-ZCA routing, **this same activation is also passed as the input to Block 4 and the subsequent late layers!**
- The subsequent weights ($W_{\text{base}}^{(4)}$) and the pre-trained task-specific LoRA adapters (e.g., $A_k^{(4)}$) were trained to receive the *original, unwhitened* features $h_b^{(3)}$. 
- By passing the permanently whitened $\tilde{h}_b^{(3)}$ to Block 4 without an inverse de-whitening transformation ($W_{\text{zca}}^{-1}$), the authors completely distort the input representation space. This would completely break the feature representation of all late layers, leading to a catastrophic collapse of final classification accuracy.
- If they do perform de-whitening before Block 4 to restore the features, they must execute another matrix-vector product ($O(D^2)$), meaning the process is **not** zero-overhead as claimed. This is a severe, unaddressed system bottleneck.

### C. GMM Rejection Gate Over-Engineering
The inclusion of a diagonal GMM and a ZCA pre-whitening matrix, along with fallback policies, adds immense complexity to what should be an elegant, lightweight multi-task merge. 
- The GMM only provides a marginal improvement of **+0.50%** in joint accuracy under Soft Fallback compared to No Rejection (77.50% vs 78.00% in Table 3).
- Adding an entire secondary density estimator model, storing its means and covariances, setting a validation percentile-based threshold, and executing its likelihood calculations at test-time for a mere 0.5% gain violates the principle of elegant, minimal system design.

## 4. Reproducibility
The reproducibility of the simulated results is very high. The authors disclose all hyperparameters, training details, dimensions ($D=192$, $r=8$), and dataset calibration settings in Appendix A. However, the hardware evaluation on the STM32H7 is conducted via "cycle-accurate emulation" rather than on physical silicon. Because edge compilers can introduce non-deterministic instruction alignments and caching behavior, the lack of real physical microcontroller profiling makes the physical latency and energy claims difficult to verify precisely without the hardware.
