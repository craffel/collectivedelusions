# 1. Summary of the Submission

## 1.1 Overview and Context
The paper, titled **"PhaseMerge: Fourier Phase Interference for Noise-Cancelling Model Merging"**, addresses the problem of weight-space model merging. Existing approaches (such as Task Arithmetic, AdaMerging, and PolyMerge) operate strictly within real-valued Euclidean spaces by blending parameter matrices via linear interpolation. However, these methods are susceptible to:
1. **Destructive parameter interference** (task conflicts) when merging multiple highly conflicting experts.
2. The **Overfitting-Optimizer Paradox** during test-time calibration, where unconstrained optimization parameters overfit to small calibration streams (e.g., $M \le 16$).
3. Performance degradation under **target quantization schema shifts** (e.g., calibrating under an 8-bit post-training quantization operator but deploying under 4-bit).

## 1.2 Proposed Method (PhaseMerge)
To address these issues, the paper introduces **PhaseMerge**, a frequency-domain model merging framework that projects fine-tuning updates (task vectors $\tau_k = W_k - W_{\text{pre}}$) into the complex-valued Fourier domain using the Real 2D Fast Fourier Transform (RFFT2D). The method decouples updates into frequency-domain amplitude and phase components:
- **Amplitude ($A_k$)**: Represents the magnitude of frequency updates.
- **Phase ($\theta_k$)**: Encodes the relative feature alignments.

Rather than optimizing real-space coefficients, PhaseMerge optimizes a **differentiable, continuous phase-rotation mechanism** acting as a frequency-domain matrix-basis regularizer:
1. **Uniform Phase Rotation (U-PhaseMerge, $r=1$)**: Learns a single uniform scalar phase shift per task and layer (192 parameters), serving as a low-dimensional phase regularizer.
2. **Continuous Phase-Shift Grids (PhaseMerge, $r=2$)**: Learns a compact $2 \times 2$ phase grid per task and layer (768 parameters), bilinearly upsampled to the full frequency tensor's dimensions, ensuring spatially smooth low-frequency phase adjustments.

To guarantee that the reconstructed spatial updates are strictly real-valued, PhaseMerge employs a **symmetry-preserving frequency mask** that zeroes out learnable phase shifts on DC and Nyquist components. The phase-rotated task updates are merged in complex space via wave superposition, reconstructed to real space via the Inverse Real 2D FFT (IRFFT2D), scaled by a global learnable task-wise scalar $s_k$, and added back to the pre-trained backbone.

The framework optimizes the phase-rotation parameters by minimizing unsupervised prediction entropy on a small calibration stream ($M \in \{4, 16, 32\}$) under Post-Training Quantization (PTQ) constraints using the Straight-Through Estimator (STE).

## 1.3 Key Findings and Experimental Results
The method is evaluated using a Vision Transformer Tiny (`vit_tiny_patch16_224`) backbone across four conflicting image classification datasets (MNIST, FashionMNIST, CIFAR-10, and SVHN) and compared against:
- **Uniform Task Arithmetic (TA)** (static real-space baseline)
- **FREE-Merging** (static non-adaptive Fourier low-pass filtering)
- **AdaMerging** (unconstrained layer-wise optimization)
- **PolyMerge** (constrained quadratic depth-wise polynomial optimization)

Key empirical outcomes:
- **Absolute Performance**: **PolyMerge** is the strongest empirical baseline, achieving the highest multi-task average accuracy (e.g., $48.00\%$ under FP32 and 8-bit PTQ, $43.42\%$ under 4-bit PTQ).
- **Competitiveness of PhaseMerge**: **U-PhaseMerge ($r=1$)** is highly competitive, scoring $42.83 \pm 1.76\%$ FP32 and $42.33 \pm 1.76\%$ 8-bit PTQ, outperforming both static Uniform TA ($38.25 \pm 1.34\%$ and $37.75 \pm 1.43\%$) and adaptive AdaMerging ($42.00 \pm 0.89\%$ and $41.67 \pm 1.45\%$).
- **Calibration Efficiency**: Under extreme data scarcity ($M=4$), U-PhaseMerge achieves $42.42 \pm 1.64\%$, proving its robustness to the Overfitting-Optimizer Paradox relative to AdaMerging.
- **Failure of Static Filtering**: **FREE-Merging** performs very poorly ($27.17 \pm 1.96\%$ FP32), underscoring the necessity of *adaptive* frequency-domain phase synchronization over static hard-coded filtering.
- **Quantization Resilience**: PhaseMerge variants exhibit strong generalizability when shifting from 8-bit calibration to 4-bit deployment, demonstrating that smooth frequency-domain adjustments avoid fitting to local rounding discretization boundaries.
