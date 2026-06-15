# PhaseMerge: Fourier Phase Interference for Noise-Cancelling Model Merging

## 1. Persona Alignment
In accordance with **The Visionary** persona, this project completely rejects incremental, safe, and linear scalar-weight interpolation in favor of a radical, out-of-the-box **continuous wave superposition paradigm** in complex frequency-space. Rather than treating neural network parameters as static, Euclidean coordinates, PhaseMerge views task-specific fine-tuning updates as continuous wavefunctions. By projecting parameters into a complex-valued Hilbert space of Fourier phase angles, we introduce a highly novel **phase-cancellation mechanism** designed to actively neutralize task-specific interference and high-frequency post-training quantization noise. This is a high-risk, extremely high-novelty, and mathematically rich approach that shifts the model merging paradigm from simple scalar averaging to wave-theoretic coherence and destructive interference.

## 2. Core Techniques
- **2D Fast Fourier Transform (2D-FFT) & Inverse 2D-FFT (2D-IFFT):** Maps real-valued spatial task vectors (delta weights) to their complex-valued frequency components (amplitude and phase) and back.
- **Differentiable Learnable Phase Shifting:** Introduces complex-valued multiplication to rotate the phase angles of frequency components, enabling the optimizer to stochastically tune constructive and destructive interference between experts.
- **Low-Dimensional Phase Grid Parameterization:** Instead of optimizing high-dimensional spatial weight shifts (which triggers the Overfitting-Optimizer Paradox), the phase-shift tensor is parameterized as a tiny $r \times r$ spatial grid ($r=2$) per expert weight, bilinearly upsampled to the full complex tensor dimension. This hard smoothness constraint prevents transductive test-time overfitting and filters out high-frequency quantization noise, making the merged model uniquely robust to quantization schema shifts!

## 3. Mathematical Formulation
Let $W_{\text{pre}}^l \in \mathbb{R}^{M \times N}$ be the pre-trained weights at layer $l$, and $W_k^l \in \mathbb{R}^{M \times N}$ be the weights of fine-tuned expert $k \in \{1, \dots, K\}$. The task vector is defined as:
$$\tau_k^l = W_k^l - W_{\text{pre}}^l$$

We project $\tau_k^l$ into the Fourier frequency domain using the Real 2D Fast Fourier Transform (RFFT2D), which produces a complex-valued frequency-space tensor of shape $M \times (\lfloor N/2 \rfloor + 1)$:
$$\mathcal{F}_k^l = \text{RFFT2D}(\tau_k^l) \in \mathbb{C}^{M \times (\lfloor N/2 \rfloor + 1)}$$

We extract the amplitude $A_k^l$ and original phase angle $\theta_k^l$ of each frequency component:
$$A_k^l = |\mathcal{F}_k^l| = \sqrt{\text{Re}(\mathcal{F}_k^l)^2 + \text{Im}(\mathcal{F}_k^l)^2}$$
$$\theta_k^l = \text{angle}(\mathcal{F}_k^l) = \text{atan2}\left(\text{Im}(\mathcal{F}_k^l), \text{Re}(\mathcal{F}_k^l)\right)$$

To achieve low-dimensional, smooth parameterization, we define a tiny learnable phase-shift grid $\tilde{\phi}_k^l \in \mathbb{R}^{r \times r}$ ($r=2$). We bilinearly upsample $\tilde{\phi}_k^l$ to the shape of the frequency tensor and scale it to the interval $[-\pi, \pi]$ using a $\tanh$ activation:
$$\phi_k^l = \pi \cdot \tanh\left( \text{Upsample}\left( \tilde{\phi}_k^l, \left(M, \lfloor N/2 \rfloor + 1\right) \right) \right)$$

The phase-shifted Fourier representation is computed via Euler's formula:
$$\mathcal{F}'^l_k = A_k^l e^{i (\theta_k^l + \phi_k^l)} = A_k^l \left( \cos(\theta_k^l + \phi_k^l) + i \sin(\theta_k^l + \phi_k^l) \right)$$

We merge the expert representations by summing them in the complex frequency domain (wave superposition):
$$\mathcal{F}^l_{\text{merged}} = \sum_{k=1}^K \mathcal{F}'^l_k$$

We reconstruct the real-valued merged task vector using the Inverse Real 2D Fast Fourier Transform (IRFFT2D):
$$\tau^l_{\text{merged}} = \text{IRFFT2D}(\mathcal{F}^l_{\text{merged}})$$
The final merged weight matrix is:
$$W^l_{\text{merged}} = W^l_{\text{pre}} + \tau^l_{\text{merged}}$$

To evaluate under post-training quantization, we apply a standard PTQ operator:
$$W^l_{\text{quant}} = Q_{\text{opt}}(W^l_{\text{merged}}, b)$$

Let $p_{k, i}(c)$ represent the prediction probability for class $c$ on sample $i$ of task $k$ using the quantized network. The unsupervised phase grid parameters $\tilde{\Phi} = \{\tilde{\phi}_k^l\}$ are optimized by minimizing the prediction entropy over a small calibration stream $D_{\text{cal}}$:
$$\mathcal{L}_{\text{entropy}}(\tilde{\Phi}) = -\frac{1}{N \cdot K} \sum_{k=1}^K \sum_{i=1}^N \sum_{c=1}^C p_{k, i}(c) \log p_{k, i}(c)$$

## 4. Architecture Specifications
- **Backbone Model:** Vision Transformer Tiny (`vit_tiny_patch16_224`) containing $L = 12$ transformer blocks.
- **Layers Merged:** Linear projection layers in the Multi-Head Self-Attention (`qkv`, `proj`) and Multi-Layer Perceptron (`fc1`, `fc2`) modules, yielding approximately 48 dense weight matrices.
- **Learnable Phase Grid Size:** $r = 2$ (each weight matrix has a $2 \times 2$ grid per task).
- **Optimization Parameter Footprint:** For $K = 4$ tasks and 48 weight matrices, the total number of optimized continuous phase variables is:
$$48 \text{ matrices} \times 4 \text{ tasks} \times 4 \text{ parameters} = 768 \text{ parameters}$$
This extremely small footprint acts as a physical regularizer, avoiding the Overfitting-Optimizer Paradox.
- **Constraints & Activations:** Continuous parameters are unbounded and mapped to $[-\pi, \pi]$ using $\pi \cdot \tanh(\cdot)$. Bilinear interpolation ensures smooth spatial transitions of the phase shifts across the weight matrices.

## 5. Baselines
- **Uniform Task Arithmetic (TA):** Merges task vectors linearly: $W_{\text{merged}} = W_{\text{pre}} + 0.3 \sum_k \tau_k$.
- **AdaMerging (Yang et al., 2023):** Optimizes layer-wise linear merging coefficients at test-time via entropy minimization.
- **PolyMerge (Trial 2, Submission 3):** Parameterizes merging coefficients as low-degree polynomials across layers.
- **Q-Merge (Trial 2, Submission 6):** Linear coefficient optimization directly under quantization operators.
- **FREE-Merging (Zheng & Wang, 2024):** Non-adaptive Fourier high-pass filtering of task vectors.

## 6. Step-by-Step Interaction
1. **Caching Frequency Components:** On startup, compute and cache the RFFT2D amplitude $A_k^l$ and phase $\theta_k^l$ for all task vectors $\tau_k^l$.
2. **Differentiable Parameter Upsampling:** During the forward pass, bilinearly interpolate each $2 \times 2$ phase grid parameter $\tilde{\phi}_k^l$ to the full frequency-space tensor shape, and apply the $\pi \cdot \tanh(\cdot)$ scaling.
3. **Complex Phase Rotation & Summation:** Compute the complex task representations $\mathcal{F}'^l_k$ by rotating the phase by the upsampled values, and sum them across all $K$ tasks.
4. **Spatial Weight Reconstruction:** Apply the IRFFT2D to reconstruct the real-valued merged task vector, and add it to the pre-trained weights to obtain $W^l_{\text{merged}}$.
5. **Quantization Projection:** Apply the dynamic PTQ operator (e.g., 8-bit asymmetric channel-wise) to $W^l_{\text{merged}}$ to simulate hardware deployment.
6. **Calibration Prediction:** Run the calibration batch through the quantized merged model, compute prediction logits, and calculate the unsupervised entropy loss.
7. **Autograd Backward Pass:** Propagate gradients back through the quantized network, using the Straight-Through Estimator (STE) for the rounding operator, through the IRFFT2D, and back to the continuous phase grids $\tilde{\Phi}$.
8. **Parameter Update:** Update the phase grids using the Adam optimizer.
