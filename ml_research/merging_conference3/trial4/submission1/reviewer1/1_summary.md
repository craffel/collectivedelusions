# 1. Summary of the Paper

## Main Topic
The paper, titled **"PhaseMerge: Fourier Phase Interference for Noise-Cancelling Model Merging,"** addresses the problem of weight-space model merging—combining multiple task-specific neural network expert models (fine-tuned from a shared, pre-trained base model) into a single unified model without joint training. 

Specifically, the paper targets the issue of **destructive parameter interference** (task conflicts) where updates for one task degrade performance on another. It also addresses the **Overfitting-Optimizer Paradox** (where test-time merging parameter optimization on small calibration streams leads to overfitting to transductive noise) and performance degradation under **target post-training quantization (PTQ) schema shifts**.

## Proposed Approach
Instead of performing static or adaptive linear averaging of parameters in the real coordinate spatial domain, the paper proposes **PhaseMerge**, which projects fine-tuning weight updates (task vectors $\tau_k = W_k - W_{\text{pre}}$) into the complex-valued Fourier domain using the 2D Real Fast Fourier Transform (RFFT2D):
$$\mathcal{F}_k^l = \text{RFFT2D}(\tau_k^l) = A_k^l e^{i \theta_k^l}$$
This decouples each frequency component of the task update into its amplitude $A_k^l$ and phase angle $\theta_k^l$.

The paper introduces two differentiable, learnable phase-rotation parameterizations:
1. **Uniform Phase Rotation (U-PhaseMerge, $r=1$):** Optimizes a single uniform scalar phase-shift $\tilde{\phi}_k^l \in \mathbb{R}$ per layer $l$ and task $k$. The global phase angle $\phi_k^l = \pi \cdot \tanh(\tilde{\phi}_k^l)$ is broadcast to all frequency components. The parameter footprint is 196 parameters.
2. **Bilinear Continuous Phase Grids (PhaseMerge, $r=2$):** Optimizes a compact $2 \times 2$ phase grid $\tilde{\phi}_k^l \in \mathbb{R}^{2 \times 2}$ per layer $l$ and task $k$, which is bilinearly upsampled to the full dimensions of the frequency tensor. The parameter footprint is 772 parameters.

To preserve the conjugate symmetry of real-valued matrices during reconstruction, a **symmetry-preserving frequency mask** ($M_{\text{sym}}$) is applied to zero out phase-rotation adjustments at the DC and Nyquist components. The reconstructed real-valued spatial updates are scaled by learnable global task-wise scaling coefficients $s_k$ (initialized to $0.3$) and added back to the pre-trained backbone:
$$W^l_{\text{merged}} = W^l_{\text{pre}} + \sum_{k=1}^K s_k \cdot \text{IRFFT2D}(\mathcal{F}'^l_k)$$

The framework is optimized end-to-end using an unsupervised prediction entropy minimization objective ($\mathcal{L}_{\text{entropy}}$) over a small calibration stream ($M \in \{4, 16, 32\}$) using Adam. When validating under low-bit quantization (8-bit and 4-bit), the Straight-Through Estimator (STE) is used to propagate gradients through the rounding operator.

## Key Findings
1. **PolyMerge is the strongest baseline:** Under all tested settings, PolyMerge consistently and substantially outperforms both PhaseMerge ($r=2$) and U-PhaseMerge ($r=1$). For example, under 8-bit PTQ, PolyMerge achieves $48.00 \pm 1.47\%$ multi-task average accuracy compared to U-PhaseMerge's $42.33 \pm 1.76\%$ and PhaseMerge's $40.83 \pm 1.18\%$. Under FP32, PolyMerge achieves $48.00 \pm 1.62\%$ compared to U-PhaseMerge's $42.83 \pm 1.76\%$ and PhaseMerge's $40.75 \pm 1.43\%$.
2. **U-PhaseMerge ($r=1$) outperforms PhaseMerge ($r=2$):** The uniform $r=1$ parameterization consistently outperforms the continuous $2 \times 2$ grid ($r=2$), suggesting that restricting phase shifts to a single scalar per layer acts as a more effective regularizer for dense matrices, which lack natural spatial coordinates.
3. **U-PhaseMerge outperforms AdaMerging and Uniform TA:** U-PhaseMerge ($42.83 \pm 1.76\%$) achieves a performance improvement of $+4.58\%$ over the static Uniform Task Arithmetic baseline ($38.25 \pm 1.34\%$) and $+0.83\%$ over the unconstrained layer-wise AdaMerging baseline ($42.00 \pm 0.89\%$) in FP32.
4. **Static FREE-Merging performs poorly:** The non-adaptive frequency-filtering FREE-Merging baseline collapses, achieving only $27.17 \pm 1.96\%$ in FP32, demonstrating that adaptive continuous phase synchronization is necessary.
5. **Mitigation of the Overfitting-Optimizer Paradox:** At extreme data scarcity ($M=4$), U-PhaseMerge ($42.42 \pm 1.64\%$) outperforms AdaMerging ($40.75 \pm 1.24\%$) under 8-bit PTQ, demonstrating that the low-dimensional Fourier parameterization with the symmetry-preserving mask helps stabilize optimization.
6. **Robustness to Target Schema Shift:** Calibrating under an 8-bit schema and deploying on a 4-bit schema degrades U-PhaseMerge performance by $4.91\%$ (from $42.33\%$ to $37.42\%$) and PhaseMerge by $3.91\%$ (from $40.83\%$ to $36.92\%$), which is highly comparable to AdaMerging's $4.17\%$ drop and PolyMerge's $4.58\%$ drop.

## Explicitly Claimed Contributions and Accompanying Evidence
* **Contribution 1: Proposing a frequency-domain model merging framework (PhaseMerge) based on wave-theoretic mechanics.**
  * *Evidence:* Section 3 formalizes the projection of task vectors via RFFT2D, differentiable phase rotations, IRFFT2D reconstruction, and proves a spatial-domain dual equivalence to the directional Hilbert transform (Theorem 3.1).
* **Contribution 2: Designing a structured Uniform PhaseMerge (U-PhaseMerge, $r=1$) formulation.**
  * *Evidence:* Section 3.3 details the $r=1$ parameterization. Tables 1, 2, and 3 report that U-PhaseMerge consistently outperforms the $r=2$ grid and AdaMerging.
* **Contribution 3: Enforcing a symmetry-preserving frequency mask ($M_{\text{sym}}$).**
  * *Evidence:* Section 3.4 details the mathematical constraints on DC/Nyquist components. Section 4.3 attributes the stabilization of optimization at extreme sample scarcity ($M=4$) to this mask.
* **Contribution 4: Empirical evaluations on Vision Transformers across 4 conflicting classification tasks.**
  * *Evidence:* Section 4 presents results on a `vit_tiny_patch16_224` backbone across MNIST, FashionMNIST, CIFAR-10, and SVHN, compared against four baselines under FP32, 8-bit, and 4-bit regimes.
