# Idea Proposal: FlatQ-Merge (Flatness-Aware Quantization-Aware Model Merging)

## 1. Persona Alignment
This project directly aligns with **The Empiricist** persona. Rather than focusing on minor theoretical modifications to a merging algorithm, we investigate a fundamental, cross-axial empirical question: **How does the flatness of expert models' loss landscapes (controlled via the SAM perturbation radius $\rho$ during pre-merging training) affect their robustness to post-training quantization and test-time coefficient optimization under quantization constraints?** 

To validate this rigorously, we design an extensive empirical suite:
1. **Multi-Axial Grid Sweeps**: We train experts across 5 different SAM radii ($\rho \in \{0.0, 0.01, 0.05, 0.1, 0.2\}$).
2. **Quantization Precision Axis**: We evaluate each configuration under 8-bit (low noise) and 4-bit (extreme noise) per-channel uniform post-training weight quantization.
3. **Statistical Rigor**: All experiments are conducted across 3 independent random seeds (42, 100, 2026).
4. **Hessian Curvature Analysis**: We perform fine-grained coefficient-space perturbation sweeps to empirically map the flatness of the test-time adaptation landscape and trace the relationship between expert-level sharpness and test-time quantization robustness.

## 2. Core Techniques
The proposed framework utilizes and builds on several foundational methodologies:
- **Sharpness-Aware Minimization (SAM)**: Introduced by [Foret et al., 2020](https://arxiv.org/abs/2010.01412), SAM trains models by minimizing loss in an neighborhood of the parameters, encouraging convergence to flat minima. We apply SAM to pre-train task-specific experts before merging.
- **Quantization-Aware Model Merging (Q-Merge)**: Introduced in `trial2_submission6`, Q-Merge optimizes merging coefficients $\Lambda$ directly on quantized weights.
- **Straight-Through Estimator (STE)**: Used to propagate gradients through the non-differentiable `round` operator of post-training quantization (PTQ) during test-time adaptation.
- **Hessian/Curvature Profiling**: We compute empirical loss curves under coefficient perturbations to verify that SAM training of experts translates to a flatter and more stable test-time adaptation landscape.

## 3. Mathematical Formulation

Let $\theta_{\text{base}} \in \mathbb{R}^D$ represent the weights of a pre-trained base model, partitioned into $L$ discrete layer groups. For task $k \in \{1, \dots, K\}$, we pre-train a task-specific expert $\theta_k(\rho)$ using SAM with a perturbation radius $\rho$:
$$\theta_k(\rho) = \arg\min_{\theta_k} \max_{\|\epsilon\|_2 \le \rho} \mathcal{L}_k(\theta_k + \epsilon)$$

The task vector for layer $l \in \{1, \dots, L\}$ and task $k$ is defined as:
$$\tau^l_k(\rho) = \theta^l_k(\rho) - \theta^l_{\text{base}}$$

We parameterize the merging using a layer-wise coefficient tensor $\Lambda = \{\lambda^l_k\} \in [0, 1]^{L \times K}$. The unquantized merged weights at layer $l$ are:
$$\theta^l_{\text{merged}}(\Lambda, \rho) = \theta^l_{\text{base}} + \sum_{k=1}^K \lambda^l_k \tau^l_k(\rho)$$

We compress the merged weights to $b \in \{4, 8\}$ bits using per-channel symmetric uniform Post-Training Quantization (PTQ). For a 2D weight matrix in layer $l$, the scale factor for output channel $c \in \{1, \dots, O\}$ is computed as:
$$S^l_c(\Lambda, \rho) = \frac{\max\left(\left|\theta^{l,c}_{\text{merged}}(\Lambda, \rho)\right|\right)}{2^{b-1} - 1}$$

The quantized weights along channel $c$ are computed via:
$$\theta^{l,c}_{\text{quant}}(\Lambda, \rho) = \text{clip}\left[ \text{round}\left(\frac{\theta^{l,c}_{\text{merged}}(\Lambda, \rho)}{S^l_c(\Lambda, \rho)}\right), -2^{b-1}, 2^{b-1}-1 \right] \times S^l_c(\Lambda, \rho)$$

At test-time, we optimize the coefficients $\Lambda$ on an unlabeled calibration set $X = \{x_1, \dots, x_N\}$ by minimizing the joint Shannon entropy:
$$\mathcal{L}_{\text{entropy}}(\Lambda) = \frac{1}{N} \sum_{i=1}^N \mathcal{H}\left(f\left(x_i; \theta_{\text{quant}}(\Lambda, \rho)\right)\right)$$
where $\mathcal{H}(p) = -\sum_j p_j \log p_j$ is the prediction entropy, and $f(x; \theta)$ represents the neural network forward pass. We use Adam with the Straight-Through Estimator (STE) to update the continuous coefficients $\Lambda$:
$$\frac{\partial \text{round}(x)}{\partial x} \approx 1$$

To profile the curvature of the optimized coefficient space, we apply a Gaussian perturbation vector $\delta \sim \mathcal{N}(0, \sigma^2 I)$ to the optimized coefficients $\Lambda^*$, measuring the average prediction entropy change:
$$\Delta \mathcal{L}(\sigma) = \mathbb{E}_{\delta}\left[ \mathcal{L}_{\text{entropy}}(\Lambda^* + \delta) - \mathcal{L}_{\text{entropy}}(\Lambda^*) \right]$$
We sweep $\sigma \in [0.0, 0.1]$ to empirically compare the landscape sharpness across different expert SAM radii $\rho$.

## 4. Architecture Specifications
- **Backbone**: Vision Transformer (`vit_tiny_patch16_224`, 12 Transformer blocks, 5.7M parameters).
- **Inputs**: Standardized images of shape $3 \times 224 \times 224$ pixels.
- **Layers**: Grouped into $L = 14$ discrete parameter blocks (1 Patch Embedding block, 12 Transformer Encoder blocks, 1 Final Layer Normalization block).
- **Coefficients**: $\Lambda \in [0, 1]^{14 \times 4}$ (representing 14 layers and 4 visual tasks: MNIST, FashionMNIST, CIFAR-10, SVHN).
- **Quantization**: Signed integer per-channel weight quantization ($b=8$ or $b=4$), while bias parameters and task classification heads remain in full precision (or are quantized post-hoc to 8-bit).
- **Activations**: High-precision FP32/FP16 during inference.

## 5. Baselines
We evaluate FlatQ-Merge against a comprehensive set of baselines to isolate the exact empirical effect of expert-level flatness:
1. **SGD Experts Q-Merge ($\rho = 0.0$)**: The core baseline from `trial2_submission6` where experts are fine-tuned with standard Adam/SGD and merged using Q-Merge (Adam GD with STE).
2. **Naive Uniform Merged SAM Experts (M-then-Q)**: Merging SAM-trained experts ($\rho$-specific) with uniform static coefficients ($\lambda^l_k = 0.3$), followed by post-hoc per-channel quantization.
3. **AdaMerging on SAM Experts (Quantized)**: Optimizing merging coefficients $\Lambda$ in full-precision on SAM-trained experts and subsequently quantizing post-hoc (testing if flatness makes unquantized optimized parameters more robust to quantization noise).
4. **Individual SAM Experts (Quantized, Unmerged)**: Evaluating task-specific performance of individual $\rho$-specific experts under $b$-bit quantization to quantify the absolute upper bound of task-specific performance without weight-space interference.

## 6. Step-by-Step Interaction
1. **Expert Generation**: Fine-tune 4 independent task experts on their respective classification subsets (512 images per task) across 3 seeds under 5 distinct SAM radii $\rho \in \{0.0, 0.01, 0.05, 0.1, 0.2\}$.
2. **Task Vector Extraction**: For each $\rho$, construct task vectors $\tau^l_k(\rho)$ by subtracting the base model weights from the expert weights.
3. **Coefficient Parameterization**: Initialize continuous merging coefficients $\Lambda \in [0, 1]^{14 \times 4}$ with uniform values ($0.3$).
4. **Dynamic Weighted Blending**: Compute full-precision merged weights $\theta^l_{\text{merged}}(\Lambda, \rho)$ for each layer.
5. **Straight-Through PTQ Projection**: Apply per-channel symmetric rounding and scaling to obtain the quantized parameters $\theta^l_{\text{quant}}(\Lambda, \rho)$.
6. **Calibration Forward Pass**: Pass the tiny calibration stream $X$ (16 images per task, 64 total) through the quantized model to get output prediction probabilities.
7. **Loss & Backward Pass**: Compute the joint prediction entropy $\mathcal{L}_{\text{entropy}}(\Lambda)$ and propagate gradients back to the continuous parameters $\Lambda$ using the Straight-Through Estimator.
8. **Test-Time Optimization**: Update $\Lambda$ via Adam for 40 steps to adapt the merged quantized model.
9. **Final Evaluation**: Freeze the optimized coefficients $\Lambda^*$ and evaluate the quantized multi-task model on the full, unseen test datasets.
10. **Curvature Profiling**: Perturb the optimized coefficients $\Lambda^*$ with varying noise scales $\sigma$ to empirically measure and plot the flatness of the entropy loss landscape under each pre-training condition.
