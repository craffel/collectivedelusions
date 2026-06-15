# Soundness and Methodology Check

## Mathematical Rigor and Correctness
The mathematical formulations across the paper are highly rigorous, clear, and technically sound:

1. **Model Merging & Task Arithmetic:** The formulations of task-specific vectors $v_k = \theta_k - \theta_0$ and layer-wise linear parameter addition $w_{\text{merged}}(\Lambda) = w_0 + \sum_{k=1}^K \lambda^l_k v_{k, w}$ are mathematically precise and consistent with state-of-the-art model merging literature (such as AdaMerging).
2. **On-The-Fly Dynamic Pruning:** The dynamic sorting of absolute merged weights to compute the $p$-th percentile threshold $\tau_p(\Lambda)$ and generating the binary mask $M_w(\Lambda) = \mathbb{I}(|w_{\text{merged}}(\Lambda)| \ge \tau_p(\Lambda))$ is mathematically exact. Defining sparse parameters via the Hadamard product $\theta_{\text{sparse}}(\Lambda) = M(\Lambda) \odot \theta_{\text{merged}}(\Lambda)$ correctly shows how the pruning boundary co-evolves with the merging coefficients.
3. **Unsupervised Shannon Entropy:** The multi-task entropy loss $\mathcal{L}_{\text{entropy}}(\Lambda)$ is standard and correctly formulated. The sequential grouping of calibration images per task is a sound systems choice that preserves task boundaries during the forward pass.
4. **Alternative Test-Time Objectives:** The four proposed unsupervised objectives designed to resist degenerate collapse are mathematically rigorous and complete:
   - **Maximizing Mutual Information (MMI):** Minimizes Shannon entropy while maximizing marginal entropy over the batch to prevent single-class collapse.
   - **Temperature-Scaled Soft Pseudo-Labeling:** Regularizes updates against soft target distributions.
   - **Likelihood Ratio (LRA) Constraint:** Minimizes deviation from the unadapted dense model's high-confidence predictions, acting as a functional anchor.
   - **Class-Balanced Contrastive (CBC) Loss:** Enforces latent representation clustering on intermediate features by pulling same-class representations together and pushing different-class representations apart.
5. **Optimization Engine 1 (STE):** The paper provides a clear, detailed mathematical distinction between **Identity-pass STE** (global gradient flow) and **Mask-pass STE** (restricted gradient flow), and correctly implements Identity-pass STE using the standard `detach()` operator:
   $$w_{\text{sparse}} = w_{\text{merged}} + \text{detach}\left(M_w(\Lambda) w_{\text{merged}} - w_{\text{merged}}\right)$$
   This guarantees that the gradient of $w_{\text{sparse}}$ with respect to $w_{\text{merged}}$ is approximated as the identity ($1$) during the backward pass, allowing smooth continuous backpropagation through binarized thresholds.
6. **Optimization Engine 2 (1+1 ES):** The derivative-free Evolution Strategy is correctly formulated following standard black-box optimization theory, implementing Gaussian perturbations and step-size adjustments via the classic 1/5th success rule ($\alpha_{\text{up}}=1.22$, $\beta_{\text{down}}=0.82$).
7. **Reg-ZipMerge:** The structural distance penalty restricting coefficient drift towards the initialization is highly grounded:
   $$\mathcal{L}_{\text{Reg-ZipMerge}}(\Lambda) = \mathcal{L}_{\text{entropy}}(\Lambda) + \gamma \sum_{l=1}^L \sum_{k=1}^K \left( \lambda^l_k - \lambda^{(0), l}_k \right)^2$$
   The alternative functional KL divergence distillation penalty is also mathematically complete.
8. **Orthogonal Procrustes SVD Alignment:** This is a highlight of the methodology. It analytically solves for the optimal rotation matrix $R$ to align independently learned adapter spaces by minimizing the Frobenius norm:
   $$\min_R \|W_1 R - W_2\|_F^2 \quad \text{subject to} \quad R^T R = I$$
   The step-by-step algorithm is mathematically flawless:
   a. Cross-covariance: $C = W_1^T W_2$.
   b. SVD: $C = U \Sigma V^T$.
   c. Rotation matrix: $R = U V^T$.
   d. Rotated alignment: $W_1^{\text{aligned}} = W_1 R$.
   e. Linear average: $W_{\text{merged}} = \frac{1}{2}(W_1 R + W_2)$.
   Its SVD computational complexity is correctly analyzed as $O(d \cdot r^2 + r^3)$, showing its extreme physical efficiency.
9. **Joint Quantization-Pruning (INT8/INT4 PTQ) under STE:** Integrates rounding and dynamic scaling $q_{\text{step}}$ into the Identity-pass STE, enabling gradient propagation through both quantizers and pruning masks.

## Soundness of Experimental Design and Assumptions
The experimental setup and methodological assumptions are highly realistic and appropriate:
- **Baseline Calibration Set:** $B=16$ images per task (64 total) is highly pragmatic, matching standard on-device adaptation settings where data access is constrained.
- **Search Space Efficiency:** Grouping parameters into $L=14$ layer-wise groups is a sound choice. It keeps the search space highly parameter-efficient (56 continuous coefficients), ensuring stable optimization dynamics.
- **SVHN Expert Noise:** Initially using an under-trained SVHN expert (19.59% accuracy) is a highly clever way to analyze noisy expert noise injection. Crucially, the authors conduct a complete ablation study in Section 4.5.6 using a fully converged SVHN expert (82.15%), proving that representational collapse is driven by domain incompatibility rather than training convergence.
- **Backbone Diversity:** Evaluating both Vision Transformers (`vit_tiny_patch16_224`, `vit_base_patch16_224`) and CNNs (`resnet18`) isolates backbone architecture as a potential bias.
- **Hardware Profile Validation:** Directly measuring execution latency on an ARM mobile CPU and caching/VRAM footprints during calibration provides concrete physical evidence for the hardware arguments.

## Soundness Rating: Excellent
The paper's methodology is exceptionally sound, complete, and mathematically rigorous. The formulations are precise, the assumptions are realistic, and the alternative regularizations and analytical models are thoroughly detailed. There are no technical flaws or logical leaps in the proposed methods.
