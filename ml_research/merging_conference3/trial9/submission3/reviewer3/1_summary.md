# Summary Evaluation: Contraction-Regularized Router (CR-Router)

## 1. Paper Overview and Main Topic
The paper addresses the issue of instability and overfitting in sequential dynamic model ensembling and model merging. This scenario occurs in multi-task serving environments where specialized Parameter-Efficient Fine-Tuning (PEFT) adapters (like LoRA) are blended dynamically layer-by-layer based on intermediate representation vectors. The authors formalize this sequential feedback process across network depth as a discrete-time dynamical system. They identify a critical problem termed *sequential routing jitter*, where unregularized routing coefficients exhibit high-frequency oscillations from layer to layer, causing representation trajectories to undergo violent shifts. This degradation leads to a failure in joint classification and extreme susceptibility to transductive overfitting under severe calibration data scarcity (e.g., 16 samples per task).

## 2. Core Methodology and Approach
To solve sequential routing jitter, the paper proposes the **Contraction-Regularized Router (CR-Router)**. The core approach uses Banach’s Fixed-Point Theorem to establish stability:
- **Dynamical Systems Formulation**: The feedforward propagation of intermediate representations $h^{(l-1)}$ and Softmax gating coefficients $\alpha^{(l)}$ is modeled as a joint discrete-time feedback operator $T_l: \mathbb{R}^D \to \mathbb{R}^D$.
- **Theorem & Lipschitz Bound**: The authors prove a novel Lipschitz bound on the joint layer-wise representation-routing map $T_l$:
  $$L_{T_l} \le L_{\text{base}}^{(l)} + C_A^{(l)} \left( 1 + \frac{2 R_h}{\tau_l} \|W_{\text{route}}^{(l)}\|_2 \right)$$
  where $L_{\text{base}}^{(l)}$ is the Lipschitz constant of the base model block, $C_A^{(l)}$ bounds the spectral norm of the adapters, $R_h$ is the maximum representation norm, $\tau_l$ is the routing temperature, and $\|W_{\text{route}}^{(l)}\|_2$ is the spectral norm of the routing projection matrix.
- **Enforcing Contraction mapping**: Under Banach's theorem, if $L_{T_l} < 1$, the trajectory converges stably to a unique fixed point. The authors enforce this by penalizing the Frobenius norm of the routing weights (as a differentiable upper bound to the spectral norm) and the inverse temperature squared ($1/\tau_l^2$) via a joint objective:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{cal}} + \lambda_{\text{spec}} \sum_{l=1}^L \left\| W_{\text{route}}^{(l)} \right\|_F^2 + \lambda_{\text{temp}} \sum_{l=1}^L \frac{1}{\tau_l^2}$$
- **Extensions for Residual and Frozen Architectures**:
  1. **Scaled Residual CR-Router (SR-CR-Router)** scales the identity connection by $(1 - \gamma_l)$ to guarantee $L_{T_l} < 1$ even when $L_{\text{base}}^{(l)} = 1$.
  2. **Update-Space Quasi-Contraction** relaxes the absolute convergence guarantee for frozen pre-trained backbones by bounding the update-space Lipschitz constant ($L_{U_l} < \epsilon$).
- **Initialization and Serving Improvements**:
  1. **Centroid-Based Routing Warm-Starting**: Resolves random seed variance in scarce-data settings by initializing $W_{\text{route}}^{(l)}$ with the normalized centroids of the calibration split.
  2. **Adaptive Test-Time Temperature Annealing**: Decouples training stability from test-time performance by multiplying the learned temperatures by a scale factor $\gamma_{\text{scale}} \le 1.0$ at inference to sharpen gating.

## 3. Key Findings
- **Overfitting of Unregularized Routers**: Under severe data scarcity (16 calibration samples per task), the unregularized Linear Router overfits heavily (achieving 34.73% joint classification accuracy in orthogonal subspaces and 30.62% in overlapping subspaces on a 14-layer Sandbox).
- **CR-Router Performance**: CR-Router stabilizes parameters and achieves **53.35% $\pm$ 3.84%** accuracy in orthogonal subspaces (Experiment 1) and **43.48% $\pm$ 4.70%** in overlapping subspaces (Experiment 2). On real-world vision embedding manifolds (Experiment 3), it achieves **53.70% $\pm$ 2.37%**.
- **Test-Time Annealing Gains**: Lowering the temperature at inference time reduces "expert dilution", boosting accuracy from 53.55% up to **62.45% $\pm$ 2.98%** (at $\gamma_{\text{scale}} = 0.10$), representing a massive +8.90% absolute gain.
- **Profiling Efficiency**: On CPU, CR-Router processed **15,785.1 samples/s** at batch size 400 (compared to ~10,464.1 for non-parametric SABLE), demonstrating superior throughput and lower latency.

## 4. Explicitly Claimed Contributions and Supporting Evidence
1. **Dynamical Systems Formulation**: Formulates sequential ensembling mathematically. Support: Section 3.1 & 3.2.
2. **Theoretical Lipschitz Bounds**: Proves Theorem 3.1 & 3.2 for both linear adapter updates and interpolative coordinate systems. Support: Mathematical proofs in Section 3.4.
3. **CR-Router Design**: Proposes joint spectral and temperature regularizer. Support: Equation 15 and empirical sensitivity sweep in Section 4.5.
4. **Label-Free Tuning Heuristics**: Proposes Gating Depth-Variance, Shannon Gating Entropy, and Running Gating Lipschitz Bound. Support: Empirical validation and metrics in Table 8.
5. **Adaptive Test-Time Temperature Annealing**: Decouples training stability from test-time representation sharpness. Support: Table 9 showing up to +8.90% performance gains.
6. **Centroid-Based Routing Warm-Starting**: Proposes prototype initialization to mitigate seed variance under data scarcity. Support: Described in Section 3.8 and referenced as a mitigation for overlap sensitivity in Section 4.5.
