# Paper Outline: Contraction-Regularized Router (CR-Router) for Fixed-Point Convergence

## 1. Abstract
- Context: Dynamic model merging/ensembling over sequential layers.
- Problem: Gating coefficients oscillate violently across layers (routing jitter) and overfit under extreme data scarcity.
- Idea: Formalize feedforward ensembling as a discrete-time dynamical system and use Banach's Fixed-Point Theorem to enforce a contraction mapping.
- Solution: Contraction-Regularized Router (CR-Router), regularizing routing head spectral norm and inverse temperatures.
- Main results: Up to **+18.62%** absolute classification accuracy improvement over the unregularized linear router in orthogonal subspaces and **+12.86%** in overlapping subspaces, matching theoretical stability predictions, and visually demonstrating stable convergence of routing trajectories.

## 2. Introduction
- The Rise of Multi-Task Serving & Model Merging (e.g. LoRA).
- Dynamic Activation-Space Ensembling vs. Weight-Space Merging.
- The Core Problem: Layer-wise routing jitter and high transductive overfitting under extreme sample scarcity (16 samples/task).
- Review of existing heuristic smoothing techniques (e.g., ChemMerge kinetics) and their lack of mathematical guarantees.
- The Theorist's Perspective: Frame the layer-wise feature-coefficient feedback loop as a mathematical operator, study its Lipschitz properties, and force it to be a contraction.
- Core Contributions:
  1. Formalization of deep sequential routing as a discrete dynamical system.
  2. A novel Lipschitz and contraction bound for joint layer-wise representation-routing maps.
  3. The CR-Router algorithm with joint spectral and temperature regularizers.
  4. Exhaustive validation inside the 14-layer, 192-dimensional Sandbox across orthogonal (Exp 1) and overlapping subspaces (Exp 2), as well as real-world vision embedding manifolds (Exp 3), demonstrating performance gains and mathematical convergence.
  5. Practical label-free tuning heuristics and Adaptive Test-Time Temperature Annealing.
  6. Centroid-Based Routing Warm-Starting initialization method to mitigate seed variance.

## 3. Related Work
- Parameter-Efficient Fine-Tuning & Adapter Merging (LoRA, SABLE).
- Dynamic Routing & Mixture-of-Experts (MoE, Linear Routers).
- Trajectory Smoothing & Kinetics in Deep Models (ChemMerge).
- Mathematical Rigor in Deep Learning (Spectral Normalization, Banach Spaces, Contraction Mappings).

## 4. Mathematical Methodology
- **Notation & Formulation:**
  - Subspace Energy Projection (SEP), Logits $Z$, Softmax routing coefficients $\alpha^{(l)}$.
  - Feature blending via active low-rank expert adapters.
- **Sequential Feedback Mapping:**
  - Define the mapping $T_l(h) = F_{\text{base}}^{(l)}(h) + \sum_k R_{k, l}(h) A_k^{(l)} B_k^{(l)} h$.
- **Theorem (Contraction Bound):**
  - Detailed statement of the bound on the Lipschitz constant $L_{T_l}$:
    $$L_{T_l} \le L_{\text{base}}^{(l)} + C_A^{(l)} \left( 1 + \frac{2 R_h}{\tau_l} \|W_{\text{route}}^{(l)}\|_2 \right)$$
- **Rigorous Proof of the Theorem:**
  - Decomposition of terms using functional analysis.
  - Applying the Lipschitz constant of Softmax-linear projection.
  - Combining bounds to establish the contraction constraint.
- **Update-Space Quasi-Contraction:**
  - Formalizing the theoretical relaxation ($L_{U_l} < \epsilon$) on frozen pre-trained residual backbones.
- **Centroid-Based Routing Warm-Starting:**
  - Initializing routing weights $W_{\text{route}}^{(l)}$ with the normalized class prototypes extracted from the calibration split.
- **The Contraction-Regularized Objective:**
  - Minimizing calibration cross-entropy while penalizing Frobenius norm of $W_{\text{route}}^{(l)}$ and inverse squared temperature $1/\tau_l^2$.

## 5. Experimental Evaluation
- **Sandbox Setup:**
  - The 14-layer, 192-dimensional Analytical Coordinate Sandbox (ICS).
  - 4 orthogonal task subspaces of size 48 (MNIST, Fashion-MNIST, CIFAR-10, SVHN).
  - Data-scarce setting: 16 calibration samples per task.
- **Baselines:**
  - Expert Oracle, Uniform Merging, SABLE, ChemMerge, Shared Router, L2-Fixed Router, and Linear Router (unregularized).
- **Experiment 1: Orthogonal Task Subspaces:**
  - CR-Router achieves **53.35% ± 3.84%** classification accuracy, outperforming the unregularized Linear Router (**34.73% ± 3.78%**) and L2-Fixed Router (**38.98% ± 4.51%**).
- **Experiment 2: Overlapping Task Subspaces:**
  - CR-Router achieves **43.48% ± 4.70%** classification accuracy, outperforming the unregularized Linear Router (**30.62% ± 6.99%**) and L2-Fixed Router (**35.23% ± 5.92%**), while Uniform Merging collapses to **27.48% ± 2.88%**.
- **Experiment 3: Real-World Vision Embedding Manifolds:**
  - Extracted 512-dimensional ResNet18 representations of MNIST, Fashion-MNIST, KMNIST, and USPS.
  - CR-Router achieves **53.70% ± 2.37%** classification accuracy, outperforming L2-Fixed Router (**47.33% ± 3.24%**) and Linear Router (**39.70% ± 4.07%**), while Uniform Merging collapses to **7.70% ± 0.87%**.
- **Practical Label-Free Heuristics:**
  - Validating Gating Depth-Variance, Shannon Gating Entropy, and Running Gating Lipschitz Bound under varying joint penalty $\lambda$.
- **Qualitative Trajectory Analysis (Fixed-Point Convergence):**
  - Showing unregularized routing oscillations vs. CR-Router's stable convergence.
- **Adaptive Test-Time Temperature Annealing:**
  - Demonstrating up to **+8.00%** absolute test-time accuracy boost by sharpening gating decisions post-hoc.

## 6. Conclusion & Future Work
- Summary of findings: contraction regularization stabilizes deep sequential ensembling and yields robust generalization.
- Future Directions: Non-linear base blocks, infinite-depth continuous limits, and transition of the proposed framework to large-scale multi-task LLMs.
