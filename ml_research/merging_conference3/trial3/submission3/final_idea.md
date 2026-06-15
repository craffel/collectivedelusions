# Idea Proposal: FlatMerge

## 1. Persona Alignment
This project is deeply aligned with **The Pragmatist** persona. 
*   **Real-World Physical Impact:** In real-world edge deployment (e.g., autonomous systems, mobile sensors, outdoor robotics), inputs are consistently corrupted by sensor noise, defocus blur, dust, weather shifts, and compression artifacts. FlatMerge directly targets the robustness of merged models under these realistic test-time corruptions.
*   **Deployment-Focused Efficiency:** Traditional Test-Time Adaptation (TTA) models are highly prone to "Noise-Entropy Collapse" where input corruptions distort entropy-minimization landscapes. FlatMerge solves this not by training heavy models or caching massive activations, but by introducing a computationally lightweight Flatness-Aware Minimization (SAM) perturbation inside the extremely compact coefficient space (e.g., just $K \times (d+1)$ parameters in the polynomial subspace). This results in virtually zero extra FLOPs or memory overhead during test-time calibration, meeting strict edge-compute latency and memory budgets.
*   **Ease of Integration:** FlatMerge is a simple, plug-and-play modification to any gradient-based test-time model merging framework. It does not require modifying the base model weights, adding auxiliary layers, or altering the model's forward path, making it extremely robust, reliable, and easy to deploy in production.

## 2. Core Techniques
*   **Polynomial Subspace Constrained Blending (PolyMerge):** Restricts the search space of layer-wise merging coefficients to a low-degree polynomial function of normalized depth, mathematically enforcing depth-wise smoothness and reducing parameter search dimensionality from $L$ to $d+1$ \cite{polymerge}.
*   **Coefficient-Space Flatness-Aware Minimization (FlatMerge):** Applies a modified Sharpness-Aware Minimization (SAM) \cite{sam} optimization directly to the low-dimensional polynomial coefficient space during test-time adaptation. Instead of optimizing the coefficients for a single point estimate of prediction entropy, we perturb the coefficients along the gradient direction to find a neighborhood of high flatness, preventing transductive overfitting to high-frequency test-time input noise.
*   **Test-Time Entropy Minimization:** Serves as the unsupervised calibration objective \cite{adamerging}, utilizing a small batch of unlabeled target-task samples at runtime.

## 3. Mathematical Formulation
Let $\Theta_{\text{base}} \in \mathbb{R}^M$ represent the weights of a pre-trained base model, and $\mathbf{\Delta}_k = \Theta_k - \Theta_{\text{base}}$ represent the task vector for task $k \in \{1, \dots, K\}$.
The merged weights at layer $l \in \{1, \dots, L\}$ are defined as:
$$\Theta^l_{\text{merged}}(\mathbf{W}) = \Theta^l_{\text{base}} + \sum_{k=1}^K \lambda^l_k(\mathbf{w}_k) \mathbf{\Delta}^l_k$$
where $\mathbf{w}_k = [w_{k, 0}, w_{k, 1}, \dots, w_{k, d}]^\top$ represents the polynomial coefficients for task $k$, and the layer blending coefficient $\lambda^l_k$ is parameterized as:
$$\lambda^l_k(\mathbf{w}_k) = \sum_{j=0}^d w_{k, j} \cdot \left(\frac{l-1}{L-1}\right)^j$$
Let $\mathbf{W} = \{\mathbf{w}_k\}_{k=1}^K$ represent the complete set of optimized polynomial parameters.

For a test-time calibration batch $X$, the standard unsupervised optimization objective is the multi-task prediction entropy:
$$\mathcal{L}_{\text{ent}}(\mathbf{W}; X) = -\frac{1}{|X|} \sum_{x \in X} \sum_{c} P_{\mathbf{W}}(c|x) \log P_{\mathbf{W}}(c|x)$$
where $P_{\mathbf{W}}(c|x)$ is the predicted probability distribution over classes for sample $x$ using the merged model parameterized by $\mathbf{W}$.

Under **FlatMerge**, we optimize the minimax objective:
$$\min_{\mathbf{W}} \max_{\|\mathbf{E}\|_F \le \rho} \mathcal{L}_{\text{ent}}(\mathbf{W} + \mathbf{E}; X)$$
where $\mathbf{E} = \{\mathbf{e}_k\}_{k=1}^K$ is the parameter-space perturbation bounded by radius $\rho > 0$.

Using a first-order Taylor expansion, the optimal perturbation $\mathbf{E}^* = \{\mathbf{e}^*_k\}$ is computed analytically as:
$$\mathbf{E}^* = \rho \frac{\nabla_{\mathbf{W}} \mathcal{L}_{\text{ent}}(\mathbf{W}; X)}{\|\nabla_{\mathbf{W}} \mathcal{L}_{\text{ent}}(\mathbf{W}; X)\|_F}$$
The polynomial coefficients are then updated via gradient descent with learning rate $\eta$:
$$\mathbf{W} \leftarrow \mathbf{W} - \eta \nabla_{\mathbf{W}} \mathcal{L}_{\text{ent}}(\mathbf{W} + \mathbf{E}^*; X)$$

## 4. Architecture Specifications
*   **Backbone Network:** Pre-trained Vision Transformer backbone (CLIP ViT-B/32, 86M parameters, $L=13$ layer groups consisting of the pre-projection and 12 Transformer blocks; or timm ViT-Tiny, 5.7M parameters, $L=14$ layers).
*   **Subspace Dimension:** Low-degree polynomial parameterization with $d=2$ (quadratic), resulting in exactly $3$ parameters ($w_{k,0}, w_{k,1}, w_{k,2}$) per task.
*   **Optimization Search Space:** For $K=4$ tasks and $d=2$, the total number of optimized parameters in $\mathbf{W}$ is $4 \times 3 = 12$, compressed from $14 \times 4 = 56$ in standard AdaMerging.
*   **Classification Heads:** Pre-trained linear classification heads for each task. The logits from all heads are concatenated during multi-task evaluation.
*   **Perturbation Hyperparameters:** Perturbation radius $\rho \in [0.01, 0.1]$ (default $\rho = 0.05$), optimized via a light grid search on a validation split.

## 5. Baselines
*   **Task Arithmetic (No Optimization) \cite{ilharco2022editing}:** Merges models using a single, global task vector scaling coefficient per task (typically $\lambda_k = 0.3$), serving as the foundational training-free baseline.
*   **AdaMerging (Unconstrained Layer-wise Optimization) \cite{adamerging}:** Optimizes $L \times K$ independent layer-wise coefficients via standard gradient descent on test-time prediction entropy.
*   **PolyMerge (Subspace-Constrained TTA) \cite{polymerge}:** Optimizes $K \times (d+1)$ polynomial parameters using standard test-time entropy minimization, establishing the clean-data performance ceiling and demonstrating the effect of subspace regularization without flatness optimization.
*   **RegCalMerge (Regularized Entropy Weighting) \cite{regcalmerge}:** Employs Elastic Spatial Regularization (ESR) and Scale-Normalized Entropy Weighting (SNEW) to regularize AdaMerging, serving as a state-of-the-art baseline for robust model merging.

## 6. Step-by-Step Interaction
At test-time, a small stream of unlabeled samples $X$ is processed through the system:
1.  **Forward Pass (Perturbation Phase):** 
    *   Compute the merged model weights $\Theta_{\text{merged}}(\mathbf{W})$ using the current polynomial parameters $\mathbf{W}$.
    *   Feed the input batch $X$ through the merged model to compute predictions $P_{\mathbf{W}}(c|x)$.
    *   Compute the entropy loss $\mathcal{L}_{\text{ent}}(\mathbf{W}; X)$ and its gradients with respect to the low-dimensional parameters $\nabla_{\mathbf{W}} \mathcal{L}_{\text{ent}}(\mathbf{W}; X)$.
2.  **Perturbation Calculation:**
    *   Compute the optimal coefficient-space perturbation $\mathbf{E}^* = \rho \frac{\nabla_{\mathbf{W}} \mathcal{L}_{\text{ent}}(\mathbf{W}; X)}{\|\nabla_{\mathbf{W}} \mathcal{L}_{\text{ent}}(\mathbf{W}; X)\|_F}$.
3.  **Forward Pass (Update Phase):**
    *   Construct the perturbed merged model weights $\Theta_{\text{merged}}(\mathbf{W} + \mathbf{E}^*)$ using the perturbed polynomial coefficients.
    *   Feed the same input batch $X$ through the perturbed merged model to compute perturbed predictions.
    *   Compute the perturbed entropy loss $\mathcal{L}_{\text{ent}}(\mathbf{W} + \mathbf{E}^*; X)$ and its gradients $\nabla_{\mathbf{W}} \mathcal{L}_{\text{ent}}(\mathbf{W} + \mathbf{E}^*; X)$.
4.  **Parameter Update:**
    *   Update the core polynomial parameters $\mathbf{W} \leftarrow \mathbf{W} - \eta \nabla_{\mathbf{W}} \mathcal{L}_{\text{ent}}(\mathbf{W} + \mathbf{E}^*; X)$ using the Adam or SGD optimizer.
5.  **Multi-Task Inference:**
    *   Deploy the updated merged model $\Theta_{\text{merged}}(\mathbf{W})$ to perform highly robust, noise-resistant multi-task predictions on subsequent test-time samples.
