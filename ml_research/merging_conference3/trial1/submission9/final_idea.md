# Standard-Deviation Scaling (SD-Scale): Unifying Model Merging via Minimalist Scale Calibration

## 1. Persona Alignment
This technical implementation perfectly embodies the values of **The Minimalist** (found in `persona.md`):
*   **Occam's Razor:** Instead of resorting to complex, high-overhead operations like singular value decomposition (SVD in SAIM) or test-time gradient descent with robust self-labeling (SyMerge), SD-Scale solves task vector imbalance using only standard deviations and basic arithmetic.
*   **No Training/Optimization:** It is completely training-free and non-parametric. It requires no learning rate, no optimizer, no batch size, and no iterations.
*   **Elegance and Readability:** The entire merging operation can be implemented in a single, readable line of PyTorch code per layer, making it highly maintainable, reproducible, and robust against hyperparameters.

---

## 2. Core Techniques
The core technique is **Standard-Deviation Scaling (SD-Scale)** applied to task vectors during model merging. It modifies:
1.  **Task Vector Normalization:** Normalizing individual task vectors by their standard deviations to achieve a scale-balanced, isotropic update direction across different tasks.
2.  **Global Scale Calibration:** Re-scaling the merged task vector by the average standard deviation of the individual task vectors at each layer to preserve the natural scale of the adaptations.

This method builds directly upon the concept of **Task Arithmetic** (Ilharco et al., 2022) but introduces a minimalist scaling mechanism to prevent task dominance and representation bias.

---

## 3. Mathematical Formulation
Let $\Theta_{\text{pre}}^l \in \mathbb{R}^{d_1 \times d_2}$ be the pretrained model weights at layer $l \in \{1, \dots, L\}$.
Let $\Theta_k^l \in \mathbb{R}^{d_1 \times d_2}$ be the fine-tuned model weights for task $k \in \{1, \dots, K\}$ at layer $l$.

The task vector $\tau_k^l$ represents the weight difference for task $k$ at layer $l$:
$$\tau_k^l = \Theta_k^l - \Theta_{\text{pre}}^l$$

We define the standard deviation of the task vector $\tau_k^l$ as:
$$\sigma_k^l = \sqrt{\frac{1}{N^l} \sum_{i=1}^{N^l} (\tau_{k, i}^l - \mu_k^l)^2 + \epsilon}$$
where $\mu_k^l = \frac{1}{N^l} \sum_{i=1}^{N^l} \tau_{k, i}^l$ is the mean parameter update, $N^l$ is the total number of parameters in layer $l$, and $\epsilon = 10^{-8}$ is a constant for numerical stability.

We compute the normalized task vector $\hat{\tau}_k^l$ having unit standard deviation:
$$\hat{\tau}_k^l = \frac{\tau_k^l}{\sigma_k^l}$$

We compute the average adaptation scale $\bar{\sigma}^l$ at layer $l$ as:
$$\bar{\sigma}^l = \frac{1}{K} \sum_{k=1}^K \sigma_k^l$$

The merged task vector $\tau_{\text{merged}}^l$ is then defined as the rescaled average of the normalized task vectors:
$$\tau_{\text{merged}}^l = \bar{\sigma}^l \cdot \left( \frac{1}{K} \sum_{k=1}^K \hat{\tau}_k^l \right)$$

Finally, the merged model weights are constructed via:
$$\Theta_{\text{merged}}^l = \Theta_{\text{pre}}^l + \lambda \tau_{\text{merged}}^l$$
where $\lambda$ is a global scaling coefficient (default is 1.0).

---

## 4. Architecture Specifications
*   **Input Representations:** Standard pretrained model weights $\Theta_{\text{pre}}$ and $K$ independently fine-tuned expert weights $\Theta_1, \dots, \Theta_K$ (e.g., CLIP ViT-B/32 or RoBERTa-base).
*   **Layer-wise Application:** The scaling is applied independently to every weight tensor in the model, including:
    *   Multi-head self-attention projection weights ($W_q, W_k, W_v, W_o$)
    *   MLP intermediate and output projection weights ($W_1, W_2$)
    *   Linear classification heads or task-specific layers ($W_{\text{clf}}$)
*   **Intermediate Representations:** The standard deviation $\sigma_k^l$ (a single scalar per weight tensor) and the normalized tensor $\hat{\tau}_k^l$ (same dimension as $\tau_k^l$).
*   **Output Representations:** The merged weights $\Theta_{\text{merged}}$, which have the same architectural shape as the base pretrained model, ensuring zero inference latency or parameter overhead.

---

## 5. Baselines
Our method will be compared against the following baselines:
1.  **Task Arithmetic (Ilharco et al., 2022):** The standard linear averaging baseline: $\tau_{\text{merged}}^l = \frac{1}{K} \sum_{k=1}^K \tau_k^l$. This baseline is appropriate because SD-Scale is a direct, minimalist improvement over it.
2.  **Ties-Merging (Yadav et al., 2024):** A popular heuristic-based merging baseline that prunes small weights and resolves sign conflicts. It is a suitable baseline to see if our purely scale-based calibration matches or outperforms heuristic pruning.
3.  **AdaMerging (Yang et al., 2024b):** A test-time adaptive merging method that learns merging coefficients via entropy minimization. Comparing against AdaMerging will demonstrate whether our training-free parameter calibration can match the performance of active test-time optimization.

---

## 6. Step-by-Step Interaction
The merging process operates at the parameter level prior to inference. The data flows as follows:

1.  **Extract Task Vectors:** For each fine-tuned expert model, compute the parameter difference from the pretrained base model for every weight tensor.
2.  **Calculate Scales:** For each weight tensor, compute its standard deviation $\sigma_k^l$.
3.  **Normalize Directions:** Divide each task vector by its standard deviation to get $\hat{\tau}_k^l$. This ensures that direction vectors contribute equally, regardless of their original magnitude.
4.  **Calibrate Magnitude:** Average the normalized task vectors and multiply the result by the average standard deviation $\bar{\sigma}^l$ to project the update back onto the natural adaptation scale of the model.
5.  **Reconstruct Model:** Add the calibrated updates to the pretrained base weights to obtain the merged weights.
6.  **Inference:** During evaluation, test data is passed through the merged model in a single standard forward pass, requiring zero extra computation.
