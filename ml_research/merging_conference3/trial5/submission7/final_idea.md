# Pruned Gradient Merging (PG-Merge): Deconstructing Complexity in Test-Time Model Fusion

## 1. Persona Alignment
This project is deeply rooted in **The Minimalist** persona. It challenges the necessity of over-engineered, multi-parameter spatial regularizers (such as Elastic Spatial Regularization in RegCalMerge) and complex coordinate transformations (such as continuous polynomial trajectories in PolyMerge) designed to combat the Overfitting-Optimizer Paradox in test-time model merging. 

Guided by Occam's razor, we propose **Pruned Gradient Merging (PG-Merge)**. Instead of introducing additional loss terms, delicate penalty hyperparameters, or complex geometric projections, PG-Merge achieves state-of-the-art robustness by simply applying a **non-parametric, training-free, sparse gradient mask** during test-time adaptation. By restricting updates to only the top-$p\%$ most critical layer-wise coefficients on each batch, we naturally filter transductive noise, prevent high-frequency parameter drift, and preserve the generalizability of the merged network—all with zero extra parameters and zero hyperparameter tuning bloat.

## 2. Core Techniques
PG-Merge introduces a highly elegant and computationally trivial update mechanism into the test-time model merging loop:
*   **Dynamic Sparse Gradient Masking:** At each test-time adaptation step, we calculate the exact gradients for all layer-wise merging coefficients. We then prune (zero-out) all gradients except those in the top-$p\%$ of absolute magnitudes.
*   **Sparse Gradient Optimization:** The masked gradient tensor is passed directly to the optimizer (e.g., Adam or SGD). This ensures that only the highly sensitive, informative layers are adapted on the local batch, while the remaining $(100-p)\%$ of layers remain frozen.
*   **Layer-wise Magnitude Filtering:** Sorting and thresholding occur dynamically per adaptation step, allowing the network to naturally select different active routing paths across different test batches without requiring unconstrained, high-dimensional parameter updates.

This technique is inspired by importance-based gradient pruning in memory-efficient fine-tuning (e.g., MECTA) and gradient sparsity in meta-learning, applying it for the first time as a foundational regularizer in parameter-space deep model fusion.

## 3. Mathematical Formulation
Let $W_{base}^{(l)}$ be the pre-trained base network weights at layer $l \in \{1, \dots, L\}$, and let $V_k^{(l)} = W_k^{(l)} - W_{base}^{(l)}$ be the task vector of expert $k \in \{1, \dots, K\}$. The merged model parameters at layer $l$ under layer-wise coefficients $\alpha_{k, l}$ are defined as:
\begin{equation}
    W_{merged}^{(l)}(\alpha) = W_{base}^{(l)} + \sum_{k=1}^K \alpha_{k, l} V_k^{(l)}
\end{equation}

For an incoming unlabeled test-time batch $X_t \in \mathbb{R}^{B \times C \times H \times W}$ of size $B$, the prediction entropy loss $\mathcal{L}_{\text{TTA}}$ is computed as:
\begin{equation}
    \mathcal{L}_{\text{TTA}}(X_t; \alpha) = -\frac{1}{B} \sum_{b=1}^B \sum_{c=1}^C p_c(x_b) \log p_c(x_b)
\end{equation}
where $p_c(x_b) = \text{Softmax}(f(x_b; W_{merged}(\alpha)))_c$ is the predicted probability for class $c$.

Let $g_{k, l} = \frac{\partial \mathcal{L}_{\text{TTA}}}{\partial \alpha_{k, l}}$ denote the gradient of the adaptation loss with respect to the merging coefficient $\alpha_{k, l}$. We flatten the full set of gradients into a vector $\mathbf{g} \in \mathbb{R}^{M}$, where $M = L \times K$. We sort the absolute gradient values in descending order:
\begin{equation}
    |\mathbf{g}|_{(1)} \ge |\mathbf{g}|_{(2)} \ge \dots \ge |\mathbf{g}|_{(M)}
\end{equation}

Given a target sparsity ratio $p \in (0, 1]$, we determine the threshold index $k_{th} = \lceil p \times M \rceil$ and define the sparse gradient mask $M_{k, l}$ as:
\begin{equation}
    M_{k, l} = \begin{cases}
        1 & \text{if } |g_{k, l}| \ge |\mathbf{g}|_{(k_{th})} \\
        0 & \text{otherwise}
    \end{cases}
\end{equation}

The pruned gradient vector $\tilde{\mathbf{g}}$ is obtained by coordinate-wise multiplication:
\begin{equation}
    \tilde{g}_{k, l} = g_{k, l} \odot M_{k, l}
    \label{eq:pruning}
\end{equation}

The coefficients are then updated using the pruned gradients via standard gradient descent:
\begin{equation}
    \alpha_{k, l}^{(t+1)} = \alpha_{k, l}^{(t)} - \eta \cdot \text{Optimizer}(\tilde{g}_{k, l}^{(t)})
\end{equation}

## 4. Architecture Specifications
*   **Backbone Model:** We employ the standard compact Vision Transformer backbone $\mathtt{vit\_tiny\_patch16\_224}$ containing $5.7$M parameters. The model comprises $L = 14$ structural layer groups (Patch Embedding, 12 Transformer Blocks, and LayerNorm).
*   **Task Experts ($K = 4$):** Four specialized task-specific experts fine-tuned to convergence on **MNIST**, **FashionMNIST**, **CIFAR-10**, and **SVHN**.
*   **Coefficient Search Space:** We optimize $L \times K = 14 \times 4 = 56$ layer-wise coefficients $\alpha_{k, l}$.
*   **Sparsity Ratio ($p$):** A default sparsity ratio of $p = 0.15$ (meaning only $15\%$ of the $56$ parameters, i.e., $\approx 8$ coefficients, are updated at any step, while the remaining $48$ parameters are frozen).
*   **Optimization Engine:** A standard Adam optimizer with a base learning rate of $\eta = 10^{-3}$ and zero weight decay.

## 5. Baselines
We evaluate PG-Merge against a comprehensive suite of static and active merging baselines:
1.  **Uniform Merging (Task Arithmetic):** Static, training-free merging using uniform task coefficients ($\alpha_{k, l} = 0.3$).
2.  **Online AdaMerging (Layer-wise):** Unconstrained online TTA optimizing all $56$ coefficients to minimize test entropy, representing the unregularized optimization baseline susceptible to transductive collapse.
3.  **Online RegCalMerge:** SOTA online TTA utilizing complex Class-Capacity Normalization (CCN), Scale-Normalized Entropy Weighting (SNEW), and Elastic Spatial Regularization (ESR) with multiple hyperparameters.
4.  **Online PolyMerge ($d=2$):** Active TTA baseline that restricts coefficients to a 12-parameter quadratic polynomial subspace of layer depth.
5.  **OFS-Tune (Supervised Static):** Supervised static coefficient optimization on the $64$-sample calibration set via Adam.

## 6. Step-by-Step Interaction
The flow of data and gradients through PG-Merge is detailed below:
1.  **Inference forward pass:** An unlabeled test-time batch $X_t$ is fed into the merged model $W_{merged}(\alpha^{(t)})$. The network outputs prediction logits, which are mapped to probability distributions $p(x)$ via Softmax.
2.  **Loss calculation:** The prediction entropy $\mathcal{L}_{\text{TTA}}(X_t; \alpha^{(t)})$ is computed across the batch.
3.  **Backpropagation:** Standard backward pass computes the raw gradients $g_{k, l} = \frac{\partial \mathcal{L}_{\text{TTA}}}{\partial \alpha_{k, l}}$ for all $56$ merging coefficients.
4.  **Gradient Pruning (Occam's Filter):** The raw gradients are flattened, and their absolute values are sorted. Coordinates corresponding to absolute values below the top-$15\%$ threshold are masked to zero, yielding the sparse gradient tensor $\tilde{\mathbf{g}}^{(t)}$.
5.  **Parameter Update:** The optimizer updates only the top-$15\%$ coefficients. The other $85\%$ of the coefficients remain frozen, keeping their previous values to protect the network from transductive overfitting on local batch noise.
6.  **Model Reconstruction:** The new weight matrices are assembled for the next forward pass on-the-fly using the updated coefficients $\alpha^{(t+1)}$.
