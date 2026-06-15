# Orthogonal-Residual Isotropic Merging (ORIM)

## 1. Persona Alignment
As **The Empiricist**, our research philosophy is grounded in exhaustive, large-scale empirical validation, multi-hyperparameter sweeps, multi-seed evaluation, and detailed ablation studies. We believe that minor theoretical modifications must be backed by overwhelming empirical evidence across diverse scenarios to be proven effective.

**ORIM is designed specifically to maximize empirical evaluability:**
*   **Hyperparameter Sweeps:** It introduces the residual isotropy factor $\gamma \in [0, 1]$ and the orthogonal scaling coefficient $\alpha$, allowing for dense grid sweeps to map the exact trade-offs of geometry-preservation vs. spectral balance.
*   **Scale-up and Robustness:** We evaluate ORIM across three different CLIP backbones: ViT-B/32, ViT-B/16, and ViT-L/14, across different numbers of tasks (8, 14, and 20 tasks). This allows us to empirically verify if our method scales to larger capacities and task complexities.
*   **Ablation Space:** The decoupled nature of ORIM allows us to cleanly isolate the contribution of:
    1.  The orthogonal merging component ($\gamma = 1.0$, varying $\alpha$).
    2.  The isotropic balancing component (no decoupling, pure isotropic).
    3.  The combination (varying both).
    4.  The decoupling strategy (Global vs. Conflict-Aware).
This massive experimental surface area is a perfect match for our persona's strengths.

## 2. Core Techniques
ORIM introduces a hybrid model-merging paradigm that combines geometric manifold preservation with isotropic spectral balancing. The core techniques are:
1.  **Orthogonal-Residual Decoupling (via Procrustes Analysis):** Extracting the high-dimensional rotation components from the weights by solving the orthogonal Procrustes problem, following **OrthoMerge (Yang et al., 2026)**.
2.  **Lie Algebra Aggregation:** Mapping the extracted orthogonal matrices to the Lie algebra $so(d)$ using the inverse Cayley transform, performing magnitude-corrected averaging, and returning to the orthogonal manifold, following **OrthoMerge**.
3.  **Isotropic Spectrum Balancing on Residuals:** Rather than performing a standard additive merge on the linear residuals in Euclidean space, we apply an isotropic scaling operation to the singular value spectrum of the residual matrices, inspired by the isotropic alignment principles of **SAIM (Anonymous, 2026)**.
4.  **Ecosystem Codebase to Clone:** We recommend cloning the official Task Arithmetic repository: `https://github.com/mlfoundations/task_vectors`. This codebase provides robust scripts for downloading pre-trained and task-finetuned CLIP models (ViT-B/32, ViT-L/14) and evaluating them across 8, 14, and 20 classification benchmarks.

## 3. Mathematical Formulation

Let $W_0 \in \mathbb{R}^{d_{out} \times d_{in}}$ be the pre-trained base model weights of a specific layer, and $\{W_i\}_{i=0}^{N-1}$ be the fine-tuned weights for the $N$ downstream tasks.

### Step 3.1: Orthogonal extraction
For each task $i$, we define the target weight $W^{target}_i$ according to the chosen decoupling strategy:
*   **Global Decoupling:** $W^{target}_i = W_i$
*   **Conflict-Aware Decoupling:** We identify conflicting column/neuron updates $\tau^{conf}_i$ by checking the cosine similarity between the individual task update $\tau_i = W_i - W_0$ and the mean update $\tau_{mean} = \frac{1}{N} \sum_{j=0}^{N-1} \tau_j$. If the cosine similarity of the $j$-th column is negative, we keep it as a conflict:
    $$\tau^{conf}_i[:, j] = \begin{cases} \tau_i[:, j] & \text{if } \cos(\tau_i[:, j], \tau_{mean}[:, j]) < 0 \\ 0 & \text{otherwise} \end{cases}$$
    And set $W^{target}_i = W_0 + \tau^{conf}_i$.

We solve the Orthogonal Procrustes problem to extract the orthogonal rotation $R_i \in \mathbb{R}^{d_{out} \times d_{out}}$:
$$U_i, \Sigma_i, V_i^T = \text{SVD}(W^{target}_i W_0^T)$$
$$R_i = U_i V_i^T$$

### Step 3.2: Residual acquisition
We obtain the linear residual component $\rho_i \in \mathbb{R}^{d_{out} \times d_{in}}$ representing task-specific updates not captured by rotation:
$$\rho_i = W_i - R_i W_0$$

### Step 3.3: Isotropic Spectrum Balancing of Residuals
We compute the SVD of the task-specific residual component:
$$\tilde{U}_i, \tilde{\Sigma}_i, \tilde{V}_i^T = \text{SVD}(\rho_i)$$
Let $r = \min(d_{out}, d_{in})$ be the rank of the layer. We compute the mean singular value:
$$\bar{\sigma}_i = \frac{1}{r} \text{Tr}(\tilde{\Sigma}_i) = \frac{1}{r} \sum_{j=1}^r \sigma^i_j$$
We adjust the singular value spectrum to be more isotropic via the residual isotropy factor $\gamma \in [0, 1]$:
$$\hat{\Sigma}_i = \bar{\sigma}_i I + \gamma (\tilde{\Sigma}_i - \bar{\sigma}_i I)$$
The balanced residual is reconstructed as:
$$\hat{\rho}_i = \tilde{U}_i \hat{\Sigma}_i \tilde{V}_i^T$$

### Step 3.4: Lie Manifold Merging of Rotations
The extracted orthogonal matrices $R_i$ are converted to skew-symmetric matrices $Q_i$ in the Lie algebra $so(d)$ via the inverse Cayley transform:
$$Q_i = (R_i - I)(R_i + I)^{-1}$$
We aggregate them using the magnitude-corrected average:
$$Q_{merged} = \frac{1}{N} \cdot \frac{\sum_{i=0}^{N-1} \|Q_i\|_F}{\| \sum_{i=0}^{N-1} Q_i \|_F} \sum_{i=0}^{N-1} Q_i$$
We then map $Q_{merged}$ back to the orthogonal manifold via the Cayley transform:
$$R_{merged} = (I + Q_{merged})(I - Q_{merged})^{-1}$$

### Step 3.5: Hybrid Merging
We merge the balanced residuals in Euclidean space using a standard weighted average (with task coefficients $c_i$, typically $\frac{1}{N}$):
$$\hat{\rho}_{merged} = \sum_{i=0}^{N-1} c_i \hat{\rho}_i$$
The final merged weight matrix is given by:
$$W_{final} = R_{merged} W_0 + \hat{\rho}_{merged}$$

## 4. Architecture Specifications
We apply ORIM to the weight matrices of the **self-attention projection layers** (queries, keys, values, and output projections) and **feed-forward neural network layers** (gate, up, and down projections) of the transformer blocks, as these contain the bulk of the task-specific representation learning.

*   **Inputs:**
    *   Pre-trained base layer weights $W_0 \in \mathbb{R}^{d_{out} \times d_{in}}$ (e.g., $768 \times 768$ for ViT-B/32).
    *   $N$ task-finetuned layer weights $W_i \in \mathbb{R}^{d_{out} \times d_{in}}$.
*   **Intermediate Representations:**
    *   $U_i \in \mathbb{R}^{d_{out} \times d_{out}}$, $V_i^T \in \mathbb{R}^{d_{out} \times d_{out}}$ (orthogonal components from SVD).
    *   $R_i, R_{merged} \in \mathbb{R}^{d_{out} \times d_{out}}$ (orthogonal rotations).
    *   $Q_i, Q_{merged} \in \mathbb{R}^{d_{out} \times d_{out}}$ (skew-symmetric matrices in Lie algebra).
    *   $\tilde{U}_i \in \mathbb{R}^{d_{out} \times r}$, $\tilde{V}_i^T \in \mathbb{R}^{r \times d_{in}}$ (residual SVD components).
    *   $\hat{\Sigma}_i \in \mathbb{R}^{r \times r}$ (balanced residual singular values).
*   **Outputs:**
    *   Merged layer weights $W_{final} \in \mathbb{R}^{d_{out} \times d_{in}}$ of the same shape as the base weights.

## 5. Baselines
To empirically validate the benefits of ORIM, we compare against several highly relevant baselines:
1.  **Task Arithmetic (Ilharco et al., 2023):** Standard linear merging in Euclidean space. It serves as the primary baseline.
2.  **TIES-Merging (Yadav et al., 2024):** Resolves task interference by sign agreement and top-k magnitude selection.
3.  **Standard OrthoMerge (Yang et al., 2026):** Orthogonal-Residual Decoupling *without* any isotropic balancing of the residuals (equivalent to ORIM with $\gamma = 1.0$).
4.  **Standard Isotropic Merging (SAIM / Marczak et al., 2025):** Pure isotropic scaling of task vectors in Euclidean space *without* orthogonal-residual decoupling.
5.  **Individual Task-Specific Models:** Serves as the empirical upper bound for each task.

## 6. Step-by-Step Interaction

The flow of parameters and data during the execution of ORIM is as follows:

```
                  [Pretrained W0]       [Finetuned Wi]
                         |                     |
                         +--------+------------+
                                  |
                   [Orthogonal-Residual Decoupling]
                                  |
                +-----------------+-----------------+
                |                                   |
       [Orthogonal Component Ri]          [Residual Component \rho_i]
                |                                   |
      [Inverse Cayley Transform]              [SVD Decomposition]
                |                                   |
         [Lie Algebra Qi]               [Isotropic Spectrum Balancing]
                |                                   |
  [Magnitude-Corrected Average Q_merged]      [Balanced Residuals \hat{\rho}_i]
                |                                   |
         [Cayley Transform]                [Euclidean Linear Average]
                |                                   |
       [Rotation R_merged]                 [Merged Residual \hat{\rho}_merged]
                |                                   |
                +-----------------+-----------------+
                                  |
                        [Hybrid Combination]
                    W_final = R_merged W0 + \hat{\rho}_merged
```

1.  **Load Weights:** Load pre-trained model weights $W_0$ and the $N$ task-finetuned expert weights $\{W_i\}$.
2.  **Decouple:** Perform SVD on $W^{target}_i W_0^T$ to solve the orthogonal Procrustes problem, obtaining orthogonal rotation $R_i$ for each expert. Compute the residual $\rho_i = W_i - R_i W_0$.
3.  **Isotropic Balancing of Residuals:** Compute SVD on each residual $\rho_i$, compute the average singular value $\bar{\sigma}_i$, interpolate the singular value spectrum towards the average using the factor $\gamma$, and reconstruct the balanced residual $\hat{\rho}_i$.
4.  **Map to Lie Algebra:** Convert $R_i$ to skew-symmetric Lie algebra representation $Q_i$ using the inverse Cayley transform.
5.  **Aggregate Lie Algebra:** Compute the magnitude-corrected average $Q_{merged}$ from the individual task Lie algebra matrices.
6.  **Cayley Transform Back:** Convert $Q_{merged}$ back to the orthogonal rotation $R_{merged}$ on the manifold via the Cayley transform.
7.  **Merge Residuals:** Average the balanced residuals $\{\hat{\rho}_i\}$ in Euclidean space to obtain $\hat{\rho}_{merged}$.
8.  **Combine & Reconstruct:** Compute the final merged weight matrix $W_{final} = R_{merged} W_0 + \hat{\rho}_{merged}$ and inject it into the model architecture.
9.  **Evaluate:** Run the zero-shot multi-task evaluation scripts from the cloned `task_vectors` codebase across all downstream evaluation datasets.
