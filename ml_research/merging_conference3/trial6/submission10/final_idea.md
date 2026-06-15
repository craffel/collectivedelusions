# Idea Proposal: Task-Correlation Prior Regularization (TCPR)

## 1. Persona Alignment
As **The Empiricist**, our core philosophy dictates that progress is achieved through rigorous, scale-controlled baseline optimization, exhaustive sweeps, and robust ablation studies rather than unnecessarily complex theoretical metaphors. 

This proposal directly aligns with this worldview:
*   **Baseline Enhancement:** Instead of introducing elaborate quantum or wave equations, we optimize the calibration of the highly parameter-efficient, classical Sigmoidal/Linear routing head by introducing a task-relationship prior.
*   **Exhaustive Sweeps:** The success of this regularizer will be verified through a massive parallel hyperparameter sweep of the regularization coefficient $\beta \in [10^{-6}, 10^2]$ across log-space.
*   **Empirical Rigor:** We plan to validate the method across 10 distinct calibration dataset splits, multiple model seeds, and task-expert combinations, delivering undeniable empirical evidence of its generalizability.
*   **Abundant Ablations:** We incorporate structured ablations comparing parameter-space similarity priors against representation-space similarity priors, and analyze their joint behavior with standard L2 weight decay.

## 2. Core Techniques
The primary technique introduced is **Task-Correlation Prior Regularization (TCPR)**, which injects a pre-computed cross-task similarity matrix $S \in \mathbb{R}^{K \times K}$ as a prior to regularize the trainable routing projection weights during low-data calibration.

We define two distinct variants of the similarity prior matrix:
1.  **Parameter-Space Similarity prior (TCPR-Param):**
    The cosine similarity of task vectors $V_k^{(l)}$ averaged across all layers $l \in \{1, \dots, L\}$:
    $$S^{\text{param}}_{i, j} = \frac{1}{L} \sum_{l=1}^L \frac{\langle V_i^{(l)}, V_j^{(l)} \rangle}{\|V_i^{(l)}\|_F \|V_j^{(l)}\|_F}$$
2.  **Representation-Space Similarity Prior (TCPR-Rep):**
    The cosine similarity of the intermediate representations (activations) extracted from the pretrained base model on a generic validation subset. Let $Z_i, Z_j \in \mathbb{R}^{M \times D}$ be the representations of task expert $i$ and $j$ evaluated on $M$ validation samples:
    $$S^{\text{rep}}_{i, j} = \frac{\sum_{m=1}^M \langle Z_{i, m}, Z_{j, m} \rangle}{\sqrt{\sum_{m=1}^M \|Z_{i, m}\|^2_2} \sqrt{\sum_{m=1}^M \|Z_{j, m}\|^2_2}}$$

Using $S$, we regularize the routing projection matrix $W_{\text{route}} \in \mathbb{R}^{D \times K}$. Let the $k$-th column of $W_{\text{route}}$ be $\mathbf{w}_k \in \mathbb{R}^D$, representing the routing signature for task $k$. We penalize the routing head during calibration via:
$$\mathcal{L}_{\text{prior}}(W_{\text{route}}) = \beta \sum_{i=1}^K \sum_{j \ne i}^K S_{i, j} (\mathbf{w}_i^T \mathbf{w}_j)$$
This enforces that tasks with high positive correlation (high similarity $S_{i, j}$) have aligned routing weights ($\mathbf{w}_i^T \mathbf{w}_j > 0$), while high-conflict tasks are forced to have independent, orthogonal routing pathways, mitigating the zero-sum competitive bottleneck.

## 3. Mathematical Formulation
Let $x \in \mathbb{R}^{B \times C \times H \times W}$ be an input batch. The patch embeddings are spatially averaged to form the global representation $z(x) \in \mathbb{R}^{B \times D}$. 

The dynamic routing logits $o(x) \in \mathbb{R}^{B \times K}$ are computed linearly:
$$o(x) = z(x) W_{\text{route}} + b_{\text{route}}$$
where $W_{\text{route}} \in \mathbb{R}^{D \times K}$ and $b_{\text{route}} \in \mathbb{R}^K$.

To obtain sample-level routing coefficients $\alpha_{k, b}$ with a scale ceiling, we employ the Softmax-free **Sigmoid-based Bounded Router (BSigmoid-Router)** activation:
$$\alpha_{k, b} = \lambda_{\text{max}} \times \text{Sigmoid}(o(x)_{b, k}) \quad \text{with } \lambda_{\text{max}} = 0.3$$
These are averaged across the batch to generate the collapsed batch merging coefficients:
$$\bar{\alpha}_k = \frac{1}{B} \sum_{b=1}^B \alpha_{k, b}$$

During calibration on the tiny 64-sample set, the optimization objective is:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \beta \sum_{i=1}^K \sum_{j \ne i}^K S_{i, j} (\mathbf{w}_i^T \mathbf{w}_j) + \gamma \|W_{\text{route}}\|_F^2$$
where $\mathcal{L}_{\text{CE}}$ is the joint cross-entropy loss over the calibration tasks, $\beta$ is the TCPR regularization scaling factor, and $\gamma$ is the L2 weight decay factor (fixed at $10^{-4}$).

## 4. Architecture Specifications
*   **Backbone:** `vit_tiny_patch16_224` with $L = 14$ layers (Patch Embedding, 12 Transformer blocks, final Layer Norm) and hidden dimension $D = 192$.
*   **Routing Head Parameters:**
    *   $W_{\text{route}} \in \mathbb{R}^{192 \times 4}$ (768 parameters).
    *   $b_{\text{route}} \in \mathbb{R}^4$ (4 parameters).
    *   **Total Trainable Footprint:** Exactly 772 parameters.
*   **Inputs:** Spatially averaged patch embedding tokens $z(x) \in \mathbb{R}^D$ ($D=192$).
*   **Intermediate Representation:** Raw logits $o(x)_k$ ($K=4$).
*   **Final Output:** Batch-averaged bounded merging coefficients $\bar{\alpha}_k \in [0, 0.3]$ ($K=4$), used to linearly interpolate the task vectors.

## 5. Baselines
We evaluate TCPR against a highly rigorous set of baselines:
1.  **Uniform Merge:** Static model merging with task vectors scaled by a uniform $\lambda = 0.3$.
2.  **Quantum Wavefunction Superposition Merging (QWS-Merge):** The complex wave-interference dynamic routing model.
3.  **Linear Router (Classical, Unregularized):** Softmax-based classical dynamic routing without weight decay or bounding.
4.  **Bounded Linear Router (BL-Router):** Softmax-based linear routing with a strict maximum scale of $0.3$.
5.  **Bounded Sigmoidal Router (BSigmoid-Router):** Softmax-free independent sigmoidal dynamic routing without prior regularization.
6.  **L2-Regularized Baselines:** BL-Router and BSigmoid-Router calibrated with only isotropic L2 regularization (to isolate the specific benefit of task-correlation prior information).

## 6. Step-by-Step Interaction
1.  **Off-line Prior Pre-computation:**
    *   Compute the task vectors $V_k^{(l)} = W_k^{(l)} - W_{\text{base}}^{(l)}$ for each expert $k$ at each layer $l$.
    *   Compute the similarity prior matrix $S \in \mathbb{R}^{K \times K}$ (either parameter-space similarity $S^{\text{param}}$ or representation-space similarity $S^{\text{rep}}$).
2.  **Calibration Phase (On 64-sample set):**
    *   Pass the input batch through the patch embedding to extract global representations $z(x) \in \mathbb{R}^{B \times D}$.
    *   Feed $z(x)$ into the routing head to obtain raw task logits.
    *   Apply the Sigmoid activation and scale by $\lambda_{max} = 0.3$ to get the dynamic merging coefficients $\bar{\alpha}_k$.
    *   Construct the merged weights: $W_{\text{merged}}^{(l)} = W_{\text{base}}^{(l)} + \sum_k \bar{\alpha}_k V_k^{(l)}$.
    *   Compute predictions, calculate $\mathcal{L}_{\text{CE}}$, and evaluate the TCPR penalty $\mathcal{L}_{\text{prior}}$ using $S$ and routing projection columns $\mathbf{w}_k$.
    *   Minimize $\mathcal{L}_{\text{total}}$ using Adam for 100 steps to optimize $W_{\text{route}}$ and $b_{\text{route}}$.
3.  **Deployment & Inference Phase:**
    *   Deploy the frozen routing head ($W_{\text{route}}, b_{\text{route}}$) alongside the base model and task vectors.
    *   For any incoming test batch, dynamically compute the bounded coefficients $\bar{\alpha}_k$ in a single forward pass through the routing head.
    *   Assemble the merged model $W_{\text{merged}}^{(l)}$ and run the inference forward pass, achieving zero test-time active optimization overhead.
