# Grassmannian Subspace Consensus Merging (GSC-Merge)

## 1. Persona Alignment
As **The Theorist**, I approach the challenge of weight-space model merging from the first principles of linear algebra, spectral theory, and Grassmannian manifold geometry. Traditional model merging methods rely heavily on heuristic weight manipulation (such as coordinate-wise sign voting, random zeroing, or hard magnitude-based thresholding as seen in TIES and STA), which lack formal mathematical justification or error bounds.

**GSC-Merge** replaces these heuristic pipelines with a mathematically rigorous spectral consensus mechanism. By framing collective task updates as a joint multi-task parameter space, we employ Singular Value Decomposition (SVD) and the **Eckart-Young-Mirsky Theorem** to find the optimal low-rank projection onto a shared Grassmannian subspace. This projection minimizes representation distortion under the Frobenius norm, providing a provable upper bound on the representational drift of task updates. This theoretical approach satisfies our core mandate: ensuring that empirical success is built upon a provably sound mathematical framework rather than arbitrary heuristics.

---

## 2. Core Techniques
1. **Joint Multi-Task Update Matrix Construction:** Horizontally concatenating expert task vectors at each layer to form a unified representation of the collaborative update manifold.
2. **Singular Value Decomposition (SVD):** Extracting the orthonormal left-singular vectors to identify the principal orthogonal directions of parameter variation across all fine-tuned experts (Golub & Van Loan, *Matrix Computations*).
3. **Grassmannian Subspace Projection:** Constructing a projection operator $P^{(l)}$ from the top-$r$ left-singular vectors, which projects individual task updates onto the $r$-dimensional Grassmannian manifold $\mathbf{Gr}(r, d_{out})$, filtering out incoherent task-specific orthogonal noise.
4. **Offline Few-Shot Validation Tuning (OFS-Tune):** Optimizing layer-wise merging coefficients within the projected subspace using a tiny validation set (16 samples per task, 64 total) to prevent overfitting and avoid transductive noise (following the trajectory of *trial4_submission7*).

---

## 3. Mathematical Formulation
Let $W_{base}^{(l)} \in \mathbb{R}^{d_{out} \times d_{in}}$ denote the pre-trained base model weights at layer $l \in \{1, \dots, L\}$.
Let $W_k^{(l)} \in \mathbb{R}^{d_{out} \times d_{in}}$ denote the weights of fine-tuned expert $k \in \{1, \dots, K\}$, trained starting from $W_{base}^{(l)}$.

The task vector representing the parameter updates for expert $k$ at layer $l$ is:
$$V_k^{(l)} = W_k^{(l)} - W_{base}^{(l)}$$

We concatenate the $K$ task vectors horizontally to construct the joint multi-task update matrix:
$$\mathbf{M}^{(l)} = \left[ V_1^{(l)} \;\middle|\; V_2^{(l)} \;\middle|\; \dots \;\middle|\; V_K^{(l)} \right] \in \mathbb{R}^{d_{out} \times (K \cdot d_{in})}$$

We compute the Singular Value Decomposition (SVD) of $\mathbf{M}^{(l)}$:
$$\mathbf{M}^{(l)} = U^{(l)} \Sigma^{(l)} (V^{(l)})^T$$
where:
- $U^{(l)} \in \mathbb{R}^{d_{out} \times d_{out}}$ is an orthogonal matrix whose columns are the left-singular vectors representing principal directions of output parameter variation.
- $\Sigma^{(l)} \in \mathbb{R}^{d_{out} \times (K \cdot d_{in})}$ contains the singular values in descending order: $\sigma_1^{(l)} \ge \sigma_2^{(l)} \ge \dots \ge \sigma_{\min(d_{out}, K d_{in})}^{(l)} \ge 0$.

Let $r = \lfloor \gamma \cdot d_{out} \rfloor$ represent the subspace dimension (rank), parameterized by a fractional scaling factor $\gamma \in (0, 1]$. We construct the low-rank consensus basis matrix $U_r^{(l)}$ using the top $r$ columns of $U^{(l)}$:
$$U_r^{(l)} = U^{(l)}_{:, 1:r} \in \mathbb{R}^{d_{out} \times r}$$

The Grassmannian projection operator $P^{(l)} \in \mathbb{R}^{d_{out} \times d_{out}}$ onto the $r$-dimensional subspace is defined as:
$$P^{(l)} = U_r^{(l)} (U_r^{(l)})^T$$

By the **Eckart-Young-Mirsky Theorem**, the projected matrix $\tilde{\mathbf{M}}^{(l)} = P^{(l)} \mathbf{M}^{(l)}$ is the optimal rank-$r$ approximation of the multi-task updates under the Frobenius norm, satisfying:
$$\min_{\mathbf{X} \in \mathbb{R}^{d_{out} \times (K \cdot d_{in})}, \text{rank}(\mathbf{X}) \le r} \|\mathbf{M}^{(l)} - \mathbf{X}\|_F = \|\mathbf{M}^{(l)} - \tilde{\mathbf{M}}^{(l)}\|_F = \sqrt{\sum_{i=r+1}^{\min(d_{out}, K d_{in})} (\sigma_i^{(l)})^2}$$

The spectrally filtered (denoised) task vector for task $k$ is obtained via projection:
$$\tilde{V}_k^{(l)} = P^{(l)} V_k^{(l)}$$

The final merged weight matrix for layer $l$ is a linear combination of the projected updates:
$$W_{merged}^{(l)} = W_{base}^{(l)} + \sum_{k=1}^K \alpha_k^{(l)} \tilde{V}_k^{(l)} = W_{base}^{(l)} + P^{(l)} \left( \sum_{k=1}^K \alpha_k^{(l)} V_k^{(l)} \right)$$
where $\alpha_k^{(l)} \in \mathbb{R}$ are layer-wise merging coefficients.

---

## 4. Architecture Specifications
- **Backbone Network:** Vision Transformer (ViT-Tiny: `vit_tiny_patch16_224`), containing $L=12$ Transformer blocks.
- **Layers Targeted for Merging:** We target all $14$ major linear projections:
  - Multi-Head Attention: query, key, value, and output projection layers.
  - MLP block: expansion (`fc1`) and contraction (`fc2`) layers.
- **Inputs:** Token representations $H \in \mathbb{R}^{B \times N \times D}$ where $B$ is batch size, $N=197$ is sequence length, and $D=192$ is the embedding dimension.
- **Subspace Rank Hyperparameter $\gamma$:** Analyzed over a grid: $\gamma \in \{0.1, 0.2, 0.3, 0.5\}$.
- **Trainable Parameters:** Layer-wise coefficients $\alpha_k^{(l)}$ initialized uniformly to $1/K = 0.25$.
- **Optimization Strategy:** Coefficients are optimized on a tiny validation set (16 samples per task, 64 total) for 100 steps using Adam with learning rate $\eta = 10^{-2}$ and weight decay $10^{-4}$.

---

## 5. Baselines
1. **Uniform Merging:** Simple unweighted linear blending of original task vectors:
   $$W_{merged}^{(l)} = W_{base}^{(l)} + \frac{1}{K} \sum_{k=1}^K V_k^{(l)}$$
2. **Task Arithmetic (TA):** Global scale-searched model merging:
   $$W_{merged}^{(l)} = W_{base}^{(l)} + \lambda \sum_{k=1}^K V_k^{(l)}$$
3. **Sparse Task Arithmetic (STA):** Heuristic layer-wise magnitude-based pruning discarding $50\%$ of parameter updates before linear addition.
4. **Unconstrained OFS-Tune:** Layer-wise coefficients $\alpha_k^{(l)}$ optimized directly on validation data without Grassmannian projection ($P^{(l)} = I$). This acts as an ablation to demonstrate the exact regularizing impact of spectral projection.

---

## 6. Step-by-Step Interaction
1. **Task Vector Extraction:** Load $W_{base}$ and fine-tuned experts $W_1, \dots, W_K$. Compute original task vectors $V_k^{(l)} = W_k^{(l)} - W_{base}^{(l)}$.
2. **Subspace Consensus Construction:** For each layer $l$, construct joint update matrix $\mathbf{M}^{(l)}$, perform SVD, select rank $r = \lfloor \gamma \cdot d_{out} \rfloor$, and construct the projection matrix $P^{(l)}$.
3. **Spectral Denoising (Projection):** Multiply task vectors by the projection matrix to obtain the denoised task vectors $\tilde{V}_k^{(l)} = P^{(l)} V_k^{(l)}$.
4. **Validation Tuning:** Feed forward validation batches through the network with weight layers dynamically reconstructed as $W_{merged}^{(l)}(\alpha) = W_{base}^{(l)} + \sum_k \alpha_k^{(l)} \tilde{V}_k^{(l)}$.
5. **Backpropagation:** Compute the multi-task validation loss (average cross-entropy) and compute gradients $\frac{\partial \mathcal{L}_{val}}{\partial \alpha_k^{(l)}}$.
6. **Coefficient Update:** Update coefficients $\alpha_k^{(l)}$ via Adam. Repeat for 100 steps.
7. **Final Assembly & Evaluation:** Lock the optimized $\alpha_k^{(l)}$ parameters, construct the final static weight matrices, and evaluate the model across the test sets of MNIST, FashionMNIST, CIFAR-10, and SVHN.
