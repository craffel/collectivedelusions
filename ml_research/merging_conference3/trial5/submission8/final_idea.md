# EpiMerge: Epigenetic Weight Masking for True Sample-Wise Dynamic Model Merging

## 1. Persona Alignment
EpiMerge embodies the philosophy of **The Visionary** (found in `persona.md`) by rejecting the incremental, compromise-driven mindset of static weight averaging and the batch-dependent shortcuts of existing dynamic routers. Instead of finding a static consensus weight vector or resorting to a lossy batch-averaged "wavefunction collapse" (such as QWS-Merge), we draw a profound, paradigm-shifting inspiration from molecular biology: **Epigenetics**. 

In biology, the physical genetic sequence (the static pre-trained weights) remains completely unchanged, but the physical expression of individual genes is dynamically and continuously modulated on-the-fly by chemical markers (methylation and histone modification) in response to environmental stimuli. EpiMerge imports this biological mechanism into deep weight spaces. We propose a completely fresh architecture and training paradigm where the input sample dynamically acts as an environmental stimulus, generating custom, low-rank row-wise and column-wise "epigenetic masks" that selectively activate or silence specific coordinate-wise expert parameter pathways. This radical approach enables **true sample-wise dynamic merging** in a single vectorized forward pass, representing an exciting step toward adaptive, biological-mimicry deep learning systems.

---

## 2. Core Techniques
*   **Epigenetic Reader Head (ERH):** A highly parameter-efficient, low-rank routing module placed at each layer group. The ERH projects the global latent input representation down to coordinate-wise row and column gating masks.
*   **Low-Rank Row-Column Dual Gating:** Instead of constructing expensive 2D weight-shaped masks (which would cause parameter explosion), we parameterize coordinate-wise gating using the outer product of a low-rank row mask $\mathbf{r} \in \mathbb{R}^{D_{out}}$ and column mask $\mathbf{c} \in \mathbb{R}^{D_{in}}$.
*   **True Parallel Sample-Wise Weight Contraction:** We bypass the batch-dependency and "heterogeneity collapse" that affects prior dynamic methods by using highly vectorized PyTorch tensor contractions (`torch.einsum`). This allows every sample in a mixed-task batch to be processed by its own customized weight matrix in parallel, maintaining perfect I.I.D. inference.
*   **Optimization-Regularized Calibration:** The ERH parameters are optimized offline on a tiny 64-sample calibration dataset (16 samples per task) for 100 steps, utilizing standard backpropagation without any test-time computational overhead.

---

## 3. Mathematical Formulation

### A. Global Input Representation Extraction
Let $x \in \mathbb{R}^{B \times C \times H \times W}$ be a batch of input images. The patch embedding layer extracts tokens:
$$H_0 = \text{PatchEmbed}(x) \in \mathbb{R}^{B \times N \times D}$$
where $N$ is the number of patch tokens and $D$ is the backbone's embedding dimension. We compute a global representation $z(x)_b \in \mathbb{R}^D$ via spatial average pooling:
$$z(x)_b = \frac{1}{N} \sum_{n=1}^N H_{0, b, n, :}$$
We project this into a low-dimensional latent space ($d = K = 4$) via a frozen random projection matrix $P \in \mathbb{R}^{D \times d}$ and apply L2-normalization to yield the unit-sphere representation $\psi(x)_b$:
$$\tilde{\psi}(x)_b = z(x)_b P \in \mathbb{R}^d$$
$$\psi(x)_b = \frac{\tilde{\psi}(x)_b}{\|\tilde{\psi}(x)_b\|_2 + \epsilon}$$

### B. Low-Rank Epigenetic Mask Generation
For each layer group $l \in \{1, \dots, L\}$, task expert $k \in \{1, \dots, K\}$, and linear layer weight matrix $W^{(l)} \in \mathbb{R}^{D_{out} \times D_{in}}$, we introduce trainable epigenetic reader weights:
$$U_k^{(l)} \in \mathbb{R}^{D_{out} \times d}$$
$$V_k^{(l)} \in \mathbb{R}^{D_{in} \times d}$$
The sample-specific row gating mask $\mathbf{r}_{k, b}^{(l)}(x) \in \mathbb{R}^{D_{out}}$ and column gating mask $\mathbf{c}_{k, b}^{(l)}(x) \in \mathbb{R}^{D_{in}}$ are:
$$\mathbf{r}_{k, b}^{(l)}(x) = \text{Sigmoid}\left( U_k^{(l)} \psi(x)_b \right)$$
$$\mathbf{c}_{k, b}^{(l)}(x) = \text{Sigmoid}\left( V_k^{(l)} \psi(x)_b \right)$$

### C. Sample-Specific Weight Reconstruction
We denote the static expert task vectors as $T_k^{(l)} = W_k^{(l)} - W_{base}^{(l)}$. The dynamic, epigenetically modulated task vectors are combined with the base weights to construct sample-specific merged weight matrices:
$$W_{merged, b}^{(l)}(x) = W_{base}^{(l)} + \sum_{k=1}^K \left( \mathbf{r}_{k, b}^{(l)}(x) \otimes \mathbf{c}_{k, b}^{(l)}(x) \right) \odot T_k^{(l)}$$
where $\otimes$ is the vector outer product and $\odot$ is the element-wise Hadamard product. In index form:
$$[W_{merged, b}^{(l)}(x)]_{i, j} = [W_{base}^{(l)}]_{i, j} + \sum_{k=1}^K \mathbf{r}_{k, b, i}^{(l)}(x) \cdot \mathbf{c}_{k, b, j}^{(l)}(x) \cdot [T_k^{(l)}]_{i, j}$$

### D. Vectorized Parallel Forward Pass
Let $X \in \mathbb{R}^{B \times N \times D_{in}}$ be the incoming activation tensor for layer $l$. The sample-specific forward pass is executed in parallel using batched tensor contractions:
$$Y_b = X_b \cdot \left[ W_{merged, b}^{(l)}(x) \right]^T + b_{merged}^{(l)}$$
In PyTorch, this is expressed as:
$$Y = \text{torch.einsum}('bni,boi->bno', X, W_{merged}) + b_{merged}$$

---

## 4. Architecture Specifications
*   **Backbone:** $\mathtt{vit\_tiny\_patch16\_224}$ ($5.7$M parameters), containing $L=14$ layer groups.
*   **Latent Space Dimension ($d$):** $d = K = 4$ tasks (MNIST, FashionMNIST, CIFAR-10, SVHN).
*   **Low-Rank Projection Matrices:**
    *   For a Linear layer with weight shape $[D_{out}, D_{in}]$:
        *   Row reader matrix $U_k^{(l)}$ of shape $[D_{out}, 4]$.
        *   Column reader matrix $V_k^{(l)}$ of shape $[D_{in}, 4]$.
    *   For the Multi-Head Attention QKV projection ($576 \times 192$):
        *   $U_k$ has size $576 \times 4 = 2304$ parameters.
        *   $V_k$ has size $192 \times 4 = 768$ parameters.
        *   Total ERH parameters per expert per QKV projection: $3072$.
*   **Global Random Projector ($P$):** A single frozen matrix of shape $[192, 4]$ shared across the entire model.

---

## 5. Baselines
*   **Uniform Merging (Task Arithmetic):** Standard static baseline ($\lambda = 0.3$), showing the performance ceiling of unoptimized weight merging.
*   **AdaMerging (Unsupervised TTA):** Standard test-time adaptive baseline using entropy minimization.
*   **OFS-Tune (Supervised Static):** Static layer-wise coefficient optimization on the $64$-sample calibration set.
*   **Linear Router (Classical Dynamic):** Classical input-dependent dynamic router that maps global pooled features to global scaling coefficients, showing vulnerability to "heterogeneity collapse" and SVHN task conflict.
*   **QWS-Merge (Quantum-Inspired Dynamic):** The primary baseline for dynamic merging, which suffers from batch dependency and "heterogeneity collapse" due to batch averaging.

---

## 6. Step-by-Step Interaction
1.  **Input:** A batch of images $x \in \mathbb{R}^{B \times 3 \times 224 \times 224}$.
2.  **Patch Embedding:** The Patch Embedding block projects $x$ to $H_0 \in \mathbb{R}^{B \times 197 \times 192}$.
3.  **Global Pooling:** We compute $z(x) \in \mathbb{R}^{B \times 192}$ via spatial averaging.
4.  **Unit Sphere Projection:** $z(x)$ is projected via frozen random matrix $P$ and L2-normalized to yield $\psi(x) \in \mathbb{R}^{B \times 4}$.
5.  **Epigenetic Masking (At Layer $l$):**
    *   Inputs $X \in \mathbb{R}^{B \times 197 \times D_{in}}$ enter the layer.
    *   For each of the $K=4$ tasks, the trainable epigenetic reader heads process $\psi(x)$ to output row masks $R \in \mathbb{R}^{B \times 4 \times D_{out}}$ and column masks $C \in \mathbb{R}^{B \times 4 \times D_{in}}$.
    *   The modulated task vectors are combined with the base weights to construct the batch of custom, sample-specific weight matrices $W_{merged} \in \mathbb{R}^{B \times D_{out} \times D_{in}}$.
6.  **Parallel Execution:** The tensor contraction $Y = \text{torch.einsum}('bni,boi->bno', X, W_{merged})$ is computed in parallel.
7.  **Classification:** The final output of the ViT backbone is passed to specialized classification heads to yield predictions $\hat{y} \in \mathbb{R}^{B \times 10}$.
