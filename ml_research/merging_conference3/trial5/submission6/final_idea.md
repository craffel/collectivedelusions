# Sparse Low-Rank Dynamic Merging (SLD-Merge)

## 1. Persona Alignment
*Sparse Low-Rank Dynamic Merging (SLD-Merge)* is designed from the ground up to align perfectly with the core principles of **The Pragmatist**:
1.  **Eliminating Batch Dependency (Real-World Deployment):** Previous dynamic model merging methods like QWS-Merge average routing coefficients across the batch dimension to construct a single merged parameter set per batch. This results in "heterogeneity collapse" at larger batch sizes and introduces a severe I.I.D. violation: a single sample's prediction varies based on other samples in the batch. SLD-Merge operates completely batch-independently. The inference for any sample is stateless, deterministic, and identical whether run individually ($B=1$) or in massive, heterogeneous batches.
2.  **Resource-Constrained Optimization (Memory and Storage):** Standard multi-task expert systems require storing $K$ full-rank specialized models, leading to linear storage growth ($O(K \cdot N)$). SLD-Merge uses offline Singular Value Decomposition (SVD) to decompose the parameter task vectors of the $K$ experts into low-rank adapters of rank $r \ll D$ (e.g., $r=8$). This reduces the extra parameter storage by over **91%**, making it feasible for edge and embedded deployment.
3.  **Low Latency and Compute Costs:** By implementing a Top-1 sparse selection mechanism, SLD-Merge activates only the single most relevant expert's low-rank adapter path per sample. The computational overhead is restricted to a single low-rank forward pass ($2 \times B \times N \times D \times r$), adding only about **8.3%** more FLOPs compared to a single static model, compared to the huge overhead of running $K$ full models or dynamically reconstructing massive dense layers at runtime.
4.  **Robustness to Overfitting:** The routing parameters are mapped into a bounded, low-dimensional spherical cosine space rather than unconstrained high-dimensional linear projections. This acts as a heavy regularizer, preventing overfitting during the few-shot calibration phase on tiny offline datasets.

---

## 2. Core Techniques
SLD-Merge combines the following core techniques to achieve efficient, batch-independent dynamic multi-task inference:
1.  **Offline SVD Task Vector Decomposition:** We compute the task vectors $V_k = W_k - W_{base}$ for each pre-trained expert model. Since these are fixed, we run Singular Value Decomposition (SVD) offline once to decompose $V_k \approx B_k A_k$, keeping only the top-$r$ singular vectors.
2.  **Bounded Cosine-Similarity Router:** We define a learned routing basis vector $\Phi_k^{(l)}$ for each expert $k$ at layer $l$. We compute the cosine similarity between the layer's pooled input activations and the routing basis, ensuring routing scores are strictly bounded in $[-1, 1]$ to suppress representation noise.
3.  **Top-1 Sparse Routing Selection (Hard Gating):** We apply a hard argmax over the cosine similarity scores. This disables all except the highest-scoring expert for that specific input, ensuring that activations only pass through the most specialized adapter path.
4.  **Batch-Independent Parallel Execution:** We perform the selective update using element-wise multiplication in a fully vectorized PyTorch forward pass, ensuring that no batch-level averaging occurs.

---

## 3. Mathematical Formulation

### 3.1. Task Vector Low-Rank SVD Approximation
Let $W_{base}^{(l)} \in \mathbb{R}^{D_{out} \times D_{in}}$ be the pre-trained base model weights at layer $l$, and $W_k^{(l)}$ be the weights of the fine-tuned expert on task $k \in \{1, \dots, K\}$. The dense task vector is:
$$V_k^{(l)} = W_k^{(l)} - W_{base}^{(l)}$$
We perform offline SVD on $V_k^{(l)}$:
$$V_k^{(l)} = U_k^{(l)} \Sigma_k^{(l)} (V'_k^{(l)})^T$$
We truncate this decomposition to rank $r$ to form the low-rank matrices $B_k^{(l)} \in \mathbb{R}^{D_{out} \times r}$ and $A_k^{(l)} \in \mathbb{R}^{r \times D_{in}}$:
$$B_k^{(l)} = U_k^{(l)}[:, :r] \sqrt{\Sigma_k^{(l)}[:r]}$$
$$A_k^{(l)} = \sqrt{\Sigma_k^{(l)}[:r]} (V'_k^{(l)})^T[:, :r]^T$$
This ensures $V_k^{(l)} \approx B_k^{(l)} A_k^{(l)}$ with minimal reconstruction loss under the L2 norm.

### 3.2. Cosine-Similarity Routing Scores
Let $X \in \mathbb{R}^{B \times N \times D}$ be the input activation tensor to layer $l$, where $B$ is the batch size, $N$ is the sequence/token length, and $D$ is the embedding dimension. We extract a global sample representation $z(x)_b \in \mathbb{R}^D$ by computing the spatial average over tokens:
$$z(x)_b = \frac{1}{N} \sum_{n=1}^N X_{b, n, :} \quad \forall b \in \{1, \dots, B\}$$
We define learned, layer-wise task-routing basis vectors $\Phi_k^{(l)} \in \mathbb{R}^D$. For each sample $b$ and expert $k$, we compute the cosine similarity score:
$$s_{k, b}^{(l)} = \frac{\langle z(x)_b, \Phi_k^{(l)} \rangle}{\|z(x)_b\|_2 \|\Phi_k^{(l)}\|_2 + \epsilon}$$
where $\epsilon = 10^{-8}$ is a numerical stabilizer.

### 3.3. Top-1 Sparse Routing and Forward Pass
Rather than computing soft weights, we enforce extreme computational efficiency and clean isolation of expert features by selecting only the single best-aligned expert for sample $b$:
$$\hat{k}_b^{(l)} = \arg\max_{k \in \{1, \dots, K\}} s_{k, b}^{(l)}$$
The sample-wise routing coefficient is represented as a sparse one-hot vector:
$$\alpha_{k, b}^{(l)} = \begin{cases} 1 & \text{if } k = \hat{k}_b^{(l)} \\ 0 & \text{otherwise} \end{cases}$$
The output activation $Y_b \in \mathbb{R}^{N \times D}$ of layer $l$ for sample $b$ is computed as:
$$Y_b = X_b W_{base}^{(l)} + \alpha_{\hat{k}_b, b}^{(l)} \cdot \left( (X_b A_{\hat{k}_b}^{(l)}) B_{\hat{k}_b}^{(l)} \right)$$
where $X_b \in \mathbb{R}^{N \times D}$. In parallel batched form, this is implemented as:
$$Y = X W_{base}^{(l)} + \sum_{k=1}^K \alpha_k \odot \left( (X A_k^{(l)}) B_k^{(l)} \right)$$
where $\alpha_k \in \mathbb{R}^{B \times 1 \times 1}$ is the broadcasted routing coefficient tensor for task $k$.

---

## 4. Architecture Specifications
1.  **Backbone Network:** We use the compact `vit_tiny_patch16_224` vision transformer (5.7M parameters) with $12$ Transformer blocks ($L=12$), plus Patch Embedding and LayerNorm layers (total 14 layer groups). The hidden embedding dimension is $D=192$.
2.  **Low-Rank Adapters:** 
    *   **Matrix Dimensions:** For a dense layer of shape $192 \times 192$, the low-rank matrices are $A_k^{(l)} \in \mathbb{R}^{r \times 192}$ and $B_k^{(l)} \in \mathbb{R}^{192 \times r}$.
    *   **Target Rank:** We evaluate ranks $r \in \{4, 8, 16\}$.
3.  **Routing Parameters:**
    *   **Basis Vectors:** $\Phi_k^{(l)} \in \mathbb{R}^{D}$ ($D=192$).
    *   **Temperature Parameter:** A learned temperature scale $\tau^{(l)} > 0$, initialized to $1.0$.
    *   **Parameter Footprint:** Across $14$ layer groups and $K=4$ experts, the total trainable routing parameters are $14 \times 4 \times 192 = 10,752$ parameters. This is highly compact and can be robustly optimized on a tiny calibration split.
4.  **Inputs and Outputs:**
    *   **Input:** Multi-task classification images of size $224 \times 224 \times 3$.
    *   **Intermediate Activation:** Token representations of shape $B \times 197 \times 192$.
    *   **Final Output:** Logits over $10$ classes per dataset task.

---

## 5. Baselines
We compare SLD-Merge against the following baselines:
1.  **Uniform Merging (Static):** The arithmetic mean of all expert weights: $W_{merged} = \frac{1}{K} \sum_k W_k$.
2.  **Task Arithmetic (Static):** Linear addition of task vectors scaled by a fixed global coefficient: $W_{merged} = W_{base} + \lambda \sum_k V_k$.
3.  **Linear Router (Dynamic):** A classical fully-connected projection router that uses Softmax to compute routing coefficients. We evaluate both the standard batch-averaged version and a sample-wise vectorized version.
4.  **QWS-Merge (Dynamic):** The state-of-the-art quantum wavefunction dynamic merging.
5.  **Individual Expert Ceiling (Task-Specific):** The upper-bound performance where each task is evaluated on its own dedicated fully fine-tuned expert model.

*Critically, we evaluate all dynamic methods across various evaluation streams to demonstrate our advantages under:*
-   **Varying Batch Sizes ($B \in \{1, 4, 16, 64, 256\}$):** To expose the "heterogeneity collapse" of batch-dependent methods.
-   **Mixed-Task (Heterogeneous) Batches:** Where samples from different datasets (MNIST, SVHN, etc.) are mixed in the same batch, violating standard I.I.D. assumptions for QWS-Merge but handled seamlessly by SLD-Merge.

---

## 6. Step-by-Step Interaction

The flow of data through SLD-Merge at inference time for a batch $x$ is as follows:

```
[Input Batch: x (B x C x H x W)]
               │
               ▼
[Patch Embedding & LayerNorm] ──► Global Pooled representation: z(x) (B x D)
               │
               ├─────────────────────────────────────────┐
               ▼                                         ▼
[Base Layer Forward Pass]                  [Cosine-Similarity Score Calc]
  Compute X_base = X @ W_base                Compute s_k,b = CosSim(z(x)_b, Phi_k)
               │                                         │
               │                                         ▼
               │                           [Top-1 Sparse Selection]
               │                             Select hat_k_b = argmax(s_k,b)
               │                             Generate sparse coefficient alpha_k,b
               │                                         │
               ├─────────────────────────────────────────┘
               ▼
[Parallel Low-Rank Adapter Computation]
  For each sample b:
    Run forward pass ONLY through selected adapter path: alpha * (X_b @ A_hat_k @ B_hat_k)
               │
               ▼
[Element-wise Summation]
  Y_b = (X_b @ W_base) + (X_b @ A_hat_k @ B_hat_k)
               │
               ▼
[Layer Output Activation: Y (B x N x D)]
```

This interaction is fully vectorized in PyTorch, executing in parallel without loops or branches across the batch dimension, ensuring peak GPU/NPU utilization.
