# Cross-Attention Multi-Expert Routing (CAM-Router)

## 1. Persona Alignment
As **The Empiricist**, our research philosophy is built upon the conviction that architectural choices must be validated by exhaustive, large-scale experimentation rather than elegant theoretical abstractions alone. The **Cross-Attention Multi-Expert Router (CAM-Router)** is custom-tailored for this philosophy:
1. **Massive Multi-Dimensional Sweeps:** CAM-Router introduces discrete structural hyperparameters (number of attention heads $h$, query dimensions, query initialization strategies) that allow us to run extensive, multi-node parameter sweeps using available compute (via Slurm QoS `--qos=low`).
2. **Robust Noise and Occlusion Stress Testing:** Unlike global pooling methods, CAM-Router's spatial cross-attention mechanism is uniquely testable. We will evaluate its performance under systematic patch-level spatial occlusions and Gaussian noise across varying intensities, creating robust empirical noise-resilience curves to verify our claims.
3. **Extensive Baseline Benchmarking:** We will compare CAM-Router against an exhaustive array of baselines (Static Uniform, Unregularized/Regularized Global Linear, QWS-Merge, BL-Router, BSigmoid-Router, and L3-Router) across multiple seeds (5+ seeds) and multiple backbones (ViT-Tiny and CLIP-ViT-B/16) to ensure overwhelming statistical evidence of its benefits.

---

## 2. Core Techniques
CAM-Router replaces the standard flat global average pooling step used in classical and quantum routing with a lightweight, multi-head cross-attention mechanism:
1. **Un-pooled Spatial Token Sequences:** Instead of collapsing the spatial representations immediately via global average pooling, we retain the full sequence of patch tokens $H_0 \in \mathbb{R}^{B \times N \times D}$ from the initial transformer layer of the backbone.
2. **Trainable Task-Expert Queries:** We introduce a set of $K$ trainable query embeddings $Q \in \mathbb{R}^{K \times D}$, where each query $Q_k$ acts as a specialized template trained to detect feature structures relevant to task $k$.
3. **Multi-Head Cross-Attention (MHCA):** The task queries $Q$ attend to the patch tokens $H_0$. This allows each task expert to dynamically concentrate on the spatial regions most relevant to its domain (e.g., focusing on the digits in MNIST or the cluttered background in SVHN).
4. **Independent Bounded Sigmoidal Routing:** To avoid the zero-sum competitive bottleneck of Softmax bounding (as identified in `trial5_submission4`), the task-specific contextual representations are projected to logits and passed through independent, Softmax-free Sigmoid activations, scaled by $\lambda_{max} = 0.3$.

---

## 3. Mathematical Formulation

### 1. Multi-Head Cross-Attention (MHCA) Routing
Let $H_{0, b} \in \mathbb{R}^{N \times D}$ be the spatial token representations for sample $b \in \{1, \dots, B\}$, and $Q \in \mathbb{R}^{K \times D}$ be the trainable task-expert queries, where $K$ is the number of tasks and $D$ is the feature dimension. 

For each attention head $j \in \{1, \dots, h\}$, let the head dimension be $d_h = D/h$. We project the queries, keys, and values:
\begin{align}
    Q^{(j)} &= Q W_Q^{(j)} \in \mathbb{R}^{K \times d_h} \\
    K_{b}^{(j)} &= H_{0, b} W_K^{(j)} \in \mathbb{R}^{N \times d_h} \\
    V_{b}^{(j)} &= H_{0, b} W_V^{(j)} \in \mathbb{R}^{N \times d_h}
\end{align}
where $W_Q^{(j)}, W_K^{(j)}, W_V^{(j)} \in \mathbb{R}^{D \times d_h}$ are the trainable query, key, and value projection matrices for head $j$.

The attention score matrix for sample $b$ and head $j$ is computed via scaled dot-product:
\begin{equation}
    S_{b}^{(j)} = \text{Softmax}\left( \frac{Q^{(j)} (K_{b}^{(j)})^T}{\sqrt{d_h}} \right) \in \mathbb{R}^{K \times N}
\end{equation}

The contextual representation for head $j$ is:
\begin{equation}
    O_{b}^{(j)} = S_{b}^{(j)} V_{b}^{(j)} \in \mathbb{R}^{K \times d_h}
\end{equation}

We concatenate the outputs of all $h$ heads and project back to $D$-dimensional space to produce the joint task-specific representation $A_b$:
\begin{equation}
    A_b = \left[ O_{b}^{(1)}; \dots; O_{b}^{(h)} \right] W_O \in \mathbb{R}^{K \times D}
\end{equation}
where $W_O \in \mathbb{R}^{D \times D}$ is the trainable output projection matrix.

### 2. Independent Gating and Bounded Activation
For each task expert $k \in \{1, \dots, K\}$, we extract its contextualized representation $A_{b, k, :} \in \mathbb{R}^D$ (the $k$-th row of $A_b$). We pass this representation through an independent linear routing head to compute a scalar logit:
\begin{equation}
    o_{k, b} = \langle A_{b, k, :}, W_{route, k} \rangle + b_{route, k}
\end{equation}
where $W_{route, k} \in \mathbb{R}^D$ and $b_{route, k} \in \mathbb{R}$ are trainable routing weights and biases.

We apply a Softmax-free, independent Bounded Sigmoid activation to obtain sample-wise coefficients:
\begin{equation}
    \alpha_{k, b} = \lambda_{max} \cdot \text{Sigmoid}(o_{k, b}) \quad \text{with } \lambda_{max} = 0.3
\end{equation}

### 3. Batch Collapse and Dynamic Weight Merging
We average the sample-level routing coefficients across the batch dimension to achieve the batch-level collapsed coefficients:
\begin{equation}
    \bar{\alpha}_k = \frac{1}{B} \sum_{b=1}^B \alpha_{k, b}
\end{equation}

The multi-task merged model weights for each layer $l \in \{1, \dots, L\}$ are assembled dynamically:
\begin{equation}
    W_{merged}^{(l)}(x) = W_{base}^{(l)} + \sum_{k=1}^K \bar{\alpha}_k V_k^{(l)}
\end{equation}
where $V_k^{(l)} = W_k^{(l)} - W_{base}^{(l)}$ is the task vector of task expert $k$.

### 4. Loss Function and Regularization
We optimize the parameters $\Theta = \{Q, W_Q, W_K, W_V, W_O, W_{route}, b_{route}\}$ by minimizing the Cross-Entropy loss on the calibration set under standard $L_2$ weight decay:
\begin{equation}
    \mathcal{L}_{total} = \mathcal{L}_{CE} + \lambda_{wd} \left( \sum_{j=1}^h \left( \|W_Q^{(j)}\|_F^2 + \|W_K^{(j)}\|_F^2 + \|W_V^{(j)}\|_F^2 \right) + \|W_O\|_F^2 + \|Q\|_F^2 + \sum_{k=1}^K \left( \|W_{route, k}\|_2^2 + b_{route, k}^2 \right) \right)
\end{equation}

---

## 4. Architecture Specifications

The CAM-Router operates with the following specific architectural parameters:
- **Backbone Network:** Compact Vision Transformer (`vit_tiny_patch16_224`) with $L=14$ layer groups and $D=192$ hidden dimensions.
- **Input Tokens:** Spatial patch token sequence $H_0 \in \mathbb{R}^{B \times N \times D}$ where $N = 196$ patch tokens.
- **Trainable Queries:** $Q \in \mathbb{R}^{K \times D}$ with $K=4$ (representing MNIST, FashionMNIST, CIFAR-10, SVHN) and $D=192$.
- **Attention Projections:** Query, Key, Value weights $W_Q, W_K, W_V$ of shape $192 \times 192$, and Output weight $W_O$ of shape $192 \times 192$.
- **Routing Classifier:** $K=4$ linear layers mapping $D=192$ to $1$ logit.
- **Parameter Overhead:** 
  - $Q$: $4 \times 192 = 768$ params.
  - MHCA projections: $4 \times 192 \times 192 = 147,456$ params.
  - Routing layers: $4 \times 192 + 4 = 772$ params.
  - **Total Trainable Params:** $148,996$ (~0.15M parameters). This is an extremely lightweight addition (~2.6% of the 5.7M parameter ViT-Tiny backbone).

---

## 5. Baselines
We evaluate CAM-Router against a comprehensive set of baselines:
1. **Static Uniform Merging:** Parameter fusions without dynamic routing ($\alpha_k = 0.3$).
2. **Unregularized Global Linear Router:** Maps the average-pooled representation $z(x)_b$ directly to task logits via a single linear layer, optimized with no weight decay.
3. **Regularized Global Linear Router:** Average-pooled routing optimized with AdamW and $L_2$ weight decay ($\lambda_{wd} = 10^{-3}$).
4. **QWS-Merge SOTA:** Quantum wave-superposition routing model utilizing cosine phase activations on the unit sphere.
5. **BSigmoid-Router:** The state-of-the-art classical alternative mapping average-pooled representations to task coefficients using independent Bounded Sigmoids.
6. **L3-Router:** Layer-wise low-dimensional classical linear and non-linear routers.

---

## 6. Step-by-Step Interaction

1. **Feature Extraction:** The input batch $x \in \mathbb{R}^{B \times C \times H \times W}$ is passed through the patch embedding layer of the ViT backbone to produce the sequence of patch tokens $H_0 \in \mathbb{R}^{B \times N \times D}$.
2. **Cross-Attention Projection:** The spatial patch tokens $H_0$ and the trainable task queries $Q \in \mathbb{R}^{K \times D}$ are projected into query, key, and value matrices across $h$ attention heads.
3. **Query-Token Attention:** For each sample $b$, the task-expert queries $Q$ attend to the key representations $K_{b}^{(j)}$ to compute attention weights $S_{b}^{(j)}$. The attention outputs are concatenated and projected to assemble the joint contextual task representations $A_b \in \mathbb{R}^{K \times D}$.
4. **Logit Projection:** For each task expert $k$, its specialized feature vector $A_{b, k, :}$ is projected via $W_{route, k}$ to produce raw logit $o_{k, b}$.
5. **Independent Sigmoid Gating:** Raw logits are mapped through the Bounded Sigmoid activation to obtain sample-wise routing coefficients $\alpha_{k, b} \in [0, 0.3]$.
6. **Batch-Level Averaging:** The coefficients are averaged over the batch dimension to produce collapsed batch coefficients $\bar{\alpha}_k$.
7. **On-the-Fly Merging:** The merged model weights $W_{merged}^{(l)}(x)$ are dynamically assembled layer-by-layer.
8. **Forward Prediction:** The input batch is processed by the newly assembled model parameters $W_{merged}$ to produce the final classification predictions.
