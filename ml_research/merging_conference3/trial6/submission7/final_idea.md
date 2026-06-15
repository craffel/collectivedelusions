# Idea Proposal: Micro-Batch Homogenization & Parameter-Free Subspace Routing

## 1. Persona Alignment
This proposal is a direct and relentless application of **The Minimalist** philosophy (Occam's razor). Rather than designing increasingly complex multi-layer or wave-inspired routing architectures to survive heterogeneous streams (which leads to the "Robustness-Accuracy Illusion" where accuracy is sacrificed for relative stability), we resolve the issue at the **data-stream level**. By partitioning heterogeneous streams into homogeneous micro-batches, we achieve perfect task specificity. Furthermore, we eliminate **100% of the trainable routing parameters** ($W$, $B$) by using the unsupervised subspace projections directly as coefficients. This completely bypasses both out-of-distribution (SVHN) parameter overfitting and the optimization overhead of AdamW, requiring **zero training/calibration split** data.

---

## 2. Core Techniques
We introduce two synergistic, completely non-parametric mechanisms:
1. **Parameter-Free Subspace Routing (PFSR):** We leverage the frozen classification weights of the pre-trained expert models to project feature representations onto a $K$-dimensional task coordinate space using cosine similarity, completely bypassing the need for parametric routing.
2. **Micro-Batch Homogenization (MBH):** To bypass **heterogeneity collapse** (the degradation of coefficients under mixed-task batches), we dynamically partition the incoming batch into homogeneous micro-batches on the fly based on their dominant task alignment. Each micro-batch is processed using a model merged with its own highly specialized, collapsed-free coefficients.

---

## 3. Mathematical Formulation

### 3.1 Input Subspace Projection
Let $z(x)_b \in \mathbb{R}^D$ be the globally pooled feature representation for sample $b$ in a batch of size $B$. Let the $k$-th block of feature $z(x)_b$ of size $d = D//K$ be denoted as $z_b^{(k)} \in \mathbb{R}^d$. Rather than training a parametric routing network or using unaligned global PCA, we project the block features onto the semantic task subspace learned by the pre-trained expert models.

Specifically, let the weights of Expert $k$ be denoted as $W_k \in \mathbb{R}^{C \times d}$, where the $C$ rows represent the learned class prototypes. We compute the maximum cosine similarity between the block feature $z_b^{(k)}$ and the rows of $W_k$:
$$u_{k, b} = \max_{c \in \{1, \dots, C\}} \frac{W_{k, c} \cdot z_b^{(k)}}{\|W_{k, c}\|_2 \|z_b^{(k)}\|_2}$$
This yields a $K$-dimensional task coordinate vector $u_b = [u_{1, b}, \dots, u_{K, b}]^T \in \mathbb{R}^K$. This projection requires **zero training** and **zero trainable parameters**, utilizing the frozen, pre-trained expert weights to perform robust task identification.

### 3.2 Dynamic Stream Partitioning & Micro-Batch Assembly
For each sample $b$ in the batch, we determine its dominant task coordinate:
$$k_b^* = \arg\max_{k \in \{1, \dots, K\}} u_{k, b}$$
We partition the heterogeneous batch $X = \{x_1, \dots, x_B\}$ into $G \le K$ homogeneous micro-batches $X^{(1)}, \dots, X^{(G)}$ based on these dominant coordinates:
$$X^{(g)} = \{x_b \in X \mid k_b^* = g\}$$

### 3.3 Parameter-Free Subspace Routing (PFSR)
The sample-wise merging coefficient for task $k$ and sample $x_b \in X^{(g)}$ is computed directly from the projected coordinates $u_b$ scaled by a static temperature hyperparameter $\tau > 0$:
$$\alpha_{k, b} = \frac{\exp(u_{k, b} / \tau)}{\sum_{j=1}^K \exp(u_{j, b} / \tau)}$$

### 3.4 Batch Coefficient Aggregation & Parameter Merging
For each active micro-batch $X^{(g)}$, we aggregate its sample-wise coefficients by averaging across the micro-batch size $|X^{(g)}|$:
$$\bar{\alpha}_k^{(g)} = \frac{1}{|X^{(g)}|} \sum_{x_b \in X^{(g)}} \alpha_{k, b}$$
Because all samples in $X^{(g)}$ map to the same task $g$, the average coefficient vector $\bar{\alpha}^{(g)}$ is highly task-specific and does not suffer from heterogeneity collapse. 
The merged parameters at layer $l \in \{1, \dots, L\}$ for micro-batch $g$ are:
$$W_{merged}^{(l), (g)} = W_{base}^{(l)} + \sum_{k=1}^K \bar{\alpha}_k^{(g)} V_k^{(l)}$$

---

## 4. Architecture Specifications
*   **Trainable Parameters:** **0** (completely parameter-free).
*   **Hyperparameters:** $\tau$ (scaling temperature, defaults to $0.1$).
*   **Dimensions:**
    *   Input representation dimension: $D = 192$.
    *   Subspace projection dimension: $d = K = 4$.
*   **Inputs:** High-dimensional feature batch $Z \in \mathbb{R}^{B \times D}$.
*   **Intermediate Representations:** Projected coordinate matrix $U \in \mathbb{R}^{B \times K}$.
*   **Outputs:** Concatenated model predictions of size $B \times C$ (where $C$ is the class dimension), re-assembled to match the original input batch ordering.

---

## 5. Baselines
We will evaluate our proposed PFSR + MBH method against a highly rigorous set of baselines:
1. **Static Uniform Merging:** Standard task arithmetic fusions without dynamic routing (Uniform Merging).
2. **Global Classical Linear Router:** The unregularized baseline that maps $z(x)_b$ directly to a $K$-dimensional score space via a trainable linear layer (achieved SOTA 67.20% on homogeneous streams).
3. **QWS-Merge SOTA:** The wave-inspired dynamic merging router.
4. **L3-Linear Router:** Our regularized layer-wise classical alternative (achieved 63.10% on homogeneous streams and 52.30% on heterogeneous streams).
5. **L3-Softmax Router:** The normalized variant that suffers from the Robustness-Accuracy Illusion.

All models will be evaluated under three deployment stream configurations:
*   **Homogeneous Batching ($B=256$):** Task-wise streams.
*   **Heterogeneous Batching ($B=256$):** Mixed-task streams.
*   **Sample-wise Batching ($B=1$):** High-precision streaming.

---

## 6. Step-by-Step Interaction
1. **Feature Extraction:** A heterogeneous batch $X$ of size $B$ is processed by the backbone, extracting globally pooled representation matrix $Z \in \mathbb{R}^{B \times D}$ from the first block.
2. **Coordinate Projection:** $Z$ is multiplied by the frozen, unsupervised projection matrix $P$, yielding low-dimensional task coordinates $U \in \mathbb{R}^{B \times K}$.
3. **Stream Partitioning:** For each sample $b$, we compute $k_b^* = \arg\max_k u_{k, b}$. Samples are grouped into homogeneous micro-batches $X^{(1)}, \dots, X^{(G)}$.
4. **Dynamic Coefficient Generation:** For each active micro-batch $X^{(g)}$:
    a. Sample-wise coefficients $\alpha_{k, b}$ are computed via PFSR: $\alpha_{k, b} = \text{Softmax}(u_{k, b} / \tau)$.
    b. Coefficients are averaged over the micro-batch to form $\bar{\alpha}_k^{(g)}$.
5. **Parameter Merging:** The model weights are dynamically merged on the fly for each micro-batch: $W_{merged}^{(l), (g)} = W_{base}^{(l)} + \sum_k \bar{\alpha}_k^{(g)} V_k^{(l)}$.
6. **Inference:** Each micro-batch $X^{(g)}$ is passed through its corresponding merged model $W_{merged}^{(g)}$, producing logit predictions.
7. **Re-assembly:** The logits from all micro-batches are concatenated and re-sorted to match the original index ordering of the input batch $X$.
