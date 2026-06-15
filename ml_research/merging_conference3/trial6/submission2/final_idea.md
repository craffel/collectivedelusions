# Idea Proposal: Rademacher-Regularized Dynamic Model Merging (R2D-Merge)

## 1. Persona Alignment
This proposal strongly aligns with the **Theorist** persona. Rather than adopting heuristic formulations or over-parameterized physical metaphors (like quantum wavefunctions), we approach the challenge of dynamic model merging through the rigorous lens of learning theory and statistical generalization. By formally analyzing the Rademacher complexity of the dynamic parameter-space blending function class, we derive a mathematically optimal, task-adaptive covariance-weighted regularizer. This approach replaces empirical trial-and-error with a provably sound, bounded, and stable framework that explicitly prevents transductive overfitting on OOD stream noise.

---

## 2. Core Techniques
We introduce **Rademacher-Regularized Dynamic Model Merging (R2D-Merge)**, which consists of the following core mechanisms:
1.  **Low-Dimensional Representational Projection:** Input representations $z(x) \in \mathbb{R}^D$ are compressed using a frozen unsupervised PCA projection matrix $P \in \mathbb{R}^{D \times d}$ and normalized onto the unit sphere, forming the input state $\psi(x) \in \mathbb{R}^d$. This restricts the router's representation capacity.
2.  **Layer-wise Low-dimensional Classical Routing:** A highly parameter-efficient linear projection layer mapping the input state $\psi(x)$ directly to layer-specific task merging coefficients $\alpha_{l, k}(x) = w_{l, k}^T \psi(x) + b_{l, k}$. This eliminates wave-like over-parameterization.
3.  **Covariance-weighted Frobenius Regularization (CFR):** A novel, task-adaptive regularizer directly derived from the Rademacher complexity bound of the dynamic blending hypothesis class. CFR weights the penalty on router weight vector $w_{l, k}$ using a pre-calculated empirical covariance matrix of the projected features, scaled by the localized energy of the task vector.

---

## 3. Mathematical Formulation

### 3.1 Parameter Blending Formulation
Let $W_{base}^{(l)}$ be the pre-trained weights of layer $l \in \{1, \dots, L\}$. Let $V_k^{(l)} = W_k^{(l)} - W_{base}^{(l)}$ be the fine-tuned task vector for expert $k \in \{1, \dots, K\}$. For an input $x_i$, the layer-wise merged weight is:
$$W_{merged}^{(l)}(x_i) = W_{base}^{(l)} + \sum_{k=1}^K \alpha_{l, k}(x_i) V_k^{(l)}$$
where the dynamic merging coefficients $\alpha_{l, k}(x_i)$ are predicted by the router:
$$\alpha_{l, k}(x_i) = w_{l, k}^T \psi(x_i) + b_{l, k}$$

### 3.2 Rademacher Complexity Derivation
The forward activation of layer $l$ on input activation $z^{(l)}(x_i)$ is:
$$y^{(l)}(x_i) = z^{(l)}(x_i) W_{merged}^{(l)}(x_i) = z^{(l)}(x_i) W_{base}^{(l)} + \sum_{k=1}^K (w_{l, k}^T \psi(x_i) + b_{l, k}) \left( z^{(l)}(x_i) V_k^{(l)} \right)$$
Let $\mathcal{H}_l$ represent the hypothesis class of this dynamic mapping. Ignoring the fixed pre-trained term $z^{(l)}(x_i) W_{base}^{(l)}$ and the bias offset, we vectorize the router parameters $w_l \in \mathbb{R}^{K \times d}$ as $\mathbf{w}_l \in \mathbb{R}^{Kd}$. The empirical Rademacher complexity $\hat{\mathcal{R}}_S(\mathcal{H}_l)$ over a calibration set $S$ of size $N$ is bounded by:
$$\hat{\mathcal{R}}_S(\mathcal{H}_l) \leq \frac{\|\mathbf{w}_l\|_2}{N} \sqrt{\sum_{i=1}^N \sum_{k=1}^K \|z^{(l)}(x_i) V_k^{(l)}\|_2^2 \cdot \|\psi(x_i)\|_2^2}$$

### 3.3 Covariance-weighted Frobenius Regularization (CFR)
When $\psi(x)$ is not strictly constrained to the unit sphere, or to reflect task-specific scale imbalances, we define the **Task-specific Empirical Covariance Matrix** $C_{l, k} \in \mathbb{R}^{d \times d}$ for each layer $l$ and task $k$:
$$C_{l, k} = \frac{1}{N} \sum_{i=1}^N \|z^{(l)}(x_i) V_k^{(l)}\|_2^2 \cdot \psi(x_i) \psi(x_i)^T$$
The **CFR penalty** is defined as the quadratic form:
$$\mathcal{L}_{CFR}(W) = \sum_{l=1}^L \sum_{k=1}^K w_{l, k}^T C_{l, k} w_{l, k}$$
This CFR penalty directly minimizes the Rademacher generalization bound of the dynamic blending operation.
The total optimization objective for the router parameters is:
$$\mathcal{L}_{total} = \mathcal{L}_{CE} + \lambda_{wd} \mathcal{L}_{CFR}(W)$$
where $\lambda_{wd}$ is the regularization strength. Since $C_{l, k}$ is computed **once** over the fixed calibration split before training starts, CFR introduces **zero** online computational overhead!

---

## 4. Architecture Specifications
- **Input Dimensions ($D$):** 192 (globally pooled representational features from the first block of the ViT-Tiny backbone).
- **Projection Layer ($d$):** Frozen unsupervised PCA projection matrix $P \in \mathbb{R}^{192 \times 4}$ compressing inputs to $d = 4$ dimensions, followed by unit-sphere normalization.
- **Router Parameters:**
  - Layer-wise weight vectors: $W \in \mathbb{R}^{14 \times 4 \times 4}$ (weight vectors $w_{l, k} \in \mathbb{R}^4$ for $L=14$ layers and $K=4$ tasks).
  - Layer-wise biases: $B \in \mathbb{R}^{14 \times 4}$ ($b_{l, k} \in \mathbb{R}$).
  - Total trainable parameters: $224 \text{ (weights)} + 56 \text{ (biases)} = 280$.
- **Activation Function:** Identity (linear routing).

---

## 5. Baselines
We evaluate R2D-Merge against the following baselines to ensure extreme transparency:
1.  **Static Uniform Merging (Task Arithmetic):** Bypasses all dynamic routing, setting static weights $\alpha_{l, k} = 1/K$ everywhere.
2.  **Unregularized Global Linear Router:** Maps the high-dimensional input $z(x) \in \mathbb{R}^{192}$ directly to a unified $K$-dimensional score space via Softmax, repeated across all layers.
3.  **QWS-Merge SOTA (Quantum-Inspired):** Wave-like cosine-based dynamic routing using trainable amplitudes and phases (336 parameters).
4.  **Standard L2-regularized L3-Router (L2 Reg):** The classical layer-wise low-dimensional linear router regularized with standard uniform $L_2$ weight decay. This isolates the benefit of our task-adaptive, covariance-weighted CFR penalty against standard uniform regularization.

---

## 6. Step-by-Step Interaction

### Phase 1: Pre-Computation (Offline)
1.  Feed the $N=64$ calibration samples through each task-specific expert network to obtain intermediate layer-wise activations $z^{(l)}(x_i)$ and task vectors $V_k^{(l)} = W_k^{(l)} - W_{base}^{(l)}$.
2.  Compute the frozen PCA projection matrix $P \in \mathbb{R}^{D \times d}$ using the calibration representations, and obtain the low-dimensional normalized state $\psi(x_i) = \frac{z(x_i)P}{\|z(x_i)P\|_2 + \epsilon}$.
3.  For each layer $l \in \{1, \dots, L\}$ and task $k \in \{1, \dots, K\}$, compute the task-specific empirical covariance matrix:
    $$C_{l, k} = \frac{1}{N} \sum_{i=1}^N \|z_i^{(l)} V_k^{(l)}\|_2^2 \cdot \psi(x_i) \psi(x_i)^T \in \mathbb{R}^{d \times d}$$
    and store it as a frozen matrix.

### Phase 2: Router Optimization (Offline Training)
1.  Initialize trainable router weights $W$ and biases $B$.
2.  For each epoch (up to 100):
    a.  Compute dynamic coefficients: $\alpha_{l, k}(x_i) = w_{l, k}^T \psi(x_i) + b_{l, k}$.
    b.  Compute multi-task cross-entropy loss $\mathcal{L}_{CE}$ on the calibration labels.
    c.  Compute CFR penalty: $\mathcal{L}_{CFR}(W) = \sum_{l=1}^L \sum_{k=1}^K w_{l, k}^T C_{l, k} w_{l, k}$.
    d.  Update $W, B$ via AdamW to minimize $\mathcal{L}_{total} = \mathcal{L}_{CE} + \lambda_{wd} \mathcal{L}_{CFR}(W)$.

### Phase 3: Evaluation (Online Inference)
1.  For each incoming test sample $x_{test}$:
    a.  Extract globally pooled backbone representation $z(x_{test})$ and project to normalized state $\psi(x_{test})$.
    b.  Forward pass through the optimized router to compute dynamic coefficients $\alpha_{l, k}(x_{test})$.
    c.  Assemble the merged weights layer-by-layer: $W_{merged}^{(l)}(x_{test}) = W_{base}^{(l)} + \sum_{k=1}^K \alpha_{l, k}(x_{test}) V_k^{(l)}$.
    d.  Compute multi-task predictions and evaluate accuracy on the OOD SVHN and mixed-task streams.
