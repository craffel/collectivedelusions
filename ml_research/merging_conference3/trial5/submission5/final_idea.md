# Idea Proposal: Deconstructing QWS-Merge via Layer-wise Low-dimensional Classical Routing

## 1. Persona Alignment
This project directly aligns with the core philosophy of **The Methodologist**:
*   **Skepticism of Hype:** We challenge the recent "quantum-inspired" claims of QWS-Merge. We hypothesize that modeling parameters as "quantum superpositions" collapsing via "wave-like phase interference" (cosine modulation) is an unnecessary and over-engineered mathematical gimmick.
*   **Exposing Confounders:** QWS-Merge compared its model against a crippled classical baseline (the Linear Router) that was global (not layer-wise) and completely unregularized. We isolate the true driver of success by constructing a classical, regularized counterpart: a **Layer-wise Low-dimensional Linear Router (L3-Router)**.
*   **Rigorous and Fair Evaluation:** We establish a fair comparison where both the classical L3-Router and QWS-Merge operate in the exact same low-dimensional projected space, with the same layer-wise capacity, but we apply standard classical regularization (L2 weight decay) to show that classical routing completely avoids the SVHN collapse without needing any quantum-inspired machinery.

---

## 2. Core Techniques
We introduce and evaluate three variants of a classical, highly regularized alternative to QWS-Merge:
1.  **Layer-wise Low-dimensional Linear Router (L3-Router):** Projects the $d$-dimensional unit-norm input state linearly to task coefficients.
2.  **Bounded Tanh L3-Router (L3-Tanh):** Applies a $\tanh$ activation to bound classical routing weights in the $[-1, 1]$ interval.
3.  **Softmax L3-Router (L3-Softmax):** Applies Softmax over the task dimension per layer to enforce a classical probability distribution.
4.  **Classical L2 Regularization (Weight Decay):** Injects standard weight decay on the router weights to penalize unconstrained scaling, addressing the primary overfitting failure mode of classical routing under low-data (64-sample) calibration.

---

## 3. Mathematical Formulation

### 3.1 Input State Extraction and Low-Dimensional Projection
Let $z(x) \in \mathbb{R}^{B \times D}$ be the global pooled patch-representation of a batch $x$. We project $z(x)$ into a low-dimensional space via a frozen random projection matrix $P \in \mathbb{R}^{D \times d}$ and normalize onto the unit sphere:
$$\psi(x)_b = \frac{z(x)_b P}{\|z(x)_b P\|_2 + \epsilon} \in \mathbb{R}^d \quad \forall b \in \{1, \dots, B\}$$
where $d = K = 4$ is the number of expert tasks and $\epsilon = 10^{-8}$.

### 3.2 L3-Router Formulations
For each layer group $l \in \{1, \dots, L\}$ and task expert $k \in \{1, \dots, K\}$, we define trainable routing vectors $W_k^{(l)} \in \mathbb{R}^d$ and biases $B_k^{(l)} \in \mathbb{R}$.

We formulate three distinct classical routing channels:
1.  **Linear Routing (L3-Linear):**
    $$\alpha_{k, b}(l) = \langle \psi(x)_b, W_k^{(l)} \rangle + B_k^{(l)}$$
2.  **Tanh Routing (L3-Tanh):**
    $$\alpha_{k, b}(l) = \tanh\left( \langle \psi(x)_b, W_k^{(l)} \rangle + B_k^{(l)} \right)$$
3.  **Softmax Routing (L3-Softmax):**
    $$\boldsymbol{\alpha}_{:, b}(l) = \text{Softmax}\left( \mathbf{W}^{(l)} \psi(x)_b + \mathbf{B}^{(l)} \right)$$

### 3.3 Batch Average & Weight Assembly
To process batch $x$ efficiently on standard hardware accelerators, we perform a mean measurement of the routing coefficients over the batch dimension:
$$\bar{\alpha}_k(l) = \frac{1}{B} \sum_{b=1}^B \alpha_{k, b}(l)$$
The dynamic weight matrix at layer $l$ is then assembled as:
$$W_{merged}^{(l)}(x) = W_{base}^{(l)} + \sum_{k=1}^K \bar{\alpha}_k(l) V_k^{(l)}$$

### 3.4 Regularized Optimization Objective
The router parameters are trained on the tiny $64$-sample calibration split by minimizing the multi-task cross-entropy loss augmented with L2 weight decay:
$$\mathcal{L}_{total} = \mathcal{L}_{CE} + \lambda_{wd} \sum_{l=1}^L \sum_{k=1}^K \left( \|W_k^{(l)}\|_2^2 + (B_k^{(l)})^2 \right)$$
where $\lambda_{wd}$ is a tunable regularization hyperparameter (e.g., $10^{-4}$ or $10^{-3}$).

---

## 4. Architecture Specifications
*   **Backbone:** `vit_tiny_patch16_224` containing $L = 14$ layer groups (Patch Embedding, $12$ Transformer Blocks, LayerNorm).
*   **Routing Dimension:** $d = 4$ (matching the task dimensionality $K = 4$).
*   **Task Experts:** $K = 4$ (MNIST, FashionMNIST, CIFAR-10, SVHN).
*   **Trainable Router Parameters:**
    *   Routing weights $W_k^{(l)}$: $14 \times 4 \times 4 = 224$ parameters.
    *   Biases $B_k^{(l)}$: $14 \times 4 = 56$ parameters.
    *   **Total Parameters:** exactly $280$ parameters.
    *   *Comparison:* This is even smaller than QWS-Merge ($336$ parameters), demonstrating superior parameter efficiency while eliminating the non-linear cosine wave activation.

---

## 5. Baselines
Our evaluation is designed to be highly rigorous, comparing our L3-Router directly against all historical baselines under identical experimental settings:
1.  **Individual Experts (Ceiling):** Specialized models establishing the maximum performance.
2.  **Uniform Merging (Task Arithmetic):** Static uniform weights ($\lambda = 0.3$).
3.  **AdaMerging:** Unsupervised online Test-Time Adaptation.
4.  **OFS-Tune:** Supervised offline static layer-wise coefficients (56 parameters).
5.  **Linear Router (Crippled Global Classical):** Unregularized global projection mapping $z(x)$ directly to routing ($772$ parameters).
6.  **QWS-Merge (Quantum-Inspired SOTA):** Layer-wise quantum wave-interference routing ($336$ parameters).
7.  **L3-Router (Ours - Classical Low-dimensional):** Highly regularized layer-wise classical low-dimensional routing ($280$ parameters).

---

## 6. Step-by-Step Interaction

### Step 1: Input Feature Processing
A batch of images $x \in \mathbb{R}^{B \times C \times H \times W}$ is passed through the frozen patch embedding layer of `vit_tiny_patch16_224` to extract patch tokens:
$$H_0 = \text{PatchEmbed}(x) \in \mathbb{R}^{B \times N \times D}$$

### Step 2: Global Representation Pooling
A global representation vector $z(x)_b \in \mathbb{R}^D$ is extracted for each sample $b$ in the batch by performing spatial average pooling over the token sequence:
$$z(x)_b = \frac{1}{N} \sum_{n=1}^N H_{0, b, n, :}$$

### Step 3: Low-Dimensional Unit Projection
To prevent overfitting in the low-data regime, $z(x)_b$ is projected into a low-dimensional $d$-dimensional space via the frozen projection matrix $P$ and normalized onto the unit sphere to construct the input state vector $\psi(x)_b \in \mathbb{R}^d$.

### Step 4: Classical Layer-Wise Routing
For each layer group $l$, the low-dimensional input state $\psi(x)_b$ is mapped to task routing coefficients $\alpha_{k, b}(l)$ via the trainable parameters $W_k^{(l)}$ and $B_k^{(l)}$ using the Linear, Tanh, or Softmax routing formulations.

### Step 5: Batch-Level Weight Assembly
The sample-level coefficients are averaged across the batch to yield the collapsed batch-level coefficient $\bar{\alpha}_k(l)$. The physical weights $W_{merged}^{(l)}(x)$ are dynamically reconstructed in parameter-space using the task vectors.

### Step 6: Backbone Forward Pass
The batch representation is passed through the $l$-th layer block of the backbone using the dynamically assembled weights $W_{merged}^{(l)}(x)$, proceeding to the next block in sequence until final outputs are generated.
