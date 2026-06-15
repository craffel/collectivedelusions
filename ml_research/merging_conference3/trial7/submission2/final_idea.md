# Fisher-Information Optimal Subspace Routing (FIOSR)

## 1. Persona Alignment
As **The Theorist**, we care deeply about the mathematical and statistical foundations of our algorithms. Empirical heuristics like standard cosine similarity or unconstrained parameter ensembling assume that neural network weight spaces are flat, isotropic Euclidean spaces. This assumption is geometrically and information-theoretically misspecified. 

**Fisher-Information Optimal Subspace Routing (FIOSR)** resolves this by treating the parameter space as a Riemannian manifold. By incorporating the **diagonal Fisher Information Matrix (dFIM)** of the specialized expert models, we define a local Riemannian metric that scales the projection coordinates based on parameter sensitivities. This ensures that:
1. **Geometric Rigor:** We measure task alignment in a mathematically principled, information-geometric space.
2. **Robustness to Overfitting:** The routing is entirely parameter-free and training-free, requiring zero optimization at test-time. This mathematically bypasses both the *Dynamic Routing Paradox* and *Vectorization Collapse* without adding a single trainable parameter.
3. **Information-Theoretic Noise Suppression:** Features that are highly noisy or irrelevant for a task (low Fisher values) are automatically suppressed, while highly critical features (high Fisher values) are amplified during the similarity projection.

---

## 2. Core Techniques
The proposed framework introduces and integrates the following mathematical techniques:
*   **Diagonal Fisher Information Matrix (dFIM):** Pre-computed or analytically estimated diagonal Fisher coordinates of the expert classification heads, representing local parameter curvature.
*   **Fisher-Weighted Cosine Similarity:** A novel, local Riemannian inner product metric that scales representations and prototypes by their Fisher Information coordinates before projection.
*   **Class-Size Scaling Calibration (CSC):** Analytical normalization of projection coordinates to account for statistical maximum bias across highly asymmetrical output class dimensions.
*   **Micro-Batch Homogenization (MBH):** Dynamic, unsupervised batch partitioning at the stream level based on dominant Fisher-weighted task coordinates, completely shielding the model from *heterogeneity collapse*.

---

## 3. Mathematical Formulation

### 3.1 Diagonal Fisher Information Matrix (dFIM)
Let $W_k \in \mathbb{R}^{C_k \times d}$ be the classification weight matrix of Expert $k \in \{1, \dots, K\}$, where $C_k$ is the class vocabulary size and $d = D // K$ is the task block size of the penultimate representation. Let $W_{k, c} \in \mathbb{R}^d$ represent the class prototype vector for class $c \in \{1, \dots, C_k\}$.

The diagonal empirical Fisher Information vector $F_{k, c} \in \mathbb{R}^d$ for class prototype $c$ of expert $k$ measures the variance of the log-likelihood gradient with respect to the parameters:
\begin{equation}
F_{k, c, j} = \mathbb{E}_{(x, y) \sim D_{\text{cal}}} \left[ \left( \frac{\partial \log p(y | x; W_k)}{\partial W_{k, c, j}} \right)^2 \right] \quad \forall j \in \{1, \dots, d\}
\end{equation}
where $D_{\text{cal}}$ is the small calibration split of size $N=64$. To ensure numerical stability, we add a tiny regularizing constant $\epsilon = 10^{-5}$ and normalize the Fisher vector for each class prototype to sum to 1, representing a valid local probability distribution of parameter sensitivity:
\begin{equation}
\tilde{F}_{k, c, j} = \frac{F_{k, c, j} + \epsilon}{\sum_{m=1}^d (F_{k, c, m} + \epsilon)}
\end{equation}

### 3.2 Fisher-Weighted Cosine Similarity Projection
Let $z_b \in \mathbb{R}^D$ be the globally pooled penultimate feature representation for sample $b$. Let $z_{k, b} \in \mathbb{R}^d$ be the $k$-th block of feature $z_b$ corresponding to the expert block.
Instead of computing standard cosine similarity, we compute the **Fisher-Weighted Cosine Similarity** between the feature block $z_{k, b}$ and the class prototypes $W_{k, c}$ scaled by their sensitivity $\tilde{F}_{k, c}$:
\begin{equation}
\text{Sim}_{\tilde{F}}(z_{k, b}, W_{k, c}; \tilde{F}_{k, c}) = \frac{\sum_{j=1}^d \tilde{F}_{k, c, j} \cdot W_{k, c, j} \cdot z_{k, b, j}}{\sqrt{\sum_{j=1}^d \tilde{F}_{k, c, j} \cdot W_{k, c, j}^2} \sqrt{\sum_{j=1}^d \tilde{F}_{k, c, j} \cdot z_{k, b, j}^2}}
\end{equation}
This formulation corresponds to a local Riemannian inner product $\langle z_{k,b}, W_{k,c} \rangle_{\tilde{F}_{k,c}}$ under the diagonal metric tensor $\mathbf{g} = \text{diag}(\tilde{F}_{k,c})$.

The raw task coordinate $u_{k, b}$ for expert $k$ is the maximum Fisher-weighted similarity across all class prototypes:
\begin{equation}
u_{k, b} = \max_{c \in \{1, \dots, C_k\}} \text{Sim}_{\tilde{F}}(z_{k, b}, W_{k, c}; \tilde{F}_{k, c})
\end{equation}

### 3.3 Class-Size Scaling Calibration (CSC)
To prevent statistical maximum bias under asymmetrical vocabulary sizes $C_k$, the coordinates are normalized using their expected analytical maximum under a random Gaussian assumption:
\begin{equation}
u'_{k, b} = \frac{u_{k, b}}{\sqrt{2 \log C_k / d}}
\end{equation}

### 3.4 Fisher-Information Subspace Routing (FISR) Coefficients
We derive the routing coefficients $\alpha_{k, b}$ via a temperature-scaled Softmax over the calibrated coordinates:
\begin{equation}
\alpha_{k, b} = \frac{\exp(u'_{k, b} / \tau)}{\sum_{j=1}^K \exp(u'_{j, b} / \tau)}
\end{equation}
where $\tau > 0$ is a static scaling temperature (e.g., $\tau = 0.001$ for near-discrete routing).

---

## 4. Architecture Specifications

### 4.1 Backbone and Experts
*   **Backbone:** $L=14$ layers, intermediate representation dimension $D=192$.
*   **Experts:** $K=4$ specialized experts representing MNIST, FashionMNIST, CIFAR-10, and SVHN.
*   **Adapter Parameters:** Parameter-efficient fine-tuning via LoRA with rank $r=10$ at each layer, or full-parameter task vectors:
    \begin{equation}
    V_k^{(l)} = W_k^{(l)} - W_{base}^{(l)}
    \end{equation}

### 4.2 Parameter Footprint & Overhead
*   **Trainable Parameters:** **0** (completely non-parametric and training-free).
*   **Storage Overhead:** Requires storing only the diagonal Fisher vectors $\tilde{F}_k \in \mathbb{R}^{C_k \times d}$ alongside the classification head weights $W_k$. For $K=4$ tasks and maximum class size $C_k = 10$, this requires storing a tiny matrix of size $10 \times 48$ per expert, adding virtually zero storage overhead ($1.92$ KB in half-precision).
*   **Computational Complexity:** The similarity projection scales as $O(K \cdot C \cdot d)$ per sample. With $C=10$ and $d=48$, this is mathematically negligible, running in less than 1 millisecond per batch.

---

## 5. Baselines
We will evaluate FIOSR against the following established baselines to isolate its causal benefit:
1.  **Static Uniform Merging:** The standard non-adaptive benchmark where weights are arithmetically averaged with a fixed weight of $1/K = 0.25$.
2.  **Linear Router (Unregularized):** An unconstrained, parametric linear projection router trained on the 64-sample calibration split.
3.  **QWS-Merge SOTA:** The highly complex, unregularized quantum wavefunction superposition router with cosine activations.
4.  **L3-Softmax (Well-Regularized):** The classical layer-wise Softmax router equipped with zero-initialization and $L_2$ weight decay, representing the SOTA for regularized parametric routers.
5.  **PFSR + MBH (Standard Parameter-Free):** The parameter-free baseline using raw unweighted cosine similarity ($\tilde{F}_{k,c,j} = 1/d$). This comparison is of high diagnostic importance as it directly measures the performance gain of our Fisher Information metric over flat Euclidean projections.

---

## 6. Step-by-Step Interaction

The flow of data and mathematical transformations through the FIOSR framework is detailed below:

```
                  [ Input Batch X (Size B) ]
                              │
                              ▼
            [ Base Model Backbone Forward Pass ]
                              │
                              ▼
      [ Penultimate Representation z_b (D-dimensional) ]
                              │
             ┌────────────────┴────────────────┐
             ▼                                 ▼
      [ Task Partitioning ]            [ Adapter Scaling ]
    For each expert block k:           (During Micro-Batch forward)
    Extract block feature z_k,b
             │
             ▼
    Compute Fisher-Weighted Sim:
    Sim_F(z_k,b, W_k,c)
             │
             ▼
    Find Max over Classes: u_k,b
             │
             ▼
    Apply CSC Normalization: u'_k,b
             │
             ▼
    Apply Softmax Routing: α_k,b
             │
             ▼
    Identify Dominant Task:
    k*_b = argmax_k u'_k,b
             │
             ▼
    [ Micro-Batch Homogenization (MBH) ]
    Partition batch into G groups:
    X^(g) = {x_b | k*_b = g}
             │
             ▼
    [ Weight Merging & Forward Pass ]
    For each active group g:
    1. Average coefficients: \bar{α}^(g) = Mean_{b \in g}(α_b)
    2. Assemble weights: W_merged^(g) = W_base + \sum_k \bar{α}_k^(g) V_k
    3. Forward pass: Y^(g) = Model(X^(g); W_merged^(g))
             │
             ▼
    [ Output Re-assembly via Scatter ]
    Reassemble outputs: Y = Scatter( { Y^(g) } )
                              │
                              ▼
                        [ Final Logits Y ]
```

1.  **Penultimate Feature Extraction:** An input batch $X$ of size $B=256$ is passed through the pre-trained base backbone, yielding globally pooled intermediate representations $Z = \{z_1, \dots, z_B\} \in \mathbb{R}^{B \times D}$ from the penultimate layer.
2.  **Fisher-Weighted Projection:** Each sample's feature $z_b$ is split into blocks $z_{k,b}$ of size $d = D//K$. We compute its Fisher-weighted similarity against all class prototypes of expert $k$ using pre-computed diagonal Fisher weights $\tilde{F}_{k,c}$.
3.  **Statistical Calibration:** The raw maximum coordinates are normalized by their analytical random expected maximums to eliminate class-size scaling bias.
4.  **Coefficient Derivation:** Calibrated coordinates are passed through a temperature-scaled Softmax to compute sample-specific merging coefficients $\alpha_{k,b}$.
5.  **Micro-Batch Partitioning:** The dominant task for each sample is identified as $k_b^* = \arg\max_k u'_{k,b}$. The heterogeneous batch $X$ is dynamically partitioned into homogeneous micro-batches $X^{(1)}, \dots, X^{(G)}$.
6.  **Dynamic Parameter Assembly:** For each active micro-batch $g$, the sample-wise coefficients are averaged to form $\bar{\alpha}^{(g)}$. The merged weights $W_{merged}^{(g)}$ are dynamically assembled by adding the scaled task vectors back to the base model parameters.
7.  **Symmetric Forward and Scatter:** A forward pass is executed on each micro-batch $X^{(g)}$ using its specific merged parameters $W_{merged}^{(g)}$ to obtain predictions $Y^{(g)}$. The predictions are re-assembled using index scatter operations to preserve the original batch sequence.
