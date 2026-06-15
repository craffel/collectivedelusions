# Idea Proposal: Variance-Regularized Classical Routing (VR-Router)

## 1. Persona Alignment
This proposal is designed to perfectly embody **The Empiricist** persona. Rather than relying on speculative mathematical metaphors (such as quantum wave superposition), our approach is grounded in classical statistical regularization and extreme empirical validation. 
To prove the superiority and robustness of VR-Router, we establish a massive, multi-seed comparative framework:
1. **Statistical Significance Sweep:** We evaluate all methods across 10 independent random seeds (generating unique feature subspaces, prototypes, and calibration splits for each).
2. **Regularization Sensitivity Audit:** We perform an exhaustive 10-value parameter sweep on the variance penalty weight $\lambda_{var} \in [0.0, 10.0]$ to identify the optimal regularization frontier.
3. **Stream Heterogeneity Stress Test:** We sweep 5 different batch sizes ($B \in \{1, 8, 32, 128, 512\}$) and 5 task mixture entropy levels to thoroughly evaluate performance under varying deployment conditions.
4. **Exhaustive Ablation:** We perform a complete ablation of all loss components ($\mathcal{L}_{CE}$, $\mathcal{L}_{reg}$, and $\mathcal{L}_{VR}$) to empirically isolate the exact drivers of generalization.

---

## 2. Core Techniques
The proposed **Variance-Regularized Classical Routing (VR-Router)** introduces four core techniques to the dynamic model-merging pipeline:
1. **Low-Dimensional Unit-State Projection ($\psi(x)_b$):** Compressing high-dimensional feature representations onto a $d$-dimensional unit sphere via frozen unsupervised PCA or random projections (Johnson-Lindenstrauss lemma) to avoid low-data overfitting.
2. **Layer-wise Classical Projections:** Linear mappings from the unit state to dynamic layer-specific routing coefficients, avoiding the non-monotonic optimization landscapes of quantum equations.
3. **Simulated Heterogeneous Calibration:** Actively training the router on simulated mixed-task calibration batches of varying size and entropy to prepare the model for real-world deployment.
4. **Task-Variance Regularization ($\mathcal{L}_{VR}$):** A novel penalty function that minimizes the sample-wise variance of predicted coefficients within each task group in a batch. This forces the router to produce highly consistent, low-entropy task coordinates, mitigating the skew of outlier samples during batch-averaging.

---

## 3. Mathematical Formulation

### 3.1. Dynamic Model Merging
Let $W_{base}^{(l)}$ be the base parameters at layer $l \in \{1, \dots, L\}$, and let $V_k^{(l)} = W_k^{(l)} - W_{base}^{(l)}$ be the task vector for expert $k \in \{1, \dots, K\}$. For an input batch $x$ of size $B$, the merged weights are:
$$W_{merged}^{(l)}(x) = W_{base}^{(l)} + \sum_{k=1}^K \bar{\alpha}_k(l) V_k^{(l)}$$
where the batch-averaged merging coefficient is:
$$\bar{\alpha}_k(l) = \frac{1}{B} \sum_{b=1}^B \alpha_{k, b}(l)$$

### 3.2. Low-Dimensional Unit State Projection
Let $z(x)_b \in \mathbb{R}^D$ be the globally pooled visual representation for sample $b$. Using a static projection matrix $P \in \mathbb{R}^{D \times d}$ (computed via PCA on an unlabeled split or initialized via JL random projection), we compute the unit state $\psi(x)_b$:
$$\psi(x)_b = \frac{z(x)_b P}{\|z(x)_b P\|_2 + \epsilon} \in \mathbb{R}^d$$
where $\epsilon = 10^{-8}$.

### 3.3. Layer-wise Classical Routing
For each layer $l$, the dynamic sample-wise coefficient for task $k$ is computed via linear projection:
$$\alpha_{k, b}(l) = \langle \psi(x)_b, W_{l, k} \rangle + B_{l, k}$$
where $W_{l, k} \in \mathbb{R}^d$ and $B_{l, k} \in \mathbb{R}$ are the trainable routing weights and biases.

### 3.4. Multi-Objective Optimization Loss
We train the router parameters ($W$ and $B$) using a joint objective function on simulated calibration batches:
$$\mathcal{L}_{total} = \mathcal{L}_{CE} + \mathcal{L}_{reg} + \mathcal{L}_{VR}$$

1. **Cross-Entropy Loss ($\mathcal{L}_{CE}$):**
   $$\mathcal{L}_{CE} = -\frac{1}{B} \sum_{b=1}^B \log P(y_b \mid x_b; W_{merged}(x))$$
2. **$L_2$ Parameter Weight Decay ($\mathcal{L}_{reg}$):**
   $$\mathcal{L}_{reg} = \lambda_{wd} \sum_{l=1}^L \sum_{k=1}^K \left( \|W_{l, k}\|_2^2 + B_{l, k}^2 \right)$$
3. **Task-Variance Regularization Penalty ($\mathcal{L}_{VR}$):**
   Let $S_k \subseteq \{1, \dots, B\}$ be the set of sample indices in the batch belonging to task $k$. We compute the intra-task sample variance of predicted routing coefficients at layer $l$:
   $$\sigma^2_{task}(k, l) = \frac{1}{|S_k|} \sum_{b \in S_k} \left( \alpha_{k, b}(l) - \mu_k(l) \right)^2$$
   where the mean task coefficient is $\mu_k(l) = \frac{1}{|S_k|} \sum_{b \in S_k} \alpha_{k, b}(l)$. The variance regularization penalty is defined as:
   $$\mathcal{L}_{VR} = \lambda_{var} \frac{1}{L \cdot K} \sum_{l=1}^L \sum_{k=1}^K \sigma^2_{task}(k, l)$$

---

## 4. Architecture Specifications
- **Input Dimension ($D$):** $D = 192$ (for the controlled representation sandbox) or $D = 768$ (for CLIP-ViT-B/16 scale validation).
- **Projection Dimension ($d$):** $d = K = 4$ tasks.
- **Layers ($L$):** $L = 14$ layer groups.
- **Activation Functions:** None in the routing path (L3-Linear) or standard Tanh/Softmax depending on the chosen variant.
- **Router Parameter Shape:**
  - Weights ($W$): shape $[L, K, d]$ (total $14 \times 4 \times 4 = 224$ parameters).
  - Biases ($B$): shape $[L, K]$ (total $14 \times 4 = 56$ parameters).
  - Total Trainable Parameters: **280 parameters** (a structural saving of **16.7%** over QWS-Merge's 336 parameters).

---

## 5. Baselines
We evaluate VR-Router against an exhaustive and scientific set of baselines:
1. **Expert Ceiling:** The upper bound of task-specific non-merged experts.
2. **Uniform Merging:** Zero-optimization, parameter-averaged baseline (static).
3. **Global Classical Linear Router:** Unregularized & $L_2$ regularized global routing mapping $z(x)_b \to \alpha$ directly.
4. **Quantum Waveface Superposition Merging (QWS-Merge):** The state-of-the-art quantum-inspired model merging method.
5. **Layer-wise Classical Routers (L3-Linear, L3-Tanh, L3-Softmax):** Unregularized and standard $L_2$-regularized layer-wise routers (without VR penalty).

---

## 6. Step-by-Step Interaction
1. **Extraction:** High-dimensional feature vector $z(x)_b$ is extracted from the backbone's first block for all samples in the batch.
2. **Projection:** Each $z(x)_b$ is projected via the static projection matrix $P$ and $L_2$-normalized onto the unit sphere to form the compressed state representation $\psi(x)_b$.
3. **Routing:** The unit state $\psi(x)_b$ is multiplied by the trainable layer-specific routing weights $W_l$ and added to biases $B_l$ to output sample-wise coefficients $\alpha_{k, b}(l)$.
4. **Batch Averaging:** Sample-wise coefficients are averaged across the batch dimension to obtain the single unified layer-wise coefficients $\bar{\alpha}_k(l)$.
5. **Merging:** The task vector parameters $V_k^{(l)}$ are scaled by $\bar{\alpha}_k(l)$ and added to $W_{base}^{(l)}$ to dynamically assemble the active merged weights $W_{merged}^{(l)}$.
6. **Forward Pass & Objective Evaluation:** The input batch is processed by the newly merged parameters to output logits and compute cross-entropy loss $\mathcal{L}_{CE}$. Simultaneously, sample-wise coefficients are grouped by task, and their intra-task variances are computed to formulate the task-variance penalty $\mathcal{L}_{VR}$. All loss components are backpropagated to update $W_l$ and $B_l$ during calibration.
