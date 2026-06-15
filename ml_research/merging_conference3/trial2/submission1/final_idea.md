# Calibrated & Regularized Test-Time Model Merging (RegCalMerge)

## 1. Persona Alignment
This proposal is designed by and for **The Empiricist**. It prioritizes comprehensive, systematic evaluation and dense multi-axial sweeps over narrow theoretical claims. Specifically, we deconstruct the test-time model merging landscape across two major axes:
1. **Regularization Space**: A dense grid sweep crossing the Proximity Penalty ($\beta \in [0, 2]$) and our novel **Spatial Deviation Penalty** ($\gamma \in [0, 2]$) across multiple independent optimization seeds.
2. **Entropy Scale Normalization**: Comparing 3 scaling regimes (Unnormalized, Class-Capacity Normalized, and Scale-Normalized) to evaluate their capacity to resolve "sacrificial task bias" on imbalanced multi-task suites.

We do not merely propose a singular heuristic; we run extensive parallel sweeps with both first-order gradient descent (Adam GD) and derivative-free optimization (1+1 ES) to map out the entire empirical loss and generalization landscape. This ensures overwhelming, multi-seed statistical proof for our claims.

---

## 2. Core Techniques
We introduce and evaluate a dual-component framework, **RegCalMerge**, that resolves the twin failure modes of test-time entropy-based model merging—namely, **transductive overfitting (parameter drift)** and **sacrificial task bias**:
1. **Elastic Spatial Regularization (ESR)**: Standard layer-wise optimization (AdaMerging) overfits calibration data by introducing high-frequency noise across layers. Instead of completely collapsing the layer dimension (which sacrifices fine-grained adaptability), we introduce a **Spatial Deviation Penalty** that penalizes the variance of layer-wise coefficients around their task-wise mean, alongside a **Proximity Penalty** that keeps coefficients close to their uniform task arithmetic initialization. This acts as a smooth, adjustable spatial regularizer.
2. **Class-Capacity Normalization (CCN)**: Different tasks have unequal numbers of categories, making their entropy limits unequal. We normalize prediction entropy by the maximum theoretical capacity of each task ($\log C_k$, where $C_k$ is the class count), mapping task-wise entropy onto a uniform $[0, 1]$ interval.
3. **Scale-Normalized Entropy Weighting (SNEW)**: Weights task entropy by the inverse of its baseline uniform task arithmetic entropy, preventing low-complexity tasks from dominating the joint gradients and sacrificing high-complexity domains.

---

## 3. Mathematical Formulation
Let there be $K$ tasks and $L$ discrete parameter layers. The layer-wise merging coefficients are parameterized as $\Lambda \in [0, 1]^{K \times L}$, initialized at uniform Task Arithmetic weights $\lambda_{\text{init}}$ (e.g., $0.3$).

### Spatial Regularization Penalty
For each task $k$, the task-wise mean coefficient across all layers is:
$$\bar{\lambda}_k = \frac{1}{L} \sum_{l=1}^L \lambda^l_k$$

The **Elastic Spatial Regularization** term $\mathcal{R}_{\text{spatial}}(\Lambda)$ is:
$$\mathcal{R}_{\text{spatial}}(\Lambda) = \beta \sum_{k=1}^K \sum_{l=1}^L (\lambda^l_k - \lambda_{\text{init}})^2 + \gamma \sum_{k=1}^K \sum_{l=1}^L (\lambda^l_k - \bar{\lambda}_k)^2$$
where $\beta \ge 0$ is the proximity penalty weight and $\gamma \ge 0$ is the spatial deviation penalty weight.

### Normalized Prediction Entropy
For a task $k$ with $C_k$ classes and a batch of $N$ calibration samples, the **Class-Capacity Normalized Entropy** is:
$$\bar{\mathcal{H}}_k(\Lambda) = \frac{-1}{N \log C_k} \sum_{i=1}^N \sum_{c=1}^{C_k} P(y=c \mid X_i; \theta_{\text{merged}}(\Lambda)) \log P(y=c \mid X_i; \theta_{\text{merged}}(\Lambda))$$

### Scale-Normalized Joint Loss
We define the total optimization objective as:
$$\min_{\Lambda} \mathcal{L}(\Lambda) = \sum_{k=1}^K w_k \bar{\mathcal{H}}_k(\Lambda) + \mathcal{R}_{\text{spatial}}(\Lambda)$$
where the scale weights $w_k$ are:
$$w_k = \frac{1}{\bar{\mathcal{H}}_k(\Lambda_{\text{init}})}$$
which are computed at step 0 of optimization and held constant.

---

## 4. Architecture Specifications
We evaluate our framework on the Vision-Language Transformer benchmark (using CLIP ViT-B/32 as the backbone) across 8 datasets (MNIST, FashionMNIST, CIFAR-10, SVHN, DTD, GTSRB, NWPU-RESISC45, SUN397).
- **Target Layers**: The visual projection layer of the image encoder (`model.visual.proj`), divided into $L = 13$ discrete layer groups.
- **Parameters**: Matrix $\Lambda$ of shape $K \times L$ (with $K = 8$ and $L = 13$, totaling 104 continuous parameters).
- **Inputs**: Unlabeled calibration images $X_i \in \mathbb{R}^{3 \times 224 \times 224}$ passed through the merged encoder, producing visual embeddings.
- **Outputs**: Logits computed via cosine similarity against text classification heads for each category.

---

## 5. Baselines
Our empirical study compares our method against a comprehensive set of rigorous baselines to isolate the true causal drivers of performance:
1. **Task Arithmetic (Uniform)**: Static baseline with uniform merging weights $\lambda^l_k = 0.3$.
2. **Unconstrained AdaMerging (Adam GD & 1+1 ES)**: The standard, unregularized layer-wise test-time adaptation.
3. **Spatially Averaged AdaMerging (Spatial Mean)**: The 92.3% parameter-reduced baseline from literature where coefficients are collapsed to a single mean per task.
4. **Proximity-Regularized AdaMerging**: The standard regularized baseline with $\gamma = 0$ (proximity-only).
5. **Ablation Configurations**: Mapping the $\beta \times \gamma$ grid to isolate the individual contributions of ESR, CCN, and SNEW.

---

## 6. Step-by-Step Interaction
1. **Load and Extract**: Load pre-trained CLIP base model and $K$ task experts. Extract the layer-wise task vectors: $\tau^l_k = \theta^l_k - \theta^l_{\text{pre}}$.
2. **Initialize Parameters**: Instantiate the continuous coefficient matrix $\Lambda \in [0, 1]^{K \times L}$ with values set to $0.3$.
3. **Compute Weights**: Compute a forward pass on the unlabeled calibration set with $\Lambda_{\text{init}}$ to find $\bar{\mathcal{H}}_k(\Lambda_{\text{init}})$ and initialize the scale weights $w_k = 1 / \bar{\mathcal{H}}_k(\Lambda_{\text{init}})$.
4. **Optimization Loop (Test-Time Adaptation)**:
   - Construct the merged model differentiably:
     $$\theta^l_{\text{merged}}(\Lambda) = \theta^l_{\text{pre}} + \sum_{k=1}^K \lambda^l_k \tau^l_k$$
   - Forward Pass: Compute predictions $P(y \mid X; \theta_{\text{merged}})$ and calculate the capacity-normalized prediction entropy $\bar{\mathcal{H}}_k(\Lambda)$.
   - Regularize: Calculate the task mean $\bar{\lambda}_k$ and compute the Elastic Spatial Regularization penalty $\mathcal{R}_{\text{spatial}}(\Lambda)$.
   - Loss & Backward Pass: Compute $\mathcal{L}(\Lambda) = \sum_k w_k \bar{\mathcal{H}}_k(\Lambda) + \mathcal{R}_{\text{spatial}}(\Lambda)$.
   - Update: Perform parameter updates on $\Lambda$ using Adam GD gradients or 1+1 ES mutations, and clamp the resulting coefficients to $[0, 1]$.
5. **Final Evaluation**: Freeze $\Lambda^*$ and evaluate the final merged model's multi-task accuracy on the unseen test splits.
