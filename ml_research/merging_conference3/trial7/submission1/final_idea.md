# Idea Proposal: The Layer-Averaging Collapse Paradox

An empirical deconstruction and physical systems audit of layer-wise dynamic model merging to expose the boundaries of representational collinearity in deep hierarchical architectures.

## 1. Persona Alignment
This project directly embodies the traits and core philosophy of **The Methodologist**:
* **Skepticism of "Analytical Collapse" Claims:** Prior work (`trial6_submission7`) proposed a mathematical proof of "Layer-Averaging Collapse" to claim that layer-wise dynamic routing is entirely redundant and collapses to a single global dimension. However, this proof relies on a simplified 14-layer linear sandbox and several strong assumptions (collinear base representations, contractive linear Jacobians). We are highly skeptical that this collapse holds in deeply hierarchical, non-linear physical architectures.
* **Exposing Flaws in Current Practices:** We challenge the widespread practice of evaluating routing mechanics on toy synthetic sandboxes, showing that sandboxes hide depth-dependent semantic features.
* **Rigorous Evaluation Protocols:** We introduce a rigorous mathematical evaluation protocol (Singular Value Decomposition and Pairwise Cosine Similarity of the Layer-wise Coefficient Matrix) to empirically measure and map the true dimensionality of the layer-wise routing space in deep physical models (ViT-B/16 and ResNet-50) across diverse task suites of varying representational conflict.

## 2. Core Techniques
We introduce and modify the following algorithms, layers, and diagnostic mechanisms:
1. **Physical Layer-Wise Dynamic Router Optimization:** Unlike previous trials that optimize a 14-dimensional synthetic vector, we optimize the physical parameters of a layer-wise linear router directly coupled with a real deep neural network backbone (e.g., Vision Transformer ViT-B/16 with $L=12$ blocks, or ResNet-50 with $L=4$ residual stages).
2. **Singular Value Decomposition (SVD) of Layer Coefficients:** We construct the **Layer-wise Routing Coefficient Matrix** $A \in \mathbb{R}^{L \times K}$ and analyze its singular value spectrum to measure its effective rank and verify if it collapses to a rank-1 (collinear) subspace.
3. **Pairwise Inter-Layer Cosine Similarity Matrix:** We compute the cosine similarity of the routing coefficient vectors between all pairs of layers $(l, l')$ to map how routing decisions shift along the depth of the network.
4. **Task-Conflict Suite Partitioning:** We systematically evaluate the router across three distinct suites of varying domain distance and representational conflict (inspired by `trial4_submission7`):
   * **Low-Conflict Suite:** MNIST + FashionMNIST (homogeneous representation).
   * **High-Conflict Suite:** CIFAR-10 + SVHN (heterogeneous representation with high semantic clash).
   * **Cross-Domain Suite:** Imagenette + CIFAR-100 (high-dimensional, diverse classification).

## 3. Mathematical Formulation

Let $L$ be the number of layers/blocks in the physical model backbone, and $K$ be the number of specialized expert models to merge. Let $X$ be a batch of samples.

### 3.1. Layer-wise Coefficient Matrix
For each sample $b \in \{1, \dots, B\}$ and layer $l \in \{1, \dots, L\}$, let $\alpha_{k, b}^{(l)}$ be the unconstrained dynamic routing coefficient for expert $k$ computed via:
$$\alpha_{k, b}^{(l)} = \langle \psi(x)_b, W_{l, k} \rangle + B_{l, k}$$
where $\psi(x)_b$ is the low-dimensional projected state representation of sample $b$, and $W_{l, k}, B_{l, k}$ are the trainable router weights and biases for layer $l$ and expert $k$.

We define the **Batch-Averaged Layer-wise Coefficient Matrix** $A \in \mathbb{R}^{L \times K}$ as:
$$A_{l, k} = \frac{1}{B} \sum_{b=1}^B \alpha_{k, b}^{(l)}$$
Each row $A_l = [A_{l, 1}, \dots, A_{l, K}]$ represents the effective merging weight vector applied at layer $l$.

### 3.2. SVD and Dimensionality Metrics
We perform Singular Value Decomposition (SVD) on the matrix $A$:
$$A = U \Sigma V^T$$
where $\Sigma = \text{diag}(\sigma_1, \sigma_2, \dots, \sigma_{\min(L, K)})$ is the diagonal matrix of singular values sorted in descending order ($\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_{\min(L, K)}$).

To quantify the degree of **Layer-Averaging Collapse**, we define the **Collinearity Ratio** (or Explored Variance of the First Principal Component) as:
$$\rho_{collinear} = \frac{\sigma_1}{\sum_{i=1}^{\min(L, K)} \sigma_i}$$
* **If $\rho_{collinear} \approx 1.0$ (Rank-1 Collapse):** The routing trajectories across all layers are perfectly collinear, verifying the Layer-Averaging Collapse theorem in physical networks.
* **If $\rho_{collinear} \ll 1.0$ (Multi-dimensional Routing):** The routing trajectories occupy a higher-dimensional subspace, proving that layers specialize in task routing and that the collinearity proof fails in real networks.

### 3.3. Inter-Layer Cosine Similarity
We compute the pairwise cosine similarity matrix $S \in \mathbb{R}^{L \times L}$:
$$S_{l, l'} = \frac{A_l \cdot A_{l'}}{\|A_l\|_2 \|A_{l'}\|_2}$$
$S_{l, l'}$ quantifies the directional alignment between the merging weights of layer $l$ and layer $l'$. Collinearity implies $S_{l, l'} \approx \pm 1.0$ for all pairs, while layer-specificity implies $S_{l, l'} \approx 0$ for distant layers.

## 4. Architecture Specifications
* **Backbone Models:** 
  1. **Vision Transformer (ViT-B/16):** $L=12$ self-attention blocks, hidden dimension $D=768$, merged over $K=4$ vision experts.
  2. **ResNet-50:** $L=4$ main residual stages (layer1 to layer4), merged over $K=4$ vision experts.
* **Routing Network:** Single-layer linear projection mapping from the $d=4$ projection space to $K=4$ coefficients per layer.
  * **For Layer-wise Routing:** Total parameters are $L \times (d \cdot K + K) = L \times 20$ parameters (e.g., 240 parameters for ViT-B/16).
  * **For Global Routing:** Total parameters are $1 \times 20 = 20$ parameters.
* **Inputs & Outputs:**
  * **Inputs:** Low-dimensional normalized routing states $\psi(x)_b \in \mathbb{R}^d$ obtained via a data-independent Random Gaussian projection of the penultimate features.
  * **Outputs:** A tensor of dynamic, sample-specific merging coefficients $\alpha_b \in \mathbb{R}^{L \times K}$ mapping to physical weight-space interpolation.

## 5. Baselines
We will evaluate and compare our proposed audit against the following key baselines:
1. **Static Uniform Merging:** Standard arithmetic average of expert weights ($\alpha_k = \frac{1}{K}$ for all $l, k$) with zero routing parameters.
2. **Offline Few-Shot Validation Tuning (OFS-Tune):** A powerful, offline, regularized static baseline where a single global set of weights is validated. This serves as the benchmark for single-forward-pass model merging.
3. **Single-Layer Global Routing (L1-Linear Router):** A single set of routing coefficients applied uniformly to all layers ($L=1$), which is the recommended configuration under the collapse assumption.
4. **Oracle Multi-Model Routing:** Routing the inputs directly to the specialized individual expert models (no parameter merging, $O(K)$ forward passes), representing the performance upper-bound.

## 6. Step-by-Step Interaction
The flow of data and transformations through our diagnostic and model-merging system:

1. **Feature Extraction:** A batch of samples $X$ of size $B$ is passed through the pre-trained base model backbone to extract penultimate representations $z_b \in \mathbb{R}^D$.
2. **State Projection:** The high-dimensional features $z_b$ are projected into a low-dimensional space via a frozen Random Gaussian matrix $P \in \mathbb{R}^{D \times d}$ and normalized to obtain routing states $\psi(x)_b = \frac{z_b P}{\|z_b P\|_2 + \epsilon}$.
3. **Coefficient Generation:** The routing states $\psi(x)_b$ are passed to the layer-wise router to compute unconstrained, sample-specific coefficients $\alpha_{k, b}^{(l)}$ for each layer $l \in \{1, \dots, L\}$.
4. **Batch Aggregation:** The sample-wise coefficients are averaged over the batch to obtain the layer-wise coefficient matrix $A \in \mathbb{R}^{L \times K}$, where $A_{l, k} = \frac{1}{B} \sum_b \alpha_{k, b}^{(l)}$.
5. **Spectral SVD Diagnostics (Our Focus):** SVD is performed on $A$ to compute the singular values $\sigma_i$, the Collinearity Ratio $\rho_{collinear}$, and the Pairwise Cosine Similarity Matrix $S \in \mathbb{R}^{L \times L}$.
6. **Physical Model Merging:** The physical model parameters are merged layer-by-layer:
   $$W_{merged}^{(l)} = W_{base}^{(l)} + \sum_{k=1}^K A_{l, k} V_k^{(l)}$$
7. **Forward Pass and Evaluation:** The batch $X$ is run through the merged physical model in a single forward pass, and its classification accuracy is measured on the target multi-task dataset suite.
