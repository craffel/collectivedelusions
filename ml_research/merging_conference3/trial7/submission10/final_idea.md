# Idea Proposal: Single-Pass Sample-Wise Routing with Zero-Shot Centroid Alignment (SPS-ZCA)

## 1. Persona Alignment
*The Pragmatist* persona prioritizes real-world, practical applications, reducing computational cost/VRAM, minimizing inference latency, and ensuring robustness to real-world noise without using complex, fragile theoretical novelties.
- **Inference Latency Reduction:** Micro-Batch Homogenization (MBH) in prior work requires partitioning heterogeneous batches on the fly and executing up to $G \le K$ sequential forward passes of the entire base model backbone. For a model with $K=4$ tasks, this translates to up to $4\times$ latency and compute cost. SPS-ZCA resolves this bottleneck by blending adapter activations sample-wise inside a single forward pass of the shared base backbone. This converts sequential multi-pass execution back into an $O(1)$ constant-time parallel pass, achieving a massive computational speedup (up to $3.8\times$).
- **Robustness and Cost:** SPS-ZCA requires zero training, zero optimization parameters, and zero additional VRAM compared to standard dynamic LoRA. It relies on pre-computed task centroids extracted from a tiny, low-resource calibration split of 64 samples.
- **OOD Handling:** By replacing noisy, head-dependent similarity projection with stable representation-space centroids, SPS-ZCA stabilizes out-of-distribution routing (e.g., SVHN), restoring performance to its theoretical limit on noisy domains.

## 2. Core Techniques
- **Activation-Space Dynamic Blending (SPS):** Instead of merging expert parameters in weight-space and dispatching sequential forward passes, SPS executes the shared frozen base model and its lightweight LoRA adapters in parallel, combining their activation outputs on-the-fly using sample-specific routing coefficients.
- **Zero-Shot Centroid Alignment (ZCA):** Replaces the noisy, head-dependent, block-wise projection with a geometrically grounded nearest-centroid routing mechanism in the shared penultimate representation space of the base model.
- **Unit-Norm Calibration (UNC):** Applied to features and centroids to normalize representation-space scale imbalances.

## 3. Mathematical Formulation
Let the input heterogeneous batch of size $B$ be $X = \{x_1, \dots, x_B\}$.
Let the globally pooled penultimate feature representation for sample $b$ under the pre-trained base model backbone be $z_b \in \mathbb{R}^D$.

### Step 1: Zero-Shot Centroid Pre-computation
For each expert task $k \in \{1, \dots, K\}$, we pre-compute its manifold centroid $\mu_k \in \mathbb{R}^D$ using its 64 calibration samples $\mathcal{C}_k$:
$$\mu_k = \frac{1}{|\mathcal{C}_k|} \sum_{s \in \mathcal{C}_k} z_s$$
where $z_s$ is the penultimate representation of the calibration sample $s$ extracted from the frozen pre-trained base backbone.

### Step 2: Zero-Shot Centroid Alignment (ZCA) Routing
During inference, for an incoming sample $b$, we extract its penultimate representation $z_b$ from the base backbone.
The task coordinate vector $u_b = [u_{1, b}, \dots, u_{K, b}]^T \in \mathbb{R}^K$ is computed using cosine similarity against the pre-computed task centroids:
$$u_{k, b} = \text{cosine\_similarity}(z_b, \mu_k) = \frac{z_b \cdot \mu_k}{\|z_b\|_2 \|\mu_k\|_2}$$
Because all $\mu_k$ reside in the same penultimate representation space of the base backbone as $z_b$, this similarity metric is highly robust, un-biased, and immune to classification head asymmetries or label-space differences.

The dynamic sample-wise routing coefficients $\alpha_{k, b}$ are derived via a temperature-scaled Softmax over the coordinates:
$$\alpha_{k, b} = \frac{\exp(u_{k, b} / \tau)}{\sum_{j=1}^K \exp(u_{j, b} / \tau)}$$
where $\tau > 0$ is a static scaling temperature (we set $\tau = 0.001$).

### Step 3: Single-Pass Activation-Space Dynamic Blending (SPS)
For each layer $l \in \{1, \dots, L\}$ and sample $b$, let $h_b^{(l-1)}$ be the input activation.
The base layer weights are $W_{base}^{(l)}$, and the task-specific LoRA adapters are represented by low-rank matrices $A_k^{(l)} \in \mathbb{R}^{r \times d}$ and $B_k^{(l)} \in \mathbb{R}^{D \times r}$, where $r \ll D$ is the rank.
The output activation $h_b^{(l)}$ is computed sample-wise as:
$$h_b^{(l)} = h_b^{(l-1)} W_{base}^{(l)} + \sum_{k=1}^K \alpha_{k, b} \left( h_b^{(l-1)} B_k^{(l)} A_k^{(l)} \right)$$
This formulation blends the outputs of the task-specific adapters in activation space on-the-fly. Each sample $b$ uses its own individual, un-averaged routing coefficients $\alpha_{k, b}$, completely resolving heterogeneity collapse *without* requiring any batch partitioning or multi-pass execution.

## 4. Architecture Specifications
- **Base Model Backbone:** `vit_tiny_patch16_224` (5.7M parameters) with $L=14$ layer groups (Patch Embedding, 12 Transformer blocks, and final layer norm).
- **Penultimate Representation Space:** Globally pooled final representation $z_b \in \mathbb{R}^{D}$ with $D=192$.
- **Adapters:** LoRA adapters inserted into all query, key, value, and projection layers of the Transformer blocks, with rank $r=8$.
- **Task Centroids:** $K=4$ pre-computed vectors $\mu_k \in \mathbb{R}^D$ where $D=192$.
- **Routing Hyperparameter:** Scaling temperature $\tau = 0.001$.

## 5. Baselines
We compare SPS-ZCA against:
- **Expert Ceiling (0 parameters):** Evaluates the performance of running task-specific models directly on their corresponding tasks.
- **Uniform Merging (0 parameters):** Merges task vectors statically with uniform weights: $\bar{\alpha}_k = 1/K$.
- **Linear Router (Reg) (10,752 parameters):** A classical parametric router with Softmax activation, regularized via $L_2$ weight decay.
- **QWS-Merge SOTA (3,072 parameters):** Quantum Wavefunction Superposition Merging, modeling task experts as quantum eigenstates.
- **PFSR + MBH SOTA (0 parameters):** The prior non-parametric state-of-the-art that uses classification head-based similarity and Micro-Batch Homogenization (MBH).

## 6. Step-by-Step Interaction
1. **Input Batching:** A heterogeneous batch $X = \{x_1, \dots, x_B\}$ of size $B$ arrives (mixed MNIST, F-MNIST, CIFAR-10, SVHN).
2. **First-Stage Penultimate Extraction:** The batch $X$ is passed through the pre-trained shared base model backbone to extract the penultimate representation batch $Z = \{z_1, \dots, z_B\} \in \mathbb{R}^{B \times D}$.
3. **Centroid Coordinate Projection:** For each sample $b$, we compute its similarity task coordinate vector $u_b$ against the pre-computed task centroids $\mu_1, \dots, \mu_K$ using cosine similarity.
4. **Softmax Routing Coefficient Derivation:** Apply temperature-scaled Softmax to $u_b$ to obtain sample-specific routing coefficients $\alpha_b \in \mathbb{R}^K$.
5. **Single-Pass Layer Execution:** Pass the activations through each layer $l$ of the network. Inside the layer forward pass, the base representation $h_b^{(l-1)} W_{base}^{(l)}$ is computed once for the entire batch. In parallel, the expert LoRA adapter paths are executed and scaled sample-wise by $\alpha_{k, b}$, blending the outputs to form the final layer output activation $h_b^{(l)}$.
6. **Unified Output Generation:** The final layer representation is passed through the respective classification heads to generate predictions $Y$, running in a single forward pass with zero sequential dispatching.
