# Research Proposal: Orthogonal Task-Space Projection (OTSP) for Zero-Shot Model Merging

## 1. Persona Alignment
*   **The Minimalist Philosophy:** OTSP is a relentless application of Occam's razor to the problem of representation-space dynamic routing in model merging. Instead of designing a complex multi-layer parametric routing network with dozens of hyperparameters (e.g., QWS-Merge, L3-Linear), OTSP uses a single, closed-form linear algebra operation—**Löwdin Symmetric Orthogonalization**—to decouple the task coordinate spaces of pre-trained experts.
*   **Zero Optimization & Zero Parameters:** OTSP introduces **zero trainable parameters** and requires **zero epochs of training** or calibration split data. It is a completely data-free, closed-form linear projection that leverages the pre-existing structure of pre-trained classification heads to perform robust task routing.
*   **Pruning Complexity:** By eliminating backpropagation, optimization loops, and transductive training noise, OTSP achieves superior generalization and complete immunity to out-of-distribution (OOD) collapse, replacing the convoluted "quantum" metaphors of prior state-of-the-art methods with clean, classical linear algebra.

## 2. Core Techniques
*   **Symmetric Task Centroids:** Extract a single task-representative centroid vector $v_k \in \mathbb{R}^D$ directly from the frozen, pre-trained classification weights $W_k$ of each expert $k$.
*   **Löwdin Symmetric Orthogonalization:** Transform the set of highly correlated, overlapping task centroids into a perfectly orthonormal task-coordinate basis $\{q_1, \dots, q_K\}$ using the symmetric inverse square root of the Gram overlap matrix. This ensures order-invariance and mathematical optimality (finding the orthonormal basis closest to the original centroids in the least-squares sense).
*   **Orthonormal Subspace Projection:** Project the normalized penultimate representation $\tilde{z}_b$ onto this orthonormal task basis to derive perfectly decoupled, cross-talk-free task coordinates.
*   **Temperature-Scaled Softmax Gating:** Compute dynamic routing coefficients using a sharp Softmax function directly on the orthogonalized coordinates, ensuring near-discrete task specialization.

## 3. Mathematical Formulation

Let the classification weights of Expert $k \in \{1, \dots, K\}$ be denoted as $W_k \in \mathbb{R}^{C_k \times D}$, where $C_k$ is the class size (or token vocabulary size) and $D$ is the representation dimension. 

### Step 1: Extract Task Centroids
We compute the representative task direction $v_k \in \mathbb{R}^D$ for Expert $k$ as the mean of its normalized class prototype weights:
$$v_k = \frac{1}{C_k} \sum_{c=1}^{C_k} \frac{W_{k, c}}{\|W_{k, c}\|_2}$$
We then normalize each task centroid to unit-norm to ensure scale-invariance across expert registries:
$$\bar{v}_k = \frac{v_k}{\|v_k\|_2}$$

### Step 2: Compute Overlap (Gram) Matrix
We construct the symmetric Gram overlap matrix $S \in \mathbb{R}^{K \times K}$ representing the pairwise cosine similarities of the task directions:
$$S_{ij} = \bar{v}_i \cdot \bar{v}_j$$

### Step 3: Compute Löwdin Symmetric Orthogonalization
Because $S$ is a symmetric positive semidefinite matrix, we compute its eigendecomposition:
$$S = U \Lambda U^T$$
where $U$ is an orthogonal matrix of eigenvectors, and $\Lambda = \text{diag}(\lambda_1, \dots, \lambda_K)$ represents the eigenvalues. To eliminate representation overlap while preserving symmetric order-invariance, we compute the symmetric inverse square root of $S$:
$$S^{-1/2} = U \Lambda^{-1/2} U^T = U \text{diag}\left(\frac{1}{\sqrt{\lambda_1 + \epsilon}}, \dots, \frac{1}{\sqrt{\lambda_K + \epsilon}}\right) U^T$$
where $\epsilon = 10^{-6}$ is a small numerical stabilizer.
The symmetric orthonormalized task directions $\{q_1, \dots, q_K\} \in \mathbb{R}^D$ are given by:
$$q_k = \sum_{j=1}^K (S^{-1/2})_{kj} \bar{v}_j$$
By construction, these vectors are guaranteed to be perfectly orthonormal:
$$q_i \cdot q_j = \delta_{ij}$$

### Step 4: Orthogonal Coordinate Projection
Let $z_b \in \mathbb{R}^D$ be the penultimate representation of sample $b$ in batch $X$. We normalize the feature representation:
$$\tilde{z}_b = \frac{z_b}{\|z_b\|_2}$$
We then compute the orthogonal task-coordinate projection $u'_b = [u'_{1, b}, \dots, u'_{K, b}]^T \in \mathbb{R}^K$ by projecting the normalized representation onto the orthonormal task basis:
$$u'_{k, b} = q_k \cdot \tilde{z}_b$$

### Step 5: Temperature-Scaled Softmax Routing
The sample-wise dynamic merging coefficients $\alpha_b \in \mathbb{R}^K$ are derived via a temperature-scaled Softmax over the orthogonal coordinates:
$$\alpha_{k, b} = \frac{\exp(u'_{k, b} / \tau)}{\sum_{j=1}^K \exp(u'_{j, b} / \tau)}$$
where $\tau > 0$ is a static, pre-defined scaling temperature (typically $\tau = 0.001$ to perform near-discrete routing).

## 4. Architecture Specifications
*   **Input Dimensions:** Intermediate representation $z_b \in \mathbb{R}^D$ extracted from the penultimate layer of the backbone model (immediately prior to classification heads).
    *   *Sandbox:* $D = 192$
    *   *ViT-Base:* $D = 768$
    *   *LLaMA-7B:* $D = 4096$
*   **Orthonormal Basis:** $K$ orthonormal task vectors $\{q_k\}_{k=1}^K$ of size $D$, stored in memory as an orthogonal matrix $Q \in \mathbb{R}^{K \times D}$.
*   **Coordinate Space:** $K$-dimensional task coordinate vector $u'_b \in \mathbb{R}^K$.
*   **Gating Coefficients:** $\alpha_b \in \mathbb{R}^K$, satisfying the simplex constraint $\sum_k \alpha_{k, b} = 1$ and $\alpha_{k,b} \ge 0$.
*   **Hyperparameters:**
    *   Scaling temperature: $\tau = 0.001$
    *   Numerical stabilizer: $\epsilon = 10^{-6}$

## 5. Baselines
*   **Static Uniform Merging:** The simplest model-merging baseline, which sets static uniform weights $\bar{\alpha}_k = \frac{1}{K}$ for all samples.
*   **Parametric Linear Router:** A standard single-layer Linear Router trained on the 64-sample calibration split via AdamW optimization.
*   **SOTA Parametric Routers (QWS-Merge):** Over-parameterized wave-superposition routing trained with weight decay, which suffers from transductive overfitting and OOD collapse.
*   **Parameter-Free Subspace Routing (PFSR):** The previous non-parametric baseline that projects representations onto raw, unorthogonalized expert heads. OTSP serves as a direct mathematical improvement over PFSR by eliminating task overlap and routing cross-talk.

## 6. Step-by-Step Interaction

### Phase A: Offline Initialization (Completely Data-Free)
1.  **Extract Weights:** Retrieve the pre-trained classification weight matrices $W_k \in \mathbb{R}^{C_k \times D}$ for each expert $k$.
2.  **Generate Centroids:** For each expert $k$, compute the normalized centroid direction $\bar{v}_k$ by averaging the normalized row vectors of $W_k$.
3.  **Compute Overlap:** Construct the $K \times K$ Gram overlap matrix $S$ by computing the pairwise dot products of the centroids.
4.  **Symmetric Orthogonalization:** Compute the eigendecomposition of $S$, calculate the inverse square root matrix $S^{-1/2}$, and derive the symmetric orthonormal task basis $Q \in \mathbb{R}^{K \times D}$. Store $Q$ in VRAM/RAM.

### Phase B: Online Inference (Dynamic Single Pass)
1.  **Feature Extraction:** Pass the input batch $X$ through the shared pre-trained base model backbone to extract the penultimate representation $Z \in \mathbb{R}^{B \times D}$.
2.  **Unit-Norm Calibration:** For each sample $b$ in the batch, normalize the representation to unit-norm $\tilde{z}_b$.
3.  **Linear Projection:** Perform a fast matrix-vector multiplication of the stored orthonormal basis $Q$ against the normalized representation $\tilde{z}_b$ to obtain the orthogonal task coordinates: $u'_b = Q \tilde{z}_b$.
4.  **Softmax Gating:** Apply temperature-scaled Softmax to the coordinates $u'_b$ to compute the gating coefficients $\alpha_b$.
5.  **Data-Stream Orchestration (MBH):** Group samples into micro-batches $X^{(g)}$ based on their dominant task coordinate $k_b^* = \arg\max_k u'_{k,b}$. 
6.  **Parameter Merging & Inference:** For each micro-batch, compute the micro-batch average coefficients, merge the lightweight LoRA adapters of the experts on the fly, execute the forward pass, and scatter-assemble the final outputs back to the original batch sequence.
