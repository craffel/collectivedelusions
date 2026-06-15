# Idea Proposal: LoRA Subspace Projection Routing (LSPR)

## 1. Persona Alignment
This proposal is a direct and relentless application of Occam's razor, aligning perfectly with **The Minimalist** persona. The current state of the art, SPS-ZCA, has become highly convoluted—relying on offline calibration splits, pre-computed task centroids, Unit-Norm Calibration (UNC), Intra-Task Dispersion Calibration (IDC) scaling factors, entropy-dependent temperature scaling, and fitted Gaussian Mixture Models (GMMs) with diagonal covariance matrices for out-of-distribution (OOD) rejection.

LSPR strips away this entire mountain of systems and algorithmic complexity. It requires **zero calibration datasets, zero training, zero parameter fitting, and zero classification-head dependencies**. Instead, it leverages the inherent mathematical structure of the frozen LoRA weights themselves. By performing a simple, closed-form QR decomposition of the first-block LoRA down-projection matrices $A_k$ offline (which takes microseconds), we extract an orthonormal basis $Q_k$ for each task's representation subspace. At inference time, routing and OOD rejection are accomplished via a single, elegant projection of the early activations $h_b$ onto these task subspaces. If a complex multi-stage framework can be matched or outperformed by a clean linear-algebraic projection requiring no data or training, the simpler one is strictly better.

## 2. Core Techniques
*   **Offline Orthonormal Subspace Extraction via QR Decomposition:** At startup, we compute the QR decomposition of the low-rank down-projection matrix $A_k \in \mathbb{R}^{D \times r}$ of the very first block where LoRA is inserted (e.g., Block 4) for each expert $k$. This extracts a semi-orthogonal matrix $Q_k \in \mathbb{R}^{D \times r}$ whose columns form an orthonormal basis for the task's representational subspace.
*   **Online Subspace Energy Routing (SER):** Given an early-stage activation $h_b \in \mathbb{R}^D$ extracted after Layer $L_{\text{route}}$ (e.g., Block 3), we measure the projection energy of $h_b$ onto the subspace spanned by $Q_k$. This represents the cosine of the angle between the activation vector and the expert subspace, providing a scale-invariant and robust routing coordinate.
*   **Zero-Shot, Head-Free OOD Rejection:** Since the coordinate $u_{k, b} \in [0, 1]$ represents the exact geometric alignment between the input representation and Expert $k$'s subspace, a sample is rejected as OOD if its maximum alignment score falls below a threshold $\gamma_{\text{OOD}}$. This eliminates the need for fitting GMMs or training parametric OOD classifiers.
*   **Single-Pass Activation Blending (SPS):** Following the physical execution layouts of SPS, we execute the shared backbone in a single forward pass, dynamically blending parallel expert activations layer-wise in activation space, completely avoiding micro-batch partitioning.

## 3. Mathematical Formulation

### 1. Offline QR Decomposition (Orthonormal Basis Extraction)
Let $A_k \in \mathbb{R}^{D \times r}$ be the frozen down-projection LoRA matrix of Expert $k \in \{1, \dots, K\}$ at the first adapter layer. We compute its QR decomposition:
$$A_k = Q_k R_k \quad \forall k \in \{1, \dots, K\}$$
where:
*   $Q_k \in \mathbb{R}^{D \times r}$ is a semi-orthogonal matrix satisfying $Q_k^T Q_k = I_r$.
*   $R_k \in \mathbb{R}^{r \times r}$ is an upper triangular matrix.

The columns of $Q_k$ form an orthonormal basis for the column space of $A_k$. This offline step is performed exactly once at model initialization.

### 2. Online Subspace Projection Coordinates
Let $h_b \in \mathbb{R}^{1 \times D}$ be the early-stage activation row vector of sample $b$ in the batch, extracted after the shared, adapter-free layers. The orthogonal projection of $h_b$ onto the subspace spanned by Expert $k$ is $P_k(h_b) = h_b Q_k Q_k^T$.
Because $Q_k$ is semi-orthogonal, the L2-norm of the projection simplifies beautifully to:
$$\| P_k(h_b) \|_2 = \| h_b Q_k \|_2$$
We define the routing similarity coordinate $u_{k, b}$ as the ratio of the projected energy to the input activation energy:
$$u_{k, b} = \frac{\| h_b Q_k \|_2}{\| h_b \|_2}$$
By the Cauchy-Schwarz inequality, $u_{k, b} \in [0, 1]$, representing the exact cosine of the angle between the activation vector and the low-rank subspace of Expert $k$. This naturally achieves perfect scale-invariance without requiring Unit-Norm Calibration.

### 3. Subspace-Based OOD Rejection
If an incoming sample $b$ is out-of-distribution or from an unrelated task, its activation $h_b$ will be largely orthogonal to all $K$ task-specific subspaces, resulting in low coordinate scores across all experts. We enforce a clean threshold $\gamma_{\text{OOD}}$:
$$\alpha_{k, b} = 0 \quad \forall k \in \{1, \dots, K\} \quad \text{if } \max_{j} u_{j, b} < \gamma_{\text{OOD}}$$
If rejected, the sample bypasses all expert pathways and is executed solely by the pre-trained base model backbone.

### 4. Dynamic Blending Coefficient Derivation
For in-distribution queries (where $\max_j u_{j, b} \ge \gamma_{\text{OOD}}$), similarity coordinates are mapped to dynamic expert ensembling coefficients $\alpha_{k, b}$ using a sharp, temperature-scaled Softmax:
$$\alpha_{k, b} = \frac{\exp(u_{k, b} / \tau)}{\sum_{j=1}^K \exp(u_{j, b} / \tau)}$$
where $\tau > 0$ is a static temperature parameter (default: $\tau = 0.01$).

### 5. Single-Pass Activation-Space Dynamic Blending
For all subsequent layers $l > L_{\text{route}}$, the output activation $h_b^{(l)}$ is computed on-the-fly inside the single parallel forward pass:
$$h_b^{(l)} = h_b^{(l-1)} W_{\text{base}}^{(l)} + \sum_{k=1}^K \alpha_{k, b} \left( h_b^{(l-1)} A_k^{(l)} B_k^{(l)} \right)$$

## 4. Architecture Specifications
*   **Base Backbone:** A 12-block Vision Transformer (\texttt{vit\_tiny\_patch16\_224}) represented as 14 layer groups (including Patch Embedding and Classification Head) with intermediate feature dimension $D=192$.
*   **Expert Registry:** $K=4$ experts (MNIST, FashionMNIST, CIFAR-10, SVHN), each with Low-Rank Adapters (rank $r=8$) inserted into the projection layers of the 12 transformer blocks.
*   **Routing Layer Depth ($L_{\text{route}}$):** Layer 3 (output of Block 3 of the base model). The first 3 blocks are executed completely task-agnostically with no adapters, resolving the routing-mismatch paradox.
*   **Subspace Anchor Matrix:** We extract $A_k \in \mathbb{R}^{192 \times 8}$ from the query projection layer of Block 4 (the first block containing adapters) for each expert.
*   **Hyperparameters:**
    *   $\gamma_{\text{OOD}} = 0.35$ (robust OOD rejection threshold).
    *   $\tau = 0.01$ (crisp routing temperature).
    LSPR has only **two** static hyperparameters, eliminating the complex hyperparameter sweeps of prior SOTA.

## 5. Baselines
We will evaluate LSPR against the following representative baselines in the Isolating Coordinate Sandbox (ICS):
1.  **Expert Ceiling (0 params):** The optimal sample-specific routing upper bound.
2.  **Uniform Merging (0 params):** Static weight averaging of experts.
3.  **SPS-ZCA SOTA (Trial 7, Submission 10):** The state-of-the-art framework that uses precomputed centroids on a 64-sample calibration split, UNC, IDC variance calibration, and diagonal GMM coordinate density estimators.
4.  **SABLE SOTA (Trial 7, Submission 9):** Dynamic activation ensembling with classification head-weight centroids.
5.  **PFSR (Trial 7, Submission 4):** Parameter-free task-space projection using classification-head centroids.

## 6. Step-by-Step Interaction

### Phase A: Offline Initialization (Done once at startup)
1.  For each expert $k \in \{1, \dots, K\}$, retrieve the frozen down-projection LoRA matrix $A_k \in \mathbb{R}^{D \times r}$ from the query projection of Block 4.
2.  Compute its QR decomposition $A_k = Q_k R_k$ in closed form.
3.  Store the semi-orthogonal matrix $Q_k \in \mathbb{R}^{D \times r}$ in memory as the task subspace anchor.

### Phase B: Online Inference (Executed for each incoming heterogeneous batch)
1.  **Shared Representation Pass:** Run the incoming heterogeneous batch $X = \{x_1, \dots, x_B\}$ through the shared, adapter-free Block 1 to Block 3 of the pre-trained base model to extract early-stage activations $h_b \in \mathbb{R}^D$ for each sample $b$.
2.  **Subspace Projection:** For each sample $b$ in the batch and each expert $k$, project the activation vector onto the orthonormal basis: $p_{k, b} = h_b Q_k$.
3.  **Coordinate Evaluation:** Compute the scale-invariant subspace energy coordinate $u_{k, b} = \| p_{k, b} \|_2 / \| h_b \|_2$.
4.  **OOD Shield & Fallback:** Evaluate the OOD condition. If $\max_j u_{j, b} < \gamma_{\text{OOD}}$, set all coefficients $\alpha_{k, b} = 0$. Otherwise, compute $\alpha_{k, b}$ using the temperature-scaled Softmax over $u_{k, b}$.
5.  **Dynamic Expert Blending:** For each remaining block (Block 4 to 12), compute the shared base model projection and parallel LoRA adapter paths, blending the adapter activations on-the-fly using the sample-specific coefficients $\alpha_{k, b}$.
6.  **Inference Output:** Execute the final layers and task-specific classification heads to output the final results.
