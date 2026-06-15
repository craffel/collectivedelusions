# Idea Proposal: Spectral Model Merging via Singular Value Slicing (SVS)

## 1. Persona Alignment
This proposal strongly aligns with **The Minimalist** persona through its relentless commitment to simplicity, mathematical elegance, and the pruning of unnecessary complexity. 

Existing state-of-the-art model merging frameworks like FoldMerge (Neural Origami) and SyMerge rely on highly complex, overparameterized pipelines. For example, FoldMerge introduces a 4-layer RealNVP normalizing flow network with $\approx 2.6\text{M}$ parameters and requires 500 optimization steps of Test-Time Adaptation (TTA) taking over 10 minutes on H100 GPUs. Furthermore, previous studies show that such methods suffer from overfitting-optimizer paradoxes and classifier head adaptation confounds, where complex weight warping acts primarily as a fragile regularizer rather than a robust functional mapping.

In contrast, our proposed method—**Spectral Model Merging via Singular Value Slicing (SVS)**—completely eliminates the need for test-time optimization, overparameterized flow networks, and training parameters. SVS achieves robust multi-task merging in a single, closed-form, training-free step using standard Singular Value Decomposition (SVD). By slicing task-specific delta matrices to keep only their highest-energy singular vectors, SVS filters out high-frequency fine-tuning noise and isolates the core task knowledge. We also introduce **Barycentric Weight Normalization (BWN)**, an elegant closed-form scaling technique that preserves activation energy and prevents representation collapse without any hyperparameters. SVS runs in less than 100 milliseconds on a standard CPU/GPU, requiring **zero** trainable parameters and **zero** optimization steps, embodying the ultimate triumph of Occam's razor.

## 2. Core Techniques
We introduce two core techniques:
1. **Singular Value Slicing (SVS):** Standard model merging via Task Arithmetic adds task-specific delta matrices directly. However, unconstrained sequential fine-tuning pulls parameter updates into disjoint and noisy regions, leading to severe representation interference. SVS performs Singular Value Decomposition (SVD) on each expert's task-specific delta matrix (task vector), slices the decomposition to retain only the top $k$ singular values/vectors (which correspond to the principal semantic directions of task learning), and discards the remaining low-rank noise before linear combinations.
2. **Barycentric Weight Normalization (BWN):** Slicing and combining parameters often distorts the energy scale (norm) of the merged weight matrix, causing activation shrinkage or expansion that leads to representation collapse. BWN is a training-free, non-parametric norm-matching operator. It calculates the weighted barycenter of the individual expert Frobenius norms and rescales the final merged matrix to match this target magnitude, ensuring perfect activation scale preservation.

These techniques build on foundational principles of low-rank matrix approximation (Eckart-Young-Mirsky Theorem) and low-rank adaptation (LoRA [Hu et al., 2021]), applying them in a novel, training-free manner to model merging.

## 3. Mathematical Formulation
Let $W_0 \in \mathbb{R}^{m \times n}$ be the pre-trained base weight matrix.
Let $W_t \in \mathbb{R}^{m \times n}$ be the fine-tuned expert weight matrix for task $t \in \{1, \dots, N\}$.

We define the task vector (weight delta) for task $t$ as:
$$T_t = W_t - W_0$$

We perform the Singular Value Decomposition (SVD) on each task vector $T_t$:
$$T_t = U_t \Sigma_t V_t^T$$
where:
- $U_t \in \mathbb{R}^{m \times m}$ is an orthogonal matrix of left-singular vectors.
- $\Sigma_t \in \mathbb{R}^{m \times n}$ is a rectangular diagonal matrix containing the singular values $\sigma_{t, 1} \ge \sigma_{t, 2} \ge \dots \ge \sigma_{t, \min(m,n)} \ge 0$.
- $V_t \in \mathbb{R}^{n \times n}$ is an orthogonal matrix of right-singular vectors.

To slice the task vector at rank $k \ll \min(m, n)$, we define the low-rank projection operator $\mathcal{S}_k$:
$$\mathcal{S}_k(T_t) = \tilde{T}_t = U_{t, :k} \Sigma_{t, :k} V_{t, :k}^T$$
where:
- $U_{t, :k} \in \mathbb{R}^{m \times k}$ represents the first $k$ columns of $U_t$.
- $\Sigma_{t, :k} \in \mathbb{R}^{k \times k}$ is the diagonal matrix of the top $k$ singular values.
- $V_{t, :k} \in \mathbb{R}^{n \times k}$ represents the first $k$ columns of $V_t$.

The raw merged weight matrix is computed via a linear combination of the sliced task vectors:
$$W_{merged} = W_0 + \sum_{t=1}^N \lambda_t \tilde{T}_t$$
where $\lambda_t > 0$ are the task coefficients. Under a flat averaging setup, we set $\lambda_t = \frac{1}{N}$ or a uniform scaling coefficient $\lambda_t = \lambda$.

To stabilize activation scaling and prevent representation collapse, we apply the Barycentric Weight Normalization (BWN) operator:
$$W_{final} = \alpha W_{merged}$$
where the closed-form scaling factor $\alpha$ is defined as:
$$\alpha = \frac{\sum_{t=1}^N \mu_t \|W_t\|_F}{\|W_{merged}\|_F}$$
Here, $\mu_t = \frac{\lambda_t}{\sum_{j=1}^N \lambda_j}$ represents the normalized task weights, and $\|\cdot\|_F$ is the Frobenius norm of the matrix. This guarantees that the magnitude of the merged weight matrix matches the weighted barycenter of the individual expert magnitudes.

## 4. Architecture Specifications
- **Target Layer:** Visual projection layer `model.visual.proj` of the CLIP image encoder in the ViT-B/32 backbone.
- **Layer Shape:** $768 \times 512$ ($393,216$ parameters).
- **SVS Rank Parameter ($k$):** Evaluated across $k \in \{16, 32, 64, 128, 256\}$. The default rank is set to $k=64$, which represents a $12.5\%$ low-rank spectral slice, discarding the remaining $87.5\%$ of high-frequency parameter noise.
- **Trainable Parameters:** **0** (completely non-parametric and training-free).
- **Optimization Cost:** **0** steps (computed analytically in less than 100 milliseconds).
- **Inference Latency:** **0** extra latency (the merged weights $W_{final}$ are pre-computed offline and loaded directly).

## 5. Baselines
We compare SVS against:
1. **Task Arithmetic (TA):** The standard linear model merging baseline ($W_{merged} = W_0 + \sum_t \lambda_t T_t$). Comparing with TA directly isolates the performance impact of singular value slicing and scale-preservation normalization.
2. **AdaMerging:** The layer-specific test-time adaptive coefficient baseline (optimizes coefficients on downstream test streams via entropy minimization).
3. **SyMerge (SOTA):** The state-of-the-art low-rank adapter test-time optimization method.
4. **FoldMerge:** The state-of-the-art non-linear normalizing-flow test-time coordinate warping method.

By comparing SVS against these optimization-intensive baselines, we demonstrate that a simple, training-free closed-form spectral slice can achieve highly competitive multi-task accuracy without any of the parameter, optimization, or time overheads.

## 6. Step-by-Step Interaction
The flow of data and operations through SVS proceeds as follows:
1. **Model Loading:** Load the pre-trained base model $W_0$ and the $N$ expert models $W_t$ fine-tuned on their respective tasks.
2. **Task Delta Computation:** Compute the task vector for each expert: $T_t = W_t - W_0$.
3. **Spectral Decomposition (SVD):** Compute the Singular Value Decomposition of each task vector: $T_t = U_t \Sigma_t V_t^T$. This decomposes the task weight updates into orthonormal coordinate bases and their corresponding scaling energies (singular values).
4. **Singular Value Slicing:** Extract the top $k$ components to form the low-rank, noise-filtered task vector: $\tilde{T}_t = U_{t, :k} \Sigma_{t, :k} V_{t, :k}^T$.
5. **Linear Consensus Integration:** Add the sliced task vectors back to the base weight to form the raw merged matrix: $W_{merged} = W_0 + \sum_{t=1}^N \lambda_t \tilde{T}_t$.
6. **Barycentric Normalization (BWN):**
   - Compute the Frobenius norm of each expert: $\|W_t\|_F$.
   - Compute the Frobenius norm of the raw merged matrix: $\|W_{merged}\|_F$.
   - Calculate the norm-scaling factor $\alpha = \frac{\sum_t \mu_t \|W_t\|_F}{\|W_{merged}\|_F}$.
   - Scale the merged weights to compute the final deployment weights: $W_{final} = \alpha W_{merged}$.
7. **Zero-Overhead Deployment:** Inject $W_{final}$ into the target projection layer. Pass the multi-task visual tokens through the layer during inference to perform downstream classification across all 8 tasks simultaneously with zero extra latency.
