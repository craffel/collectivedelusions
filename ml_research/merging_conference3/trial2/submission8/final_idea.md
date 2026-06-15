# Idea Proposal: Flatness-Guided Budgeted Task-Vector Pruning (FG-BTVP) for Resource-Constrained Model Merging

## 1. Persona Alignment
This project directly aligns with **The Pragmatist** persona by addressing a critical real-world constraint in deep learning deployment: **storage and memory bandwidth limitations on edge devices**. 

When deploying multi-task expert systems in the wild, storing multiple independent, fully-parameterized foundation models is computationally and financially prohibitive (e.g., storing five CLIP backbones requires gigabytes of storage). Model merging via Task Arithmetic mitigates this by storing only lightweight "task vectors" ($\Delta \theta_k$) relative to a shared base model. However, even these task vectors can be extremely large when scaled to multiple downstream tasks. 

Our proposed method, **Flatness-Guided Budgeted Task-Vector Pruning (FG-BTVP)**, compresses these task vectors by **80% to 95%** using magnitude-based pruning post-hoc. This enables storing the expert weights in highly compressed sparse formats (like CSR), directly reducing storage and transmission costs by up to 20x. Crucially, to make this aggressive compression viable without collapsing task performance, we leverage **optimizer-driven flatness (SAM-trained experts)** during training. SAM training is a standard, robust, and reliable practice that introduces **zero test-time latency or parameter overhead**. This guarantees an extremely simple, robust, and easy-to-integrate deployment-ready multi-task system.

---

## 2. Core Techniques
We introduce and evaluate a training-free, post-hoc compression and merging pipeline consisting of:
1. **Flatness-Aware Expert Training:** Fine-tuning task-specific expert models using standard, globally-perturbed Sharpness-Aware Minimization (SAM) as the optimizer. This ensures the expert parameters reside in wide, flat loss basins.
2. **Magnitude-Based Task Vector Sparsification:** Pruning task-specific updates $\tau_k = \theta_k - \theta_{base}$ by keeping only the top $p\%$ highest-magnitude parameters and resetting the rest to $0$.
3. **Adaptive Saliency-Based Budget Allocation:** Instead of uniform pruning, we dynamically allocate the total parameter budget $p$ across layers based on the relative magnitude (L1-norm) of layer updates, prioritizing layers with more significant task-specific adaptation.
4. **Sparse Multi-Task Merging:** Fusing the compressed task vectors onto the base model weights on-the-fly using Task Arithmetic.

---

## 3. Mathematical Formulation
Let $\theta_{base} \in \mathbb{R}^d$ be the parameters of the shared pre-trained base model.
Let $\theta_k \in \mathbb{R}^d$ be the weights of the fine-tuned expert model for task $k \in \{1, \dots, K\}$.

The task vector for task $k$ is defined as:
$$\tau_k = \theta_k - \theta_{base}$$

For a Transformer model with $L$ layers, let $\tau_{k, l}$ be the slice of the task vector corresponding to layer $l$. 
Given a target global retention budget $p \in (0, 1]$ (where $p=0.10$ means we keep only 10% of updates, achieving 90% sparsification), we define two pruning schemes:

### A. Uniform Pruning (FG-BTVP-U)
Each task vector is independently pruned to retain exactly the top $p\%$ absolute values globally:
$$\tilde{\tau}_k^{(p)} = \tau_k \odot M_k^{(p)}$$
where $M_k^{(p)} \in \{0, 1\}^d$ is a binary mask defined by:
$$M_{k, i}^{(p)} = \begin{cases} 1 & \text{if } |\tau_{k, i}| \ge \text{Percentile}(|\tau_k|, 100(1-p)) \\ 0 & \text{otherwise} \end{cases}$$

### B. Adaptive Saliency-Based Budget Allocation (FG-BTVP-S)
Rather than uniform sparsification across all layers, we allocate more parameter budget to layers that underwent more significant updates. 
We define the layer-wise update saliency $S_l$ as the average L1-norm of the task vectors at layer $l$:
$$S_l = \frac{1}{K} \sum_{k=1}^K \|\tau_{k, l}\|_1$$
The layer-wise budget allocation factor is:
$$w_l = \frac{S_l}{\frac{1}{L} \sum_{j=1}^L S_j}$$
We scale the target retention rate $p$ for layer $l$ as:
$$p_l = \text{clip}(p \cdot w_l, 0, 1.0)$$
The mask $M_{k, l}^{(p_l)}$ for layer $l$ is then computed using the custom percentile $100(1-p_l)$ of that layer's task vector slice. The sparse task vector becomes:
$$\tilde{\tau}_{k, l}^{(p)} = \tau_{k, l} \odot M_{k, l}^{(p_l)}$$

### C. Multi-Task Merging
The final merged model parameters are reconstructed on-the-fly via Task Arithmetic:
$$\theta_{MTL}^{(p)} = \theta_{base} + \sum_{k=1}^K \lambda_k \tilde{\tau}_k^{(p)}$$
where $\lambda_k$ is the task merging coefficient.

---

## 4. Architecture Specifications
- **Model Backbone:** CLIP ViT-B/32 or standard Vision Transformer (ViT-B/16).
- **Target Parameters:** Vision encoder weights, specifically:
  - Visual projection layer weights (`model.visual.proj`).
  - Attention projection layers across all transformer blocks.
- **Input Modality:** Multi-task image classification datasets (MNIST, FashionMNIST, CIFAR-10, SVHN).
- **Task Classifiers:** Zero-shot text-prompt classifier heads (computing cosine similarity between image embeddings and class text embeddings).
- **Output:** Multi-class logit predictions.

---

## 5. Baselines
To rigorously evaluate our claims, we compare against the following baselines:
1. **Unpruned Task Arithmetic (TA-AdamW):** Linear merging of dense expert models trained with standard AdamW.
2. **Unpruned Task Arithmetic (TA-SAM):** Linear merging of dense expert models trained with SAM (representing the uncompressed upper bound).
3. **Standard Magnitude Pruning (TA-Pruned-AdamW):** Magnitude-based task-vector pruning applied directly to standard AdamW-trained experts.
4. **TIES-Merging (TIES-AdamW vs. TIES-SAM):** Comparing the standard TIES pruning/consensus pipeline under standard vs. flat optimization.
5. **DARE (Drop and Rescale):** Random dropout of delta parameters with scaling on AdamW vs. SAM experts.

---

## 6. Step-by-Step Interaction
1. **Training (Offline):** Independent task experts $\theta_k$ are trained from the shared backbone $\theta_{base}$ on separate datasets. We compare experts trained with **AdamW** versus **SAM** to evaluate pruning resilience.
2. **Extraction (Offline):** Extract task-specific delta updates: $\tau_k = \theta_k - \theta_{base}$.
3. **Sparsification & Allocation (Offline):** Compute the layer-wise update saliency $S_l$ across all tasks and determine layer budgets $p_l$. Apply magnitude pruning to produce highly sparse task vectors $\tilde{\tau}_k$.
4. **Compression (Offline):** Save the sparse task vectors using compressed sparse formats (like CSR or coordinate lists), achieving up to 20x storage reduction.
5. **On-the-Fly Merging (Inference):** When deploying the multi-task model on edge hardware, load the shared base model $\theta_{base}$ and the lightweight compressed task vectors. Reconstruct the merged weights on-the-fly: $\theta_{MTL} = \theta_{base} + \sum_k \lambda_k \tilde{\tau}_k$.
6. **Inference (Online):** Pass a batch of multi-task images through the merged model $\theta_{MTL}$ to generate zero-shot predictions.
