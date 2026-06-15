# Idea Proposal: Sparse Task Arithmetic (STA) - Deconstructing the Redundancy of Sign-Resolution in Model Merging

## 1. Persona Alignment
This proposal is highly aligned with the traits and goals of **The Minimalist** (as described in `persona.md`). Modern model merging research has become needlessly complex, introducing convoluted multi-stage heuristics (e.g., sign voting, coordinate-wise sign resolution, random dropping, and adaptive rescaling) to resolve parameter interference. Guided by Occam's razor, we hypothesize that the complex sign-election and sign-resolution steps of methods like TIES-Merging (Yadav et al., 2023) are entirely redundant. 

By stripping away this unnecessary complexity, we show that simply applying magnitude-based pruning (to remove small, noisy task vector updates) followed by standard linear Task Arithmetic (addition) matches or outperforms complex pipelines. We value clean architectures, code readability (implementable in just 3 lines of PyTorch), and rigorous ablation, demonstrating that the simplest, stripped-down solution is strictly better.

## 2. Core Techniques
We introduce **Sparse Task Arithmetic (STA)**, a streamlined weight-space fusion method that directly modifies the delta parameter updates of task-specific expert models. The core techniques modified/used are:
1.  **Task Vector Formulation (Task Arithmetic):** Expressing fine-tuned models as "task vectors" relative to the pre-trained base model, following Ilharco et al. (2022).
2.  **Isotropic Magnitude Pruning (Trimming):** Retaining only the top-$s$\% largest elements of each task vector by absolute value, discarding the small background parameter shifts as orthogonal task-specific noise.
3.  **Linear Summation (Standard Addition):** Directly adding the sparse task vectors without any sign voting, dominant sign enforcement, or random drop-and-rescale operations.

We compare this against:
-   **Task Arithmetic (TA)** (Ilharco et al., 2022)
-   **TIES-Merging** (Yadav et al., 2023)
-   **DARE** (Yu et al., 2024)

Our codebase will build upon the official AdaMerging codebase (`https://github.com/Shuai0302/AdaMerging`) or a clean PyTorch-based model merging simulation environment to run the multi-seed evaluation.

## 3. Mathematical Formulation

Let $\theta_0 \in \mathbb{R}^D$ represent the parameters of the shared pre-trained base model.
Let $\theta_k \in \mathbb{R}^D$ represent the parameters of task expert $k$ (for $k = 1, \dots, K$), fine-tuned from $\theta_0$.

### 3.1 Task Vector Computation
For each task expert $k$, we define the task vector $v_k$ as:
$$v_k = \theta_k - \theta_0$$

### 3.2 Isotropic Magnitude Pruning
We compute a binary magnitude mask $M_k \in \{0, 1\}^D$ for each task vector $v_k$ layer-wise. Let $v_{k, l}$ be the parameters of layer $l$ for task $k$, and let $s \in (0, 100]$ be the survival density percentage.
The threshold $\tau_{k, l}$ for layer $l$ is defined as the $(100 - s)$-th percentile of the absolute task vector elements:
$$\tau_{k, l} = \text{percentile}(|v_{k, l}|, 100 - s)$$

The binary mask $M_{k, l}$ is defined coordinate-wise as:
$$[M_{k, l}]_j = \begin{cases} 1 & \text{if } |[v_{k, l}]_j| \ge \tau_{k, l} \\ 0 & \text{otherwise} \end{cases}$$

The sparse task vector $v^{\text{sparse}}_{k, l}$ is obtained by element-wise multiplication:
$$v^{\text{sparse}}_{k, l} = v_{k, l} \odot M_{k, l}$$

### 3.3 Linear Merging (No Sign Resolution)
Unlike TIES-Merging (which performs coordinate-wise sign voting and discards all updates whose sign disagrees with the majority), STA directly merges the sparse task vectors:
$$v_{\text{merged}, l} = \sum_{k=1}^K \lambda_k v^{\text{sparse}}_{k, l}$$

where $\lambda_k > 0$ represents the scaling coefficient for task $k$ (e.g., uniformly set to $0.3$ or optimized).
The final merged model parameters are:
$$\theta_{\text{merged}} = \theta_0 + v_{\text{merged}}$$

### 3.4 Hypothesis & Deconstruction of Sign-Resolution Redundancy
TIES-Merging argues that if $[v_a]_j > 0$ and $[v_b]_j < 0$, their conflicting signs will damage performance, requiring sign voting to select one dominant sign. We hypothesize that:
1.  **Sparsity Reduces Collision:** For a sufficiently small density $s$ (e.g., $s \le 20\%$), the intersection of active coordinates $\text{supp}(M_a) \cap \text{supp}(M_b)$ is extremely small.
2.  **Sign Conflicts are Rare:** In the rare cases where active coordinates overlap, the magnitude-largest update represents the dominant functional feature, and allowing the smaller opposite update to be summed does not damage performance as much as TIES's aggressive zeroing-out of active weights.
3.  **Occam's Razor:** The extra complexity of electing and enforcing sign consensus is mathematically unnecessary and structurally harmful.

## 4. Architecture Specifications
The system is evaluated on a standard vision transformer or deep neural network backbone.
-   **Backbone:** Vision Transformer (`ViT-B/32` or `vit_tiny_patch16_224`) or standard 5-layer Convolutional Neural Networks (DeepCNN).
-   **Layers:** 12-layer Transformer backbone with Multi-Head Self-Attention (MSA) and Multi-Layer Perceptrons (MLP).
-   **Input:** Batched images rescaled to $224 \times 224$ (ViT) or $28 \times 28$ (DeepCNN).
-   **Output Heads:** Separate linear classification heads (one for each task) attached to the unified backbone representations.
-   **Tasks:** 4-task classification suite: **MNIST**, **FashionMNIST**, **CIFAR-10**, and **SVHN**.
-   **Sparsity Parameter $s$:** Swept across $\{5\%, 10\%, 20\%, 50\%, 100\%\}$ to evaluate the optimal trade-off between noise removal and representation preservation.

## 5. Baselines
We compare our proposed **Sparse Task Arithmetic (STA)** against the following baseline paradigms:
1.  **Task Arithmetic (Uniform):** Full-density model merging using linear addition ($s = 100\%$). Shows performance under unpruned interference.
2.  **TIES-Merging:** The standard multi-stage heuristic baseline (Trimming + Sign Election + Disjoint Merging). Shows whether sign consensus provides any benefit over simple sparse addition.
3.  **DARE-Merging:** The random-dropout baseline (Delta-Agnostic Drop and Rescale). Shows whether deterministic magnitude-based selection is superior to stochastic coordinate dropping.
4.  **Online AdaMerging:** Unpruned, active test-time adaptation. Shows how static, training-free sparse merging compares to compute-intensive, online optimization methods.

These baselines are appropriate because they cover the entire landscape of training-free, sparse, and adaptive weight-space merging.

## 6. Step-by-Step Interaction
Data flows through the proposed training-free system as follows:

1.  **Initialization:** Load the pre-trained base model $\theta_0$ and the $K$ task-specific expert weights $\theta_1, \dots, \theta_K$.
2.  **Delta Extraction:** Compute the dense task vector $v_k = \theta_k - \theta_0$ for each task.
3.  **Magnitude Filtering:** Compute the layer-wise percentile threshold $\tau_{k, l}$ and create binary mask $M_{k, l}$ containing 1s for the top-$s$\% largest coordinates and 0s elsewhere.
4.  **Pruning Application:** Obtain the sparse task vectors $v^{\text{sparse}}_{k} = v_k \odot M_k$.
5.  **Linear Summation:** Construct the merged delta representation $v_{\text{merged}} = \sum_k \lambda_k v^{\text{sparse}}_k$.
6.  **Parameter Assembly:** Construct the merged network parameters $\theta_{\text{merged}} = \theta_0 + v_{\text{merged}}$.
7.  **Inference Evaluation:** Load the merged parameters $\theta_{\text{merged}}$ into the ViT backbone. For a test sample belonging to task $k$, pass the input through the merged backbone to generate features, and then route the features through the task-specific linear classifier head $\text{Head}_k$ to yield final class predictions.
