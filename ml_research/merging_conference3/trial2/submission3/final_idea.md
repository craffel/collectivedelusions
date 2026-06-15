# Idea Proposal: PolyMerge (Polynomial Spline Parameterization of Layer-Wise Merging Strengths)

## 1. Persona Alignment
As **The Empiricist**, our core philosophy is that true progress in machine learning comes from exhaustive, large-scale empirical validation. We believe that overparameterized test-time adaptation frameworks (like AdaMerging and SyMerge) are highly prone to transductive overfitting, and that unconstrained optimization leads to "delicate, overfit" layer configurations that fail to generalize. 

Rather than relying on unvalidated theoretical assumptions or minor qualitative descriptions, **PolyMerge** is designed to directly regularize test-time optimization by constraining the search space to a low-dimensional polynomial manifold. To prove its effectiveness, we will execute a massive, multi-axis grid sweep across:
- Polynomial degrees $D \in \{0, 1, 2, 3\}$.
- Optimizers (first-order Adam GD vs zero-order 1+1 ES).
- Multiple independent random seeds (at least 3).
- Multiple image classification benchmarks on CLIP ViT-B/32 (including MNIST, FashionMNIST, CIFAR-10, SVHN).

This rigorous empirical approach will map the trade-off between parameter expressiveness (degrees of freedom) and generalization capability, providing overwhelming empirical evidence for our hypothesis.

## 2. Core Techniques
The proposed framework, **PolyMerge**, introduces the following core techniques:
1.  **Polynomial Spline Parameterization:** Instead of directly and independently learning a coefficient $\lambda_{k, l}$ for each task $k$ and layer $l$, we parameterize the layer-wise coefficient profile as a continuous, low-degree polynomial of the normalized layer depth.
2.  **Normalized Depth Scale:** To ensure numerical stability and translation/scale invariance across different model depths, we normalize the layer indexes to a standard $[0, 1]$ range.
3.  **Low-Dimensional Subspace Optimization:** By optimizing only $D+1$ parameters per task (where $D \in \{0, 1, 2, 3\}$) instead of $L$ parameters (where $L \ge 12$), we mathematically prune high-frequency optimization noise and prevent transductive overfitting.

**Foundational Citations:**
- **Task Arithmetic:** Ilharco et al., "Editing Models with Task Arithmetic", ICLR 2023.
- **AdaMerging:** Yang et al., "AdaMerging: Adaptive Model Merging for Multi-Task Learning", ICLR 2024.
- **SyMerge:** Jung et al., "SyMerge: Exploring and Exploiting Cross-Task Synergy in Model Merging", ICML 2026.
- **Overfitting-Optimizer Paradox:** Trial 1, Submission 7 (reproduced in this workspace).

## 3. Mathematical Formulation
Let $K$ be the number of distinct downstream tasks, and $L$ be the total number of layer-specific parameter blocks being merged. For task $k$ and layer block $l \in \{0, 1, \dots, L-1\}$, we define the merging coefficient $\lambda_{k, l}$ as a polynomial of degree $D$:

$$\lambda_{k, l} = \sum_{d=0}^D \alpha_{k, d} \cdot \left( \frac{l}{L-1} \right)^d$$

where:
- $\alpha_{k, d} \in \mathbb{R}$ represent the $D+1$ learnable parameters for task $k$.
- $\bar{l} = \frac{l}{L-1}$ is the normalized layer depth, bounding the polynomial basis to the domain $[0, 1]$.

Let $\Theta_{\text{base}}$ be the pre-trained base model weights, and $\mathbf{\Delta}_{k, l} = \Theta_{k, l} - \Theta_{\text{base}, l}$ be the task vector (expert parameters minus base parameters) for task $k$ at layer block $l$. The merged model weights $\Theta_{\text{merged}}$ are computed as:

$$\Theta_{\text{merged}, l} = \Theta_{\text{base}, l} + \sum_{k=1}^K \lambda_{k, l} \mathbf{\Delta}_{k, l}$$

For test-time adaptation (TTA), we optimize the polynomial coefficients $\mathbf{\alpha} = \{ \alpha_{k, d} \}$ by minimizing the unsupervised entropy of predictions over unlabeled target data streams:

$$\mathcal{L}_{\text{TTA}}(\mathbf{\alpha}) = \sum_{k=1}^K \mathbb{E}_{x \sim \mathcal{D}_k^{\text{unlabeled}}} \left[ H(f_{\Theta_{\text{merged}}}(x)) \right]$$

where $H(\mathbf{p}) = -\sum_c p_c \log p_c$ is the entropy of the predicted class distribution $\mathbf{p} = f_{\Theta_{\text{merged}}}(x)$.

## 4. Architecture Specifications
- **Base Model Backbone:** CLIP ViT-B/32 (Vision Transformer with 12 layers).
- **Layer Parameter Blocks ($L$):** We support two granularities of layer parameter blocks:
  - **Transformer Layer Granularity ($L=12$):** One set of merging coefficients per transformer layer.
  - **Projection Layer Granularity ($L=52$):** Separate coefficients for individual weight projections (e.g., self-attention query/key/value projections, MLP projection layers) following standard AdaMerging.
- **Learnable Parameter Space:** For each task, we maintain a small parameter vector $\mathbf{\alpha}_k \in \mathbb{R}^{D+1}$. For $K=4$ tasks and degree $D=2$, the total learnable parameters optimized at test-time is only $4 \times 3 = 12$, compared to $4 \times 52 = 208$ in unconstrained layer-wise AdaMerging (a 94.2% parameter reduction).
- **Inputs:** Batches of unlabeled images from each task.
- **Intermediate Representations:** Output class logits from the merged model $\Theta_{\text{merged}}$.
- **Outputs:** Softmax probability distributions over target classes.

## 5. Baselines
We will compare PolyMerge against the following baselines:
1.  **Task Arithmetic (Uniform Baseline):** Merging task vectors using a fixed, manually selected uniform scalar coefficient $\lambda_k$ across all layers ($D=0$, no test-time adaptation).
2.  **Unconstrained AdaMerging (Layer-wise):** Direct, unconstrained optimization of all $L$ coefficients per task using entropy minimization under Adam GD or 1+1 ES.
3.  **Spatial Mean Baseline (Mean Treatment):** First optimizing unconstrained AdaMerging coefficients, and then replacing them with their spatial mean per task to verify if smoothing acts as a strong post-hoc regularizer.
4.  **SyMerge (SOTA):** Low-rank adapter-based adaptive model merging.

## 6. Step-by-Step Interaction
1.  **Initialization:** Initialize the polynomial coefficients $\alpha_{k, d}$ such that the initial merging coefficients default to a standard Task Arithmetic baseline (e.g., $\alpha_{k, 0} = 0.3$ and $\alpha_{k, d > 0} = 0.0$, making $\lambda_{k, l} = 0.3$ for all layers).
2.  **Batch Sampling:** Retrieve a batch of unlabeled images $x_k$ from the test stream of each task $k$.
3.  **Coefficient Synthesis:** For each task $k$ and layer block $l$, compute the merging coefficient:
    $$\lambda_{k, l} = \sum_{d=0}^D \alpha_{k, d} \cdot \left(\frac{l}{L-1}\right)^d$$
4.  **Model Fusing:** Construct the active merged model parameters $\Theta_{\text{merged}}$ by scaling the task vectors $\mathbf{\Delta}_{k, l}$ with the synthesized coefficients $\lambda_{k, l}$ and adding them to the base model $\Theta_{\text{base}}$.
5.  **Forward Pass:** Pass the batch $x_k$ through the merged model to get class logits, and apply Softmax to obtain class probabilities $p_k$.
6.  **Loss Evaluation:** Compute the entropy loss $\mathcal{L} = H(p_k)$ across the batch.
7.  **Backward Pass & Optimization:** 
    - Compute the gradients of $\mathcal{L}$ with respect to the learnable polynomial coefficients $\alpha_{k, d}$.
    - Update $\alpha_{k, d}$ using the Adam optimizer (or mutate them using the zero-order 1+1 ES).
8.  **Repeat:** Iterate steps 2–7 for 500 optimization steps.
9.  **Evaluation:** Evaluate the final merged model on the held-out test sets across all tasks to measure multi-task accuracy, generalization performance under distribution shift, and stability across random seeds.
