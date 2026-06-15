# GranMerge: Deconstructing the Generalization-Granularity Trade-off in Adaptive Model Merging

## 1. Persona Alignment
As **The Empiricist**, our research philosophy is that progress in machine learning comes from exhaustive empirical validation, large-scale sweeps, and comprehensive statistical evidence over multiple datasets, seeds, and hyperparameters. We do not trust a method until it has been rigorously stressed-tested in multiple dimensions.

The investigation of the **Generalization-Granularity Trade-off** in model merging is the quintessential empiricist project. Rather than introducing a minor heuristic tweak, we are systematically mapping out the complete multi-dimensional landscape of weight merging over 5 levels of structural granularity (Global, Block-wise, Layer-wise, Component-wise, and Tensor-wise) across 4 diverse visual classification tasks (MNIST, FashionMNIST, CIFAR-10, SVHN), crossed with 2 completely different optimization families (first-order Adam gradient descent with backpropagation vs. zero-order 1+1 Evolution Strategies), under a wide range of test-time calibration budgets, evaluated over dozens of seeds to ensure statistical significance. This work will deliver massive empirical datasets, rigorous ablation studies, and comprehensive evidence regarding where, when, and at what structural level merging coefficients are physically meaningful or merely overfitting.

## 2. Core Techniques
We introduce **GranMerge**, a unified framework to study the **Generalization-Granularity Trade-off** in model merging.
We define and implement five nested levels of parameter resolution for merging coefficients:
- **Level 1: Global Merging (Task Arithmetic):** 1 scalar coefficient $\lambda_k$ per task. Merges all parameters with a uniform scale.
- **Level 2: Block-wise Merging:** 2 coefficients per layer per task: $\lambda_{k, \text{Attention}}$ and $\lambda_{k, \text{MLP}}$. Isolates representation routing inside attention blocks vs. MLP blocks.
- **Level 3: Layer-wise Merging (AdaMerging):** 1 coefficient per layer per task: $\lambda_{k, l}$ where $l \in [0, L-1]$. Allows individual layers to vary.
- **Level 4: Component-wise Merging:** 4 coefficients per layer per task: $\lambda_{k, QKV}$, $\lambda_{k, \text{AttnOut}}$, $\lambda_{k, \text{MLP1}}$, $\lambda_{k, \text{MLP2}}$.
- **Level 5: Tensor-wise Merging:** Unique coefficient $\lambda_{k, t}$ for every parameter tensor (weights and biases) in the model, e.g., 10 parameters per layer per task.

We optimize these coefficients using:
- **Adam Gradient Descent:** Backpropagating unsupervised Shannon entropy loss from task classification outputs directly to the coefficients.
- **1+1 Evolution Strategies:** Black-box stochastic mutation searches that evaluate prediction entropy to guide steps.

To prevent overparameterized transductive overfitting at higher granularities, we introduce and benchmark:
- **Elastic Spatial Regularization (ESR):** L2 penalty on coefficient drift from their layer-wise or block-wise spatial means.
- **Total Variation (TV) Smoothness Penalty:** L1/L2 penalties on depth-wise parameter differences.
- **Coefficient Dropout:** Randomly zeroing out coefficient updates during calibration steps.

Foundational citations:
- Task Arithmetic: Ilharco et al. (2022) "Editing models with task arithmetic"
- AdaMerging: Yang et al. (2024) "AdaMerging: Adaptive Model Merging for Multi-Task Learning"
- PolyMerge: Jung et al. (2025/2026) "PolyMerge: A Controlled Simulation and Optimization Study..."
- RegCalMerge: (2026) "RegCalMerge: Overcoming Transductive Overfitting..."

## 3. Mathematical Formulation
Let $W_{base}$ be the pre-trained base model weights, and $W_k$ be the fine-tuned expert model weights for task $k \in \{1, \dots, K\}$.
The task vector for task $k$ is defined as $\theta_k = W_k - W_{base}$.
For a specific parameter tensor $t \in T$, where $T$ is the set of all matchable tensors in the model, the merged weight $W^{(t)}$ is:
$$W^{(t)} = W_{base}^{(t)} + \sum_{k=1}^K \lambda_{k, t} \theta_k^{(t)}$$
where $\lambda_{k, t}$ is the merging coefficient for task $k$ at tensor $t$.

### Unsupervised Surrogate Loss Function
During test-time adaptation, we do not have access to true labels. Thus, we optimize the coefficients by minimizing the multi-task unsupervised prediction entropy on a small calibration batch $X_{cal} = \{X_{cal, 1}, \dots, X_{cal, K}\}$:
$$\mathcal{L}(\Lambda) = \frac{1}{K} \sum_{k=1}^K \mathcal{H}(P(Y | X_{cal, k}; W(\Lambda))) + \beta \mathcal{R}(\Lambda)$$
where:
- $\Lambda = \{\lambda_{k, t}\}$ is the set of all merging parameters.
- $P(Y | X_{cal, k}; W(\Lambda))$ is the softmax probability distribution outputted by the merged model $W(\Lambda)$ on task $k$.
- $\mathcal{H}(P) = -\sum_{c=1}^C P_c \log P_c$ is the Shannon entropy.
- $\mathcal{R}(\Lambda)$ is the regularization penalty.
- $\beta$ is the regularization strength.

### Regularization Penalty Definitions
1. **Elastic Spatial Regularization (ESR):** Pulls fine-grained parameters towards their task-specific spatial average:
   $$\mathcal{R}_{\text{ESR}}(\Lambda) = \sum_{k=1}^K \sum_{t \in T} \left( \lambda_{k, t} - \bar{\lambda}_{k} \right)^2$$
   where $\bar{\lambda}_k = \frac{1}{|T|} \sum_{t \in T} \lambda_{k, t}$.
2. **Total Variation (TV) Regularization:** Enforces depth-wise smoothness across transformer layers:
   $$\mathcal{R}_{\text{TV}}(\Lambda) = \sum_{k=1}^K \sum_{l=1}^{L-1} \sum_{c \in \text{Components}} \left( \lambda_{k, l, c} - \lambda_{k, l-1, c} \right)^2$$

## 4. Architecture Specifications
We evaluate GranMerge on a 12-layer Vision Transformer (specifically, pre-trained multimodal CLIP ViT-B/32 or ViT-Tiny backbones).
- **Inputs:** Vision features or raw images of size $224 \times 224$ (or $28 \times 28$ for resized MNIST-like datasets).
- **Model Parameters ($L = 12$):**
  - Patch embedding projection: `patch_embed.proj.weight` (size $[768, 3, 32, 32]$ for ViT-B/32).
  - Attention blocks: 12 layer blocks, each containing:
    - Query projection: `attn.q_proj` (size $[768, 768]$)
    - Key projection: `attn.k_proj` (size $[768, 768]$)
    - Value projection: `attn.v_proj` (size $[768, 768]$)
    - Out projection: `attn.out_proj` (size $[768, 768]$)
  - MLP blocks: 12 layer blocks, each containing:
    - Layer 1 projection: `mlp.fc1` (size $[3072, 768]$)
    - Layer 2 projection: `mlp.fc2` (size $[768, 3072]$)
  - Output Classifier Heads: Task-specific linear layers mapped to task label counts ($C = 10$).
- **Intermediate Representations:** Hidden dimensions $d_{model} = 768$ (for ViT-B/32) or $d_{model} = 192$ (for ViT-Tiny). Number of attention heads is 12 (ViT-B/32) or 3 (ViT-Tiny).
- **Number of Merging Parameters per Task:**
  - **Level 1 (Global):** 1 parameter.
  - **Level 2 (Block-wise):** $2 \times L = 24$ parameters.
  - **Level 3 (Layer-wise):** $L = 12$ parameters.
  - **Level 4 (Component-wise):** $4 \times L = 48$ parameters.
  - **Level 5 (Tensor-wise):** $\approx 10 \times L = 120$ parameters.

## 5. Baselines
We compare our GranMerge structural settings against a robust suite of prior methods:
1. **Uniform Task Arithmetic (TA) baseline:** Manually selected uniform task-wise coefficient ($\lambda_k = 0.3$), which represents a static baseline.
2. **Tuned Task Arithmetic (Global-Opt):** Level 1 global optimization of 1 parameter per task. This is the optimal uniform scalar benchmark.
3. **AdaMerging (Layer-Opt):** Level 3 layer-wise coefficient optimization (12 parameters per task), which represents the SOTA adaptive baseline.
4. **PolyMerge (d=2):** Parameterizing coefficients using a quadratic polynomial of layer depth, representing a hard-constrained smoothness baseline.
5. **RegCalMerge (with ESR):** Adaptive layer-wise merging with CCN, SNEW, and spatial regularization.

## 6. Step-by-Step Interaction
1. **Initialization:**
   - Load pre-trained base model $W_{base}$ and fine-tuned expert models $W_k$ for $K = 4$ tasks (MNIST, FashionMNIST, CIFAR-10, SVHN).
   - Compute task vectors $\theta_k = W_k - W_{base}$.
   - Initialize merging parameters $\Lambda = \{\lambda_{k, t}\}$ (e.g., uniformly at $0.3$ or zero).
2. **Test-Time Adaptation Stream:**
   - Sample a tiny unlabeled calibration stream batch $X_{cal, k}$ of size $N=64$ for each task.
3. **Weight Merging Step:**
   - Blend the expert weights to construct the multi-task model $W(\Lambda)$ at the selected level of structural granularity (e.g., component-wise or tensor-wise), matching each tensor's coefficient to its group's parameter.
4. **Forward Pass & Loss Evaluation:**
   - Run a forward pass of the merged model $W(\Lambda)$ on $X_{cal, k}$.
   - Compute softmax class probabilities and calculate prediction entropy $\mathcal{H}_k$.
   - Average entropy across tasks and add the regularization term $\beta \mathcal{R}(\Lambda)$ to compute total loss $\mathcal{L}(\Lambda)$.
5. **Optimization Update:**
   - **Adam GD:** Backpropagate gradients $\nabla_{\Lambda} \mathcal{L}$ and update continuous coefficients $\Lambda \leftarrow \text{Adam}(\Lambda, \nabla_{\Lambda} \mathcal{L})$.
   - **1+1 ES:** Randomly mutate coefficients, evaluate loss, and keep mutations that decrease loss.
6. **Inference & Validation:**
   - Deploy the final merged model $W(\Lambda^*)$ on the full held-out test sets of all tasks to record true generalization accuracies and compute average multi-task performance.
