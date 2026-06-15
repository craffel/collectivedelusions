# Idea Proposal: ZipMerge (Post-Merge Joint Weight Pruning and Coefficient Tuning)

## 1. Persona Alignment
As **The Pragmatist**, my research is driven by a focus on real-world utility, extreme efficiency gains, and practical deployment constraints. While model merging successfully fuses multiple specialized experts into a single multi-task model without training data, the resulting model remains dense (100% parameter size). For actual edge deployment (e.g., on mobile phones, IoT nodes, and automotive chips), memory and storage are extremely constrained, and pruning is a critical tool to reduce model footprint.

However, standard post-hoc pruning of a merged model is highly destructive, as it naively removes weights without consideration of task-specific alignments. Conversely, pruning individual experts before merging is also suboptimal because it breaks structural alignment. 

**ZipMerge** solves this practical dilemma by integrating magnitude pruning directly into the test-time adaptation loop. By co-optimizing the layer-wise merging coefficients $\Lambda$ and the non-differentiable pruning boundaries on a tiny, unlabeled calibration dataset (16 images per task), ZipMerge creates a sparse, high-performance, multi-task model ready for low-memory edge deployment. This avoids expensive retraining, requires zero labeled data, and directly reduces storage and memory costs.

---

## 2. Core Techniques
ZipMerge introduces and evaluates the following mechanisms and algorithms:
1. **Pruning-Aware Test-Time Adaptation:** Formulating the magnitude-pruning operator as part of the forward graph during test-time adaptation, allowing the model to adapt merging coefficients under the explicit constraint of a target sparsity level.
2. **Straight-Through Estimator (STE) for Pruning Gradients:** Employs a Straight-Through Estimator to bypass the non-differentiable step-function of the pruning mask, allowing first-order gradient descent (Adam) to flow through the sparse mask and optimize the continuous merging coefficients.
3. **Zero-Order Subspace Exploration (1+1 ES):** Evaluates a black-box 1+1 Evolution Strategy to optimize the coefficients, which completely sidesteps the non-differentiability of the pruning threshold.
4. **Task-Specific Gradient Routing via Sparsity Masks:** Analytically studies how magnitude pruning alters task-specific gradient flows and prevents interference between experts.

**Citations to Foundational Work:**
- **Task Arithmetic / Model Merging:** *Editing Models with Task Arithmetic* (Ilharco et al., 2023)
- **AdaMerging (Test-Time Coefficient Tuning):** *AdaMerging: Adaptive Model Merging for Multi-Task Learning* (Yang et al., 2024)
- **Magnitude Pruning:** *Learning both Weights and Connections for Efficient Neural Networks* (Han et al., 2015)
- **Straight-Through Estimator:** *Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation* (Bengio et al., 2013)

---

## 3. Mathematical Formulation

Let:
- $K$ be the number of tasks, $L$ be the number of layers, and $N$ be the total number of parameters in the model.
- $\theta_0 \in \mathbb{R}^N$ represent the pre-trained base model weights.
- $\theta_k \in \mathbb{R}^N$ represent the task-specific fine-tuned expert weights for task $k \in \{1, \dots, K\}$.
- $v_k = \theta_k - \theta_0 \in \mathbb{R}^N$ represent the task-specific parameter vector.
- $\Lambda = \{ \lambda^l_k \}$ represent the layer-wise merging coefficients, where $\lambda^l_k \in \mathbb{R}$ is the coefficient for task $k$ in layer $l \in \{1, \dots, L\}$.

### Step 1: Weight Merging
The merged parameters $\theta_{merged}(\Lambda)$ for a weight $w$ in layer $l$ are defined as:
$$
w_{merged}(\Lambda) = w_0 + \sum_{k=1}^K \lambda^l_k v_{k, w}
$$

### Step 2: Sparsity Mask Generation
We define a target global or layer-wise sparsity ratio $p \in [0, 1)$ (e.g., $p = 0.5$ for 50% pruning). The absolute merged weights are sorted to find the $p$-th percentile threshold $\tau_p(\Lambda)$:
$$
\tau_p(\Lambda) = \text{percentile}\left( \{ |w_{merged}(\Lambda)| : w \in \text{Model} \}, 100 \cdot p \right)
$$
The binary pruning mask $M(\Lambda) \in \{0, 1\}^N$ is defined as:
$$
M_w(\Lambda) = \mathbb{I}(|w_{merged}(\Lambda)| \ge \tau_p(\Lambda))
$$
where $\mathbb{I}(\cdot)$ is the indicator function.

### Step 3: Sparse Model Construction
The sparse merged parameters $\theta_{sparse}(\Lambda)$ are:
$$
\theta_{sparse}(\Lambda) = M(\Lambda) \odot \theta_{merged}(\Lambda)
$$
where $\odot$ represents the element-wise Hadamard product.

### Step 4: Unsupervised Test-Time Objective
The model is optimized using an unsupervised minimum entropy objective on a small calibration set of $B$ unlabeled images per task:
$$
\mathcal{L}_{entropy}(\Lambda) = -\frac{1}{K \cdot B} \sum_{k=1}^K \sum_{i=1}^B \sum_{c=1}^C p_{k,i,c}(\Lambda) \log p_{k,i,c}(\Lambda)
$$
where $p_{k,i,c}(\Lambda)$ is the predicted probability of the $c$-th class for the $i$-th calibration image of task $k$ using the sparse merged model $\theta_{sparse}(\Lambda)$.

### Step 5: Optimization with Straight-Through Estimator (STE)
To optimize $\Lambda$ using gradient descent, we bypass the non-differentiable indicator function of $M(\Lambda)$ by approximating:
$$
\frac{\partial \theta_{sparse}(\Lambda)}{\partial \theta_{merged}(\Lambda)} \approx M(\Lambda)
$$
Thus, the gradient of the loss with respect to a coefficient $\lambda^l_k$ is computed as:
$$
\frac{\partial \mathcal{L}_{entropy}}{\partial \lambda^l_k} \approx \sum_{w \in \text{layer } l} \frac{\partial \mathcal{L}_{entropy}}{\partial w_{sparse}} \cdot M_w(\Lambda) \cdot v_{k, w}
$$

---

## 4. Architecture Specifications

### Model Backbone
We utilize the **timm ViT-Tiny** backbone (`vit_tiny_patch16_224`), which contains 5.7M parameters across:
- **1 Patch Embedding Layer**
- **12 Transformer Blocks** (each containing Multi-Head Attention and MLP sub-layers)
- **1 Final Layer Normalization Layer**
- **4 Task-Specific Linear Heads** (one for each dataset: MNIST, FashionMNIST, CIFAR-10, SVHN).

### Parameter Grouping ($L=14$ layers)
Consistent with prior work, we group the parameters into $L=14$ discrete layers to define our layer-wise coefficients $\Lambda$:
- Layer 1: Patch embeddings.
- Layers 2–13: The 12 Transformer blocks.
- Layer 14: The final layer normalization layer.

This results in a parameter coefficient matrix $\Lambda$ of dimension $14 \times 4$ (total of 56 optimization parameters).

### Target Sparsity $p$
We evaluate under three strict edge-deployment constraints:
- **No Pruning ($p=0.0$):** To verify baseline performance.
- **Moderate Sparsity ($p=0.5$):** 50% of weights set to zero.
- **Aggressive Sparsity ($p=0.8$):** 80% of weights set to zero.

---

## 5. Baselines
We evaluate ZipMerge against the following highly appropriate baselines:
1. **FP16 Merged Model (Uniform, Dense):** Standard Task Arithmetic merging with uniform coefficients ($\lambda^l_k = 0.3$), representing the unpruned, unoptimized dense starting point.
2. **AdaMerging (FP16 Optimized, Dense):** Optimizing $\Lambda$ via standard AdaMerging, representing the unpruned optimized performance ceiling.
3. **Merge-then-Prune (M-then-P) (Uniform, Sparse):** Naive pipeline where experts are merged uniformly ($\lambda^l_k = 0.3$), and the merged checkpoint is subsequently magnitude-pruned post-hoc. This evaluates if joint optimization is necessary.
4. **AdaMerging-then-Prune (Ada-then-P) (Optimized, Sparse):** Merging coefficients $\Lambda$ are optimized first on the dense model, and then the resulting joint checkpoint is magnitude-pruned post-hoc. This tests if dense-optimized coefficients are robust to subsequent pruning.
5. **Prune-then-Merge (P-then-M) (Sparse):** Task-specific experts are individually magnitude-pruned first, and then their remaining active weights are merged.

---

## 6. Step-by-Step Interaction

From data loading to final prediction, the data flows as follows:

1. **Test-Time Input Stream:**
   A batch of $B=16$ unlabeled calibration images from task $k$ is loaded and preprocessed to shape `(16, 3, 224, 224)`.

2. **Dynamic Weight Reconstruction:**
   - The system retrieves the base model parameters $\theta_0$ and task vectors $v_k = \theta_k - \theta_0$.
   - The current continuous coefficients $\Lambda$ are used to compute the full-precision dense merged parameters $\theta_{merged}(\Lambda)$ layer-by-layer.

3. **On-the-Fly Masking (Magnitude Pruning):**
   - The absolute values of $\theta_{merged}(\Lambda)$ are computed.
   - The $p$-th percentile threshold $\tau_p$ is calculated globally.
   - The binary mask $M(\Lambda) = \mathbb{I}(|\theta_{merged}(\Lambda)| \ge \tau_p)$ is generated.
   - The sparse weights $\theta_{sparse}(\Lambda) = M(\Lambda) \odot \theta_{merged}(\Lambda)$ are loaded into the active model structure.

4. **Forward Propagation:**
   - The calibration batch is propagated through the sparse network $\theta_{sparse}(\Lambda)$.
   - The patch embedding layer maps the image to visual tokens.
   - The 12 Transformer blocks perform self-attention and MLP transformations using the sparse weights.
   - The representations are pooled, and the task-specific linear head for task $k$ outputs the class logits.

5. **Loss Computation:**
   - Softmax is applied to the logits to produce probability distributions $p_{k,i,c}(\Lambda)$.
   - The multi-task minimum entropy loss $\mathcal{L}_{entropy}(\Lambda)$ is calculated.

6. **Backward Propagation & Parameter Update:**
   - The gradients are backpropagated through the network.
   - The gradient $\frac{\partial \mathcal{L}_{entropy}}{\partial \theta_{sparse}}$ is computed.
   - The Straight-Through Estimator routes the gradient to the merged weights: $\frac{\partial \mathcal{L}_{entropy}}{\partial \theta_{merged}} = M(\Lambda) \odot \frac{\partial \mathcal{L}_{entropy}}{\partial \theta_{sparse}}$.
   - Gradients with respect to the merging coefficients $\lambda^l_k$ are derived and used by the Adam optimizer (or ES perturbations) to update $\Lambda$.
   - Steps 2-6 are repeated for a small number of test-time optimization epochs (e.g., 40 steps).

7. **Final Inference Evaluation:**
   - The optimized coefficients $\Lambda^*$ and the final mask $M(\Lambda^*)$ are frozen.
   - The sparse model is evaluated on the full multi-task test sets to report final classification accuracies.
