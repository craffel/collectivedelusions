# Exclusive Parameter Merging (EPM): Zero-Interference Multi-Task Model Fusion

## 1. Persona Alignment
Exclusive Parameter Merging (EPM) is the ultimate realization of **The Minimalist** research philosophy. Guided strictly by Occam's razor, EPM rejects the hyper-complex, unstable, and token-expensive test-time adaptation (TTA) frameworks of previous trials (such as ZipMerge's joint optimization of 112 layer-wise merging coefficients and pruning masks). Instead, EPM achieves state-of-the-art sparse multi-task performance by pruning away unnecessary algorithmic bloat and resolving weight-space interference at its most fundamental level:
- **Relentless Simplification:** EPM replaces backpropagation-heavy online optimization loops with a single, elegant, closed-form parameter-allocation step.
- **Absolute Reject of Overfitting:** By optimizing only $K$ global scaling factors (one per task, e.g., $K=4$) instead of 56 layer-wise parameters, EPM completely bypasses the Overfitting-Optimizer Paradox that plagued prior attempts.
- **Elegance and Readability:** The core algorithm of EPM is implemented in under 10 lines of standard PyTorch code, offering perfect reproducibility, zero test-time overhead, and clear theoretical grounding.

## 2. Core Techniques
EPM introduces two core techniques that revolutionize weight-space model fusion under extreme domain shift:
1. **Exclusive Parameter Allocation (EPA):** Unlike Task Arithmetic (Ilharco et al., 2023), TIES-Merging (Yadav et al., 2023), or DARE (Yu et al., 2024)—which average or blend parameter updates across models—EPA guarantees **zero spatial interference**. For every coordinate in the weight space, EPA allocates the update *exclusively* to the single expert that exhibits the largest scaled magnitude. This creates a soft, parameter-level routing of representations, where each expert owns a disjoint, non-interfering subset of the model's parameters.
2. **Task-Level Coefficient Tuning (TLC-Tune):** Since EPM eliminates layer-wise coefficients, we only tune $K$ global scaling factors $\Lambda = [\lambda_1, \dots, \lambda_K]^T \in \mathbb{R}^K$. This extremely parameter-efficient search space is optimized on a tiny calibration set using a simple zero-order (1+1) Evolution Strategy, making it perfectly stable, training-free, and immune to transductive noise.

## 3. Mathematical Formulation
Let $\theta_{\text{base}}$ represent the pre-trained base model weights, and let $\theta_k$ represent the fine-tuned dense expert model for task $k \in \{1, \dots, K\}$.

### Step 1: Task Vector Extraction
We extract individual task vectors $\tau_k$:
$$\tau_k = \theta_k - \theta_{\text{base}}$$

### Step 2: Exclusive Parameter Allocation (EPA)
We introduce global task-specific scaling factors $\Lambda = \{\lambda_1, \dots, \lambda_K\}$. For each parameter coordinate $j$ across the entire network weight space, we calculate the absolute scaled update magnitude for each expert:
$$M_{k, j} = |\lambda_k \tau_{k, j}|$$

We identify the index of the dominant expert at coordinate $j$ using an argmax operator:
$$k^*(j) = \arg\max_{k \in \{1, \dots, K\}} M_{k, j}$$

The unpruned, exclusive merged task vector $\tau^{\text{exclusive}}$ is defined by retaining only the dominant expert's update at each coordinate, completely eliminating averaging and sign conflicts:
$$\tau^{\text{exclusive}}_j = \lambda_{k^*(j)} \tau_{k^*(j), j}$$

### Step 3: Sparsity Enforcement and Global Thresholding
To enforce an overall target sparsity $p \in (0, 1)$ (e.g., $p=0.5$ or $p=0.8$), we compute the saliency score $S_j$ for each coordinate:
$$S_j = |\tau^{\text{exclusive}}_j| = \max_{k \in \{1, \dots, K\}} |\lambda_k \tau_{k, j}|$$

We find a global threshold $T$ such that exactly $(1 - p)$ fraction of the coordinates satisfy $S_j \ge T$. The final sparse exclusive task vector is:
$$\tau^{\text{final}}_j = \begin{cases} \tau^{\text{exclusive}}_j & \text{if } S_j \ge T \\ 0 & \text{otherwise} \end{cases}$$

The merged sparse model weights are:
$$\theta_{\text{merged}} = \theta_{\text{base}} + \tau^{\text{final}}$$

### Step 4: Optimization via TLC-Tune
Since the search space consists of only $K$ global scaling factors $\Lambda$, we optimize $\Lambda$ to minimize cross-entropy loss $\mathcal{L}$ on a tiny validation set (or Shannon entropy on a tiny calibration set) using a (1+1) Evolution Strategy (ES). In each step $t$:
1. Perturb the current coefficients: $\Lambda^{(t)} = \Lambda + \epsilon$, where $\epsilon \sim \mathcal{N}(0, \sigma^2 I_K)$.
2. Compute the loss $\mathcal{L}(\theta_{\text{merged}}(\Lambda^{(t)}))$.
3. If $\mathcal{L}(\theta_{\text{merged}}(\Lambda^{(t)})) < \mathcal{L}(\theta_{\text{merged}}(\Lambda))$, accept the update and scale up $\sigma \leftarrow \sigma \cdot \alpha_{\text{up}}$. Otherwise, reject and scale down $\sigma \leftarrow \sigma \cdot \beta_{\text{down}}$.

## 4. Architecture Specifications
- **Backbone Model:** Vision Transformer (`vit_tiny_patch16_224` from the `timm` library), featuring 5.7 million parameters.
- **Internal Specs:** 12 Transformer blocks, hidden dimension of 192, 3 attention heads, and MLP expansion factor of 4.
- **Layer Grouping:** Parameter routing is applied globally across all layers, preserving the base normalization layers to maintain activation stability.
- **Inputs:** Batches of size 16 per task (64 total images for 4 tasks) containing standard $224 \times 224 \times 3$ images.
- **Outputs:** Softmax probability distributions over 10 classes per task, evaluated via task-specific classification heads.

## 5. Baselines
We evaluate EPM against a rigorous suite of baselines:
1. **Uniform Dense Merge (Task Arithmetic):** Linearly averages task vectors with a fixed weight ($\lambda = 0.3$), evaluating dense, unpruned performance.
2. **AdaMerging (Dense):** Optimizes layer-wise continuous merging coefficients on the dense model.
3. **Prune-then-Merge (P-then-M):** Magnitude-prunes each task vector independently before uniform merging. This is the strongest baseline under high conflict.
4. **ZipMerge (STE & ES):** Jointly co-optimizes layer-wise coefficients and magnitude-pruning boundaries at test-time.
5. **TIES-Merging:** Trim, Elect Sign, and Average. This serves to evaluate whether our average-free EPA formulation is superior to sign-elected averaging.

## 6. Step-by-Step Interaction
1. **Extraction:** Read pre-trained base model weights $\theta_{\text{base}}$ and expert weights $\theta_1, \dots, \theta_K$ to extract task vectors $\tau_k$.
2. **Scaling:** Apply global scale factors $\Lambda$ to task vectors.
3. **EPA Routing:** For every coordinate in the weight tensors, identify the expert with the maximum absolute scaled update. Zero out all other expert updates at that coordinate, forming the exclusive task vector $\tau^{\text{exclusive}}$.
4. **Global Pruning:** Sort the magnitudes of $\tau^{\text{exclusive}}$, locate the global threshold $T$ corresponding to the target sparsity $p$, and zero out any parameters below $T$.
5. **Update Base:** Add the final sparse exclusive task vector to the base model weights to yield $\theta_{\text{merged}}$.
6. **Evaluate/Tune:** Pass a tiny validation batch through the merged model, compute loss, and update $\Lambda$ using TLC-Tune (1+1 ES) over 40 steps.
7. **Deploy:** Deploy the final, static sparse merged model for zero-overhead multi-task inference.
