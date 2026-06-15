# Idea Proposal: 2D-STEM (2D Spatio-Temporal Exponential Moving Average)

## 1. Persona Alignment
This proposal strongly adopts the philosophy of **The Minimalist** by applying Occam's razor to the highly convoluted field of stateful dynamic model merging. Recent state-of-the-art frameworks—such as ChemMerge (biochemical kinetics ODEs) and PAC-Kinetics (learning-theoretic state-space optimization)—rely on complex continuous-time metaphors, virtual step sizes, activation energies, and extensive optimization routines. 

By deconstructing these frameworks under a minimalist lens, we reveal that they are mathematically equivalent to localized Exponential Moving Average (EMA) smoothing. Symmetrically, Momentum-Merge restricts EMA to the spatial (depth-wise) dimension, resetting at each sample, while PAC-Kinetics restricts it to the temporal (sample-wise) dimension, using a static routing weight across depth. 

**2D-STEM** completely prunes this artificial complexity. It introduces a training-free, single-parameter-controlled, 2D bilinear Spatio-Temporal Exponential Moving Average that filters noise across both depth and time simultaneously. It strips away all biochemical parameters, virtual integrations, and optimization loops, demonstrating that a simple, elegant 2-dimensional recurrence filter achieves superior routing stability with zero serving latency or training overhead.

## 2. Core Techniques
1. **Zero-Shot Layer-wise Centroid Calibration:** Extracted from SABLE and Momentum-Merge, we pre-compute task-specific centroids $\mu_k^{(l)}$ at each adapted layer $l$ using a tiny offline calibration split ($N_{\text{cal}} = 64$) to resolve semantic representational shift across backbone depth.
2. **2D Bilinear State Propagation (Spatio-Temporal EMA):** The routing state at time step $t$ and layer $l$ is recursively updated as a 2D bilinear combination of the state from the previous layer $s_t^{(l-1)}$ and the state from the previous sample at the same layer $s_{t-1}^{(l)}$, combined with the current layer similarity.
3. **Adaptive Temporal Gating (Inertia Damping):** To prevent temporal transition lag under abrupt task switches, we measure stream-level homogeneity via cosine similarity between consecutive early-layer coordinate projections, dynamically scaling down the temporal momentum when task switches are detected.
4. **Analytical Simplex Preservation:** We mathematically guarantee that the ensembling weights always reside on the probability simplex with zero computational projection overhead, simply by enforcing a linear inequality on the momentum coefficients: $\beta_{\text{depth}} + \beta_{\text{temp}, 0} \le 1$.

## 3. Mathematical Formulation

Let $w_k^{(l)}(t) \in [0, 1]$ denote the raw similarity-routing weight for task $k$ at adapted layer $l$ and sample time step $t$. We define the 2D Spatio-Temporal Exponential Moving Average (2D-STEM) recurrence for the final ensembling coefficients $\alpha_k^{(l)}(t)$ as:

$$\alpha_k^{(l)}(t) = \beta_{\text{depth}} \alpha_k^{(l-1)}(t) + \beta_{\text{temp}, t} \alpha_k^{(l)}(t-1) + \left(1 - \beta_{\text{depth}} - \beta_{\text{temp}, t}\right) w_k^{(l)}(t)$$

where:
- $\beta_{\text{depth}} \in [0, 1]$ is the constant spatial (depth-wise) smoothing coefficient.
- $\beta_{\text{temp}, t} \in [0, 1]$ is the dynamic temporal (sample-wise) smoothing coefficient at step $t$.
- $w_k^{(l)}(t)$ is the raw Softmax similarity-routing weight:

$$w_k^{(l)}(t) = \frac{\exp\left(S\left(h^{(l-1)}(t), \mu_k^{(l)}\right) / \tau\right)}{\sum_{j=1}^K \exp\left(S\left(h^{(l-1)}(t), \mu_j^{(l)}\right) / \tau\right)}$$

$$S\left(h^{(l-1)}(t), \mu_k^{(l)}\right) = \frac{h^{(l-1)}(t) \cdot \mu_k^{(l)}}{\left\|h^{(l-1)}(t)\right\|_2 \left\|\mu_k^{(l)}\right\|_2}$$

### Dynamic Temporal Gating
To eliminate transition lag under rapid switches, we compute the local stream similarity $Sim_t$ using early-layer coordinates (Layer $L_{\text{frozen}}$) between the current sample $t$ and previous sample $t-1$:

$$Sim_t = \frac{\mathbf{e}_t^T \mathbf{e}_{t-1}}{\|\mathbf{e}_t\|_2 \|\mathbf{e}_{t-1}\|_2 + \epsilon} \in [0, 1]$$

where $\mathbf{e}_t = \left[ S\left(h^{(L_{\text{frozen}})}(t), \mu_1^{(L_{\text{frozen}})}\right), \dots, S\left(h^{(L_{\text{frozen}})}(t), \mu_K^{(L_{\text{frozen}})}\right) \right]^T$ is the task similarity vector.
The temporal momentum is scaled on-the-fly:

$$\beta_{\text{temp}, t} = \beta_{\text{temp}, 0} \cdot Sim_t$$

where $\beta_{\text{temp}, 0} \in [0, 1]$ is the baseline temporal smoothing coefficient.

### Boundary and Initialization Conditions
- **Temporal Boundary ($t=1$):** At the start of serving, there is no history, so we set:
  $$\beta_{\text{temp}, 1} = 0$$
- **Depth Boundary ($l = L_{\text{frozen}} + 1$):** To prevent artificial damping in early layers, the spatial recurrence is initialized directly with the raw similarity weights:
  $$\alpha_k^{(L_{\text{frozen}})}(t) = w_k^{(L_{\text{frozen}} + 1)}(t)$$

### Simplex-Preserving Constraint
To guarantee $\sum_k \alpha_k^{(l)}(t) = 1$ and $\alpha_k^{(l)}(t) \in [0, 1]$ analytically without normalization projections:
$$\beta_{\text{depth}} + \beta_{\text{temp}, 0} \le 1, \quad \beta_{\text{depth}} \ge 0, \quad \beta_{\text{temp}, 0} \ge 0$$

## 4. Architecture Specifications
- **Backbone Network:** Simulated 14-layer deep Transformer (ViT-Tiny configuration) with representation dimension $D = 192$.
- **Frozen Feature Extractor:** First $L_{\text{frozen}} = 3$ layers.
- **Adapted Layers:** Layers $l \in [4, 14]$. target Query and Value matrices with LoRA rank $r = 8$.
- **Downstream Expert Pool:** $K = 4$ task experts (MNIST, Fashion-MNIST, CIFAR-10, SVHN).
- **Hyperparameters:**
  - Spatial momentum: $\beta_{\text{depth}} = 0.40$
  - Baseline temporal momentum: $\beta_{\text{temp}, 0} = 0.40$
  - Routing Softmax temperature: $\tau = 0.005$ or $\tau = 0.100$

## 5. Baselines
We compare 2D-STEM against the complete lineage of static and dynamic merging methods:
1. **Expert Ceiling (Oracle):** Pure executing of correct expert (100% accuracy, 0.0 Jitter).
2. **Uniform Merging (Static):** Static weight parameter average ($\alpha_k = 0.25$).
3. **SABLE (Stateless):** Nearest-centroid similarity routing with layer centroids (SABLE + Layer Centroids).
4. **ChemMerge (Stateful Biochemical):** Integrates non-equilibrium continuous kinetics across depth (ChemMerge + Layer Centroids).
5. **Momentum-Merge (Stateful Depth-only):** Spatial-only constant EMA across network depth.
6. **PAC-Kinetics (Stateful Temporal-only):** Stateful first-order recurrence optimized via PAC-Bayesian bound across samples.

## 6. Step-by-Step Interaction
1. **Input Arrival:** Sample $X_t$ arrives sequentially at serving step $t$ ($B=1$).
2. **Early Extraction:** The sample is propagated through the frozen shared extractor (Layers 1--3) to obtain activation $h^{(L_{\text{frozen}})}(t)$.
3. **Coordinate Scoring:** Compute similarity coordinate vector $\mathbf{e}_t$ at Layer $L_{\text{frozen}}$.
4. **Homogeneity Gating:** Compute stream-level similarity $Sim_t$ and dynamically scale temporal momentum $\beta_{\text{temp}, t} = \beta_{\text{temp}, 0} \cdot Sim_t$.
5. **Layer-wise Propagation:** For each adapted layer $l \in [4, 14]$:
   a. Compute raw cosine similarity of current activation $h^{(l-1)}(t)$ to layer-specific centroids $\mu_k^{(l)}$.
   b. Apply gated Softmax with temperature $\tau$ to obtain raw routing weights $w_k^{(l)}(t)$.
   c. Update ensembling weights $\alpha_k^{(l)}(t)$ via the 2D-STEM bilinear recurrence (combining previous layer $\alpha_k^{(l-1)}(t)$ and previous sample $\alpha_k^{(l)}(t-1)$).
   d. Blend intermediate LoRA expert activations in parallel using the updated $\alpha_k^{(l)}(t)$ coefficients:
      $$h^{(l)}(t) = h^{(l-1)}(t) W_{\text{base}}^{(l)} + \sum_{k=1}^K \alpha_k^{(l)}(t) \left( h^{(l-1)}(t) A_k^{(l)} B_k^{(l)} \right)$$
6. **Output Generation:** Deep blended features $h^{(L)}(t)$ are fed to classification heads to produce final predictions.
