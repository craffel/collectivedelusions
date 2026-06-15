# Paper Summary: FoldMerge (Neural Origami)

## 1. Overview and Core Concept
The paper proposes **FoldMerge (Neural Origami)**, a novel and exploratory framework for multi-task model merging that departs from the traditional linear Euclidean paradigm. Instead of performing linear parameter interpolation (as in Task Arithmetic, Ties-Merging, or SyMerge) or rigid Riemannian manifold projections (as in OrthoMerge), FoldMerge treats model merging as a **non-linear weight-space coordinate warping process**. 

By training a highly expressive, differentiable coordinate transformation network $g_\phi: \mathbb{R}^d \to \mathbb{R}^d$ (parameterized using a sequence of RealNVP normalizing flow affine coupling layers), the method maps disjoint task-specific parameter spaces into a latent, shared coordinate system called **"Origami Space."** In Origami Space, the separate low-loss basins of attraction are aligned. The task parameters are combined in this latent space, and the merged coordinates are decoded back to the original parameter space via the analytical inverse diffeomorphism $g_\phi^{-1}$.

## 2. Key Mathematical Formulations
- **Forward Coordinate Transformation:**
  For each task expert $k$, the weights $\theta_k$ are mapped into Origami Space via:
  $$z_k = g_\phi(\theta_k)$$
  Similarly, the pre-trained base model weights $\theta_{base}$ are mapped:
  $$z_{base} = g_\phi(\theta_{base})$$

- **Origami Space Merging Formulations:**
  The paper explores three alternative formulations for coordinate combination in Origami Space:
  1. **Absolute Additive (Default Exploratory Baseline):**
     $$\bar{z} = 1.0 \cdot z_{base} + \sum_{k=1}^K \lambda_k z_k$$
     where $\lambda_k \geq 0$ are the adaptive task-specific merging coefficients.
  2. **Barycentric Latent Merging (Scale-Preserving):**
     $$\bar{z} = \left(1.0 - \sum_{k=1}^K \lambda_k\right) z_{base} + \sum_{k=1}^K \lambda_k z_k$$
     which constrains the coordinates to lie on a convex simplex that preserves the original base model's energy scale.
  3. **Latent Task Vector Warping (Scale-Preserving):**
     Warping the task updates (task vectors $\tau_k = \theta_k - \theta_{base}$) directly instead of absolute parameter weights, bypassing base model scale distortion:
     $$\theta_{MTL} = \theta_{base} + g_\phi^{-1}\left( \sum_{k=1}^K \lambda_k g_\phi(\tau_k) \right)$$

- **Inverse Diffeomorphism Decoding:**
  For the default absolute additive formulation, the merged parameters are reconstructed via:
  $$\theta_{MTL}(\phi, \lambda) = g_\phi^{-1}(\bar{z})$$

- **Unsupervised Test-Time Optimization Loss:**
  The flow parameters $\phi$ and coefficients $\lambda$ are optimized at test-time on unlabeled downstream validation streams $\mathcal{X}^{te}_k$ using expert teacher predictions:
  $$\min_{\phi, \lambda} \mathcal{L}(\phi, \lambda) = \sum_{k=1}^K \mathbb{E}_{x \in \mathcal{X}^{te}_k} \left[ \mathcal{D}_{KL}\left( f(x; \theta_{MTL}(\phi, \lambda)) \parallel f(x; \theta_k) \right) \right] + \gamma \mathcal{R}(\phi)$$
  where $\mathcal{R}(\phi)$ is the implicit flow regularization penalty.

- **Implicit Flow Regularization:**
  To ensure smooth, volume-preserving transformations and prevent chaotic warping without the prohibitive $O(d^2)$ cost of computing Jacobian matrices, the authors apply parameter-wise $\ell_2$ weight decay on the MLP parameters of the normalizing flow:
  $$\mathcal{R}(\phi) = \sum_{p \in \phi} \|p\|_2^2$$
  This encourages the scale ($s_\phi$) and translation ($t_\phi$) networks to remain close to zero, anchoring $g_\phi$ near the identity mapping.

- **LoRA-Flow Parameterization:**
  To reduce the flow's parameter footprint, the authors parameterize the scale and translation networks' MLP layers using low-rank decompositions:
  $$W = W_0 + \frac{\alpha}{r} AB$$
  where $W_0$ is frozen and only low-rank matrices $A$ and $B$ are updated, compressing trainable parameters by $27\times$ (to 96K parameters).

## 3. Main Empirical Findings
- Evaluated on the standard **8-task Vision-Language ViT-B/32 benchmark**, targeting the visual projection matrix `model.visual.proj` ($768 \times 512 = 393,216$ parameters).
- **Primary Results:** The default absolute additive FoldMerge achieves an Average Accuracy of **89.76%**, performing on par with the highly competitive, state-of-the-art linear adaptive baseline **SyMerge** (89.74%). It achieves slight improvements over SyMerge on 5 out of 8 individual tasks (Stanford Cars, RESISC45, EuroSAT, GTSRB, DTD), which are primarily fine-grained or structured classification domains.
- **Scale-Preserving Formulations:**
  - **Latent Task Vector Warping** achieves **89.77%** average accuracy, establishing a new state-of-the-art on the benchmark.
  - **Barycentric Latent Merging** achieves **89.74%** average accuracy, matching SyMerge while preserving the energy scale in Origami Space.
- **LoRA-Flow Performance:**
  - Achieves **89.82%** Average Accuracy under the Latent Task Vector Warping formulation, outperforming full-rank flow (+0.05%) while compressing trainable parameters to just 96,256 weights.
- **Frozen Classifier Head Ablation:**
  To isolate representational alignment from classifier head adaptation (a potential confound), both methods are evaluated with frozen classification heads. Under this setting, FoldMerge (Ours) achieves **83.56%** ($83.5597\%$) vs. SyMerge's **83.56%** ($83.5572\%$), showing that the coordinate warp performs genuine, functional representation alignment matching or exceeding linear baselines.
- **Zero Inference Overhead:** Once optimized, the merged parameters are decoded via $g_\phi^{-1}$ and frozen, introducing **zero extra parameters or latency** during actual inference.
- **Theoretical Limitations:** The paper transparently discusses critical limitations: coordinate-dependence of coupling layers (violating permutation symmetry), the row-wise slicing heuristic (a weight-space "category error"), the classifier head training confound, and parameter/computational overhead.
