# Idea Proposal: Riemannian Curvature-Regularized Test-Time Model Merging (RCR-Merge)

## 1. Persona Alignment
As **The Theorist**, I hold that empirical success in model merging is meaningless without an analytical, mathematically rigorous foundation. Direct weight addition of independently fine-tuned models is a crude heuristic that ignores the complex non-linear geometry of deep network loss landscapes. To resolve the **Overfitting-Optimizer Paradox** and **degenerate entropy minimization** during test-time adaptation, we must ground our regularizer in the second-order curvature properties of the loss landscape. 

**RCR-Merge** implements this philosophy. Rather than assuming a flat parameter space, we model the network as a Riemannian manifold where distance is scaled by the **Fisher Information Matrix (FIM)** (representing the local Hessian curvature). By weighting the spatial total variation of merging coefficients across depth with the geometric mean of layer-wise base curvatures, we mathematically restrict coefficient variations in sharp, sensitive regions of the landscape while preserving specialization capacity in flat basins. This brings formal, curvature-based guarantees to test-time adaptation.

## 2. Core Techniques
RCR-Merge introduces the following core algorithms and mechanisms:
1. **Empirical Base Curvature Estimation:** Prior to merging, we compute the trace of the diagonal Fisher Information Matrix (FIM) for each layer $l$ of the pre-trained base model $\theta_0$ using a tiny set of calibration inputs. This acts as our localized second-order curvature metric $c_l$, measuring the sensitivity of layer $l$ to parameter shifts.
2. **Riemannian Curvature-Weighted Total Variation Regularization (RCR-TV):** During unsupervised test-time adaptation, we apply a novel spatial regularizer to the layer-wise merging coefficients $\boldsymbol{\lambda} \in \mathbb{R}^{K \times L}$. Unlike standard Total Variation (TV) penalties, RCR-TV dynamically scales the penalty on coefficient discrepancies between adjacent layers $l$ and $l-1$ using the geometric mean of their curvatures, $\sqrt{c_l c_{l-1}}$.
3. **Curvature-Guided Coefficient Adaptation:** We optimize the coefficients online or offline using gradient-based (e.g., Adam) or derivative-free (e.g., Nelder-Mead) methods to minimize predicted Shannon entropy under our curvature constraint.

### Key Citations & Foundations:
- **Task Vectors & Arithmetic:** Ilharco et al. (2023), *Editing models with task arithmetic*.
- **TIES-Merging:** Yadav et al. (2023), *TIES-Merging: Resolving Interference When Merging Models*.
- **AdaMerging:** Liang et al. (2024), *AdaMerging: Adaptive Model Merging for Multi-Task Learning*.
- **Fisher/Hessian Curvature in Merging:** Matena & Raffel (2021), *Merging Models via Fisher Information*.
- **PolyMerge:** Anonymous (2025), *PolyMerge and SplineMerge: Projecting Parameter Space Merging into Continuous Subspaces*.

## 3. Mathematical Formulation

Let $\theta_0 \in \mathbb{R}^D$ represent the parameters of the shared pre-trained base model. Let $v_k = \theta_k - \theta_0$ represent the task vector for expert model $k \in \{1, \dots, K\}$, where each expert is fine-tuned from $\theta_0$.
We partition the parameter space into $L$ layer-wise blocks. Let $\theta_0^{(l)}$ and $v_{k}^{(l)}$ denote the base weights and the task vector of expert $k$ at layer $l \in \{1, \dots, L\}$.

### Step 1: Base Curvature Estimation
For each layer $l$, the local curvature $c_l \in \mathbb{R}$ is estimated as the mean diagonal element of the Fisher Information Matrix (FIM) of the pre-trained base model $\theta_0^{(l)}$ over a tiny unlabeled calibration batch $D_{\text{cal}}$:
$$c_l = \frac{1}{|D_{\text{cal}}|} \sum_{x \in D_{\text{cal}}} \mathbb{E}_{y \sim p_{\theta_0}(y|x)} \left[ \frac{1}{d_l} \|\nabla_{\theta_0^{(l)}} \log p_{\theta_0}(y|x)\|_2^2 \right]$$
where $d_l$ represents the number of parameters in layer $l$. This computation is performed exactly once prior to the optimization phase, requiring negligible computational cost.

### Step 2: Merged Representation and TTA Loss
The dynamically merged model weights at layer $l$ are parameterized by $\boldsymbol{\lambda} \in \mathbb{R}^{K \times L}$:
$$\theta_{\text{merged}}^{(l)}(\boldsymbol{\lambda}) = \theta_0^{(l)} + \sum_{k=1}^K \lambda_{k, l} v_{k}^{(l)}$$
During test-time adaptation, we minimize the Shannon entropy of the prediction probability distribution $f(x; \theta_{\text{merged}}(\boldsymbol{\lambda}))$ over an unlabeled adaptation batch $B$:
$$\mathcal{L}_{\text{TTA}}(\boldsymbol{\lambda}) = -\frac{1}{|B|} \sum_{x \in B} \sum_{c=1}^C f(x; \theta_{\text{merged}}(\boldsymbol{\lambda}))_c \log \left( f(x; \theta_{\text{merged}}(\boldsymbol{\lambda}))_c + \epsilon \right)$$
where $\epsilon = 10^{-8}$ is a numerical stabilizer.

### Step 3: Riemannian Curvature-Weighted Total Variation (RCR-TV)
To filter out high-frequency transductive noise and prevent degenerate constant-prediction states, we add our proposed second-order regularizer:
$$\mathcal{R}_{\text{curv}}(\boldsymbol{\lambda}) = \sum_{k=1}^K \sum_{l=2}^{L} \sqrt{c_l c_{l-1}} (\lambda_{k, l} - \lambda_{k, l-1})^2$$
The joint optimization objective is defined as:
$$\mathcal{L}_{\text{joint}}(\boldsymbol{\lambda}) = \mathcal{L}_{\text{TTA}}(\boldsymbol{\lambda}) + \beta \mathcal{R}_{\text{curv}}(\boldsymbol{\lambda})$$
where $\beta > 0$ is the regularization strength parameter.

### Proposition (Theoretical Guarantee against Degenerate States):
In a highly sensitive layer $l$ (large curvature $c_l \gg 0$), any sharp, discontinuous jump in the merging coefficients (which is required to disrupt functional representations and trigger degenerate constant-prediction states) will yield an extremely large spatial penalty under $\mathcal{R}_{\text{curv}}(\boldsymbol{\lambda})$. Specifically, if the optimizer attempts to set $\lambda_{k, l} \approx 0$ to mute a sensitive layer while boosting $\lambda_{k, l-1}$ to saturate logits, the local penalty is scaled by $\sqrt{c_l c_{l-1}} \gg 0$, mathematically blocking the optimization trajectory from entering these degenerate basins. Thus, RCR-TV acts as an analytical barrier protecting representation integrity.

## 4. Architecture Specifications
RCR-Merge is designed to scale across any deep architecture. For our core evaluations:
- **Unified Backbone:** We utilize the **ViT-B-32** vision transformer (or a compact **vit_tiny_patch16_224**).
- **Network Depth:** $L = 12$ transformer blocks (which translates to $12$ layer-wise blocks, or $L = 48$ individual parameter projection matrices including query, key, value, and projection layers).
- **Input Representations:** High-dimensional image features processed by the patch embedding layers, flowing through intermediate transformer representations.
- **Output Representations:** Vocabulary-level logits or $C$-class classification probabilities.
- **Learnable Parameter Space:** $\boldsymbol{\lambda} \in \mathbb{R}^{K \times L}$, initialized to uniform scaling $\lambda_{k, l} = 0.3$. For $K=4$ tasks and $L=12$ layers, the search space is $48$-dimensional, which we regularize down to a smooth Riemannian manifold of active states.

## 5. Baselines
We perform symmetric hyperparameter tuning against the following representative baselines:
1. **Flat Task Arithmetic (TA):** Linear addition with a uniform scalar coefficient $\lambda_k$ across all layers ($s = 100\%$, zero layer specialization).
2. **TIES-Merging:** The standard TRIM-ELECT-SIGN-MERGE heuristic protocol. This baseline evaluates if hard sign-consensus and coordinate zeroing-out are useful compared to smooth curvature-guided addition.
3. **DARE-TA:** Delta-dropout sparsified task arithmetic with scale-preserving correction.
4. **Unconstrained AdaMerging:** Spatial coefficient adaptation via entropy minimization without regularization, serving as the primary victim of the Overfitting-Optimizer Paradox and degenerate state-fusing.
5. **PolyMerge:** Hard continuous polynomial subspace constraint ($d \in \{1, 2, 3\}$), evaluating whether soft curvature-weighted regularization outperforms rigid polynomial trajectories.

## 6. Step-by-Step Interaction
1. **Pre-computation (Offline):**
   - Extract task vectors $v_k = \theta_k - \theta_0$ for each expert model.
   - Run a single forward pass on the pre-trained base model $\theta_0$ using a small calibration batch of 64 images to compute the diagonal Fisher Information norms across all $L$ layers. Normalize these values to obtain the localized curvature weights $\{c_l\}_{l=1}^L$.
2. **Coefficient Initialization:**
   - Initialize the merging coefficients to $\lambda_{k, l} = 0.3$ across all tasks and layers.
3. **Test-Time Adaptation Iterations:**
   - For each incoming adaptation step with batch $B$:
     - Build the merged parameter weights: $\theta_{\text{merged}}^{(l)}(\boldsymbol{\lambda}) = \theta_0^{(l)} + \sum_k \lambda_{k, l} v_k^{(l)}$.
     - Execute the forward pass on the batch $B$ using the merged weights to compute predicted class probabilities.
     - Compute the entropy of the predictions, $\mathcal{L}_{\text{TTA}}(\boldsymbol{\lambda})$.
     - Compute the Riemannian spatial path penalty, $\mathcal{R}_{\text{curv}}(\boldsymbol{\lambda})$.
     - Compute joint loss $\mathcal{L}_{\text{joint}}(\boldsymbol{\lambda}) = \mathcal{L}_{\text{TTA}}(\boldsymbol{\lambda}) + \beta \mathcal{R}_{\text{curv}}(\boldsymbol{\lambda})$.
     - Compute the gradients $\nabla_{\boldsymbol{\lambda}} \mathcal{L}_{\text{joint}}(\boldsymbol{\lambda})$ via backpropagation.
     - Update the coefficients $\boldsymbol{\lambda}$ using the Adam optimizer.
4. **Inference (Deployment):**
   - Freeze the optimized coefficients $\boldsymbol{\lambda}^*$, construct the final static merged weights, and deploy the single unified model with zero test-time computational overhead.
