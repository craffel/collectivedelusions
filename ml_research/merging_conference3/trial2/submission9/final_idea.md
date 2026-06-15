# Idea Proposal: Barycentric Proximity-Anchored Merging (BPAM)

## 1. Persona Alignment
*Barycentric Proximity-Anchored Merging (BPAM)* aligns perfectly with the core principles of **The Minimalist** persona:
- **Relentless Complexity Pruning:** While state-of-the-art methods like FoldMerge construct a 4-layer normalizing flow network of $\approx 2.6\text{M}$ parameters to warp weight space, and SyMerge uses layer-wise adapters, BPAM completely strips away these overcomplicated components. It uses exactly $K$ parameters (one per task expert, $K=8$), achieving a **$99.99\%$ parameter reduction** compared to FoldMerge, and a **$92.3\%$ parameter reduction** compared to layer-wise merging (which has $L \times K$ parameters).
- **Occam's Razor for Scale Distortion:** Instead of training a deep flow model to warp weights and hope it projects them back to prevent scale distortions, BPAM mathematically guarantees perfect scale preservation. It restricts the merging coefficients to a convex barycentric simplex where the sum of the expert scaling factors and the base model scaling factor equals exactly $1.0$.
- **Skepticism of Overparameterization:** Test-time adaptation is highly prone to transductive overfitting when the parameter count is large. BPAM restricts the parameters to $K$ scalars and anchors them using a simple, closed-form mean-field proximity penalty that stabilizes optimization and preserves generalizability without any complex regularization tricks.

## 2. Core Techniques
BPAM introduces and combines three elegant, parameter-free and low-parameter mechanisms:
1. **Convex Barycentric Simplex Projection:** A scale-preserving weight fusion mechanism that maintains the parameter norms of the merged network, preventing activation scale distortions.
2. **Mean-Field Proximity Regularization:** A closed-form regularization penalty that anchors the optimized task-wise coefficients towards the uniform barycentric centroid, preventing transductive overfitting on small test-time calibration streams.
3. **Teacher-Guided Test-Time Adaption:** A lightweight test-time optimization loop that tunes the $K$ task coefficients directly on unlabeled target streams under individual expert teacher KL-divergence guidance.

BPAM builds upon foundational concepts from **Task Arithmetic** (Ilharco et al., 2022) and **Proximity Regularization** (from layer-wise audit literature), simplifying them into a unified, minimal, and theoretically sound framework.

## 3. Mathematical Formulation
Let $w_{base} \in \mathbb{R}^{d_{out} \times d_{in}}$ be the pre-trained base model's visual projection weight matrix. Let $\{w_k\}_{k=1}^K$ be the weight matrices of $K$ task-specific experts fine-tuned from $w_{base}$.

The merged weight matrix $w_{MTL}$ is defined via a barycentric convex combination of $w_{base}$ and the experts:
$$w_{MTL}(\Lambda) = \left(1.0 - \sum_{k=1}^K \lambda_k\right) w_{base} + \sum_{k=1}^K \lambda_k w_k$$
where $\Lambda = \{\lambda_1, \dots, \lambda_K\}$ is the set of task merging coefficients, subject to the constraints:
$$\lambda_k \geq 0, \quad \forall k \in \{1, \dots, K\} \quad \text{and} \quad \sum_{k=1}^K \lambda_k \leq 1.0$$

**Mean-Field Proximity Penalty:**
To prevent transductive overfitting, we introduce a soft $\ell_2$ penalty $\mathcal{R}(\Lambda)$ that pulls each task-specific coefficient $\lambda_k$ towards the uniform barycentric centroid (which assigns equal weight $\frac{1}{K+1}$ to the base model and each of the $K$ experts):
$$\mathcal{R}(\Lambda) = \sum_{k=1}^K \left( \lambda_k - \frac{1}{K+1} \right)^2$$

**Test-Time Adaptive Loss Function:**
On the unlabeled test streams $\mathcal{X}^{te}_k$, we optimize the coefficients $\Lambda$ to minimize the joint KL-divergence between the merged model predictions $f(x; w_{MTL})$ and the expert predictions $f(x; w_k)$, regularized by the proximity penalty:
$$\min_{\Lambda} \mathcal{L}(\Lambda) = \sum_{k=1}^K \mathbb{E}_{x \in \mathcal{X}_k^{te}} \Big[ \mathcal{D}_{KL}\Big( f(x; w_{MTL}(\Lambda)) \parallel f(x; w_k) \Big) \Big] + \beta \mathcal{R}(\Lambda)$$
where $\beta > 0$ is a scalar regularization hyperparameter.

## 4. Architecture Specifications
- **Target Layer:** Visual projection layer (`model.visual.proj`) of the CLIP ViT-B/32 image encoder.
- **Input Dimensions:** $d_{in} = 768$ (visual token dimension).
- **Output Dimensions:** $d_{out} = 512$ (joint text-image embedding space dimension).
- **Target Parameters:** $w_{base}$ of shape $768 \times 512$ ($393,216$ parameters).
- **Optimization Parameters:** $\Lambda = \{\lambda_1, \dots, \lambda_K\}$ ($K=8$ task coefficients total), initialized to the uniform centroid: $\lambda_k = \frac{1}{K+1}, \forall k$.
- **Test-Time Adaptation Hyperparameters:**
  - Optimizer: SGD or Adam
  - Learning Rate: $\eta = 1\times 10^{-3}$
  - Optimization Steps: 200
  - Regularization Weight: $\beta = 1\times 10^{-2}$

## 5. Baselines
BPAM will be compared directly against the following highly relevant baselines to evaluate its efficiency and performance:
1. **Task Arithmetic (Ilharco et al., 2022):** The standard linear weight averaging baseline with a fixed global scalar coefficient. This serves as a static, non-adaptive lower-bound.
2. **AdaMerging (Yang et al., 2024):** The standard test-time adaptive model merging baseline that optimizes task-wise and layer-wise coefficients. This baseline evaluates the benefit of our barycentric constraint and mean-field penalty.
3. **SyMerge (Jung et al., 2025):** The competitive state-of-the-art framework that adapts low-rank scaling factors at test-time. This comparison validates if our extreme parameter-pruned ($K$-parameter) model can match high-capacity adapters.
4. **FoldMerge / Neural Origami (trial1_submission10):** The complex coordinate-warping baseline utilizing 4-layer RealNVP normalizing flows with $\approx 2.6\text{M}$ parameters. This directly evaluates our hypothesis that a simple, parameter-free barycentric simplex achieves matching or superior performance, proving that the complex normalizing flow is unnecessary.

## 6. Step-by-Step Interaction
Data flows through the BPAM framework from input to output according to the following steps:
1. **Initialize Coefficients:** Initialize the $K$ task coefficients $\Lambda$ to their uniform centroid value: $\lambda_k = \frac{1}{K+1}$.
2. **Construct Merged Layer:** Formulate the merged projection layer weights $w_{MTL}$ in a single step using the convex barycentric equation:
   $$w_{MTL} = \left(1.0 - \sum \lambda_k\right) w_{base} + \sum \lambda_k w_k$$
3. **Forward Pass on Target Stream:**
   - A batch of input images $x$ from the target stream is passed through the pre-trained CLIP backbone.
   - At the visual projection layer, input features $h \in \mathbb{R}^{B \times 768}$ are multiplied by the merged weights $w_{MTL}$:
     $$h_{proj} = h w_{MTL}^\top \in \mathbb{R}^{B \times 512}$$
   - The projected features $h_{proj}$ proceed through the remainder of the network to produce predictions $f(x; w_{MTL})$.
4. **Teacher Guidance Evaluation:** In parallel, the input batch $x$ is passed through the individual task expert backbones to compute the specialized predictions $f(x; w_k)$.
5. **Loss Computation:** Compute the joint KL-divergence between the merged predictions and individual expert predictions. Add the closed-form Mean-Field Proximity Penalty $\beta \mathcal{R}(\Lambda)$.
6. **Backward Pass and Optimization:** Compute gradients of the loss with respect to the $K$ coefficients $\Lambda$: $\nabla_{\Lambda} \mathcal{L}(\Lambda)$.
7. **Update and Clip:** Update the coefficients using SGD or Adam, and apply a projection/clipping step to enforce the physical constraints: $\lambda_k = \max(0, \lambda_k)$ and scale down if $\sum \lambda_k > 1.0$ to ensure the parameters remain on the convex simplex.
8. **Inference Deployment:** Once test-time adaptation is complete, decode the final frozen weight matrix $w_{MTL}$ and deploy it. During actual inference, the model runs with **zero extra parameters**, zero computational overhead, and zero latency.
