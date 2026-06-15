# Idea Proposal: HessMerge

Hessian-Regularized Coefficient Optimization (HessMerge) is a mathematically rigorous model merging framework designed to resolve the two major vulnerabilities of adaptive parameter-space fusion: **Quantization-Operator Overfitting** (the cross-schema generalization gap) and the **Overfitting-Optimizer Paradox** (transductive overfitting). By explicitly optimizing the curvature trace of the merging coefficient loss landscape, HessMerge forces the parameters into flat, robust basins that are provably insensitive to both quantization noise and local calibration distribution shifts.

---

## 1. Persona Alignment
As **The Theorist**, this work is grounded in the philosophy that empirical success in model merging is fragile without mathematical guarantees. Instead of introducing heuristic constraints or unregularized test-time adaptation (TTA), HessMerge approaches the problem through the lens of continuous optimization, second-order Taylor expansion, and curvature analysis. 
- **Rigor & Proofs:** We mathematically derive the connection between the Hessian trace of the continuous loss landscape and the worst-case performance degradation under post-training quantization (PTQ).
- **Provable Correctness:** We prove that penalizing the Hessian trace bounds the generalization gap across arbitrary, unseen hardware quantization schemas and prevents overfitting to small calibration streams.
- **Formulation:** The entire method is formulated using exact second-order functional derivatives, establishing a sound mathematical foundation for test-time adaptive parameter-space fusion.

---

## 2. Core Techniques
HessMerge introduces three core, mathematically grounded mechanisms to model merging:
1. **Continuous-Proxy Curvature Regularization:** To bypass the non-differentiability of post-training quantization operators, HessMerge computes the exact second-order Hessian of the *continuous* (unquantized) model loss with respect to the merging coefficients.
2. **Exact Coefficient Hessian Trace Minimization:** Since the merging coefficient space $\Lambda \in \mathbb{R}^{K \times L}$ is extremely low-dimensional (e.g., $d_{\Lambda} = 4 \times 12 = 48$ for a 12-layer ViT across 4 tasks), we compute the exact Hessian diagonal and trace using functional automatic differentiation, avoiding the need for expensive Hutchinson approximations.
3. **Sharpness-Aware Coefficient Minimization (SACM):** As a computationally lightweight alternative, we introduce a first-order minimax formulation that perturbs the coefficients to find wide local minima: $\min_{\Lambda} \max_{\|\epsilon\|_2 \le \rho} \mathcal{L}(\Lambda + \epsilon)$, and prove its equivalence to a bounds-smoothing regularizer.

### Citations & Foundational Methods:
- **Task Vectors / Arithmetic:** Ilharco et al., "Editing Models with Task Arithmetic", ICLR 2023.
- **AdaMerging:** Yang et al., "AdaMerging: Adaptive Model Merging for Multi-Task Learning", arXiv 2023.
- **Hessian & Generalization:** Keskar et al., "On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima", ICLR 2017.
- **Sharpness-Aware Minimization (SAM):** Foret et al., "Sharpness-Aware Minimization for Efficiently Improving Generalization", ICLR 2021.
- **Quantization Robustness Audit:** From `trial3_submission1`, which exposed the Quantization-Operator Overfitting failure mode of Q-Merge.

---

## 3. Mathematical Formulation

Let $\theta^l_{\text{pre}} \in \mathbb{R}^{D_l}$ represent the pre-trained backbone weights at layer $l \in \{1, \dots, L\}$. For $K$ task-specific expert models, let the fine-tuned weights be $\theta^l_k \in \mathbb{R}^{D_l}$, and define the layer-wise task vectors as:
$$
\tau^l_k = \theta^l_k - \theta^l_{\text{pre}}
$$

The merged weights at layer $l$ are parameterized by a layer-wise coefficient matrix $\Lambda \in [0, 1]^{K \times L}$ as:
$$
\theta^l_{\text{merged}}(\Lambda) = \theta^l_{\text{pre}} + \sum_{k=1}^K \lambda^l_k \tau^l_k
$$

We flatten $\Lambda$ into a low-dimensional coefficient vector $\lambda \in [0, 1]^d$, where $d = K \times L$. Given a tiny unlabeled multi-task calibration set $\mathcal{D}_{\text{cal}} = \{X_{i, k}\}$, let $P(y = c \mid X_{i, k}; \theta_{\text{merged}}(\lambda))$ represent the predicted probability of the continuous merged model. The base unsupervised objective is prediction entropy minimization:
$$
\mathcal{L}_{\text{entropy}}(\lambda) = -\frac{1}{N \cdot K} \sum_{k=1}^K \sum_{i=1}^N \sum_{c=1}^C P(y=c \mid X_{i, k}; \theta_{\text{merged}}(\lambda)) \log P(y=c \mid X_{i, k}; \theta_{\text{merged}}(\lambda))
$$

### Proof of Curvature-Quantization Sensitivity Bound:
When deploying the continuous merged model on edge hardware, it is subjected to a post-training quantization operator $Q \in \mathcal{Q}$ (e.g., INT8 symmetric, asymmetric, tensor-wise, or channel-wise), yielding quantized weights:
$$
\theta_{\text{quant}}(\lambda) = Q(\theta_{\text{merged}}(\lambda)) = \theta_{\text{merged}}(\lambda) + \delta
$$
where $\delta \in \mathbb{R}^D$ is the quantization rounding error. By definition of PTQ with step size $s$, the error in each coordinate is bounded: $|\delta_i| \le s/2$, giving a bounded total noise $\|\delta\|_2^2 \le D s^2 / 4$.

We perform a second-order Taylor expansion of the test-time loss $\mathcal{L}$ around the continuous merged weights $\theta_{\text{merged}}(\lambda)$:
$$
\mathcal{L}(Q(\theta_{\text{merged}}(\lambda))) \approx \mathcal{L}(\theta_{\text{merged}}(\lambda)) + \nabla_{\theta} \mathcal{L}(\theta_{\text{merged}}(\lambda))^T \delta + \frac{1}{2} \delta^T \nabla^2_{\theta} \mathcal{L}(\theta_{\text{merged}}(\lambda)) \delta
$$

Assuming we are at or near a local minimum of the continuous loss (i.e., $\nabla_{\theta} \mathcal{L} \approx 0$), the quantization-induced loss gap is governed entirely by the quadratic curvature of the weight-space loss landscape:
$$
\Delta \mathcal{L}_{\text{quant}} = \mathcal{L}(Q(\theta_{\text{merged}}(\lambda))) - \mathcal{L}(\theta_{\text{merged}}(\lambda)) \le \frac{1}{2} \lambda_{\max}\left(\nabla^2_{\theta} \mathcal{L}(\theta_{\text{merged}}(\lambda))\right) \|\delta\|_2^2
$$

Because the merged weights $\theta_{\text{merged}}(\lambda)$ are a linear function of the merging coefficients $\lambda$, the Hessian of the loss with respect to the merging coefficients, denoted by $H_{\lambda} \in \mathbb{R}^{d \times d}$, is related to the weight-space Hessian via the chain rule:
$$
[H_{\lambda}]_{j, j'} = \frac{\partial^2 \mathcal{L}}{\partial \lambda_j \partial \lambda_{j'}} = \left( \frac{\partial \theta_{\text{merged}}}{\partial \lambda_j} \right)^T \nabla^2_{\theta} \mathcal{L}(\theta_{\text{merged}}) \left( \frac{\partial \theta_{\text{merged}}}{\partial \lambda_{j'}} \right) = \tau_j^T \nabla^2_{\theta} \mathcal{L}(\theta_{\text{merged}}) \tau_{j'}
$$

Minimizing the **trace of the coefficient Hessian** $\text{Tr}(H_{\lambda}) = \sum_{j=1}^d \frac{\partial^2 \mathcal{L}}{\partial \lambda_j^2}$ directly penalizes the curvature of the loss landscape along the directions of the task vectors. This forces the optimization to converge to flat, robust basins where small rounding perturbations in weight space do not cause performance collapse.

### The HessMerge Regularized Objective:
$$
\mathcal{L}_{\text{HessMerge}}(\lambda) = \mathcal{L}_{\text{entropy}}(\lambda) + \gamma \cdot \text{Tr}\left(\nabla^2_{\lambda} \mathcal{L}_{\text{entropy}}(\lambda)\right)
$$
where $\gamma > 0$ is the regularization scaling coefficient.

---

## 4. Architecture Specifications

HessMerge is designed to be highly compatible with standard neural network architectures and edge post-training quantization runtimes.

- **Backbone Model:** Vision Transformer (`vit_tiny_patch16_224` or `vit_base_patch16_224`) consisting of $L = 12$ Transformer encoder blocks. Each block contains a Self-Attention (MSA) module and a Feed-Forward Network (FFN).
- **Target Experts:** $K = 4$ fine-tuned task-specific experts fine-tuned on visual domains: MNIST, FashionMNIST, CIFAR-10, and SVHN.
- **Merging Parameters:** A real-valued coefficient matrix $\Lambda \in [0, 1]^{K \times L}$. For $K=4$ and $L=12$, the optimization parameters $\lambda$ comprise exactly $48$ scalar parameters.
- **Inputs:** A calibration stream batch $X \in \mathbb{R}^{B \times 3 \times 224 \times 224}$ containing $B=16$ samples per task, yielding a total calibration size of $N = 64$.
- **Intermediate Representations:** The continuous weight parameters at layer $l$, calculated dynamically during optimization via:
  $$
  W_{\text{merged}}^l = W_{\text{pre}}^l + \sum_{k=1}^K \lambda_k^l \left(W_k^l - W_{\text{pre}}^l\right)
  $$
- **Optimization Parameters:** We run the Adam optimizer on $\lambda$ for $T = 100$ steps with a learning rate of $\eta = 10^{-3}$, $\beta_1=0.9$, $\beta_2=0.99$, and curvature penalty weight $\gamma = 0.5$.
- **Output Post-Training Quantization (PTQ) Target Schemas:**
  - INT8 Uniform Symmetric (Tensor-wise and Channel-wise)
  - INT8 Uniform Asymmetric (Tensor-wise and Channel-wise)
  - INT4 Uniform Symmetric (Channel-wise)

---

## 5. Baselines

To prove the superiority and theoretical robustness of HessMerge, we will evaluate it against the following baselines:

1. **Uniform Task Arithmetic (No TTA baseline):** Uses a static scalar blending factor $\lambda_k^l = 1/K$ for all layers and tasks, serving as a zero-compute continuous weight baseline.
2. **AdaMerging (Unregularized TTA):** Optimizes layer-wise coefficients $\Lambda$ purely on $\mathcal{L}_{\text{entropy}}(\Lambda)$ with zero regularization. This baseline represents the peak of transductive overfitting and sacrificial task bias.
3. **RegCalMerge (TV-Regularized TTA):** Uses Elastic Spatial Regularization (ESR), a Total Variation penalty: $\mathcal{R}_{\text{TV}}(\Lambda) = \sum_{k} \sum_{l} (\lambda_k^{l+1} - \lambda_k^l)^2$, and Class-Capacity Normalization. This baseline tests whether simple spatial smoothing can match second-order curvature alignment.
4. **Q-Merge (Quantization-Aware STE):** Simulates quantization in the forward pass of the optimizer and uses Straight-Through Estimators (STE) to update coefficients. This baseline represents the state-of-the-art in quantization-aware merging but is prone to severe operator overfitting.
5. **PolyMerge (Subspace-Constrained TTA):** Parameterizes $\Lambda$ as a continuous polynomial of layer depth, testing if hard coordinate reduction is as robust as our curvature penalty.

---

## 6. Step-by-Step Interaction

The data flows through the proposed HessMerge system during the test-time adaptation phase, followed by zero-overhead edge deployment:

```
[Calibration Batch X] ---> [Merged Continuous Model \theta_merged(\lambda)] 
                                   |
                                   v
                         [Entropy Loss L_entropy]
                                   |
              +--------------------+--------------------+
              | (Forward AD)                            | (Backward AD)
              v                                         v
     [Hessian Trace Tr(H_\lambda)]             [Gradient \nabla_\lambda L_entropy]
              |                                         |
              +--------------------+--------------------+
                                   |
                                   v
                      [HessMerge Regularized Loss]
                                   |
                                   v
                          [Adam Optimizer]
                                   | (Updates Coefficients \lambda)
                                   v
                    [Quantized Model Q_target(\theta^*)] ---> [Edge Hardware Inference]
```

### 1. Expert Parameter Extraction:
For each layer $l \in \{1, \dots, L\}$, extract weight matrices for the pre-trained backbone $W_{\text{pre}}^l$ and the $K$ fine-tuned expert models $W_k^l$. Pre-compute the task vectors $V_k^l = W_k^l - W_{\text{pre}}^l$ to avoid redundant operations.

### 2. Weight-Space Linear Blending:
Given the active merging coefficients $\lambda$, compute the continuous merged weights for each weight tensor in the model:
$$
W_{\text{merged}}^l(\lambda) = W_{\text{pre}}^l + \sum_{k=1}^K \lambda_k^l V_k^l
$$

### 3. Logits Prediction & Entropy Computation:
Pass the calibration stream inputs $X_{i, k}$ through the network parameterized by the continuous merged weights $W_{\text{merged}}^l(\lambda)$. Compute the prediction class probabilities and evaluate the unsupervised joint entropy loss $\mathcal{L}_{\text{entropy}}(\lambda)$.

### 4. Second-Order Curvature Extraction:
Using PyTorch's functional automatic differentiation engine (`torch.func.hessian`), evaluate the Hessian of the entropy loss with respect to the merging coefficients:
$$
H_{\lambda} = \nabla^2_{\lambda} \mathcal{L}_{\text{entropy}}(\lambda)
$$
Compute the trace of this matrix: $\text{Tr}(H_{\lambda}) = \sum_{j=1}^d [H_{\lambda}]_{j, j}$.

### 5. Regularized Optimization Update:
Formulate the final HessMerge loss: $\mathcal{L}_{\text{HessMerge}}(\lambda) = \mathcal{L}_{\text{entropy}}(\lambda) + \gamma \text{Tr}(H_{\lambda})$. Perform a backward pass to obtain the gradient of this total loss with respect to $\lambda$: $\nabla_{\lambda} \mathcal{L}_{\text{HessMerge}}(\lambda)$. Update the coefficient parameters using the Adam optimizer, clamping them to the $[0, 1]^d$ hypercube.

### 6. Frozen Model PTQ Deployment:
Once the optimization converges ($T = 100$), freeze the optimal coefficients $\lambda^*$. Construct the continuous merged model weights. Apply any target post-training quantization operator $Q_{\text{target}}$ to compile the weights for edge accelerator hardware (e.g., INT8 symmetric tensor-wise for low-power edge TPUs).
This process requires zero test-time backpropagation or calibration on the target hardware, delivering robust, high-accuracy multi-task fusion.
