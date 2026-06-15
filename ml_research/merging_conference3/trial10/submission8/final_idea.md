# Idea Proposal: Rademacher-Bounded Fourier Trajectory Merging (RB-FTM) for Spectral Regularization

## 1. Persona Alignment
This proposal directly aligns with the research philosophy of **The Theorist** by prioritizing mathematical safety and provable generalization guarantees over ad-hoc empirical ensembling heuristics. Adaptive model merging (specifically Test-Time Adaptation and Few-Shot Calibration) is traditionally formulated as the unconstrained optimization of independent coefficients $\alpha_k(l) \in [0, 1]$ across tasks $k$ and network layers $l$. This yields a high-capacity hypothesis class that overfits local stream noise or small calibration splits.

Rather than relying on empirical heuristics to tame this overfitting, we approach the problem through the lens of **Statistical Learning Theory**. We treat layer-wise ensembling coefficients across depth as a continuous global trajectory. By projecting this trajectory onto a low-frequency **Fourier (spectral) subspace** and deriving tight **Empirical Rademacher Complexity Bounds** for this trigonometric class, we provide a mathematically guaranteed route to limit the generalization gap. Our method, **Rademacher-Bounded Fourier Trajectory Merging (RB-FTM)**, introduces a spectral lasso ($L_1$) regularizer that directly bounds this complexity during calibration, enabling provably robust out-of-distribution generalization.

---

## 2. Core Techniques
RB-FTM introduces the following core techniques:
1.  **Fourier Trajectory Projection:** The high-dimensional ensembling coefficient matrix $\Lambda \in [0, 1]^{K \times L}$ is projected onto a low-frequency harmonic trigonometric subspace of cutoff frequency $F \ll L$. This acts as an analytical low-pass filter over layer depth, filtering out high-frequency parameter oscillations across sequential layers.
2.  **Trigonometric Generalization Bounding:** We derive the empirical Rademacher complexity of the Fourier trajectory hypothesis class $\mathcal{H}_F$. We prove that by restricting coefficients to a spectral cutoff $F$, the complexity scales with the number of sinusoids $2F+1$ rather than network depth $L$.
3.  **Spectral Lasso Regularization ($\|\Theta\|_1$):** We integrate an analytical spectral lasso penalty corresponding to the $L_1$ norm of the Fourier coefficients directly into the calibration objective. This penalty physically constrains the trajectory capacity, mapping a provable learning-theoretic bound to a practical optimization loss.
4.  **Boundary Runge's Phenomenon Mitigation:** Unlike polynomial trajectories (e.g., RBPM of degree $d \ge 3$) which suffer from severe runaway oscillations near the boundaries (first and last layers of deep backbones), our bounded sinusoids are naturally stable at $z \in \{0, 1\}$, stabilizing the crucial feature extraction and classification layers of the network.

---

## 3. Mathematical Formulation

Let $W_0^{(l)} \in \mathbb{R}^{D_l}$ represent the parameter weights of a pre-trained base model at layer $l \in \{0, \dots, L-1\}$, where $L$ is the total number of layers. Given $K$ task-specific expert models $\{W_k\}_{k=0}^{K-1}$ fine-tuned from $W_0$, the layer-wise task vectors are defined as $V_k^{(l)} = W_k^{(l)} - W_0^{(l)}$. The merged model weights at layer $l$ are constructed via:
$$W_{\text{merged}}^{(l)}(\Lambda) = W_0^{(l)} + \sum_{k=0}^{K-1} \alpha_k(l) V_k^{(l)}$$

### 3.1. Fourier Trajectory Representation
We parameterize the ensembling coefficient $\alpha_k(l)$ for task $k$ at layer $l$ using a finite Fourier series of normalized depth $z = \frac{l}{L-1} \in [0, 1]$:
$$\alpha_k(l; \theta_k) = \text{clip}\left( a_{k,0} + \sum_{f=1}^F \left( a_{k,f} \cos\left(2\pi f z\right) + b_{k,f} \sin\left(2\pi f z\right) \right), \,\, 0.0, \,\, 1.0 \right)$$
where:
-   $F \ge 1$ is the spectral cutoff frequency ($F \ll L$, typically $F=1$ or $F=2$).
-   $\theta_k = [a_{k,0}, a_{k,1}, b_{k,1}, \dots, a_{k,F}, b_{k,F}]^T \in \mathbb{R}^{2F+1}$ are the learned Fourier coefficients.
-   $\Theta = \{\theta_k\}_{k=0}^{K-1} \in \mathbb{R}^{K \times (2F+1)}$ denotes the global trajectory parameters.

### 3.2. Learning-Theoretic Complexity Guarantees
We define the Fourier trajectory hypothesis class $\mathcal{H}_F$ of spectral cutoff $F$ and bounded trajectory parameter norm as:
$$\mathcal{H}_F = \left\{ \alpha: [0, 1] \to [0, 1] \;\middle|\; \alpha(z) = a_0 + \sum_{f=1}^F \left( a_f \cos(2\pi f z) + b_f \sin(2\pi f z) \right), \;\; \|\theta\|_1 \le C_0 \right\}$$
where $\|\theta\|_1 = |a_0| + \sum_{f=1}^F (|a_f| + |b_f|)$.

**Theorem (Empirical Rademacher Complexity of Spectral Trajectories):**
Let $\mathcal{H}_F$ be the class of Fourier ensembling trajectories of cutoff frequency $F$ with $\ell_1$-bounded coefficients $\|\theta\|_1 \le C_0$. The empirical Rademacher complexity of $\mathcal{H}_F$ over the network depth coordinates of size $L$ is bounded by:
$$\widehat{\mathcal{R}}_L(\mathcal{H}_F) \le C_0 \sqrt{\frac{2 \ln(2F+1)}{L}}$$

**Proof:**
Any function $\alpha \in \mathcal{H}_F$ can be written as an inner product $\langle \theta, \Psi(z) \rangle$, where $\Psi(z) = [1, \cos(2\pi z), \sin(2\pi z), \dots, \cos(2\pi F z), \sin(2\pi F z)]^T \in \mathbb{R}^{2F+1}$. Since $\|\Psi(z)\|_\infty \le 1$ for all $z \in [0, 1]$, the coordinate vectors are bounded. The hypothesis class $\mathcal{H}_F$ is a subset of the convex hull of the base trigonometric functions scaled by $C_0$. Using Massart's Lemma for linear classes bounded over an $\ell_1$-ball of dimension $2F+1$, we obtain:
$$\widehat{\mathcal{R}}_L(\mathcal{H}_F) \le C_0 \sqrt{\frac{2 \ln(2F+1)}{L}}$$
This complexity decays with network depth $L$ and is strictly controlled by the spectral cutoff frequency $F$ rather than unconstrained layer fluctuations.

### 3.3. Optimization Objective
We formulate the spectral-regularized few-shot objective to optimize the Fourier parameters $\Theta$ on a tiny labeled calibration dataset $\mathcal{D}_{\text{cal}}$:
$$\min_{\Theta} \mathcal{L}(\Theta) = \mathcal{L}_{\text{CE}}(\mathcal{D}_{\text{cal}}; \Theta) + \gamma \sum_{k=0}^{K-1} \|\theta_k\|_1$$
where $\mathcal{L}_{\text{CE}}$ is the Cross-Entropy loss over the calibration samples, $\|\theta_k\|_1 = |a_{k,0}| + \sum_{f=1}^F (|a_{k,f}| + |b_{k,f}|)$, and $\gamma \ge 0$ is the spectral regularization coefficient.

---

## 4. Architecture Specifications

The system is evaluated on two distinct architectural scales:
1.  **Deep12LayerCNN Backbone:** 
    -   **Depth:** $L = 12$ layers (11 sequential convolutional blocks, each composed of Conv2d, BatchNorm2d, and ReLU, followed by a final task-specific Linear layer).
    -   **Hidden Dimensions:** Conv channel sizes scale sequentially from 32 to 128.
    -   **Inputs:** $32 \times 32$ 3-channel images (MNIST and FashionMNIST are upsampled and duplicated to 3 channels to match).
    -   **Outputs:** 10-class prediction logits.
2.  **CLIP ViT-B/16 Backbone:**
    -   **Depth:** $L = 13$ parameter groups (12 Transformer blocks and the final linear visual projection layer).
    -   **Hidden Dimensions:** Hidden state dimension $D = 768$.
    -   **Inputs:** $224 \times 224$ images.
    -   **Outputs:** Joint classification logits computed using cosine similarity against pre-trained text embeddings.

### Fourier Parameters Specification:
-   **Spectral Cutoff:** $F = 1$ (first harmonic) or $F = 2$ (second harmonic).
-   **Number of Parameters:** For $K = 4$ tasks:
    -   At $F=1$: $4 \times (2(1) + 1) = 12$ total continuous parameters.
    -   At $F=2$: $4 \times (2(2) + 1) = 20$ total continuous parameters.
-   **Parameter Initialization:** $a_{k,0}$ is initialized to the uniform baseline $1/K = 0.25$. All sine/cosine coefficients $a_{k,f}, b_{k,f}$ are initialized to $0.0$ to ensure optimization starts exactly at the robust uniform ensembling baseline.

---

## 5. Baselines
We evaluate RB-FTM against a comprehensive suite of static, coordinate-wise, and adaptive model-merging baselines:
1.  **Static Uniform Merging:** Sets ensembling coefficients statically to $1/K = 0.25$ for all layers, serving as the unoptimized control baseline.
2.  **Globally-Scaled Task Arithmetic ($d=0$):** Optimizes a single tuned continuous scalar ensembling coefficient per task ($\beta_k \in [0, 1]$) across all layers, equivalent to a Fourier cutoff frequency $F = 0$.
3.  **Offline Unconstrained Few-Shot Tuning:** Optimizes independent continuous layer-wise coefficients $\alpha_{k,l}$ directly on the few-shot calibration set, establishing the overparameterized overfitted baseline.
4.  **Rademacher-Bounded Polynomial Merging (RBPM) ($d=2$):** Constrains trajectories to quadratic polynomials. This acts as the direct coordinate-space trajectory competitor.
5.  **Online AdaMerging (Unconstrained):** Unsupervised online Test-Time Adaptation that optimizes $K \times L$ continuous parameters on unlabeled test streams by minimizing the prediction entropy.
6.  **TIES-Merging & DARE-Merging:** Prominent coordinate-wise parameter pruning and sign consensus heuristics, establishing standard static weight merging benchmarks.

---

## 6. Step-by-Step Interaction

During test-time inference or few-shot calibration, data flows through the proposed system as follows:

```
           [Input Data X] (Calibration split or Test stream)
                 │
                 ▼
     [Fourier Trajectory Module] (Computes layer coefficients)
                 │
                 ├─► For each task k and layer l:
                 │   z = l / (L - 1)
                 │   α_k(l) = clip( a_k,0 + Σ [a_k,f cos(2π f z) + b_k,f sin(2π f z)], 0.0, 1.0 )
                 │
                 ▼
     [Weight Assembly Phase] (Constructs merged weights per layer)
                 │
                 ├─► For each layer l:
                 │   W_merged(l) = W_0(l) + Σ_k α_k(l) * [W_k(l) - W_0(l)]
                 │
                 ▼
     [Feedforward Inference Run] (Inference through active backbone)
                 │
                 ├─► Input X propagates sequentially through W_merged(1), ..., W_merged(L)
                 │
                 ▼
     [Multi-Task Output Logits] (Computes loss or classification predictions)
                 │
                 ├─► Calibration: Compute CE Loss + γ * Σ ||θ_k||_1 ──► [Backpropagate to update Θ]
                 └─► Evaluation: Output task prediction accuracies
```

### Transformation Sequence:
1.  **Coordinate Evaluation:** For each layer index $l \in \{0, \dots, L-1\}$, the normalized depth $z = \frac{l}{L-1}$ is mapped to the Fourier trigonometric basis functions $\cos(2\pi f z)$ and $\sin(2\pi f z)$ for $f \in \{1, \dots, F\}$.
2.  **Coefficient Synthesis:** The continuous ensembling coefficients $\alpha_k(l)$ are computed as a linear combination of the trigonometric basis functions using the current learned Fourier parameters $\theta_k$, clipped to $[0, 1]$.
3.  **Parameter Interpolation:** The task vectors $V_k^{(l)}$ are scaled by their synthesized coefficients $\alpha_k(l)$ and added to the pre-trained base model $W_0^{(l)}$ to construct the merged weights $W_{\text{merged}}^{(l)}$.
4.  **Computational Propagation:** The input feature vectors propagate through the merged layers sequentially, yielding intermediate representations that are stable under depth transitions due to the smoothness of the Fourier trajectory.
5.  **Logit Projection & Optimization:** The final classification head generates class prediction logits. During calibration, gradients of the regularized cross-entropy loss are backpropagated through the feedforward graph to update the Fourier trajectory coefficients $\Theta$ via Adam GD.
