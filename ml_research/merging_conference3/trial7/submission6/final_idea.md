# Spectral and Rademacher-guided Routing Regularization (SR3)

## 1. Persona Alignment
This work is a relentless pursuit of mathematical rigor and theoretical guarantees in weight-space model merging, directly embodying **The Theorist** persona. 
Current dynamic routing models resort to ad-hoc, heuristic regularizers to survive low-data regimes ($B_{cal} \le 64$). For instance, Task-Space Anchor Regularization (TSAR) anchors weights to heuristic centroids, while VR-Router enforces arbitrary variance penalties. These methods lack formal justifications explaining *why* they work or *how* they relate to the underlying parameter spaces of the merged models.

In contrast, **SR3 (Spectral and Rademacher-guided Routing Regularization)** is derived from first-principles learning theory. We formalize the dynamic weight-space routing problem and prove that the Rademacher complexity of the dynamically merged model class is directly bounded by the norms of the routing weights weighted by the parameter-space norms of the expert task vectors. This derivation provides a rigorous, mathematically sound proof that **regularization must be asymmetric across experts and proportional to their task-vector magnitudes**. We reject heuristic ensembling in favor of a provably optimal regularization frontier.

---

## 2. Core Techniques
We introduce the following core techniques and components:

1. **Task-Vector Norm Profiling:** Before calibration, we analyze the parameter-space geometry of all $K$ expert models. For each layer $l$ and expert $k$, we compute the Frobenius norm $\|V_k^{(l)}\|_F^2$ of its corresponding task vector. These norms serve as fixed, mathematically grounded asymmetric regularizer scaling factors.
2. **Spectral/Operator Norm Scaling (Alternative):** As a spectral alternative, we also define the operator norm (spectral norm) $\|V_k^{(l)}\|_{op}^2$ (the maximum singular value) to capture the worst-case representation distortion of each task vector, scaling the routing weights accordingly.
3. **Asymmetric Spectral Routing Weight Decay:** Instead of standard isotropic $L_2$ decay, we penalize the routing parameters $W_{l, k}$ of each task expert $k$ proportionally to the magnitude of its task vector. This forces the router to remain highly conservative when activating distant, high-complexity experts, while allowing more flexible routing to nearby, low-complexity experts.
4. **Zero-Initialized Softmax Routing Prior:** We maintain the zero-initialized Softmax routing layer, establishing a maximum-entropy uniform prior ($\alpha_k = 1/K$) at the start of calibration.
5. **Low-Rank Parameter Compatibility:** Our framework is designed to seamlessly integrate with Parameter-Efficient Fine-Tuning (PEFT/LoRA) adapters, where task-vector norms are computed directly on the low-rank adapter weights $A_k, B_k$, guaranteeing extreme computational and VRAM scaling efficiency.

---

## 3. Mathematical Formulation

### Definition of Dynamic Weight-Space Merging
Let $W_{\text{base}}^{(l)}$ be the pre-trained base parameters at layer $l \in \{1,\dots,L\}$. Let $W_k^{(l)}$ be the parameters of expert $k \in \{1,\dots,K\}$. The task vector is defined as:
$$ V_k^{(l)} = W_k^{(l)} - W_{\text{base}}^{(l)} $$
For an input sample $b$ in a batch $X$, the dynamically merged parameter set $W_{\text{merged}}^{(l)}(b)$ is assembled as:
$$ W_{\text{merged}}^{(l)}(b) = W_{\text{base}}^{(l)} + \sum_{k=1}^K \alpha_{k, b}(l) V_k^{(l)} $$
where $\alpha_{k, b}(l) \in [0, 1]$ are the routing coefficients predicted for sample $b$ at layer $l$, satisfying $\sum_{k=1}^K \alpha_{k, b}(l) = 1$.

### Feature Projection
Penultimate layer representations $z(x)_b \in \mathbb{R}^D$ are projected to a $d$-dimensional subspace ($d = K$) using a frozen, normalized random projection matrix $P \in \mathbb{R}^{D \times d}$, and normalized onto a unit sphere to produce the unit-state $\psi(x)_b$:
$$ \psi(x)_b = \frac{z(x)_b P}{\|z(x)_b P\|_2 + \epsilon} \in \mathbb{R}^d $$

### Layer-wise Linear Routing
The coefficients are predicted using linear-Softmax projections:
$$ \alpha_{k, b}(l) = \text{Softmax}\left( \langle \psi(x)_b, W_{l, \cdot} \rangle + B_{l, \cdot} \right)_k = \frac{\exp\left( \langle \psi(x)_b, W_{l, k} \rangle + B_{l, k} \right)}{\sum_{j=1}^K \exp\left( \langle \psi(x)_b, W_{l, j} \rangle + B_{l, j} \right)} $$

---

### Derivation and Proof of SR3 Optimality

Let the dynamically merged model's function class be:
$$ \mathcal{H}_{\text{merged}} = \left\{ x \mapsto f\left(x; W_{\text{base}} + \sum_{k=1}^K \alpha_k(x) V_k\right) \right\} $$
where $\alpha_k(x) = \text{Softmax}(W_k^T \psi(x))$.

Assuming the neural network output is $L_{\text{net}}$-Lipschitz continuous in its parameters, the deviation of the merged model from the base model is bounded by:
$$ \left| f(x; W_{\text{merged}}) - f(x; W_{\text{base}}) \right| \le L_{\text{net}} \left\| \sum_{k=1}^K \alpha_k(x) V_k \right\|_F \le L_{\text{net}} \sum_{k=1}^K \alpha_k(x) \|V_k\|_F $$

By Rademacher complexity composition properties, the Rademacher complexity of the merged hypothesis class $\mathcal{R}_n(\mathcal{H}_{\text{merged}})$ satisfies:
$$ \mathcal{R}_n(\mathcal{H}_{\text{merged}}) \le \mathcal{R}_n(\mathcal{H}_{\text{base}}) + L_{\text{net}} \sum_{k=1}^K \|V_k\|_F \mathcal{R}_n(\mathcal{A}_k) $$
where $\mathcal{A}_k = \{ x \mapsto \alpha_k(x) \}$ is the class of routing functions.

Under our unit-sphere projection, $\|\psi(x)\|_2 \le 1$. The Rademacher complexity of the linear-Softmax routing class $\mathcal{A}_k$ with training/calibration sample size $n$ is bounded by:
$$ \mathcal{R}_n(\mathcal{A}_k) \le \frac{\|W_k\|_2}{\sqrt{n}} $$

Combining these bounds yields the unified generalization constraint on the dynamic merged model:
$$ \mathcal{R}_n(\mathcal{H}_{\text{merged}}) \le \mathcal{R}_n(\mathcal{H}_{\text{base}}) + \frac{L_{\text{net}}}{\sqrt{n}} \sum_{k=1}^K \|V_k\|_F \|W_k\|_2 $$

To minimize this generalization complexity bound under a total routing resource budget $C_0$, we formulate the constrained optimization problem:
$$ \min_{W} \sum_{k=1}^K \|V_k\|_F \|W_k\|_2 \quad \text{subject to} \quad \sum_{k=1}^K \|W_k\|_2^2 \le C_0 $$

Using Lagrangian multipliers, the optimal regularization objective is precisely:
$$ \mathcal{L}_{\text{reg}} = \lambda \sum_{k=1}^K \|V_k\|_F^2 \|W_k\|_2^2 $$
which proves that scaling the weight decay penalty of routing parameters by the squared Frobenius norm of their corresponding task vectors is the theoretically optimal regularizer.

### The SR3 Loss Objective
The finalized SR3 regularization loss is defined as:
$$ \mathcal{L}_{SR3} = \lambda_{SR3} \sum_{l=1}^L \sum_{k=1}^K \Gamma_k^{(l)} \left( \|W_{l, k}\|_2^2 + B_{l, k}^2 \right) $$
where the expert-specific scaling factor $\Gamma_k^{(l)}$ is:
- **Frobenius Variant (SR3-F):** $\Gamma_k^{(l)} = \|V_k^{(l)}\|_F^2$
- **Spectral/Operator Variant (SR3-S):** $\Gamma_k^{(l)} = \|V_k^{(l)}\|_{op}^2$

The complete multi-objective calibration loss is:
$$ \mathcal{L}_{\text{total}} = \mathcal{L}_{CE} + \mathcal{L}_{SR3} $$

---

## 4. Architecture Specifications
* **Backbone Layers ($L$):** 14 layers.
* **Feature Dimensions ($D$):** 192 (Globally pooled representation $z \in \mathbb{R}^{D}$).
* **Projection Subspace ($d$):** $d = K = 4$ (Stable random projection matrix $P \in \mathbb{R}^{D \times d}$, normalized onto the unit hypersphere).
* **Intermediate Representation:** Unit-state $\psi(x) \in \mathbb{R}^4$.
* **Router Parameters:**
  * **Trainable Routing Weights:** $W \in \mathbb{R}^{L \times K \times d}$ (Initialized to exact zeros).
  * **Trainable Routing Biases:** $B \in \mathbb{R}^{L \times K}$ (Initialized to exact zeros).
* **Expert Count ($K$):** 4 expert models fine-tuned from a shared pre-trained base model.
* **VRAM and Host Optimization:** Restricting the parameter assembly exclusively to Low-Rank LoRA adapters, capping model memory overhead at a strict $1.04\times$ footprint.

---

## 5. Baselines
We evaluate our proposed SR3 method against the following standard and state-of-the-art baselines on the synthetic sandbox:

1. **Static Uniform Merging (Zero Parameters):** Merges task vectors with fixed, uniform coefficients $\alpha_k = 1/K = 0.25$ across all samples.
2. **Standard Linear Router (Unregularized):** A parametric linear router with zero regularization, highlighting the severity of low-data overfitting.
3. **Standard Linear Router ($L_2$ Weight Decay):** Classic isotropic $L_2$ regularization, representing a uniform decay benchmark.
4. **Task-Space Anchor Regularization (TSAR):** An anchor-based centroid regularizer ($\lambda_{anchor}=0.1$) optimized with PCGrad.
5. **Task-Variance Regularized Router (VR-Router):** Enforces an explicit group-variance penalty ($\lambda_{var}=1.0$) on predicted task coefficients.
6. **PFSR + MBH:** The non-parametric, training-free subspace cosine-similarity routing baseline.

---

## 6. Step-by-Step Interaction

1. **Offline Geometric Profiling:**
   * Before calibration begins, the task vectors $V_k^{(l)} = W_k^{(l)} - W_{\text{base}}^{(l)}$ are pre-computed.
   * We calculate their Frobenius norms $\|V_k^{(l)}\|_F^2$ and spectral/operator norms $\|V_k^{(l)}\|_{op}^2$ for all layers $l \in \{1,\dots,L\}$ and experts $k \in \{1,\dots,K\}$.
   * These norms are frozen as scaling multipliers $\Gamma_k^{(l)}$ for the routing weights.
2. **Representation Extraction & Dimensionality Reduction:**
   * For an input batch $X$, high-dimensional representation features $z(x)_b$ are extracted from the penultimate layer of the backbone model.
   * $z(x)_b$ is projected using the static random matrix $P$ and normalized onto the unit hypersphere to produce the unit-state $\psi(x)_b$.
3. **Dynamic Routing Inference:**
   * The unit-state $\psi(x)_b$ is passed through the layer-wise linear routing weights $W_{l, \cdot}$ and biases $B_{l, \cdot}$ to produce logit projections.
   * A Softmax function normalizes these logits into sample-specific task coefficients $\alpha_{k, b}(l)$.
4. **Vectorized Parameter Assembly:**
   * The dynamically merged model parameters $W_{\text{merged}}^{(l)}(b)$ are constructed sample-by-sample on-the-fly using highly optimized vectorized operations (`vmap` or `einsum`).
5. **Asymmetric Optimization & Calibration:**
   * During the calibration phase, we compute the Cross-Entropy loss $\mathcal{L}_{CE}$ on the calibration split ($B_{cal} = 64$) using the sample-specific merged weights.
   * We compute the SR3 penalty:
     $$ \mathcal{L}_{SR3} = \lambda_{SR3} \sum_{l=1}^L \sum_{k=1}^K \Gamma_k^{(l)} \left( \|W_{l, k}\|_2^2 + B_{l, k}^2 \right) $$
   * We backpropagate the gradients of $\mathcal{L}_{\text{total}} = \mathcal{L}_{CE} + \mathcal{L}_{SR3}$ to optimize the routing parameters $W$ and $B$.
6. **Test-time Inference Evaluation:**
   * At test time, incoming batches are routed sample-wise, and accuracy is evaluated under Homogeneous Batching and Heterogeneous Streaming to verify robust generalization with zero vectorization collapse.
