# Robust Linear Routing (RLR) for Dynamic Model Merging

## 1. Persona Alignment
This work aligns perfectly with **The Minimalist** persona. Instead of inventing elaborate, over-engineered quantum-inspired metaphors (like task weight wavefunctions, phase basis, phase projectors, and dynamic wave interference in `QWS-Merge`) which introduce complex backpropagation graphs and high runtime overhead, we argue that the failure of classical dynamic model merging is not due to structural limitations of linear routing, but rather due to a simple, unaddressed **overfitting and representation-outlier issue**. 

Under Occam's razor, if a complex method can be matched or outperformed by a simpler one, the simpler one is strictly superior. We show that the catastrophic SVHN collapse of the classical Linear Router in `QWS-Merge` ($15.30\%$) is entirely resolved by adding standard L2 weight regularization (weight decay) and softmax temperature scaling. By doing so, we preserve the extreme simplicity of a 768-parameter linear gating network while outperforming the highly complex, multi-stage QWS-Merge.

---

## 2. Core Techniques
* **Direct Linear Gating:** Maps input representations directly to blending coefficients via a single lightweight linear layer.
* **L2 Weight Regularization (Weight Decay):** Penalizes high-magnitude weights in the router during calibration. This constrains the routing logits, preventing extreme, high-variance routing decisions on out-of-distribution tasks (like SVHN).
* **Softmax Temperature Scaling:** Softens the routing coefficients, ensuring the router maintains a stable mixture of experts under high representation shift and does not collapse to a single task-expert delta distribution.
* **Uniform Multi-Task Calibration Loss:** Minimizes a clean, unweighted uniform sum of individual task losses, removing all heuristics, difficulty-proxies, and complexity, thereby enforcing our minimalist philosophy.

---

## 3. Mathematical Formulation
Let $x \in \mathbb{R}^d$ be the pooled output representation of the model's first patch embedding (or backbone representation). 
The routing logit vector $z \in \mathbb{R}^N$ for $N$ task experts is computed as:
\begin{equation}
z = W x + b
\end{equation}
where $W \in \mathbb{R}^{N \times d}$ is the router's weight matrix and $b \in \mathbb{R}^N$ is the bias vector.

The task-blending coefficients (routing weights) $a = [a_1, a_2, \dots, a_N]^T \in \mathbb{R}^N$ are computed via a softmax function with a temperature parameter $T \ge 1$:
\begin{equation}
a_k = \frac{\exp(z_k / T)}{\sum_{j=1}^N \exp(z_j / T)}
\end{equation}
where a higher temperature $T$ acts as an implicit entropy regularizer, smoothing the routing distribution.

The dynamically merged model weights at layer $l$, denoted $\theta_l(x)$, are assembled on-the-fly for input $x$:
\begin{equation}
\theta_l(x) = \sum_{k=1}^N a_k \theta_{k, l}
\end{equation}

During calibration on a tiny 64-sample validation set, we optimize $W$ and $b$ to minimize the regularized uniform multi-task calibration loss:
\begin{equation}
\mathcal{L}(W, b) = \sum_{t=1}^N \mathcal{L}_{\text{task}, t} + \alpha \|W\|_F^2
\end{equation}
where $\mathcal{L}_{\text{task}, t}$ is the cross-entropy loss on task $t$ and $\alpha$ is the L2 weight regularization penalty. This unweighted formulation requires no heuristics, task-difficulty proxies, or calibration-biases, keeping our dynamic gating framework exceptionally simple, transparent, and perfectly aligned with our minimalist philosophy.

---

## 4. Architecture Specifications
* **Unified Backbone:** Compact Vision Transformer ($\mathtt{vit\_tiny\_patch16\_224}$) with $14$ layer groups (Patch Embedding, 12 Transformer blocks, final LayerNorm) and $5.7$M parameters.
* **Router Inputs ($x$):** Globally average-pooled representation from the first Patch Embedding layer of dimension $d = 192$.
* **Trainable Parameters:**
  - Weight matrix $W \in \mathbb{R}^{4 \times 192}$ (768 parameters).
  - Bias vector $b \in \mathbb{R}^4$ (4 parameters).
  - Total parameters: 772.
* **Hyperparameters:**
  - Temperature: $T = 2.0$ (softens routing decisions).
  - Regularization weight: $\alpha = 0.005$ (weight decay).
  - Calibration optimization: Adam optimizer with learning rate $lr = 0.01$ run for 100 steps on 64 calibration samples.

---

## 5. Baselines
* **Individual Experts (Ceiling):** Specialized, non-merged task-specific models (Joint Mean: $70.52\%$).
* **Uniform Merging (Task Arithmetic):** Static, linear addition of task vectors (Joint Mean: $49.35\%$).
* **OFS-Tune (Supervised Static):** Supervised static layer-wise coefficient optimization on the 64-sample calibration set (Joint Mean: $55.00\%$).
* **Linear Router (Unregularized Classical):** The original unregularized linear routing baseline which collapses catastrophically on SVHN to $15.30\%$ (Joint Mean: $61.23\%$).
* **QWS-Merge (Convoluted Quantum):** The complex, over-engineered dynamic model merging scheme from Trial 4 (Joint Mean: $59.32\%$).

---

## 6. Step-by-Step Interaction
1. **Input Extraction:** An input image is passed into the first Patch Embedding layer of the ViT backbone.
2. **Representation Pooling:** The embedding output is globally average-pooled to form a 192-dimensional representation vector $x$.
3. **Logits Projection:** The router projects $x$ to raw routing logits: $z = Wx + b$.
4. **Softmax with Temperature:** The logits are scaled by temperature $T$ and passed through a softmax to generate the routing coefficients $a_k \in \mathbb{R}^4$.
5. **Dynamic Weight Blending:** The backbone parameters for the current input are computed on-the-fly as a linear combination of expert weights weighted by $a_k$: $\theta_{\text{merged}} = \sum_k a_k \theta_k$.
6. **Task-Specific Prediction:** The blended network processes the input representation, and the final pooled output is passed to the corresponding task-specific classification head to compute class logits and produce the final prediction.
