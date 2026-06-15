# Idea Proposal: Demystifying Dynamic Model Merging via Bounded Classical Routing (BC-Router)

## 1. Persona Alignment
As **The Methodologist**, our core research philosophy is that bad methodology leads to false progress and that the deep learning community desperately needs better evaluation protocols, stronger baselines, and more critical analyses of existing trends. We are highly skeptical of "state-of-the-art" claims that rely on weak baselines or flawed metrics.

The recently proposed **Quantum Wavefunction Superposition Merging (QWS-Merge)** represents a growing trend in model merging: introducing complex, over-engineered mathematical metaphors (like modeling expert parameters as quantum eigenstates in a Hilbert space and using wave-like phase interference) to achieve dynamic parameter routing. QWS-Merge claims that a classical **Linear Router** baseline collapses catastrophically on high-conflict tasks like SVHN ($15.30\%$), whereas QWS-Merge preserves high performance ($31.60\%$).

However, our rigorous methodological analysis exposes **two critical confounding variables** in this comparison:
1.  **The Over-Scaling Confounder:** The Linear Router baseline uses a standard Softmax activation, which forces the routing coefficients to sum to $1.0$. When the router selects a single task expert, the corresponding coefficient approaches $1.0$. In Task Arithmetic, adding a task vector with a scale of $1.0$ is extremely large (the optimal static coefficient is $\lambda \approx 0.3$), leading to severe weight drift and representational collapse. In contrast, QWS-Merge caps its coefficients at $R_k^{(l)} = 0.3$ via its wave scaling amplitude, which naturally prevents over-scaling.
2.  **The Layer-wise Specialization Confounder:** QWS-Merge has layer-wise parameters ($336$ total), allowing distinct routing behaviors across layers. In contrast, the Linear Router baseline is global, applying a single averaged coefficient uniformly across all layers, which denies the network the ability to form layer-wise specialized representations.

By applying Occam's razor, we propose the **Bounded Classical Router (BC-Router)**. BC-Router consists of two simple, classical, and training-free/few-shot optimizable variants that directly isolate and control these confounders:
*   **Bounded Linear Router (BL-Router):** A global router that uses a Softmax projection scaled by a static hyperparameter $\lambda_{max} = 0.3$, completely eliminating the over-scaling confounder.
*   **Global Router with Layer-wise Scaling (GLS-Router):** A parameter-efficient router that shares a global linear routing head but applies learned, layer-wise task-scaling amplitudes $R_k^{(l)}$ initialized to $0.3$, isolating the layer-wise specialization confounder.

If these simple classical baselines match or outperform QWS-Merge, we will have successfully deconstructed the necessity of quantum wavefunction superposition in model merging, demonstrating that the reported "quantum" performance gains are purely a methodological artifact of an un-regularized, under-designed classical baseline.

---

## 2. Core Techniques
We introduce two classical baselines that apply direct constraints to weight-space scaling:
1.  **BL-Router (Bounded Linear Router):**
    A linear projection layer maps the global pooled representation $z(x)$ of an input batch to $K$ raw routing logits. We pass these logits through a standard Softmax to obtain relative task probabilities, and then scale the probabilities by a static factor $\lambda_{max} = 0.3$. This ensures that no individual task vector can ever be added with a coefficient greater than $0.3$, matching the optimal static Task Arithmetic scale and filtering out high-conflict parameter drift.
2.  **GLS-Router (Global Router with Layer-wise Scaling):**
    We share a single global linear projection layer across all network layers to compute task probabilities. However, for each layer $l$ and task expert $k$, we learn a small scaling amplitude $R_k^{(l)} \in \mathbb{R}$ (initialized to $0.3$). The dynamic merging coefficient for layer $l$ is the product of the shared global probability and the layer-specific amplitude $R_k^{(l)}$. This provides layer-wise routing specialization with an ultra-compact parameter footprint, completely avoiding the Overfitting-Optimizer Paradox on tiny validation sets.

---

## 3. Mathematical Formulation

Let $W_{base}^{(l)}$ be the pre-trained base model weights at layer $l \in \{1, \dots, L\}$, and $V_k^{(l)} = W_k^{(l)} - W_{base}^{(l)}$ be the task vector for expert $k \in \{1, \dots, K\}$ at layer $l$. Let $x$ be an input batch, and $z(x) \in \mathbb{R}^{B \times D}$ be the spatially averaged patch tokens extracted from the backbone's patch embedding:
$$z(x)_b = \frac{1}{N} \sum_{n=1}^N H_{0, b, n, :} \in \mathbb{R}^D \quad \forall b \in \{1, \dots, B\}$$

### 3.1 Bounded Linear Router (BL-Router)
The routing logits $o(x)_b \in \mathbb{R}^K$ for sample $b$ are computed via a linear layer parameterized by $W_{route} \in \mathbb{R}^{D \times K}$ and $b_{route} \in \mathbb{R}^K$:
$$o(x)_b = z(x)_b W_{route} + b_{route}$$

The bounded sample-level routing coefficients $\alpha_{k, b}$ are obtained by scaling the Softmax probabilities by $\lambda_{max}$:
$$\alpha_{k, b} = \lambda_{max} \times \frac{\exp(o(x)_{b, k})}{\sum_{j=1}^K \exp(o(x)_{b, j})} \quad \text{with } \lambda_{max} = 0.3$$

We perform a mean-measurement across the batch to collapse to a batch-level coefficient $\bar{\alpha}_k$:
$$\bar{\alpha}_k = \frac{1}{B} \sum_{b=1}^B \alpha_{k, b}$$

The merged weights at layer $l$ used to process batch $x$ are:
$$W_{merged}^{(l)}(x) = W_{base}^{(l)} + \sum_{k=1}^K \bar{\alpha}_k V_k^{(l)}$$

### 3.2 Global Router with Layer-wise Scaling (GLS-Router)
We use the same global routing logits $o(x)_b$ and standard Softmax probabilities $p_{k, b}$:
$$p_{k, b} = \frac{\exp(o(x)_{b, k})}{\sum_{j=1}^K \exp(o(x)_{b, j})} \in [0, 1]$$

We define layer-wise, task-specific trainable scaling amplitudes $R_k^{(l)} \in \mathbb{R}$, initialized to $0.3$. The sample-level merging coefficient for expert $k$ at layer $l$ is:
$$\alpha_{k, b}^{(l)} = R_k^{(l)} \times p_{k, b}$$

Collapsing across the batch yields the batch-level layer-wise coefficient $\bar{\alpha}_k^{(l)}$:
$$\bar{\alpha}_k^{(l)} = \frac{1}{B} \sum_{b=1}^B \alpha_{k, b}^{(l)} = R_k^{(l)} \times \left( \frac{1}{B} \sum_{b=1}^B p_{k, b} \right)$$

The merged weights at layer $l$ are:
$$W_{merged}^{(l)}(x) = W_{base}^{(l)} + \sum_{k=1}^K \bar{\alpha}_k^{(l)} V_k^{(l)}$$

---

## 4. Architecture Specifications

The backbone architecture is a compact Vision Transformer, specifically $\mathtt{vit\_tiny\_patch16\_224}$ ($5.7$M parameters), containing $L=14$ layer groups (a Patch Embedding layer, $12$ Transformer blocks, and a final Layer Normalization layer) with hidden dimension $D = 192$. We evaluate on $K = 4$ tasks.

### 4.1 Parameter Dimensions and Efficiency
*   **BL-Router:**
    *   $W_{route} \in \mathbb{R}^{192 \times 4}$ ($768$ parameters)
    *   $b_{route} \in \mathbb{R}^4$ ($4$ parameters)
    *   *Total Trainable Parameters:* **$772$ parameters**.
*   **GLS-Router:**
    *   $W_{route} \in \mathbb{R}^{192 \times 4}$ ($768$ parameters)
    *   $b_{route} \in \mathbb{R}^4$ ($4$ parameters)
    *   $R \in \mathbb{R}^{14 \times 4}$ ($56$ parameters)
    *   *Total Trainable Parameters:* **$828$ parameters**.

Both configurations are optimized using Adam on the tiny $64$-sample validation set (16 samples per task) for 100 steps. The extremely small parameter footprint guarantees that both baselines are robust to the Overfitting-Optimizer Paradox.

---

## 5. Baselines
Our primary evaluation will benchmark our proposed routers against:
1.  **Individual Experts (Ceiling):** Task-specific networks representing the upper bound of specialized capability.
2.  **Uniform Merging (Task Arithmetic):** Static uniform merging ($\lambda = 0.3$) to establish the base multi-task performance.
3.  **AdaMerging (Unsupervised TTA):** Standard unsupervised test-time adaptation.
4.  **OFS-Tune (Supervised Static):** Supervised static layer-wise coefficient optimization on the calibration set.
5.  **Classical Linear Router (Un-bounded):** The original baseline from QWS-Merge, which lacks the maximum scale cap and serves to empirically confirm the presence of the over-scaling confounder.
6.  **QWS-Merge (SOTA):** The quantum wavefunction formulation, serving to test our main hypothesis that quantum wave projections are redundant once classical over-scaling is corrected.

---

## 6. Step-by-Step Interaction

1.  **Input Extraction:**
    An input batch of images $x \in \mathbb{R}^{B \times C \times H \times W}$ is passed to the ViT patch embedding layer, yielding patch tokens $H_0 \in \mathbb{R}^{B \times N \times D}$.
2.  **Global Pooling:**
    We spatially average the patch tokens to obtain a global batch representation $z(x) \in \mathbb{R}^{B \times D}$.
3.  **Routing Logit Computation:**
    For each sample $b$, we project $z(x)_b$ through the global routing layer to compute raw task logits $o(x)_b = z(x)_b W_{route} + b_{route} \in \mathbb{R}^K$.
4.  **Coefficient Bounding & Layer Specialization:**
    *   For **BL-Router**, we apply Softmax to $o(x)_b$ and multiply by the static $\lambda_{max} = 0.3$ to get $\alpha_{k, b}$.
    *   For **GLS-Router**, we apply Softmax to $o(x)_b$ to get $p_{k, b}$, and then multiply by the trainable layer-wise task scaling amplitudes $R_k^{(l)}$ to get $\alpha_{k, b}^{(l)}$.
5.  **Batch Collapse (Wavefunction Collapse):**
    We average the sample-level coefficients across the batch to compute the batch-level merging coefficients ($\bar{\alpha}_k$ or $\bar{\alpha}_k^{(l)}$).
6.  **Weight Assembly:**
    For each layer $l$, we dynamically assemble the active weight matrix:
    $$W_{merged}^{(l)}(x) = W_{base}^{(l)} + \sum_{k=1}^K c_k^{(l)} V_k^{(l)}$$
    where $c_k^{(l)}$ represents the collapsed coefficient.
7.  **Forward Pass:**
    The input representations are propagated through each layer $l$ of the Vision Transformer using the dynamically assembled weights $W_{merged}^{(l)}(x)$, producing the final multi-task logits.
