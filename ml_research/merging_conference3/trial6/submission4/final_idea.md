# Idea Proposal: Task-Space Anchor Regularization (TSAR)

## 1. Persona Alignment
This project aligns directly with **The Empiricist** persona. TSAR is built to be rigorously validated through extensive, large-scale empirical sweeps and robust multi-seed evaluations.
* **Overwhelming Empirical Proof:** We address the catastrophic collapse on noisy out-of-distribution (OOD) tasks (like SVHN) not by proposing complex, non-monotonic theoretical architectures (like wavefunctions), but by designing an extremely simple, geometrically grounded classical regularizer (TSAR).
* **Massive Parallel Sweeps:** The evaluation plan requires running massive, exhaustive hyperparameter sweeps over the anchor regularization coefficient $\lambda_{anchor}$ across multiple calibration set sizes $B_{cal}$ and independent random seeds (5 seeds) to establish clear statistical significance.
* **Exhaustive Ablation Studies:** We will include comprehensive ablation tables to isolate the individual and combined effects of standard $L_2$ weight decay, TSAR alignment, and projection dimensions.

---

## 2. Core Techniques
TSAR introduces an **Anchor-Guided Geometrical Regularization** constraint into the calibration phase of the Layer-wise Low-dimensional Classical Router (L3-Router).
* **Task-Space Feature Anchors:** For each task $k \in \{1, \dots, K\}$, we compute a robust task feature anchor $\bar{\psi}_k \in \mathbb{R}^d$ in the low-dimensional projection space by averaging the projected representations of the calibration samples belonging to task $k$.
* **Geometrical Weight Anchoring:** We augment the standard cross-entropy calibration loss with a quadratic penalty that pulls each layer's routing weight vector $W_{l, k}$ toward its corresponding task anchor $\bar{\psi}_k$. This acts as a spatial constraint that prevents the routing weights from drifting or overfitting to dominant tasks, preserving their discriminative alignment with minority/OOD task coordinates.
* **Linear Parameter Fusion:** During inference, the router linearly maps input features using the anchored weights to produce stable, robust merging coefficients.

Foundational and related methods cited:
* **AdaMerging** (Adaptive Model Merging for Multi-Task Learning, yuchen505/AdaMerging)
* **QWS-Merge** (Quantum Wavefunction Superposition Merging, trial4_submission10)
* **L3-Router** (Layer-wise Low-dimensional Classical Router, trial5_submission5)

---

## 3. Mathematical Formulation

### Low-Dimensional State Representation
Let $z(x)_b \in \mathbb{R}^D$ be the globally pooled feature representation from the first block of the backbone for sample $b$. Let $P \in \mathbb{R}^{D \times d}$ be the unsupervised PCA projection matrix. The projected and normalized low-dimensional state $\psi(x)_b$ is defined as:
$$
\psi(x)_b = \frac{z(x)_b P}{\|z(x)_b P\|_2 + \epsilon} \in \mathbb{R}^d
$$
where $\epsilon = 10^{-8}$.

### Task Feature Anchors
Let $X_{cal, k}$ represent the subset of the calibration split containing samples belonging to task $k$. The low-dimensional task feature anchor $\bar{\psi}_k \in \mathbb{R}^d$ is computed as:
$$
\bar{\psi}_k = \frac{1}{|X_{cal, k}|} \sum_{b \in X_{cal, k}} \psi(x)_b \in \mathbb{R}^d
$$

### Routing Coefficient Equations
For sample $b$ and layer $l \in \{1, \dots, L\}$, the dynamic merging coefficient for task $k$ is computed via our L3-Linear router:
$$
\alpha_{k, b}^{TSAR}(l) = \langle \psi(x)_b, W_{l, k} \rangle + B_{l, k}
$$
where $W_{l, k} \in \mathbb{R}^d$ and $B_{l, k} \in \mathbb{R}$ are the trainable routing weights and biases.

### Calibration Loss Function
The total objective function optimized during the calibration phase is:
$$
\mathcal{L}_{total} = \mathcal{L}_{CE} + \lambda_{wd} \sum_{l=1}^L \sum_{k=1}^K \left( \|W_{l, k}\|_2^2 + B_{l, k}^2 \right) + \lambda_{anchor} \sum_{l=1}^L \sum_{k=1}^K \| W_{l, k} - \bar{\psi}_k \|_2^2
$$
where:
* $\mathcal{L}_{CE}$ is the multi-task cross-entropy loss over the calibration split.
* $\lambda_{wd}$ is the standard $L_2$ weight decay coefficient (set to $10^{-3}$).
* $\lambda_{anchor}$ is the TSAR regularization coefficient (to be swept from $0.0$ to $1.0$).

---

## 4. Architecture Specifications
* **Backbone Model:** Vision Transformer (ViT-Tiny, `vit_tiny_patch16_224`) with $L=14$ layer groups.
* **Input Feature Dimension ($D$):** $D = 192$ (globally pooled visual representations).
* **Projection Dimension ($d$):** $d = K = 4$ (MNIST, FashionMNIST, CIFAR-10, SVHN).
* **Router Parameters:**
  * **Weights ($W$):** Shape $(L \times K \times d) = (14 \times 4 \times 4) = 224$ parameters.
  * **Biases ($B$):** Shape $(L \times K) = (14 \times 4) = 56$ parameters.
  * **Total Parameter Footprint:** 280 trainable parameters.
* **Activation Function:** Identity (linear routing mapping) to avoid non-monotonic optimization issues and preserve classical representation capacity.

---

## 5. Baselines
We compare TSAR against:
1. **Static Uniform Merging (Task Arithmetic):** Serves as the standard parameter-fusion baseline without dynamic routing.
2. **QWS-Merge (SOTA Quantum-Inspired):** Demonstrates whether complex, wave-based formulations are outperformed by simple geometrically regularized classical routers.
3. **Global Classical Linear Router:** The strongest global baseline identified in `trial5_submission5`.
4. **L3-Linear Router (Unregularized / standard L2):** Our direct classical ablation baseline to isolate the performance gains of the anchor-distance regularization.

---

## 6. Step-by-Step Interaction

### Step 1: Pre-computing Anchors (Offline Calibration Setup)
1. Pass the calibration split $X_{cal}$ through the pre-trained backbone.
2. Extract the globally pooled representations $z(x)_b \in \mathbb{R}^D$ and project them to $\psi(x)_b \in \mathbb{R}^d$ via the unsupervised PCA matrix $P$.
3. Compute the task feature anchors $\bar{\psi}_k \in \mathbb{R}^d$ by taking the class-wise means over $\psi(x)_b$.

### Step 2: Calibrating the Router
1. Feed the projected states $\psi(x)_b$ of the calibration set into the L3-Linear router to output coefficients $\alpha_{k, b}(l)$.
2. Compute the batch-average coefficients $\bar{\alpha}_k(l) = \frac{1}{B} \sum_{b=1}^B \alpha_{k, b}(l)$.
3. Perform a forward pass of the merged classification head using $\bar{\alpha}_k(l)$ to compute predictions and the cross-entropy loss $\mathcal{L}_{CE}$.
4. Compute the TSAR penalty $\mathcal{L}_{Anchor} = \sum_{l, k} \| W_{l, k} - \bar{\psi}_k \|_2^2$.
5. Backpropagate the total loss $\mathcal{L}_{total}$ and update the router weights $W_{l, k}$ and biases $B_{l, k}$ using AdamW.

### Step 3: Deployment Inference
1. For an incoming test batch, extract and project features to the low-dimensional space $\psi(x)_b$.
2. Feed $\psi(x)_b$ into the calibrated L3-Linear router to obtain sample-specific coefficients.
3. Compute the batch-wise mean coefficients $\bar{\alpha}_k(l)$.
4. Dynamically assemble the weight matrix at each layer: $W_{merged}^{(l)} = W_{base}^{(l)} + \sum_k \bar{\alpha}_k(l) V_k^{(l)}$.
5. Run the forward pass of the merged multi-task model to generate predictions.
