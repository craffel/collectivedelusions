# Idea Proposal: Contraction-Regularized Router (CR-Router) for Fixed-Point Convergence in Sequential Deep Model Merging

## 1. Persona Alignment
As **The Theorist**, we approach the challenge of sequential dynamic model ensembling not as a heuristic training problem, but as a discrete-time dynamical system. Prior works like ChemMerge introduced reaction-kinetics heuristics to smooth routing trajectories, but lacked rigorous stability analysis or convergence guarantees. We argue that empirical ensembling success is fragile without a solid mathematical foundation to explain *why* representation flow converges stably across network depth. 

The **CR-Router (Contraction-Regularized Router)** directly aligns with our theoretical persona:
- **Dynamical Systems Lens:** It models the feedforward feature-coefficient propagation as a sequential mapping and analyzes its contraction properties via Banach's Fixed-Point Theorem.
- **Provable Stability:** Rather than employing hand-crafted smoothing heuristics, it derives a mathematically rigorous Lipschitz bound on the joint layer-wise representation-routing map.
- **Spectrum Constraints:** It introduces a theoretically sound spectral norm penalty on the routing projection weights and temperature scales, guaranteeing that the ensembling state converges to a stable fixed-point trajectory under depth.

---

## 2. Core Techniques
We introduce and modify the following core algorithms and mechanisms:
1. **Sequential Deep Routing Formulation:** We formalize the feedforward feature-coefficient feedback loop across sequential layers $l \in \{1, \dots, L\}$ as a joint discrete-time dynamical system.
2. **Spectral Norm Regularization of Routing Heads:** We apply an explicit spectral norm constraint $\|W_{\text{route}}^{(l)}\|_2$ to the layer-wise routing projection matrices during calibration.
3. **Inverse Temperature Bounding:** We regularize the task-specific inverse temperatures $1/\tau_l$ to prevent the routing Softmax from operating in non-Lipschitz, high-entropy, or discontinuous switching regimes.
4. **Banach Contraction Projection:** We construct a projection operator during training that restricts the Lipschitz constant of the joint layer-wise mapping $T_l$ to be strictly less than $1$ ($L_{T_l} < 1$), guaranteeing convergence to a unique fixed-point trajectory.

Our formulation builds on the following foundational methods:
- **LoRA Adapter Merging:** Using low-rank decompositions ($A_k, B_k$ of rank $r=8$) for parameter-efficient task ensembling \cite{hu2021lora}.
- **Unsupervised Subspace Projection:** Mapping features to task energy coordinates via Singular Value Decomposition (SVD) principal components \cite{zhang24spszca}.
- **Lipschitz Deep Learning:** Applying spectral normalization to guarantee functional smoothness and generalization in deep architectures \cite{bartlett2017spectrally}.

---

## 3. Mathematical Formulation

Let $h^{(l-1)}_b \in \mathbb{R}^D$ be the intermediate representation at the output of layer $l-1$ for sample $b$.
The routing head at layer $l$ predicts the ensembling coefficients $\alpha^{(l)}_b \in \Delta^{K-1}$ (the probability simplex) over $K$ experts using a Softmax-linear projection over task-specific energy coordinates extracted via Subspace Energy Projection (SEP):
$$\alpha^{(l)}_{k, b} = R_{k, l}(h^{(l-1)}_b) = \frac{\exp\left( \frac{1}{\tau_l} w_{k, \text{route}}^{(l)} h^{(l-1)}_b \right)}{\sum_{j=1}^K \exp\left( \frac{1}{\tau_l} w_{j, \text{route}}^{(l)} h^{(l-1)}_b \right)}$$
where $w_{k, \text{route}}^{(l)} \in \mathbb{R}^D$ is the routing projection vector for task $k$ at layer $l$, and $\tau_l > 0$ is the layer routing temperature.

The blended feedforward layer representation $h^{(l)}_b$ is computed by applying the shared pre-trained base block $F_{\text{base}}^{(l)}$ and superimposing the active low-rank expert adapters:
$$h^{(l)}_b = F_{\text{base}}^{(l)}(h^{(l-1)}_b) + \sum_{k=1}^K \alpha^{(l)}_{k, b} A_k^{(l)} B_k^{(l)} h^{(l-1)}_b$$
where $A_k^{(l)} \in \mathbb{R}^{D \times r}$ and $B_k^{(l)} \in \mathbb{R}^{r \times D}$ are the specialized expert adapters of rank $r$.

### The Sequential Feedback Mapping
This joint system defines a discrete-time feedforward mapping $T_l: \mathbb{R}^D \to \mathbb{R}^D$:
$$h^{(l)}_b = T_l(h^{(l-1)}_b) = F_{\text{base}}^{(l)}(h^{(l-1)}_b) + \sum_{k=1}^K R_{k, l}(h^{(l-1)}_b) A_k^{(l)} B_k^{(l)} h^{(l-1)}_b$$

### Theorem: Contraction Bound for Deep Sequential Routing
Let the shared base model block $F_{\text{base}}^{(l)}$ be Lipschitz continuous with constant $L_{\text{base}}^{(l)}$. Let the expert adapters be bounded in spectral norm by $C_A^{(l)} = \max_k \|A_k^{(l)} B_k^{(l)}\|_2$. 
Then, the Lipschitz constant $L_{T_l}$ of the sequential ensembling map $T_l$ over a bounded representation domain $\|h\|_2 \le R_h$ satisfies:
$$L_{T_l} \le L_{\text{base}}^{(l)} + C_A^{(l)} \left( 1 + \frac{2 R_h}{\tau_l} \|W_{\text{route}}^{(l)}\|_2 \right)$$
where $W_{\text{route}}^{(l)} \in \mathbb{R}^{K \times D}$ is the compiled routing projection matrix at layer $l$.

### Proof Sketch
Let $h, \tilde{h} \in \mathbb{R}^D$ with $\|h\|_2, \|\tilde{h}\|_2 \le R_h$. We bound the difference:
$$\|T_l(h) - T_l(\tilde{h})\|_2 \le \|F_{\text{base}}^{(l)}(h) - F_{\text{base}}^{(l)}(\tilde{h})\|_2 + \left\| \sum_{k=1}^K R_{k, l}(h) A_k^{(l)} B_k^{(l)} h - \sum_{k=1}^K R_{k, l}(\tilde{h}) A_k^{(l)} B_k^{(l)} \tilde{h} \right\|_2$$
Using the identity $a_1 b_1 - a_2 b_2 = a_1 (b_1 - b_2) + (a_1 - a_2) b_2$, we decompose the ensembling difference:
$$\sum_{k=1}^K \left[ R_{k, l}(h) A_k^{(l)} B_k^{(l)} (h - \tilde{h}) + (R_{k, l}(h) - R_{k, l}(\tilde{h})) A_k^{(l)} B_k^{(l)} \tilde{h} \right]$$
1. Since $\sum_k R_{k, l}(h) = 1$ and $R_{k, l}(h) \in [0, 1]$, the first term is bounded by:
   $$\left\| \sum_{k=1}^K R_{k, l}(h) A_k^{(l)} B_k^{(l)} (h - \tilde{h}) \right\|_2 \le \max_k \|A_k^{(l)} B_k^{(l)}\|_2 \|h - \tilde{h}\|_2 \le C_A^{(l)} \|h - \tilde{h}\|_2$$
2. For the second term, since the Softmax-linear projection $R_l(h)$ is Lipschitz continuous:
   $$\|R_l(h) - R_l(\tilde{h})\|_1 \le \frac{2}{\tau_l} \|W_{\text{route}}^{(l)}\|_2 \|h - \tilde{h}\|_2$$
   Applying this to the adapter-scaled sum:
   $$\left\| \sum_{k=1}^K (R_{k, l}(h) - R_{k, l}(\tilde{h})) A_k^{(l)} B_k^{(l)} \tilde{h} \right\|_2 \le \max_k \|A_k^{(l)} B_k^{(l)}\|_2 \|\tilde{h}\|_2 \|R_l(h) - R_l(\tilde{h})\|_1 \le C_A^{(l)} R_h \frac{2}{\tau_l} \|W_{\text{route}}^{(l)}\|_2 \|h - \tilde{h}\|_2$$
Combining both bounds yields the theorem. By Banach's Fixed-Point Theorem, restricting $L_{T_l} < 1$ guarantees that the representation trajectory is a contraction mapping, proving convergence to a unique fixed-point trajectory under depth and mathematically preventing ensembling jitter.

### The Contraction-Regularized Objective
To enforce the contraction property ($L_{T_l} < 1$), we define our training objective with explicit spectral and temperature regularizers:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{cal}} + \lambda_{\text{spec}} \sum_{l=1}^L \|W_{\text{route}}^{(l)}\|_F^2 + \lambda_{\text{temp}} \sum_{l=1}^L \frac{1}{\tau_l^2}$$
where $\mathcal{L}_{\text{cal}}$ is the calibration cross-entropy loss, and $\|W_{\text{route}}^{(l)}\|_F^2$ is the Frobenius norm (which acts as an analytical upper bound on the spectral norm squared, since $\|W_{\text{route}}^{(l)}\|_2 \le \|W_{\text{route}}^{(l)}\|_F$).

---

## 4. Architecture Specifications
- **Backbone Dimensions:** 14-layer sequential network depth ($L=14$), intermediate representation dimension $D=192$ (calibrated to the Analytical Coordinate Sandbox).
- **Expert Pool:** $K=4$ low-rank experts ($r=8$), representing specialized adapters for MNIST, FashionMNIST, CIFAR-10, and SVHN.
- **Routing Heads:** Independent linear routing matrices $W_{\text{route}}^{(l)} \in \mathbb{R}^{K \times D}$ placed at each layer $l \in \{1, \dots, L\}$.
- **Temperatures:** Layer-specific learnable temperature parameters $\tau_l$, initialized at $\tau_0 = 0.05$ and regularized under the inverse square penalty.
- **Inputs:** Batched streaming intermediate representation tensors $H^{(l-1)} \in \mathbb{R}^{B \times D}$.
- **Outputs:** Blended feedforward intermediate representations $H^{(l)} \in \mathbb{R}^{B \times D}$ and predicted coefficient vectors $\alpha^{(l)} \in \mathbb{R}^{B \times K}$.

---

## 5. Baselines
We will compare **CR-Router** against the following established model-merging and ensembling baselines:
1. **Expert Oracle Ceiling:** Evaluates each task's stream using its corresponding fully-specialized expert model, establishing the theoretical upper performance boundary (78.80%).
2. **Uniform Merging:** Fuses task-specific adapters statically with constant, uniform scaling coefficients $\alpha_k = 1/K$ across all layers.
3. **Linear Router (Unregularized):** Optimizes layer-wise routing weights without any spectral or temperature penalties, highlighting the baseline transductive overfitting behavior.
4. **SABLE (Late Adaptation):** Runs stateless activation-space ensembling via early-layer nearest-centroid projection, representing the state-of-the-art in stateless, robust ensembling.
5. **ChemMerge (Kinetic Routing):** Temporalizes routing coefficients using non-equilibrium chemical reaction rate equations, representing the state-of-the-art in continuous, heuristic trajectory smoothing.

---

## 6. Step-by-Step Interaction
For an incoming batched feature tensor $H^{(l-1)} \in \mathbb{R}^{B \times D}$ flowing from layer $l-1$:
1. **Routing Score Extraction:** The routing head at layer $l$ applies its projection matrix to extract logits:
   $$Z_{\text{route}}^{(l)} = H^{(l-1)} (W_{\text{route}}^{(l)})^T \in \mathbb{R}^{B \times K}$$
2. **Probability Mapping:** The logits are scaled by the layer-specific temperature $\tau_l$ and mapped to ensembling coefficients via Softmax:
   $$\alpha^{(l)}_b = \text{Softmax}\left( Z_{\text{route}, b}^{(l)} / \tau_l \right) \in \mathbb{R}^K$$
3. **Low-Rank Expert Execution:** The active low-rank expert adapters $A_k^{(l)}, B_k^{(l)}$ are executed in parallel on the incoming representation:
   $$\Delta H_k^{(l)} = H^{(l-1)} (B_k^{(l)})^T (A_k^{(l)})^T \in \mathbb{R}^{B \times D}$$
4. **Activation Blending:** The parallel expert updates are scaled element-wise by the sample-specific routing coefficients $\alpha^{(l)}_{k, b}$ and superimposed onto the shared base block forward representation:
   $$H^{(l)} = F_{\text{base}}^{(l)}(H^{(l-1)}) + \sum_{k=1}^K \text{Diag}(\alpha^{(l)}_k) \Delta H_k^{(l)} \in \mathbb{R}^{B \times D}$$
5. **State Propagation:** The stable, blended representations $H^{(l)}$ are propagated forward as input to the next layer $l+1$.
