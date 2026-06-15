# Chaos-Theoretic Attractor Merging (ChaosMerge)

## 1. Persona Alignment
ChaosMerge is a radical departure from traditional static parameter merging and standard feed-forward dynamic routers. Guided by **The Visionary** persona, it completely rethinks the fundamental assumptions of parameter alignment and routing:
- **Radical Paradigm Shift:** Instead of using flat, linear Euclidean arithmetic or classical neural networks to determine parameter fusion, ChaosMerge frames the sequence of layer groups in a neural network as discrete time-steps in a highly non-linear, chaotic dynamical system.
- **Out-of-the-Box Inspiration:** It draws inspiration from chaos theory and statistical physics, specifically modeling merging coefficients as the trajectory of a Coupled Map Lattice (CML) driven by a chaotic Logistic Map.
- **Zero Constraint Play:** Rather than playing it safe with monotonic linear routes, ChaosMerge embraces chaotic trajectories that settle into specialized task attractors, prioritizing absolute novelty and structural exploration.

## 2. Core Techniques
ChaosMerge introduces several key techniques to weight-space model fusion:
1. **Coupled Map Lattice (CML):** A spatio-temporal dynamical system where discrete state variables are arranged on a lattice and evolve through a local chaotic map coupled with neighboring nodes. In ChaosMerge, each task expert corresponds to a node in the lattice, and the lattice state at each layer group determines that expert's merging coefficient.
2. **Chaotic Logistic Map Evolution:** Nodes evolve via the fully chaotic Logistic Map $f(x) = 4x(1-x)$, which maps the interval $[0, 1]$ onto itself. This introduces high sensitivity to the input features, helping the network separate task domains.
3. **Logit-Space Feature Perturbation:** To guide the chaotic trajectory without destroying the bounded phase space, input representations are projected into low-dimensional perturbations and injected as additive biases within the logit-transformed domain of the lattice state.
4. **Frozen Random Sphere Projections:** Following QWS-Merge, we project high-dimensional pooled backbone features onto a low-dimensional unit sphere via a frozen random matrix, ensuring extreme regularization and low parameter count.

## 3. Mathematical Formulation
Let $W_{base}^{(l)}$ be the base model weights at layer group $l \in \{1, \dots, L\}$. Let $V_k^{(l)} = W_k^{(l)} - W_{base}^{(l)}$ be the task vector for expert $k \in \{1, \dots, K\}$ at layer group $l$.

For each input sample $b \in \{1, \dots, B\}$ in a batch, we define a lattice of $K$ nodes. The state of the lattice at layer group $l$ is denoted by $s_b^{(l)} \in [0, 1]^K$, where $s_{k, b}^{(l)}$ represents the state of node $k$ (the expertise level of task $k$).

### 3.1. Sphere-Projected Feature Extraction
Let $z(x)_b \in \mathbb{R}^D$ be the spatially averaged patch tokens from the backbone's frozen patch embedding layer for sample $b$. We project $z(x)_b$ into a $d$-dimensional phase-space (with $d = K = 4$) via a frozen random projection matrix $P \in \mathbb{R}^{D \times d}$ and normalize it to the unit sphere:
$$\tilde{\psi}(x)_b = z(x)_b P \in \mathbb{R}^d$$
$$\psi(x)_b = \frac{\tilde{\psi}(x)_b}{\|\tilde{\psi}(x)_b\|_2 + \epsilon} \in \mathbb{R}^d$$
where $\epsilon = 10^{-8}$.

### 3.2. Lattice State Initialization (Lattice Pre-heating)
At the initial layer step $l = 0$, we initialize the lattice state $s_b^{(0)} \in [0, 1]^K$ via a learned linear projection of the input phase state:
$$\tilde{s}_{k, b}^{(0)} = W_{init, k} \psi(x)_b + b_{init, k}$$
$$s_{k, b}^{(0)} = \sigma(\tilde{s}_{k, b}^{(0)}) = \frac{1}{1 + e^{-\tilde{s}_{k, b}^{(0)}}}$$
where $W_{init} \in \mathbb{R}^{K \times d}$ and $b_{init} \in \mathbb{R}^K$ are trainable initializers.

### 3.3. Discrete Chaotic Trajectory Update
For each subsequent layer group $l \in \{1, \dots, L\}$, the state $s_{k, b}^{(l)}$ is updated through a Coupled Map Lattice step. 
First, we compute the spatial-chaotic coupling of the nodes:
$$\bar{s}_{k, b}^{(l)} = (1 - \gamma_l) f\left(s_{k, b}^{(l-1)}\right) + \frac{\gamma_l}{K} \sum_{j=1}^K f\left(s_{j, b}^{(l-1)}\right)$$
where:
- $f(u) = 4u(1-u)$ is the chaotic logistic map.
- $\gamma_l \in [0, 1]$ is a learned layer-wise spatial coupling coefficient (initialized to $0.1$ via a clamped sigmoid), representing the local diffusion/cooperation of task nodes.

Next, we inject the localized input perturbation in logit-space to steer the chaotic orbit:
$$s_{k, b}^{(l)} = \sigma\left( \sigma^{-1}\left( \text{clip}\left(\bar{s}_{k, b}^{(l)}, \delta, 1-\delta\right) \right) + \langle \psi(x)_b, \Phi_k^{(l)} \rangle + \phi_k^{(l)} \right)$$
where:
- $\sigma^{-1}(y) = \ln\left(\frac{y}{1-y}\right)$ is the logit function.
- $\delta = 10^{-5}$ is a numerical stabilizer.
- $\Phi_k^{(l)} \in \mathbb{R}^d$ is a learned task projection key for expert $k$ at layer $l$, defining the attractor basin orientation.
- $\phi_k^{(l)} \in \mathbb{R}$ is a learned task bias for expert $k$ at layer $l$.

### 3.4. Wavefunction Collapse & Weight Assembly
To resolve accelerator limitations, we perform a mean-measurement over the batch dimension, collapsing sample-wise chaotic states into batch-level coefficients $\bar{\alpha}_k(l)$:
$$\alpha_{k, b}(l) = R_k^{(l)} \cdot s_{k, b}^{(l)}$$
$$\bar{\alpha}_k(l) = \frac{1}{B} \sum_{b=1}^B \alpha_{k, b}(l)$$
where $R_k^{(l)} \in \mathbb{R}$ is a learned layer-wise scaling amplitude initialized to $0.3$.

The dynamically assembled weights for layer $l$ are:
$$W_{merged}^{(l)}(x) = W_{base}^{(l)} + \sum_{k=1}^K \bar{\alpha}_k(l) V_k^{(l)}$$

## 4. Architecture Specifications
We employ the Vision Transformer backbone $\mathtt{vit\_tiny\_patch16\_224}$ ($L=14$ layer groups: 1 patch embedding, 12 transformer blocks, 1 LN layer; $D=192$).
- **Lattice Dimension:** $K = 4$ nodes (MNIST, FashionMNIST, CIFAR-10, SVHN).
- **Phase-space Projection:** Frozen random matrix $P \in \mathbb{R}^{192 \times 4}$, projecting backbone embeddings to $d=4$.
- **Trainable Parameters:**
  - $W_{init} \in \mathbb{R}^{4 \times 4}$ ($16$ parameters) and $b_{init} \in \mathbb{R}^4$ ($4$ parameters).
  - Layer-wise coupling coefficients $\gamma_l \in \mathbb{R}$ ($14$ parameters).
  - Scaling amplitudes $R_k^{(l)} \in \mathbb{R}$ ($14 \times 4 = 56$ parameters).
  - Attractor keys $\Phi_k^{(l)} \in \mathbb{R}^4$ ($14 \times 4 \times 4 = 224$ parameters) and biases $\phi_k^{(l)} \in \mathbb{R}$ ($14 \times 4 = 56$ parameters).
- **Total Parameter Count:** Exactly $370$ parameters. This incredibly compact footprint ensures high sample efficiency and rapid, non-overfitting optimization on tiny calibration sets.

## 5. Baselines
We evaluate ChaosMerge against five competitive baselines:
1. **Individual Experts (Ceiling):** Task-specialized networks representing the performance upper bound.
2. **Uniform Merging (Task Arithmetic):** Static uniform merging ($\lambda=0.3$).
3. **AdaMerging (Unsupervised TTA):** Test-time optimization of 56 layer-wise coefficients via entropy minimization.
4. **OFS-Tune (Supervised Static):** Supervised static layer-wise coefficient optimization on the 64-sample calibration set.
5. **Linear Router (Classical Baseline):** Input pooled representations mapped to routing weights via a soft linear layer.
6. **QWS-Merge (Quantum Wavefunction Superposition):** The previous state-of-the-art wave-like dynamic model merging baseline.

## 6. Step-by-Step Interaction
1. **Input Representation:** An input batch $x \in \mathbb{R}^{B \times C \times H \times W}$ is passed through the frozen patch embedding layer. Spatially averaged features $z(x)_b \in \mathbb{R}^D$ are computed.
2. **Phase Projection:** Average features are projected via frozen matrix $P$ and normalized to obtain unit-sphere phase states $\psi(x)_b \in \mathbb{R}^d$.
3. **Lattice Pre-heating:** Initial lattice state $s_b^{(0)}$ is set using the learned projection $W_{init}$ and Sigmoid.
4. **Dynamical Trajectory Propagation:** For each layer group $l = 1 \dots L$:
   - Apply the logistic chaotic map $f(x)$ element-wise to the previous state.
   - Blend states across the lattice nodes with coupling factor $\gamma_l$.
   - Transform states to logit space, inject the localized perturbation $\langle \psi(x)_b, \Phi_k^{(l)} \rangle + \phi_k^{(l)}$, and map back with Sigmoid to get $s_b^{(l)}$.
5. **Averaging & Scaling:** Multiply states by $R_k^{(l)}$ and average across the batch to yield classical coefficients $\bar{\alpha}_k(l)$.
6. **Weight Assembly & Forward Pass:** Construct the merged weight matrix $W_{merged}^{(l)}$ and process the layer activations. Repeat for all blocks.
