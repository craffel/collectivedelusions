# Evolutionary Symbiotic Merging via Lotka-Volterra Cooperation (ESM-LVC)

## 1. Persona Alignment
Under **The Visionary** persona, we seek to fundamentally disrupt the established paradigm of dynamic model merging. While prior methods (such as SABLE and SPS-ZCA) view dynamic routing as a static feedforward mapping or classification task, **ESM-LVC** completely rethinks this assumption. We argue that task-specific experts fine-tuned from a shared backbone do not exist in isolated, competitive vacuum states; instead, they are biological-like symbionts that co-exist within the shared parameter space of the neural network. 

By importing non-linear dynamical systems and evolutionary biology directly into activation space, ESM-LVC models model ensembling as a self-organizing ecosystem. The merging coefficients are modeled as dynamic species populations governed by a **Lotka-Volterra competition-cooperation framework**. This represents a radical paradigm shift from feedforward linear routing to dynamic, non-linear activation self-assembly, introducing an organic, self-regulating feedback mechanism that naturally resolves multi-task representation collapse.

## 2. Core Techniques
ESM-LVC introduces three interconnected, training-free, and computationally minimalist techniques:
1. **Lotka-Volterra Activation Dynamics (LVAD):** A non-linear dynamical system that models the activation levels of the $K$ task-specific experts as species populations. The expert activations evolve over a localized virtual time scale based on their mutual cooperative and competitive relationships.
2. **Symbiotic Interaction Tensor (SIT):** A pre-computed semantic matrix $\Gamma \in \mathbb{R}^{K \times K}$ that governs the ecology of the experts. Similar tasks cooperate mutualistically to reinforce each other's activation channels, while conflicting tasks compete aggressively, triggering competitive exclusion (mutual suppression) to prevent representation collapse.
3. **Discrete Euler Symbiosis Solver (DESS):** An ultra-lightweight, parameter-free step solver that integrates the Lotka-Volterra differential equations on-the-fly inside the single forward pass of the model, requiring only a few element-wise vector-matrix operations on the CPU.

## 3. Mathematical Formulation

### Lotka-Volterra Activation Dynamics
For each input sample $b$ in a heterogeneous batch, the dynamic evolution of the expert ensembling coefficients $\alpha_{k, b} \in [0, 1]$ over virtual time $\tau$ is defined by the Lotka-Volterra competition-cooperation differential equations:
\begin{equation}
    \frac{d \alpha_{k, b}}{d \tau} = \alpha_{k, b} \left( u_{k, b} + \sum_{j=1}^K \Gamma_{k, j} \alpha_{j, b} - \beta_k \alpha_{k, b} \right)
\end{equation}
where:
- $\alpha_{k, b}(\tau)$ is the population density (activation coefficient) of Expert $k$ for sample $b$.
- $u_{k, b}$ is the "environmental resource" or domain affinity of sample $b$ for task $k$, computed via Zero-Shot Centroid Alignment (ZCA) on Layer 3 features:
  \begin{equation}
      u_{k, b} = \text{cos\_sim}(h^{(3)}_b, \mu^{(3)}_k)
  \end{equation}
- $\Gamma_{k, j}$ is the symbiotic interaction coefficient between Expert $k$ and Expert $j$.
- $\beta_k > 0$ is the self-limiting crowding factor representing carrying capacity (we set $\beta_k = 1.0$ globally).

### Symbiotic Interaction Tensor (SIT)
To make the system completely training-free and training-agnostic, the matrix $\Gamma$ is pre-computed offline during a one-time calibration phase. Let $\rho_{k, j} = \text{cos\_sim}(\mu^{(3)}_k, \mu^{(3)}_j)$ be the semantic similarity between the pre-computed Layer 3 task centroids of task $k$ and task $j$. We formulate $\Gamma_{k, j}$ as:
\begin{equation}
    \Gamma_{k, j} = \tanh\left( \lambda \cdot (\rho_{k, j} - \theta) \right)
\end{equation}
where:
- $\lambda > 0$ is an interaction scaling intensity factor (we set $\lambda = 10.0$).
- $\theta \in [-1, 1]$ is a neutral conflict threshold (we set $\theta = 0.5$).
  - If $\rho_{k, j} > \theta$, the experts exhibit mutualism ($\Gamma_{k, j} > 0$), allowing them to cooperatively reinforce each other's activations.
  - If $\rho_{k, j} < \theta$, the experts exhibit competitive exclusion ($\Gamma_{k, j} < 0$), causing them to aggressively suppress each other, which isolates conflicting expert pathways and protects task-specialization accuracy.

### Discrete Euler Symbiosis Solver (DESS)
To ensure ultra-low execution latency on edge hardware, the continuous differential equations are integrated over $N$ discrete steps (e.g., $N = 5$) using a localized Euler integration step with step size $\Delta \tau$ (e.g., $\Delta \tau = 0.2$):
\begin{equation}
    \alpha_{k, b}^{(t+1)} = \alpha_{k, b}^{(t)} + \Delta \tau \cdot \alpha_{k, b}^{(t)} \left( u_{k, b} + \sum_{j=1}^K \Gamma_{k, j} \alpha_{j, b}^{(t)} - \beta_k \alpha_{k, b}^{(t)} \right)
\end{equation}
where the initial population density is seeded from a temperature-scaled Softmax over the raw affinity coordinates:
\begin{equation}
    \alpha^{(0)}_{k, b} = \text{Softmax}(u_{k, b} / \tau_{\text{init}})
\end{equation}
where $\tau_{\text{init}} = 0.03$. After $N$ steps of DESS, the population densities are L1-normalized to preserve activation scale stability:
\begin{equation}
    \alpha^{\text{final}}_{k, b} = \frac{|\alpha_{k, b}^{(N)}|}{\sum_{j=1}^K |\alpha_{j, b}^{(N)}|}
\end{equation}

## 4. Architecture Specifications
- **Base Backbone:** Vision Transformer (`vit_tiny_patch16_224`) consisting of $L=12$ transformer blocks (14 sequential layer groups, including Patch Embeddings and the Head) with an intermediate representation dimension $D=192$.
- **Expert Adapters:** $K=4$ task-specific vision experts fine-tuned via Low-Rank Adaptation (LoRA) with rank $r=8$ on MNIST, Fashion-MNIST, CIFAR-10, and SVHN, respectively.
- **Paradox-Free Execution Layout:** The first three layers (Layers 1--3) are frozen and shared across all experts during training, containing zero LoRA adapters. Layers 4 to $L$ contain the specialized task adapters.
- **Routing Module:** Built into Layer 3. It extracts the intermediate pooled features $h^{(3)}_b$, executes ZCA to obtain $u_{k, b}$, and runs DESS to obtain $\alpha^{\text{final}}_{k, b}$.
- **Activation Blending:** Inside Layers 4--$L$, the LoRA adapter output activations are dynamically blended sample-wise using $\alpha^{\text{final}}_{k, b}$ inside a single parallel forward pass:
  \begin{equation}
      h_b^{(l)} = h_b^{(l-1)} W_{\text{base}}^{(l)} + \sum_{k=1}^K \alpha^{\text{final}}_{k, b} \left( h_b^{(l-1)} A_k^{(l)} B_k^{(l)} \right)
  \end{equation}

## 5. Baselines
We evaluate ESM-LVC against five key baselines in the Isolating Coordinate Sandbox (ICS):
1. **Expert Ceiling (0 params):** The maximum possible accuracy, routing each sample directly to its respective fully isolated expert.
2. **Uniform Merging (0 params):** Static weight-space merging where expert weights are uniformly averaged: $\bar{w} = W_{\text{base}} + \frac{1}{K} \sum \Delta W_k$.
3. **Linear Router (Reg) (10,752 params):** A parametric classical routing head regularized with weight decay.
4. **SABLE (0 params):** A minimalist activation blending method using simple cosine similarity projections directly.
5. **SPS-ZCA (0 params):** The prior state-of-the-art framework that uses Zero-Shot Centroid Alignment with static sharp temperature routing. This represents our direct parent baseline; we aim to show that our ecological Lotka-Volterra dynamics achieve superior routing precision and higher robustness against noise.

## 6. Step-by-Step Interaction
1. **Calibration (Offline):** 
   - Extract intermediate pooled activations $h^{(3)}_s$ at Layer 3 from a tiny calibration split $\mathcal{C}_k$ ($|\mathcal{C}_k| = 64$) for each task $k$.
   - Compute the task centroids $\mu^{(3)}_k \in \mathbb{R}^{192}$.
   - Compute the pairwise cosine similarities between all centroids and populate the Symbiotic Interaction Tensor $\Gamma$.
2. **Early Inference Pass:** An incoming heterogeneous batch of images is fed into the shared base model. The shared, adapter-free blocks (Layers 1--3) are executed, outputting Layer 3 activations $h^{(3)}_b$.
3. **Environmental Attraction:** Compute the cosine similarity coordinates $u_{k, b} = \text{cos\_sim}(h^{(3)}_b, \mu^{(3)}_k)$ for each sample $b$.
4. **Symbiotic Adaptation:** Run $N$ discrete steps of DESS to solve the Lotka-Volterra ecosystem state, modeling the competitive and cooperative interactions of the experts. Perform L1 normalization to generate the final ensembling coefficients $\alpha^{\text{final}}_{k, b}$.
5. **Single-Pass Blending (SPS):** Propagate features through Layers 4 to $L$. At each layer, execute all expert LoRA adapters in parallel, blending their outputs dynamically using $\alpha^{\text{final}}_{k, b}$ inside a single forward pass.
6. **Task Classification:** Pass the blended penultimate activations through the expert-specific classification heads to yield the final predicted task labels.
