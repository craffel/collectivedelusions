# ThermoMerge: Thermodynamic Test-Time Diffusion for Synergistic Model Merging

## 1. Persona Alignment
ThermoMerge directly aligns with the traits and goals of **The Visionary** persona:
- **Rethinking Fundamental Assumptions**: Standard test-time model merging assumes that optimizing merging coefficients is a standard deterministic optimization task. ThermoMerge rejects this and instead views model merging as a thermodynamic phase transition where task-specific models transition from high-entropy, disorganized states to a unified, highly ordered crystalline multi-task state.
- **Unconventional Field Inspiration**: We draw deep inspiration from statistical mechanics, thermodynamic diffusion, and simulated annealing, transferring the principles of Stochastic Gradient Langevin Dynamics (SGLD) to escape sub-optimal local minima in the non-convex proxy loss landscape.
- **Willingness to Take Big Risks**: Rather than playing it safe with standard incremental gradient descent (e.g., Adam with slight hyperparameter tweaks), we introduce a completely fresh training/adaptation paradigm at test-time that explicitly injects temperature-scaled stochastic noise to explore the joint loss landscape globally.
- **Emphasizing Novelty & Paradigm Shifts**: The approach shifts the focus of model merging from "finding paths on a static landscape" to "physically cooling a chaotic multi-body system" until it crystallizes into the optimal configuration, offering a highly novel, conceptually rich perspective on the field.

---

## 2. Core Techniques
ThermoMerge introduces and modifies the following mechanisms:
- **Test-Time Joint Optimization**: Following *SyMerge (Jung et al., 2025)*, we jointly optimize both the layer-wise and task-wise merging coefficients $\Lambda = \{\lambda_k^l\}$ and the task-specific adapter/classifier layer $\Theta^{tr}$.
- **Stochastic Gradient Langevin Dynamics (SGLD)**: We replace standard deterministic optimizers (like Adam or SGD) with SGLD (*Welling & Teh, 2011*). At each update step, SGLD injects Gaussian noise scaled by a temperature $T_t$.
- **Simulated Annealing (Exponential Cooling Schedule)**: To ensure that the model shifts from global exploration (high temperature) to local convergence (low temperature) in the proxy loss landscape, we implement an exponential cooling schedule $T_t = T_0 \gamma^t$, allowing the parameters to crystallize in the flattest, most robust joint minima.
- **Stable Self-Labeling Supervision**: We adopt the expert-guided self-labeling cross-entropy objective from *SyMerge* to supervise the merged model without labels, avoiding the instability of prediction entropy minimization (*AdaMerging*).

---

## 3. Mathematical Formulation
Let $X^{te}$ be the unlabeled test dataset.
We have $K$ fine-tuned expert models, where the $k$-th model is denoted by $f(\cdot; \Theta_k)$.
The merged model encoder layers are parameterized by $\theta_{MTL}^l = \theta_{pre}^l + \sum_{k=1}^K \lambda_k^l \tau_k^l$, where $\tau_k^l = \theta_k^l - \theta_{pre}^l$ is the task vector at layer $l$, and $\lambda_k^l$ are the layer-wise merging coefficients.
The task-specific layers are parameterized by $\Theta^{tr} = \{\theta^{tr}_k\}_{k=1}^K$.

### 1. Proxy Test-Time Loss Function
We optimize our parameters using the expert-guided self-labeling cross-entropy loss:
$$\mathcal{L}_{TT}(\Lambda, \Theta^{tr}) = \frac{1}{|X^{te}|} \sum_{x \in X^{te}} \sum_{k=1}^K \mathcal{L}_{CE}\left( C_k^{merged}(x), C_k^{ft}(x) \right)$$
where $C_k^{merged}(x) = f^L(f^{1:L-1}(x; \Theta_{MTL}^{enc}); \theta^{tr}_k)$ is the merged model's output for task $k$, and $C_k^{ft}(x) = f(x; \Theta_k)$ is the fixed prediction from the corresponding fine-tuned expert teacher.

### 2. SGLD Update Equations
At each iteration $t$ of the test-time adaptation, the update rules for the merging coefficients $\Lambda$ and task-specific layers $\Theta^{tr}$ are:
$$\Lambda_{t+1} = \Lambda_t - \eta_{\Lambda} \nabla_{\Lambda} \mathcal{L}_{TT}(\Lambda_t, \Theta^{tr}_t) + \sqrt{2 \eta_{\Lambda} T_t} \cdot \epsilon_t$$
$$\Theta^{tr}_{t+1} = \Theta^{tr}_t - \eta_{\Theta} \nabla_{\Theta} \mathcal{L}_{TT}(\Lambda_t, \Theta^{tr}_t) + \sqrt{2 \eta_{\Theta} T_t} \cdot \epsilon'_t$$
where:
- $\eta_{\Lambda}, \eta_{\Theta}$ are the respective step sizes (learning rates).
- $\epsilon_t \sim \mathcal{N}(0, I_{\text{dim}(\Lambda)})$ and $\epsilon'_t \sim \mathcal{N}(0, I_{\text{dim}(\Theta^{tr})})$ are standard multivariate Gaussian noise vectors.
- $T_t$ is the temperature at step $t$.

### 3. Cooling Schedule
The temperature cools over iterations according to Simulated Annealing:
$$T_t = T_0 \cdot \gamma^t$$
where $T_0$ is the initial temperature (e.g., $1.0$), and $\gamma \in (0, 1)$ is the cooling rate (e.g., $0.95$). As $t \to \infty$, $T_t \to 0$, reducing SGLD to standard SGD/gradient updates in the crystalline phase.

---

## 4. Architecture Specifications
ThermoMerge is implemented on top of standard pre-trained architectures (e.g., CLIP ViT-B/32, ViT-L/14, or RoBERTa-base):
- **Encoder Backbone**: Shared across tasks, merged via layer-wise merging coefficients.
  - Number of layers ($L-1$): 12 (for ViT-B/32).
  - Merging coefficients $\Lambda$: Matrix of size $(L-1) \times K$. Initialized to $0.3$ (following *SyMerge*).
- **Task-Specific Adapters/Classifiers**: Unique versions for each task $k$, initialized to the fine-tuned classifier weights of task $k$.
  - Dimension for ViT-B/32 classifier: Input dimension 512, output dimension matches the number of classes for task $k$.
  - These layers are updated during test-time adaptation alongside $\Lambda$.
- **Noise Perturbations**: Applied coordinate-wise to both the scalar merging coefficients $\Lambda$ and the parameters of the task-specific layers $\Theta^{tr}$.

---

## 5. Baselines
We evaluate ThermoMerge against the following prominent, state-of-the-art model merging baselines:
1. **Task Arithmetic (Ilharco et al., 2023)**: A training-free baseline that applies simple linear combination of task vectors with a hand-tuned global scaling factor. Appropriate as the foundational starting point.
2. **AdaMerging (Yang et al., 2024b)**: A test-time adaptive model merging approach that optimizes coefficients via entropy minimization. Appropriate to show the superiority of self-labeling and Langevin diffusion over standard entropy optimization.
3. **SyMerge (Jung et al., 2025)**: The state-of-the-art test-time adaptive model merging method that jointly optimizes a single task-specific layer and merging coefficients via standard deterministic gradient descent (Adam) on self-labels. Appropriate to demonstrate that injecting thermodynamic noise (SGLD) successfully escapes sub-optimal sharp minima, achieving superior multi-task performance and better out-of-distribution robustness.

---

## 6. Step-by-Step Interaction
Data flows through the ThermoMerge system at test-time as follows:

1. **Initialization**: Initialize the merged model using the pre-trained weights, task vectors, starting merging coefficients $\Lambda_0$ (set to $0.3$), and task-specific classifiers $\Theta^{tr}_0$. Set initial temperature $T_0 = 1.0$ and cooling rate $\gamma = 0.95$.
2. **Batch Sampling**: Sample a batch of unlabeled test data $X^{te}$ from the target task distributions.
3. **Expert Forward Pass**: Pass $X^{te}$ through the $K$ fixed individual fine-tuned expert models to generate reference predictions $C_k^{ft}(x)$ (acting as soft self-labels).
4. **Merged Forward Pass**: 
   - Construct the merged encoder weights $\Theta_{MTL}^{enc}$ using current coefficients $\Lambda_t$.
   - Pass $X^{te}$ through the merged encoder to obtain shared feature representations.
   - Pass these shared features through the $K$ task-specific classifiers $\Theta^{tr}_t$ to obtain merged model predictions $C_k^{merged}(x)$.
5. **Loss Computation**: Compute the self-labeling cross-entropy loss $\mathcal{L}_{TT}(\Lambda_t, \Theta^{tr}_t)$ between $C_k^{merged}(x)$ and $C_k^{ft}(x)$.
6. **Gradient Backpropagation**: Backpropagate $\mathcal{L}_{TT}$ to compute exact gradients with respect to $\Lambda$ and $\Theta^{tr}$.
7. **Thermodynamic Perturbation**:
   - Sample Gaussian noise vectors $\epsilon_t \sim \mathcal{N}(0, I)$ and $\epsilon'_t \sim \mathcal{N}(0, I)$.
   - Multiply the noise vectors by the temperature-scaled factor $\sqrt{2 \eta T_t}$.
8. **Parameter Update (SGLD)**: Apply the gradients and noise perturbations to update the coefficients $\Lambda_{t+1}$ and task-specific classifiers $\Theta^{tr}_{t+1}$.
9. **Cooling Step**: Lower the temperature $T_{t+1} = T_t \cdot \gamma$.
10. **Iteration**: Repeat steps 2-9 for a designated number of steps (e.g., 200 steps) until the system cools to a stable crystalline state.
