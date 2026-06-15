# Idea Proposal: ThermoMerge (Thermodynamic Model Merging)

## 1. Persona Alignment
ThermoMerge is a radical departure from traditional model merging, which treats parameters as rigid Euclidean vectors and interpolates between them in a cold, zero-temperature landscape. As **The Visionary**, our core philosophy is that paradigm shifts occur by questioning foundational assumptions and drawing inspiration from diverse fields. 
Traditional model merging forces a straight-line Euclidean interpolation across highly non-convex loss boundaries, causing destructive interference and representation collapse (which, in statistical physics, is known as "system frustration"). Rather than making incremental tweaks to linear combinations, ThermoMerge completely rethinks the merging process through the lens of **statistical mechanics** and **thermodynamics**. By treating model predictions as a finite-temperature canonical ensemble, we replace static, deterministic parameter averaging with a dynamic, thermal-equilibrium process. We introduce a **Thermodynamic Annealing Schedule (TAS)** to actively flatten non-convex optimization barriers during test-time adaptation and minimize a novel **Helmholtz Free Energy Discrepancy (F-Min)** objective. This visionary approach bridges machine learning and statistical physics to unlock a completely new perspective on model merging.

---

## 2. Core Techniques
ThermoMerge introduces several foundational techniques:
- **Canonical Ensemble Mapping:** Translates the raw logits of the task-expert models into state probabilities in a canonical Boltzmann ensemble, where class-specific logits function as negative microstate energies (Boltzmann, 1868).
- **Helmholtz Free Energy Minimization (F-Min):** Derives and optimizes a physical free energy discrepancy loss on unlabeled test-time data streams, establishing a thermodynamic equilibrium between the merged model and the task experts.
- **Thermodynamic Annealing Schedule (TAS):** Employs a simulated cooling schedule (Kirkpatrick et al., 1983) during test-time optimization. High temperatures diffuse probabilities to flatten the non-convex optimization landscape and bridge disjoint basins of attraction; low temperatures sharpen the distribution to lock in specialized classification boundaries.
- **Layer-wise Thermal Coupling:** Introduces trainable, layer-specific merging coefficients $\lambda_{l,k}$ and layer-wise local temperatures $T_l$, permitting different transformer layers to operate at distinct levels of thermal excitation.

---

## 3. Mathematical Formulation
Let $f(x; \theta_k) \in \mathbb{R}^{C_k}$ represent the logits output of task-expert model $k \in \{1, \dots, K\}$ for input $x$, where $C_k$ is the number of classes.

### 3.1. Boltzmann Distribution & Partition Function
We define the canonical Boltzmann probability for class $c \in \{1, \dots, C_k\}$ under expert $k$ at temperature $T$ as:
$$p_c^{(k)}(x; T) = \frac{\exp\left( \frac{f_c(x; \theta_k)}{T} \right)}{Z_k(x; T)}$$
where $Z_k(x; T)$ is the canonical partition function:
$$Z_k(x; T) = \sum_{j=1}^{C_k} \exp\left( \frac{f_j(x; \theta_k)}{T} \right)$$
The corresponding Helmholtz Free Energy of the expert system is:
$$F_k(x; T) = -T \ln Z_k(x; T)$$

### 3.2. Merged Model Representation
For the merged model parameterized by $\theta_{MTL}$, the Boltzmann probability for task $k$ is:
$$p_c^{(MTL, k)}(x; T) = \frac{\exp\left( \frac{f_c(x; \theta_{MTL})}{T} \right)}{Z_{MTL, k}(x; T)}$$
where the partition function is:
$$Z_{MTL, k}(x; T) = \sum_{j=1}^{C_k} \exp\left( \frac{f_j(x; \theta_{MTL})}{T} \right)$$
The free energy of the merged model on task $k$ is $F_{MTL, k}(x; T) = -T \ln Z_{MTL, k}(x; T)$.

### 3.3. Helmholtz Free Energy Discrepancy Objective
During unsupervised test-time adaptation on unlabeled data streams $\mathcal{X}_k^{te}$, we minimize the Helmholtz Free Energy Discrepancy between the merged model and the expert ensemble. We show that the Free Energy Discrepancy is proportional to the Kullback-Leibler (KL) divergence at temperature $T$:
$$\mathcal{L}(\theta_{MTL}, T) = \sum_{k=1}^K \mathbb{E}_{x \in \mathcal{X}_k^{te}} \left[ T \cdot \mathcal{D}_{KL} \left( p^{(k)}(x; T) \parallel p^{(MTL, k)}(x; T) \right) \right]$$
Expanding the KL divergence:
$$T \cdot \mathcal{D}_{KL} \left( p^{(k)}(x; T) \parallel p^{(MTL, k)}(x; T) \right) = \sum_{c=1}^{C_k} p_c^{(k)}(x; T) \left( f_c(x; \theta_k) - f_c(x; \theta_{MTL}) \right) + \left( F_k(x; T) - F_{MTL, k}(x; T) \right)$$
The first term is the expectation of energy differences, and the second term is the Helmholtz free energy difference. This provides a physically grounded loss that dynamically aligns the partition functions.

### 3.4. Thermodynamic Annealing Schedule (TAS)
We parameterize the temperature as a dynamic function of optimization step $t \in \{1, \dots, N_{steps}\}$:
$$T(t) = T_{end} + (T_{start} - T_{end}) \cdot \exp(-\beta \cdot t)$$
where $T_{start} = 5.0$ (highly diffuse), $T_{end} = 1.0$ (standard inference scale), and $\beta > 0$ is the thermal cooling rate.

---

## 4. Architecture Specifications
ThermoMerge is evaluated on a pre-trained Vision Transformer (ViT-B/32) backbone.
- **Inputs:** A batch of images $x$ from unlabeled task-specific test streams.
- **Parameters to Merge:** We merge the entire network parameters layer-by-layer.
- **Merging Model:**
  Let $\theta_l^{base}$ and $\theta_{l,k}$ be the pre-trained and expert parameters of transformer layer $l \in \{1, \dots, L\}$. The merged parameters are combined via:
  $$\theta_l^{MTL} = \theta_l^{base} + \sum_{k=1}^K \lambda_{l,k} (\theta_{l,k} - \theta_l^{base})$$
  where $\lambda_{l,k}$ are the trainable layer-wise, task-wise coupling coefficients.
- **Optimization Parameters:** We jointly optimize the coupling coefficients $\lambda_{l,k}$ and the layer-wise local temperatures $T_l$ (initialized to $T_{start}$).
- **Output:** Multi-task classification logits across the combined head.

---

## 5. Baselines
We evaluate ThermoMerge against a comprehensive suite of competitive model merging baselines to demonstrate its superior performance and optimization behavior:
1. **Task Arithmetic (Ilharco et al., 2022):** The standard linear model merging baseline using a manually tuned, global scalar per task.
2. **AdaMerging (Yang et al., 2024):** A layer-wise test-time adaptive model merging framework that optimizes coefficients via entropy minimization on unlabeled target data.
3. **SyMerge (Jung et al., 2025):** The SOTA test-time adaptive model merging method using self-labeling/entropy minimization with teacher-student alignment.
4. **FoldMerge / Neural Origami (2026):** The exploratory non-linear coordinate-warping normalizing flow framework.

ThermoMerge is an appropriate, highly rigorous comparison because it operates under the exact same unsupervised test-time adaptation setting as SyMerge and FoldMerge, but replaces their heuristics with a physically grounded thermodynamic framework.

---

## 6. Step-by-Step Interaction
The flow of data and gradients through ThermoMerge proceeds as follows:
1. **Batch Sampling:** Sample a batch of unlabeled images $x_k \in \mathcal{X}_k^{te}$ for each vision classification task $k$.
2. **Expert Forward Pass:** Forward-pass $x_k$ through each frozen task expert $\theta_k$ to obtain specialized classification logits $f(x_k; \theta_k)$.
3. **Thermalization:** Compute the expert's partition functions $Z_k(x_k; T(t))$ and Boltzmann probability distributions $p^{(k)}(x_k; T(t))$ at the current step's physical temperature $T(t)$.
4. **Parameter Merging:** Merge the model parameters $\theta_l^{MTL}$ layer-by-layer using the current layer-wise coupling coefficients $\lambda_{l,k}$.
5. **Merged Forward Pass:** Forward-pass the images through the merged model $\theta^{MTL}$ to obtain logits $f(x_k; \theta^{MTL})$.
6. **Free Energy Calculation:** Compute the merged model's partition functions $Z_{MTL, k}(x_k; T(t))$ and Boltzmann distributions $p^{(MTL, k)}(x_k; T(t))$.
7. **Loss Evaluation:** Calculate the Helmholtz Free Energy Discrepancy loss $\mathcal{L}(\theta_{MTL}, T(t))$ over the mini-batch.
8. **Backward Pass & Update:** Backpropagate the gradients of the thermodynamic loss to update the trainable coupling parameters $\lambda_{l,k}$ and local temperatures $T_l$ via Adam GD, while updating the physical temperature $T(t)$ according to the annealing schedule.
