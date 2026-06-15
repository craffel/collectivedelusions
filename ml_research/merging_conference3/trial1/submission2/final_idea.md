# Demystifying SAIM: A Methodological Dissection of Sharpness-Aware Isotropic Merging

## 1. Persona Alignment
We adopt the persona of **The Methodologist**. As a Methodologist, we are highly skeptical of complex, multi-component "state-of-the-art" (SOTA) frameworks that claim synergistic benefits without rigorous ablation. SAIM (Sharpness-Aware Isotropic Merging) proposes a dual-stage solution to continual learning via model merging: a custom coordinate-wise optimizer (SA-BCD) and an SVD-based adaptive isotropic merging algorithm. 

Our goal is to critically dissect this framework to identify:
1. **Baseline Inflation:** Can standard optimizers like AdamW or standard SAM (Sharpness-Aware Minimization) achieve equivalent or superior results when basic hyperparameters (learning rate, weight decay) are properly tuned? Is the custom coordinate-wise masking in SA-BCD actually a redundant optimization constraint?
2. **Component Redundancy:** Is the SVD-based isotropic merging step truly performing meaningful subspace alignment, or is it acting as a high-frequency spectral filter/regularizer whose effects can be matched by simple scalar scaling or element-wise scaling?
3. **Decoupled Causal Drivers:** By crossing multiple optimizers with multiple merging algorithms, we will isolate whether the empirical gains are entirely driven by the optimization phase (finding flatter minima) or if the merging phase adds any statistically significant value.

This rigorous, skeptical dissection will expose potential confounding variables, evaluate the cost-benefit trade-off of SAIM's computational overhead (SVD is extremely expensive), and provide the community with a clear, honest picture of what actually drives model merging success in continual learning.

---

## 2. Core Techniques
We introduce a modular, multi-axial evaluation suite to isolate and analyze the components of SAIM. The techniques being examined and compared are:
1. **Sharpness-Aware Block Coordinate Descent (SA-BCD) Optimizer (SAIM 2026):** Selects top-$p\%$ parameters $\Omega_t$ based on absolute first-order momentum $|m_t|$, computes a SAM-style perturbation restricted to those parameters, and updates only $\Omega_t$.
2. **Standard Sharpness-Aware Minimization (SAM) (Foret et al., 2020):** Optimizes the worst-case neighborhood loss across all parameters. We use SAM as a simplified optimizer baseline to test if coordinate selection is redundant.
3. **Adaptive Isotropic Merging (SAIM 2026):** Computes SVD on combined layer weight updates, averages the singular value spectrum, and interpolates singular values toward their mean using a $1/\sqrt{t}$ decay factor.
4. **Spectral Dampening (Our Simplified Baseline):** A computationally lightweight scaling method that replaces SVD with a simple scalar decay factor $\gamma_t = \frac{1}{\sqrt{t}}$ or element-wise statistical scaling, testing if the SVD reconstruction's performance can be matched without the SVD overhead.
5. **Task Arithmetic / Euclidean Average:** A standard linear merging baseline where task vectors are averaged without modification.

---

## 3. Mathematical Formulation
Let $\theta_{t-1}$ be the model parameters after task $t-1$, and $\theta_0$ be the pre-trained initialization.
For a new task $T_t$, we fine-tune $\theta_{t-1}$ to obtain task-expert parameters $\theta_{T_t}$.
The task vector is defined as $\Delta_{T_t} = \theta_{T_t} - \theta_{t-1}$, and the cumulative historical update is $\Delta_{cum} = \theta_{t-1} - \theta_0$.

For each weight layer $k$, the combined update is:
$$\Delta^k_{com} = (1+\lambda)\Delta^k_{cum} + (1-\lambda)\Delta^k_{T_t}$$
where $\lambda$ is a balance hyperparameter (default $\lambda = 0$).

### 3.1 SAIM's SVD-based Isotropic Merging
We perform Singular Value Decomposition (SVD) on $\Delta^k_{com} \in \mathbb{R}^{d_{out} \times d_{in}}$:
$$\Delta^k_{com} = U^k \Sigma^k (V^k)^\top$$
where $\Sigma^k = \text{diag}(\sigma^k_1, \sigma^k_2, \dots, \sigma^k_r)$ are the singular values.
The mean singular value is computed as:
$$\bar{\sigma}^k = \frac{1}{r} \sum_{i=1}^r \sigma^k_i$$
We reconstruct the singular value matrix by interpolating with the mean using the current task index $t$:
$$\hat{\Sigma}^k_i = \bar{\sigma}^k + (\sigma^k_i - \bar{\sigma}^k) \times \frac{1}{\sqrt{t}}$$
The isotropic merged update is reconstructed as:
$$\Delta^k_{merged} = U^k \hat{\Sigma}^k (V^k)^\top$$
The final parameters are updated with a scaling factor $\alpha$:
$$\theta_t = \theta_0 + \alpha \Delta_{merged}$$

### 3.2 Our Spectral Dampening Alternative
To test if SVD is a redundant complexity, we propose a simple element-wise statistical scaling that mimics the spectral flattening effect. We define:
$$\Delta^k_{merged} = \gamma_t \Delta^k_{com}$$
where $\gamma_t = \frac{1}{\sqrt{t}}$ is a simple scalar decay factor. If this simple decay achieves comparable results to SVD-based interpolation, then the singular value decomposition is a redundant mathematical overhead.

### 3.3 Optimizer Formulations
We compare three optimizers:
1. **AdamW:**
   $$\theta_{t+1} = \theta_t - \eta \cdot \text{AdamW}(\nabla_\theta L(\theta_t, D))$$
2. **Standard SAM:**
   $$\epsilon^* = \rho \frac{\nabla_\theta L(\theta_t, D)}{\|\nabla_\theta L(\theta_t, D)\|_2}, \quad \theta_{t+1} = \theta_t - \eta \nabla_\theta L(\theta_t + \epsilon^*, D)$$
3. **SA-BCD (SAIM's custom optimizer):**
   $$m_t = \beta_1 m_{t-1} + (1-\beta_1)\nabla_\theta L(\theta_t, D), \quad \Omega_t = \text{Top-}p\%(|m_t|)$$
   $$\epsilon^*_{t, \Omega} = \rho \frac{\nabla_\theta L(\theta_t, D)_{\Omega_t}}{\|\nabla_\theta L(\theta_t, D)_{\Omega_t}\|_2}$$
   $$\theta_{t+1, i} = \theta_{t, i} - \eta \frac{\hat{m}_{t, i}}{\sqrt{\hat{v}_{t, i}} + \epsilon} \nabla_\theta L(\theta_t + \epsilon^*_{t,\Omega}, D)_i \quad (i \in \Omega_t)$$

---

## 4. Architecture Specifications
We evaluate our methodology on standardized continual learning architectures:
- **Model Backbone:** Vision Transformer (ViT-B/16) pre-trained on ImageNet.
  - Number of layers: 12 Transformer Blocks.
  - Hidden dimension: $D_{hidden} = 768$.
  - Number of Attention Heads: 12.
- **Classification Heads:** Task-specific linear classification heads mapping from the $768$-dimensional output CLS token.
- **Inputs:** Vision datasets (e.g., Split CIFAR-100 divided into 5 or 10 sequential tasks, or sequential diverse image datasets: CUB-200, Cars, Flowers-102). Inputs are resized to $3 \times 224 \times 224$.
- **Intermediate Representations:** Output features of the final Transformer layer ($768$ dimensions).
- **Final Outputs:** Probability distribution over task-specific class labels.

---

## 5. Baselines
Our evaluation framework crosses two independent axes: **Optimization Strategy** and **Merging Strategy**, resulting in a $3 \times 4$ grid of 12 distinct configurations.

### 5.1 Optimization Axis
1. **AdamW (Baseline):** Heavily tuned using a grid search over learning rates $\eta \in [1e-5, 1e-3]$ and weight decay $\lambda \in [1e-4, 1e-2]$.
2. **Standard SAM (Flatter Minima Baseline):** Sharpness-Aware Minimization with AdamW base optimizer, with tuned perturbation radius $\rho \in [0.01, 0.2]$.
3. **SA-BCD (SAIM Optimizer):** The proposed coordinate-wise sharpness-aware optimizer with default parameters ($p=30\%, \rho=0.05$).

### 5.2 Merging Axis
1. **Task Arithmetic (Linear Euclidean Average):** Naive average of task vectors $\Delta_{merged} = \frac{1}{t} \sum_{i=1}^t \Delta_{T_i}$.
2. **Ties-Merging (Sign-Agreed Merging):** Resolves parameter interference by keeping only parameters with sign agreement and scaling them.
3. **Isotropic Merging (SAIM Merging):** The SVD-based adaptive singular value spectrum balancing.
4. **Our Spectral Dampening Baseline:** Simple scalar scaling $\Delta^k_{merged} = \frac{1}{\sqrt{t}} \Delta^k_{com}$.

---

## 6. Step-by-Step Interaction
1. **Task Ingestion:** A new task $T_t$ and its corresponding training dataset $D_t$ are received.
2. **Task Expert Training:** The current merged parameters $\theta_{t-1}$ are fine-tuned on $D_t$ using one of the three optimizers (AdamW, SAM, or SA-BCD) to produce task expert $\theta_{T_t}$.
3. **Task Vector Extraction:** Extract the update vector for the current task: $\Delta_{T_t} = \theta_{T_t} - \theta_{t-1}$, and the cumulative historical vector: $\Delta_{cum} = \theta_{t-1} - \theta_0$.
4. **Layer-wise Update Combination:** For each layer $k$, compute the combined candidate update:
   $$\Delta^k_{com} = \Delta^k_{cum} + \Delta^k_{T_t}$$
5. **Merging Transformation:** Apply one of the merging algorithms to $\Delta^k_{com}$:
   - *If Task Arithmetic:* Keep $\Delta^k_{merged} = \Delta^k_{com}$.
   - *If Isotropic SVD:* Perform SVD on $\Delta^k_{com}$, interpolate the singular value spectrum toward its mean according to the task index $t$, and reconstruct.
   - *If Our Spectral Dampening:* Naively scale the combined weights by $\gamma_t = 1/\sqrt{t}$ to produce $\Delta^k_{merged}$.
6. **Parameter Synthesis:** Update the final model parameters:
   $$\theta_t = \theta_0 + \alpha \Delta_{merged}$$
7. **Comprehensive Multi-Task Evaluation:** Evaluate the updated model $\theta_t$ on the test sets of all tasks $T_1, T_2, \dots, T_t$ to record average accuracy, backward transfer (forgetting), and forward transfer.
