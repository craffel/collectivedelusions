# PAC-Bayesian Smooth Trajectory Merging (PAC-STM)

## 1. Persona Alignment
This proposal is designed through the lens of **The Theorist**. It rejects empirical heuristics and unregularized empirical risk minimization (ERM) on tiny calibration sets, showing they are susceptible to local noise overfitting. Instead, we establish a rigorous learning-theoretic foundation for layer-wise dynamic model merging. 

We model the routing parameters across deep network layers as a probability distribution over trajectories. By defining a Markovian random walk prior, we prove that the resulting parameter-space Kullback-Leibler (KL) complexity penalty mathematically derives a first-order ensembling smoothness regularizer. This bridges the gap between randomized generalization guarantees and stable, deterministic trajectory optimization, satisfying our core mandate for provably correct and mathematically bounded learning systems.

---

## 2. Core Techniques
Our proposed framework, **PAC-STM**, introduces three core techniques to the dynamic model merging paradigm:
1. **Unit-Norm PCA Subspace Projection (UN-PCA-SEP):** This preprocessing step normalizes early-layer hidden representations $z_b \in \mathbb{R}^D$ to the unit sphere $\tilde{z}_b = \frac{z_b}{\|z_b\|_2 + \epsilon}$, and projects them onto task-specific principal component bases $V_{k, d} \in \mathbb{R}^{D \times d}$ extracted from disjoint calibration data. This bounds the extracted coordinate energies $e_{k, b} = \|V_{k, d}^T \tilde{z}_b\|_2 \in [0, 1]$, removing coordinate magnitude ambiguity and ensuring robust inputs under extreme heteroscedastic noise.
2. **Markovian Random Walk Trajectory Prior ($P$):** Rather than modeling layer-wise routing temperatures as independent variables (which scales parameter capacity to $L \times K$ and causes transductive overfitting), we model them as an auto-regressive Markov chain. The prior penalizes abrupt transitions between consecutive layers, naturally encoding physical depth-wise smoothness.
3. **Analytical Trajectory KL-Regularizer:** We derive the exact, closed-form KL-divergence between our Markovian trajectory prior and a layer-wise Gaussian posterior. This complexity penalty acts as a learning-theoretic, parameter-free smoothness penalty during log-temperature optimization.

---

## 3. Mathematical Formulation

Let $L$ be the number of layers in the deep network, and $K$ be the number of task experts. Let $\mathbf{w} \in \mathbb{R}^{LK}$ be the trajectory of log-temperatures, partitioned layer-wise into $\mathbf{w}_l = [w_{1, l}, \dots, w_{K, l}]^T \in \mathbb{R}^K$ for $l \in \{1, \dots, L\}$. 

### 3.1. The Trajectory Prior ($P$)
We formulate our prior $P$ over the trajectory space as a Gaussian random walk starting at a neutral, uncalibrated temperature log-scale $\mathbf{w}_0 = \ln(0.05) \cdot \mathbf{1}$:
$$\mathbf{w}_1 \sim \mathcal{N}(\mathbf{w}_0, \sigma_0^2 I_K)$$
$$\mathbf{w}_l | \mathbf{w}_{l-1} \sim \mathcal{N}(\mathbf{w}_{l-1}, \sigma^2 I_K), \quad \forall l \in \{2, \dots, L\}$$
The joint probability density of the prior is:
$$p(\mathbf{w}) = \frac{1}{(2\pi \sigma_0^2)^{K/2}} \exp\left( -\frac{\|\mathbf{w}_1 - \mathbf{w}_0\|_2^2}{2\sigma_0^2} \right) \prod_{l=2}^L \frac{1}{(2\pi \sigma^2)^{K/2}} \exp\left( -\frac{\|\mathbf{w}_l - \mathbf{w}_{l-1}\|_2^2}{2\sigma^2} \right)$$

### 3.2. The Trajectory Posterior ($Q$)
We define our learned posterior distribution $Q$ centered at our optimized trajectory means $\mathbf{u} = (\mathbf{u}_1, \dots, \mathbf{u}_L) \in \mathbb{R}^{LK}$ with the same isotropic step variance:
$$\mathbf{w}_1 \sim \mathcal{N}(\mathbf{u}_1, \sigma_0^2 I_K)$$
$$\mathbf{w}_l \sim \mathcal{N}(\mathbf{u}_l, \sigma^2 I_K), \quad \text{independent across layers for } l \ge 2$$
The joint probability density of the posterior is:
$$q(\mathbf{w}) = \frac{1}{(2\pi \sigma_0^2)^{K/2}} \exp\left( -\frac{\|\mathbf{w}_1 - \mathbf{u}_1\|_2^2}{2\sigma_0^2} \right) \prod_{l=2}^L \frac{1}{(2\pi \sigma^2)^{K/2}} \exp\left( -\frac{\|\mathbf{w}_l - \mathbf{u}_l\|_2^2}{2\sigma^2} \right)$$

### 3.3. Theorem: Closed-Form Trajectory KL Divergence
The Kullback-Leibler divergence between the Markovian trajectory posterior $Q$ and prior $P$ is given exactly by:
$$\text{KL}(Q \| P) = \frac{1}{2\sigma_0^2} \|\mathbf{u}_1 - \mathbf{w}_0\|_2^2 + \frac{1}{2\sigma^2} \sum_{l=2}^L \|\mathbf{u}_l - \mathbf{u}_{l-1}\|_2^2 + \frac{L-1}{2} K$$

#### Proof:
By definition of the KL-divergence over continuous densities:
$$\text{KL}(Q \| P) = \mathbb{E}_{\mathbf{w} \sim Q} \left[ \ln \frac{q(\mathbf{w})}{p(\mathbf{w})} \right]$$
Taking the logarithm of the ratio of joint densities:
$$\ln \frac{q(\mathbf{w})}{p(\mathbf{w})} = \frac{1}{2\sigma_0^2} \left( \|\mathbf{w}_1 - \mathbf{w}_0\|_2^2 - \|\mathbf{w}_1 - \mathbf{u}_1\|_2^2 \right) + \sum_{l=2}^L \frac{1}{2\sigma^2} \left( \|\mathbf{w}_l - \mathbf{w}_{l-1}\|_2^2 - \|\mathbf{w}_l - \mathbf{u}_l\|_2^2 \right)$$
Under the posterior distribution $\mathbf{w} \sim Q$, we represent $\mathbf{w}_l = \mathbf{u}_l + \boldsymbol{\epsilon}_l$, where $\boldsymbol{\epsilon}_1 \sim \mathcal{N}(0, \sigma_0^2 I_K)$ and $\boldsymbol{\epsilon}_l \sim \mathcal{N}(0, \sigma^2 I_K)$ are independent. We evaluate the expectation of each term:
1. For the first term:
   $$\mathbb{E}_{Q} [\|\mathbf{w}_1 - \mathbf{w}_0\|_2^2] = \mathbb{E} [\|\mathbf{u}_1 + \boldsymbol{\epsilon}_1 - \mathbf{w}_0\|_2^2] = \|\mathbf{u}_1 - \mathbf{w}_0\|_2^2 + \mathbb{E}[\|\boldsymbol{\epsilon}_1\|_2^2] = \|\mathbf{u}_1 - \mathbf{w}_0\|_2^2 + \sigma_0^2 K$$
   $$\mathbb{E}_{Q} [\|\mathbf{w}_1 - \mathbf{u}_1\|_2^2] = \mathbb{E}[\|\boldsymbol{\epsilon}_1\|_2^2] = \sigma_0^2 K$$
   Subtracting these yields:
   $$\mathbb{E}_{Q} \left[ \frac{\|\mathbf{w}_1 - \mathbf{w}_0\|_2^2 - \|\mathbf{w}_1 - \mathbf{u}_1\|_2^2}{2\sigma_0^2} \right] = \frac{1}{2\sigma_0^2} \|\mathbf{u}_1 - \mathbf{w}_0\|_2^2$$
2. For each step term $l \ge 2$:
   $$\mathbb{E}_{Q} [\|\mathbf{w}_l - \mathbf{w}_{l-1}\|_2^2] = \mathbb{E} [\|\mathbf{u}_l + \boldsymbol{\epsilon}_l - \mathbf{u}_{l-1} - \boldsymbol{\epsilon}_{l-1}\|_2^2]$$
   Since $\boldsymbol{\epsilon}_l$ and $\boldsymbol{\epsilon}_{l-1}$ are independent:
   $$\mathbb{E}_{Q} [\|\mathbf{w}_l - \mathbf{w}_{l-1}\|_2^2] = \|\mathbf{u}_l - \mathbf{u}_{l-1}\|_2^2 + \mathbb{E}[\|\boldsymbol{\epsilon}_l\|_2^2] + \mathbb{E}[\|\boldsymbol{\epsilon}_{l-1}\|_2^2] = \|\mathbf{u}_l - \mathbf{u}_{l-1}\|_2^2 + 2\sigma^2 K$$
   $$\mathbb{E}_{Q} [\|\mathbf{w}_l - \mathbf{u}_l\|_2^2] = \mathbb{E}[\|\boldsymbol{\epsilon}_l\|_2^2] = \sigma^2 K$$
   Subtracting these yields:
   $$\mathbb{E}_{Q} \left[ \frac{\|\mathbf{w}_l - \mathbf{w}_{l-1}\|_2^2 - \|\mathbf{w}_l - \mathbf{u}_l\|_2^2}{2\sigma^2} \right] = \frac{1}{2\sigma^2} \|\mathbf{u}_l - \mathbf{u}_{l-1}\|_2^2 + \frac{1}{2} K$$
Summing across all $L$ layers yields the theorem. $\blacksquare$

### 3.4. The Generalized PAC-Bayesian Trajectory Bound
Let $\mathcal{C} = \{(\mathbf{e}_s, y_s)\}_{s=1}^N$ be a calibration set of size $N$, and $\delta \in (0,1)$ be the confidence parameter. The empirical risk at log-temperatures $\mathbf{w}$ is:
$$\hat{R}_N(\mathbf{w}) = \frac{1}{N} \sum_{s=1}^N \mathcal{L}_{\text{route}}(\mathbf{e}_s, y_s; \mathbf{w})$$
where $\mathcal{L}_{\text{route}}$ is the sample routing cross-entropy. By McAllester's theorem, with probability at least $1-\delta$:
$$\mathbb{E}_{\mathbf{w} \sim Q} [R(\mathbf{w})] \le \mathbb{E}_{\mathbf{w} \sim Q} [\hat{R}_N(\mathbf{w})] + \sqrt{\frac{\text{KL}(Q \| P) + \ln\left(\frac{2\sqrt{N}}{\delta}\right)}{2N}}$$
Using our derived closed-form KL-divergence, the optimized mean trajectory $\mathbf{u}^*$ is obtained by directly minimizing this learning-theoretic bound on the calibration set. Bounding the first-order difference $\|\mathbf{u}_l - \mathbf{u}_{l-1}\|_2^2$ limits the complexity of the trajectory, provably preventing local stream overfitting.

---

## 4. Architecture Specifications
* **Backbone Network:** A pre-trained, frozen 14-layer Coordinate Sandbox with representation dimensionality $D = 192$.
* **Expert Models ($K=4$):** Task experts representing MNIST, Fashion-MNIST, CIFAR-10, and SVHN, fine-tuned to convergence via LoRA.
* **Routing Layer ($l_{\text{route}}$):** Positioned early at layer $4$ to bypass the routing paradox.
* **Coordinate Space:** $K$-dimensional unit-norm PCA subspace coordinates extracted using Unit-Norm PCA Subspace Projection (UN-PCA-SEP).
* **Layer-wise Routing Policy:** For each layer $l \in \{1, \dots, L\}$, the ensembling coefficients $\alpha_{k}(l)$ for task $k$ are dynamically generated per-sample using the Gibbs routing policy:
  $$\alpha_k(l; \mathbf{w}_l) = q_{k, l}(\mathbf{e}_b; \mathbf{w}_l) = \frac{\exp(e_{k, b} / e^{w_{k, l}})}{\sum_{j=1}^K \exp(e_{j, b} / e^{w_{j, l}})}$$
* **Trajectory Parameters:** Mean log-temperature parameters $\mathbf{u} \in \mathbb{R}^{L \times K}$ ($14 \times 4 = 56$ parameters total), optimized using Adam with a learning rate of $1\times10^{-3}$ for $1000$ steps over the calibration set.
* **Hyperparameters:** Isotropic prior variance $\sigma_0^2 = 5.0$, transition variance $\sigma^2 = 0.5$, and confidence $\delta = 0.05$.

---

## 5. Baselines
We compare **PAC-STM** against a comprehensive set of baselines representing both static, wave-based, and learning-theoretic model merging:
1. **Uniform Merging:** Static weight-space average ($\alpha_k = 0.25$).
2. **QWS-Merge:** Quantum Wavefunction Superposition Merging, representing task experts as orthogonal quantum states.
3. **Linear Router (Reg):** Parameterized linear routing head optimized on the calibration set with L2 regularization.
4. **PFSR (Weight Merging):** Parameter-Free Subspace Routing, using weight-space dynamic interpolation based on head activations.
5. **SABLE (SEP-Block/PCA):** Dynamic activation-blending baseline with early-centroid cosine similarities and a static temperature scale $\tau = 0.05$.
6. **PAC-ZCA (Ours - Global Baseline):** Global temperature-only PAC-Bayesian bound calibration (representing the single-temperature trajectory constraint).
7. **Layer-wise Temp-Only ERM:** A layer-specific routing policy optimized on the calibration set via standard Empirical Risk Minimization (cross-entropy) without any smoothness or KL complexity regularizers.

---

## 6. Step-by-Step Interaction

The online inference data flow through **PAC-STM** operates as follows:
1. **Early Feature Extraction:** An input sample $x_b$ is processed by the shared, frozen base model backbone up to the routing layer $l_{\text{route}} = 4$, generating the hidden representation vector $z_b \in \mathbb{R}^{192}$.
2. **Unit-Norm Regularization:** To eliminate task-specific noise spreads and prevent norm collapse, the vector is normalized: $\tilde{z}_b = \frac{z_b}{\|z_b\|_2 + \epsilon}$.
3. **Subspace Energy Projection:** The projection energy coordinate $e_{k, b} = \|V_{k, d}^T \tilde{z}_b\|_2 \in [0, 1]$ is computed for each task $k \in \{1, \dots, K\}$ using pre-computed orthogonal projection bases $V_{k, d}$. This maps the representation to a stable, bounded coordinate vector $\mathbf{e}_b \in \mathbb{R}^K$.
4. **Trajectory Routing Generation:** At each layer $l \in \{1, \dots, L\}$, the sample-specific coordinate $\mathbf{e}_b$ is combined with the optimized layer-specific log-temperature mean $\mathbf{u}_l$ to generate the ensembling probability vector:
   $$\boldsymbol{\alpha}(l) = \text{Softmax}(\mathbf{e}_b \odot e^{-\mathbf{u}_l})$$
5. **Activation Blending:** The intermediate activation tensor $A_{l-1}$ at layer $l-1$ is passed through each independent task adapter block $E_k(l)$ to generate task-specific activations. The final input activation for layer $l+1$ is blended dynamically:
   $$A_{l} = \sum_{k=1}^K \alpha_k(l) \cdot E_k(l)(A_{l-1})$$
6. **Final Classification:** This blended activation flows feedforward through the remaining frozen backbone layers to produce the final task-specific prediction head outputs.

---
