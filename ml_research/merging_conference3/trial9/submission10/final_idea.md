# Idea Proposal: Dirichlet-PAC

## 1. Persona Alignment
As **The Theorist**, I approach the test-time model ensembling problem through the lens of rigorous statistical learning theory and formal probability. Standard routing heuristics lack mathematical guarantees and are highly susceptible to overfitting under ultra-low-data calibration regimes (e.g., $N = 64$ samples). While existing regularized routers like PAC-ZCA utilize Gaussian posteriors over unconstrained log-temperatures, they fail to model uncertainty over the probability simplex itself, and their unconstrained nature can lead to gradient explosion or extreme entropy collapse.

**Dirichlet-PAC** addresses these limitations by modeling the ensembling weights directly as a random variable drawn from a Dirichlet distribution over the probability simplex $\Delta^{K-1}$. By deriving a closed-form, mathematically rigorous PAC-Bayesian generalization bound based on the Kullback-Leibler divergence between Dirichlet distributions, we provide the first learning-theoretic framework that guarantees generalization under simplex constraints. Our approach replaces empirical temperature-tuning heuristics with a principled, bounded optimization objective, aligning perfectly with the goal of bringing solid mathematical foundations and theoretical guarantees to deep representation ensembling.

## 2. Core Techniques
*   **Subspace Energy Projection (SEP)**: A completely unsupervised, task-agnostic dimensionality reduction protocol. We perform Singular Value Decomposition (SVD) on early-layer features of a calibration set to extract orthonormal bases $V_k \in \mathbb{R}^{D \times d}$ representing the $K$ task-specific feature manifolds. The projection energy $e_{k, b} = \|V_k^T z_b\|_2$ maps the high-dimensional feature $z_b \in \mathbb{R}^D$ to a compact coordinate vector $\mathbf{e}_b \in \mathbb{R}^K$.
*   **Dirichlet Routing Policy**: Instead of mapping features directly to ensembling weights via deterministic Softmax, we treat the ensembling weight vector $\boldsymbol{\alpha}_b \in \Delta^{K-1}$ for each query as a random vector distributed according to a Dirichlet distribution:
    $$\boldsymbol{\alpha}_b \sim \text{Dirichlet}(\mathbf{a}_b)$$
    where the concentration parameters $a_{k, b} > 0$ are driven by the task-specific learned temperatures $\tau_k > 0$ and the SEP coordinates $e_{k, b}$ as $a_{k, b} = \exp(e_{k, b} / \tau_k)$.
*   **Analytical Dirichlet PAC-Bayesian Complexity Control**: We establish a Dirichlet prior $P_b$ and Dirichlet posterior $Q_b$ over the ensembling weight simplex. By minimizing a rigorous PAC-Bayesian bound containing the exact analytical KL divergence between these Dirichlet distributions, we constrain the learned temperature parameters to remain near a stable, physically grounded prior state.
*   **Activation-Space Dynamic Blending**: A single-pass, parallel forward execution of expert adapters. Rather than merging parameters in weight space (which suffers from heterogeneity collapse), we run the adapters in parallel and blend their activation outputs using the expected ensembling weights $\mathbb{E}_{Q_b}[\alpha_{k, b}]$.

## 3. Mathematical Formulation

### 3.1. Subspace Coordinate Extraction
Let $z_b \in \mathbb{R}^D$ be the intermediate activation vector extracted at the routing layer $l_{\text{route}}$. We compute the unsupervised coordinate vector $\mathbf{e}_b = [e_{1, b}, \dots, e_{K, b}]^T$ as:
$$e_{k, b} = \|V_{k, d}^T z_b\|_2$$
where $V_{k, d} \in \mathbb{R}^{D \times d}$ consists of the top $d$ right-singular vectors of the calibration feature matrix for task $k$.

### 3.2. Simplex-Constrained Dirichlet Policy
We define a Dirichlet posterior distribution $Q_b = \text{Dirichlet}(\mathbf{a}_b)$ and a Dirichlet prior distribution $P_b = \text{Dirichlet}(\mathbf{a}_{0, b})$ over the ensembling weight simplex $\Delta^{K-1}$.
The posterior concentration parameters $\mathbf{a}_b = [a_{1, b}, \dots, a_{K, b}]^T$ are parameterized by learned task-specific temperatures $\boldsymbol{\tau} = [\tau_1, \dots, \tau_K]^T > 0$ as:
$$a_{k, b} = \exp\left( \frac{\tilde{e}_{k, b}}{\tau_k} \right)$$
The prior concentration parameters $\mathbf{a}_{0, b} = [a_{0, 1, b}, \dots, a_{0, K, b}]^T$ represent the uncalibrated baseline ensembling state with a static temperature scale $\tau_0 = 0.20$:
$$a_{0, k, b} = \exp\left( \frac{\tilde{e}_{k, b}}{\tau_0} \right)$$

The expected ensembling weight $\alpha_{k, b}$ for expert $k$ under the Dirichlet posterior is given analytically by:
$$\alpha_{k, b} = \mathbb{E}_{Q_b}[\alpha_{k, b}] = \frac{a_{k, b}}{\sum_{j=1}^K a_{j, b}}$$

### 3.3. Closed-Form Dirichlet KL Divergence
The parameter-space complexity of our router is measured by the Kullback-Leibler divergence between the posterior $Q_b$ and prior $P_b$. For Dirichlet distributions, this divergence is completely analytic and is given by:
$$D_{\text{KL}}(Q_b || P_b) = \ln \Gamma\left( \sum_{k=1}^K a_{k, b} \right) - \ln \Gamma\left( \sum_{k=1}^K a_{0, k, b} \right) - \sum_{k=1}^K \ln \Gamma(a_{k, b}) + \sum_{k=1}^K \ln \Gamma(a_{0, k, b}) + \sum_{k=1}^K (a_{k, b} - a_{0, k, b})\left[ \psi(a_{k, b}) - \psi\left( \sum_{j=1}^K a_{j, b} \right) \right]$$
where $\Gamma(\cdot)$ is the Gamma function and $\psi(\cdot)$ is the digamma function (the logarithmic derivative of the Gamma function, $\psi(x) = \frac{d}{dx} \ln \Gamma(x)$).

### 3.4. Dirichlet PAC-Bayesian Generalization Bound
Under McAllester's PAC-Bayesian theorem, for any delta $\delta \in (0, 1)$, the true expected risk $\mathcal{R}(G_{\boldsymbol{\tau}})$ of our ensembling model is bounded with probability at least $1 - \delta$ over the choice of calibration set of size $N$ by:
$$\mathcal{L}_{\text{PAC}}(\boldsymbol{\tau}) = \widehat{\mathcal{L}}_{\text{cal}}(\boldsymbol{\tau}) + \sqrt{\frac{\frac{1}{N} \sum_{b=1}^N D_{\text{KL}}(Q_b || P_b) + \ln \frac{2 \sqrt{N}}{\delta}}{2 N}}$$
where $\widehat{\mathcal{L}}_{\text{cal}}(\boldsymbol{\tau})$ is the empirical cross-entropy loss evaluated on the calibration set using the expected ensembling weights $\alpha_{k, b}$:
$$\widehat{\mathcal{L}}_{\text{cal}}(\boldsymbol{\tau}) = -\frac{1}{N} \sum_{b=1}^N \ln \left( \sum_{k=1}^K \alpha_{k, b} \cdot p_k(x_b) \right)$$
where $p_k(x_b)$ is the prediction probability of expert $k$ for the correct label of sample $b$.
To ensure gradient-based optimization over unconstrained real values, we optimize the log-temperature parameters $\mathbf{w} \in \mathbb{R}^K$, where $w_k = \ln \tau_k \iff \tau_k = e^{w_k}$.

## 4. Architecture Specifications
*   **Base Network (Backbone)**: Frozen 14-layer pre-trained network (e.g., Coordinate Sandbox) or Vision Transformer (ViT-Tiny/B-16) with hidden feature dimension $D = 192$.
*   **Routing Layer**: Layer $l_{\text{route}} = 3$ is selected as the early, shared, adapter-free routing layer.
*   **Task Experts**: $K = 4$ independent task-specific experts fine-tuned via Low-Rank Adaptation (LoRA) with rank $r = 8$. The experts are active only in downstream layers $l \ge 4$.
*   **Input Dimension**: Batches of size $B$ containing images or token sequences, mapped to intermediate activations $z_b \in \mathbb{R}^D$ at Layer 3.
*   **Intermediate Representation**: Unsupervised SEP energy coordinate vector $\mathbf{e}_b = [e_{1, b}, \dots, e_{K, b}]^T \in \mathbb{R}^K$.
*   **Output Blending Layer**: In downstream layers $l \in [4, L]$, intermediate activation states are dynamically blended across expert paths as:
    $$h^{(l)} = h^{(l-1)} + \sum_{k=1}^K \alpha_{k, b} \cdot \Delta h_k^{(l)}$$
    where $\Delta h_k^{(l)} = B_k^{(l)} A_k^{(l)} h^{(l-1)}$ is the activation delta output of Expert $k$'s LoRA adapter, and $\alpha_{k, b} = \mathbb{E}_{Q_b}[\alpha_{k, b}]$ is the expected ensembling weight.

## 5. Baselines
We evaluate Dirichlet-PAC against six prominent baselines:
1.  **Expert Ceiling (Oracle)**: An upper-bound baseline where each input query is perfectly dispatched to its isolated, task-specific expert.
2.  **Uniform Merging**: A static weight-space average baseline ($\alpha_k = 1/K = 0.25$), where all expert parameters are averaged.
3.  **SABLE (Raw Coords)**: The state-of-the-art activation-blending baseline using early-centroid cosine similarity coordinates with a static, hand-tuned temperature scale $\tau = 0.05$.
4.  **SABLE (SEP-Block)**: SABLE evaluated directly on our task-agnostic Subspace Energy Projection (SEP) features with a static, hand-tuned temperature scale $\tau = 0.05$.
5.  **Temp-Only ERM (Block)**: A strictly temperature-only router optimized on the same SEP features via standard Empirical Risk Minimization (ERM) (i.e. cross-entropy loss) on the calibration set, without any complexity penalty.
6.  **PAC-ZCA (Block)**: The Gaussian log-temperature PAC-Bayesian bound minimization baseline.

## 6. Step-by-Step Interaction

1.  **Feature Extraction**: The input query $x_b$ is passed through the early frozen shared layers of the backbone (Layers 1--3), yielding the intermediate representation $z_b \in \mathbb{R}^D$ at Layer 3.
2.  **Task-Space Projection**: We project $z_b$ onto the pre-computed orthonormal bases $V_{k, d}$ for each task $k \in \{1, \dots, K\}$ to extract the unsupervised energy coordinate:
    $$e_{k, b} = \|V_{k, d}^T z_b\|_2$$
3.  **Dirichlet Concentration Mapping**: The coordinate vector $\mathbf{e}_b = [e_{1, b}, \dots, e_{K, b}]^T$ is normalized to $\tilde{\mathbf{e}}_b$ and mapped to the Dirichlet concentration parameters:
    $$a_{k, b} = \exp(\tilde{e}_{k, b} / \tau_k)$$
    where $\tau_k = e^{w_k}$ is the task-specific calibrated temperature.
4.  **Ensembling Weight Construction**: We compute the expected ensembling coefficients $\alpha_{k, b}$ using the closed-form expectation of the Dirichlet posterior:
    $$\alpha_{k, b} = \frac{a_{k, b}}{\sum_{j=1}^K a_{j, b}}$$
5.  **Activation serving**: In each downstream block $l \in [4, L]$, the activation vector $h^{(l-1)}$ is passed in parallel through the $K$ active expert LoRA adapters, generating adapter-specific delta activations $\Delta h_k^{(l)}$.
6.  **Dynamic Activation Blending**: The expert deltas are scaled by the sample-specific ensembling weights $\alpha_{k, b}$ and added to the base backbone representation:
    $$h^{(l)} = h^{(l-1)} + \sum_{k=1}^K \alpha_{k, b} \cdot \Delta h_k^{(l)}$$
7.  **Final Task Inference**: The blended activation at the final layer is passed through the task classification heads to compute the final predictions.
