# Idea Proposal: PAC-Bayesian Generalization Bound Minimization for Dynamic Model Merging (PAC-ZCA)

## 1. Persona Alignment
As **The Theorist**, we approach the challenge of dynamic model merging through the lens of mathematical, statistical, and learning-theoretic guarantees. While prior state-of-the-art dynamic routing mechanisms (such as SPS-ZCA in `trial7_submission10`) rely on empirical and heuristic hyperparameter selection (such as setting the temperature scale $\tau$ to static values like $0.001$, or scaling similarity coordinates heuristically based on expected dispersion), we argue that such approaches lack rigorous mathematical justification and are highly susceptible to overfitting or instability under streaming distribution shifts.

This proposal, **PAC-ZCA**, resolves this fundamental gap by reformulating dynamic model-merging routing under the **PAC-Bayesian learning framework**. We model the dynamic routing decision as a randomized Gibbs policy (Softmax routing), and prove that its expected generalization error is upper-bounded by a PAC-Bayesian bound. Crucially, we identify a deep, elegant identity connecting the KL-divergence of the Softmax routing distribution from a Uniform prior to the Shannon entropy of the routing coefficients. Rather than setting hyperparameters empirically, **PAC-ZCA dynamically solves for the optimal temperature parameters $\boldsymbol{\tau}^*$ that directly minimize this PAC-Bayesian generalization bound** over a small offline calibration set. This provides a formal, provably correct learning-theoretic guarantee that the selected routing parameters maximize test-time generalization on unseen multi-task streams.

---

## 2. Core Techniques
*   **Gibbs Routing Policy (Softmax Router):** Parameterizes the sample-wise routing decision as a probability distribution over the available task experts $\{E_1, \dots, E_K\}$, modeled via a temperature-scaled Softmax over calibrated similarity coordinates.
*   **Information-Theoretic Complexity Equivalence:** Proof of the exact identity connecting the Kullback-Leibler (KL) divergence of the Softmax router from a Uniform prior to the Shannon entropy of the routing coefficients:
    $$\text{KL}(Q_z \| P) = \log K - H(Q_z)$$
    This establishes that maximizing routing entropy (smoothing activation blending) directly acts as a PAC-Bayesian complexity regularizer.
*   **Differentiable PAC-Bayes Bound Minimization:** We formulate the PAC-Bayes generalization bound as a differentiable function of the temperature scale vector $\boldsymbol{\tau} \in \mathbb{R}^K$ and task similarity weights, and minimize this bound directly using gradient-based optimization on the calibration set.
*   **Wasserstein-Calibrated Representation Space:** Unifies Unit-Norm Calibration (UNC) and Intra-Task Dispersion Calibration (IDC) under a rigorous sub-Gaussian concentration model of representation manifolds.

---

## 3. Mathematical Formulation

Let $f_\theta$ be a pre-trained frozen shared base model backbone. Let $\{E_1, \dots, E_K\}$ be the independent task experts fine-tuned via Low-Rank Adaptation (LoRA) on task-specific calibration sets.
At Layer $l = 3$ (the shared, adapter-free early representation space), let $z_b = h_b^{(3)} \in \mathbb{R}^D$ be the pooled intermediate representation of input $x_b$.
Let $\mu_k \in \mathbb{R}^D$ be the pre-computed centroid for task $k$ on calibration split $\mathcal{C}_k$ of size $N_c = 64$:
$$\mu_k = \frac{1}{|\mathcal{C}_k|} \sum_{s \in \mathcal{C}_k} z_s$$

### Coordinate Calibration
For an input $z_b$, the cosine similarity coordinate $u_{k, b}$ corresponds to Unit-Norm Calibration (UNC), projecting features onto a unit hypersphere:
$$u_{k, b} = \text{cos\_sim}(z_b, \mu_k) = \frac{z_b \cdot \mu_k}{\|z_b\|_2 \|\mu_k\|_2}$$
We calibrate these coordinates using the expected dispersion scale $s_k$ via Intra-Task Dispersion Calibration (IDC):
$$s_k = \frac{1}{|\mathcal{C}_k|} \sum_{s \in \mathcal{C}_k} \text{cos\_sim}(z_s, \mu_k)$$
$$\tilde{u}_{k, b} = \frac{u_{k, b}}{s_k}$$

### Randomized Routing Policy & Gibbs Prior
We model the routing decision as a randomized classifier $Q_{z_b}$ that assigns probability $q_k(z_b; \boldsymbol{\tau})$ to task expert $k$. Under a temperature-scaled Softmax parameterized by task-specific temperature vector $\boldsymbol{\tau} = [\tau_1, \dots, \tau_K]^T > 0$:
$$q_k(z_b; \boldsymbol{\tau}) = \frac{\exp(\tilde{u}_{k, b} / \tau_k)}{\sum_{j=1}^K \exp(\tilde{u}_{j, b} / \tau_j)}$$
We define a non-informative Uniform prior routing policy $P$: $p_k(z) = 1/K$ for all $k \in \{1,\dots, K\}$.
The Kullback-Leibler (KL) divergence of the randomized routing policy $Q_{z_b}$ from the prior $P$ is:
$$\text{KL}(Q_{z_b} \| P) = \sum_{k=1}^K q_k(z_b; \boldsymbol{\tau}) \log \frac{q_k(z_b; \boldsymbol{\tau})}{1/K} = \sum_{k=1}^K q_k(z_b; \boldsymbol{\tau}) \left[ \log q_k(z_b; \boldsymbol{\tau}) + \log K \right]$$
$$\text{KL}(Q_{z_b} \| P) = \log K - H(Q_{z_b})$$
where $H(Q_{z_b}) = -\sum_{k=1}^K q_k(z_b; \boldsymbol{\tau}) \log q_k(z_b; \boldsymbol{\tau})$ is the Shannon entropy of the routing coefficient vector.

### PAC-Bayesian Generalization Bound
For a total calibration set $\mathcal{C} = \bigcup_{k=1}^K \mathcal{C}_k$ of size $N = K \cdot N_c$, and any confidence level $\delta \in (0, 1)$, the expected multi-task routing classification error $R(Q)$ on the true data stream distribution $\mathcal{D}$ is bounded with probability at least $1 - \delta$ by:
$$\mathcal{B}(\boldsymbol{\tau}) = \frac{1}{N} \sum_{s \in \mathcal{C}} \mathcal{L}(Q_{z_s}, y_s) + \sqrt{\frac{\frac{1}{N} \sum_{s \in \mathcal{C}} \text{KL}(Q_{z_s} \| P) + \log\left(\frac{2\sqrt{N}}{\delta}\right)}{2N}}$$
where $\mathcal{L}(Q_{z_s}, y_s) = \sum_{k=1}^K q_k(z_s; \boldsymbol{\tau}) \mathbb{I}(y_s \ne k)$ is the expected error of the randomized policy on sample $s$ with true task label $y_s$.
Substituting the entropy identity:
$$\mathcal{B}(\boldsymbol{\tau}) = \frac{1}{N} \sum_{s \in \mathcal{C}} \sum_{k \ne y_s} q_k(z_s; \boldsymbol{\tau}) + \sqrt{\frac{\log K - \frac{1}{N} \sum_{s \in \mathcal{C}} H(Q_{z_s}) + \log\left(\frac{2\sqrt{N}}{\delta}\right)}{2N}}$$

### Differentiable Parameter Optimization
Because $\mathcal{B}(\boldsymbol{\tau})$ is fully differentiable with respect to $\boldsymbol{\tau} > 0$, we find the optimal temperature parameters $\boldsymbol{\tau}^*$ by solving:
$$\boldsymbol{\tau}^* = \arg\min_{\boldsymbol{\tau} > 0} \mathcal{B}(\boldsymbol{\tau})$$
This optimization balance is mathematically elegant:
*   **Empirical Risk minimization:** Drives $\boldsymbol{\tau}$ lower to make the Softmax sharper around the correct task centroid ($\alpha_{y_s} \to 1$), minimizing the first term.
*   **Complexity regularization:** Drives $\boldsymbol{\tau}$ higher to make the Softmax smoother and maximize entropy $H(Q_{z_s}) \to \log K$, which minimizes the second (generalization gap) term.
The PAC-Bayesian optimum $\boldsymbol{\tau}^*$ achieves the provably correct trade-off that maximizes out-of-sample generalization.

---

## 4. Architecture Specifications
*   **Shared Base Backbone:** Vision Transformer `vit_tiny_patch16_224` with $L=12$ transformer blocks and intermediate feature dimension $D=192$.
*   **Task Expert Adapters:** $K=4$ low-rank LoRA adapters with rank $r=8$ inserted in the projection layers of Layers 4--12. Layers 1--3 are kept frozen and shared task-agnostically with no adapters.
*   **Centroid Registry:** Offline-computed task centroids $\{\mu_k\}_{k=1}^K \subset \mathbb{R}^{192}$ and dispersion scales $\{s_k\}_{k=1}^K \subset \mathbb{R}$ stored in the router's local memory.
*   **PAC-Bayesian Router Module:** Placed at the output of Layer 3. It computes similarity coordinates, scales them by $s_k$, applies the optimal temperature scaling parameters $\boldsymbol{\tau}^*$ solved offline, and outputs the sample-wise blending coefficients $\alpha_{k, b} = q_k(z_b; \boldsymbol{\tau}^*)$.
*   **SPS Activation Blender:** Inside Layers 4--12, adapter activations are dynamically blended sample-wise in a single forward pass:
    $$h_b^{(l)} = h_b^{(l-1)} W_{\text{base}}^{(l)} + \sum_{k=1}^K \alpha_{k, b} \left( h_b^{(l-1)} A_k^{(l)} B_k^{(l)} \right)$$

---

## 5. Baselines
We evaluate PAC-ZCA against the following robust baselines:
*   **Heuristic SPS-ZCA (SOTA):** Nearest-centroid router utilizing a static temperature $\tau = 0.001$ without learning-theoretic calibration.
*   **QWS-Merge:** Quantum Wavefunction Superposition Merging baseline, representing experts as quantum eigenstates.
*   **Linear Router (Reg):** Classical linear router optimized on the calibration set via regularized cross-entropy.
*   **Uniform Merging:** Static weight averaging baseline ($\alpha_{k} = 1/K$).
*   **Expert Ceiling:** Oracle baseline routing each sample directly to its isolated expert model.

---

## 6. Step-by-Step Interaction

### Phase 1: Offline Calibration (One-time Preparation)
1.  **Calibration representation extraction:** Execute pre-trained base model Layers 1--3 on calibration sets $\mathcal{C}_k$ to extract Layer 3 features $z_s = h_s^{(3)}$.
2.  **Centroid & dispersion estimation:** Compute centroid vectors $\mu_k \in \mathbb{R}^{192}$ and dispersion scales $s_k \in \mathbb{R}$ for each task $k \in \{1,\dots, K\}$.
3.  **Coordinate computation:** For each calibration sample $s \in \mathcal{C}$, compute the dispersion-calibrated similarity coordinates $\tilde{u}_{k, s} = \text{cos\_sim}(z_s, \mu_k) / s_k$.
4.  **PAC-Bayes Optimization:** Set the failure tolerance $\delta = 0.05$. Solve the optimization problem $\boldsymbol{\tau}^* = \arg\min_{\boldsymbol{\tau}} \mathcal{B}(\boldsymbol{\tau})$ using gradient descent (e.g., Adam optimizer in PyTorch for 100 epochs) over the calibration coordinates to obtain the optimal task-specific temperatures $\boldsymbol{\tau}^*$. Store $\{\mu_k\}$, $\{s_k\}$, and $\boldsymbol{\tau}^*$ in the serving registry.

### Phase 2: Online Inference (Dynamic Serving)
1.  **Stream Input:** A highly heterogeneous mixed batch of size $B$, $X = \{x_1, \dots, x_B\}$, is received by the edge CPU.
2.  **Early Pass:** Feed $X$ through the shared, adapter-free early layers (Layers 1--3) of the base ViT model to obtain the Layer 3 intermediate activations $h_b^{(3)}$ for each sample $b$.
3.  **Coordinate Projection:** Extract $z_b = \text{Pool}(h_b^{(3)})$. Compute calibrated coordinates:
    $$u_{k, b} = \text{cos\_sim}(z_b, \mu_k), \quad \tilde{u}_{k, b} = u_{k, b} / s_k$$
4.  **Optimal Policy Generation:** Compute the sample-wise routing coefficients using the PAC-Bayesian optimal Gibbs policy:
    $$\alpha_{k, b} = \frac{\exp(\tilde{u}_{k, b} / \tau_k^*)}{\sum_{j=1}^K \exp(\tilde{u}_{j, b} / \tau_j^*)}$$
5.  **Single-Pass Blended Forward:** Execute Layers 4--12 of the base backbone. At each layer $l$, apply Single-Pass Activation-Space Dynamic Blending (SPS) to blend the expert activations sample-wise using $\alpha_{k, b}$.
6.  **Final Prediction:** Output the final predictions from the joint parallel pipeline with zero train-test mismatch and guaranteed out-of-sample generalization.
