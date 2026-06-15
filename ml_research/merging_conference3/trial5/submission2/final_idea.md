# Rademacher-Bounded Polynomial Merging (RBPM): Provable Generalization Bounds for Adaptive Model Merging

## 1. Persona Alignment
This proposal is directly aligned with **The Theorist** persona. Rather than introducing heuristic mechanisms or relying on intuitive physical analogies, RBPM tackles the core challenge of model merging through the rigorous framework of **Statistical Learning Theory**. Specifically, we address the "Task Suite Bias" and the transductive stream overfitting observed in online Test-Time Adaptation (TTA) by:
1.  Formulating the space of layer-wise merging coefficients as a structured hypothesis class.
2.  Deriving formal **Rademacher Complexity Bounds** for this hypothesis space.
3.  Proving mathematically that constraining the trajectory of merging coefficients across network depth to a low-degree polynomial shatters high-frequency noise and guarantees generalization.
4.  Introducing a theoretically-sound **Rademacher-Regularized Few-Shot Objective** that directly minimizes a generalization upper-bound.

## 2. Core Techniques
RBPM introduces three key technical and theoretical innovations:
1.  **Polynomial Trajectory Projection ($P_d$):** We map the high-dimensional, unconstrained layer-wise scaling coefficient space ($L$ parameters per task) to a low-dimensional polynomial subspace of degree $d \ll L$ ($d+1$ parameters per task).
2.  **Rademacher Complexity Regularization ($\mathcal{R}_{\text{rad}}$):** An analytical penalty term added to the few-shot offline validation objective that explicitly bounds the Rademacher complexity of the merging trajectory, penalizing rapid oscillations in weight-space interpolation.
3.  **Fisher-Weighted Empirical Loss:** We weight the empirical validation loss by the diagonal Fisher Information Matrix (FIM) of the layer weights to represent the local loss landscape curvature, aligning parameter deviations with functional output stability.

These methods build on and extend the empirical findings of `SuiteMerge` (OFS-Tune) and `Sparse Task Arithmetic` (noise-filtering perspective).

## 3. Mathematical Formulation

Let $L$ be the number of layers in the backbone, and $K$ be the number of task experts. Let $v_k^{(l)} \in \mathbb{R}^{d_l}$ be the task vector for task $k \in \{1, \dots, K\}$ at layer $l \in \{1, \dots, L\}$.

### 3.1. Hypothesis Space and Polynomial Trajectory Constraint
We define the hypothesis space of merging trajectories for task $k$ as a polynomial function of normalized depth $z = \frac{l}{L} \in [0, 1]$:
\begin{equation}
\mathcal{H}_d = \left\{ \alpha_k: \{1, \dots, L\} \to [0, 1] \;\middle|\; \alpha_k(l) = \sum_{j=0}^d \theta_{k, j} \left(\frac{l}{L}\right)^j, \;\; \sum_{j=0}^d |\theta_{k, j}| \le C_0 \right\}
\end{equation}
where $d$ is the polynomial degree (typically $d \le 2$), and $C_0$ is a positive constant bounding the $L_1$-norm of the polynomial coefficients.

### 3.2. Rademacher Complexity Bound Theorem
We prove that the empirical Rademacher complexity $\widehat{\mathcal{R}}_N(\mathcal{H}_d)$ of the polynomial trajectory space over a sample of size $N$ is strictly bounded and scales with the polynomial degree $d$ rather than the network depth $L$.

**Theorem 1.** *Let $\mathcal{H}_d$ be the class of polynomial merging coefficient trajectories of degree $d$ with $L_1$-bounded coefficients $\|\theta_k\|_1 \le C_0$. Then, the empirical Rademacher complexity of $\mathcal{H}_d$ satisfies:*
\begin{equation}
\widehat{\mathcal{R}}_N(\mathcal{H}_d) \le C_0 \sqrt{\frac{2 \log(2 d + 2)}{N}}
\end{equation}

*Proof Sketch:* Since the trajectory $\alpha_k(l)$ is a linear combination of the base monomials $x_j(l) = (l/L)^j$ for $j \in \{0, \dots, d\}$, the hypothesis class $\mathcal{H}_d$ is a subset of the convex hull of the base monomial functions scaled by $C_0$. By Massart's Lemma and the properties of Rademacher complexity of linear classes over bounded $L_1$ ball in $\mathbb{R}^{d+1}$, the complexity is bounded by $C_0 \sqrt{\frac{2 \log(2(d+1))}{N}}$. $\square$

Contrast this with the unconstrained layer-wise hypothesis space $\mathcal{H}_{\text{unconstrained}}$ where each layer has an independent coefficient in $[0, 1]$:
\begin{equation}
\widehat{\mathcal{R}}_N(\mathcal{H}_{\text{unconstrained}}) \le \sqrt{\frac{L \log(2)}{N}}
\end{equation}
Since $d \ll L$ (e.g., $d=1$ or $2$ vs. $L=12$ or $14$), the polynomial constraint shrinks the Rademacher complexity by a factor of $\mathcal{O}(\sqrt{L / \log(d)})$, mathematically explaining why it suppresses high-frequency transductive stream noise.

### 3.3. Rademacher-Regularized Few-Shot Objective
We formulate the offline optimization objective for the polynomial coefficients $\Theta = \{\theta_{k,j}\}$ over a small validation set of size $N = K \times M$ (where $M=10$ samples per task):
\begin{equation}
\min_{\Theta} \frac{1}{K} \sum_{k=1}^K \left[ \mathcal{L}_{\text{val}}^{(k)}(\Theta) \right] + \lambda_{\text{rad}} \sum_{k=1}^K \sum_{j=0}^d |\theta_{k, j}|
\end{equation}
where:
*   $\mathcal{L}_{\text{val}}^{(k)}(\Theta)$ is the cross-entropy classification loss on task $k$ using weights merged via $\alpha_k(l; \Theta)$.
*   $\lambda_{\text{rad}}$ is the Rademacher regularization coefficient, which controls the $L_1$-bound of the coefficients, directly corresponding to minimizing the generalization upper-bound:
\begin{equation}
\text{Gen-Error} \le \text{Emp-Error} + 2 \widehat{\mathcal{R}}_N(\mathcal{H}_d) + 3\sqrt{\frac{\log(2/\delta)}{2N}}
\end{equation}

## 4. Architecture Specifications

The backbone model is a standard Vision Transformer (ViT-Tiny/B) with $L=12$ or $14$ layers and hidden dimension $D$.
*   **Inputs:** Batches of images $x \in \mathbb{R}^{B \times 3 \times H \times W}$.
*   **Intermediate Representations:** Layer-wise activations $H_l \in \mathbb{R}^{B \times N_{\text{patches}} \times D}$.
*   **Weight Construction:** At each layer $l$, the active parameters are assembled dynamically before the forward pass of that layer:
\begin{equation}
W_{\text{merged}}^{(l)} = W_0^{(l)} + \sum_{k=1}^K \alpha_k\left(\frac{l}{L}\right) V_k^{(l)}
\end{equation}
where $\alpha_k(z) = \theta_{k,0} + \theta_{k,1} z + \theta_{k,2} z^2$ (for quadratic $d=2$).
*   **Outputs:** Multi-task task-specific logits $y_k = f(x; W_{\text{merged}}) \in \mathbb{R}^{B \times C_k}$ where $C_k$ is the class count of task $k$.

## 5. Baselines
We evaluate RBPM against five highly relevant baselines to isolate and prove its theoretical advantages:
1.  **Static Uniform Merging:** The parameter consensus baseline with $\alpha_k(l) = 1/K$ for all $l$. This represents zero parameter optimization.
2.  **Unconstrained Test-Time Adaptation (AdaMerging):** Unsupervised online TTA that optimizes $K \times L$ parameters on test streams. This isolates the effect of the polynomial constraint vs. unconstrained search under stream noise.
3.  **PolyMerge ($d=2$):** Online TTA with quadratic constraints, isolating the value of offline few-shot supervision vs. online unsupervised optimization.
4.  **Offline Unconstrained Few-Shot Tuning (OFS-Unconstrained):** Optimizes $K \times L$ unconstrained layer-wise coefficients offline on the same few-shot set. This isolates the regularizing value of the polynomial constraint under validation-set sampling noise.
5.  **Quantum Superposition Merging (QWS-Merge):** A dynamic, wave-inspired router, allowing us to compare our mathematically-principled static trajectory constraint against heuristic dynamic routing.

## 6. Step-by-Step Interaction

1.  **Offline Calibration Phase:**
    *   **Step 1.1:** Standard stratified sampling is executed on the training/validation pool to draw $M=10$ samples per task, creating a balanced calibration set of size $N=K \times M$.
    *   **Step 1.2:** The polynomial coefficients $\Theta \in \mathbb{R}^{K \times (d+1)}$ are initialized to the Uniform baseline (i.e., $\theta_{k,0} = 1/K$ and $\theta_{k,j>0} = 0$).
    *   **Step 1.3:** We perform derivative-free optimization (Nelder-Mead simplex) or gradient-based search (Adam) to minimize the Rademacher-Regularized Few-Shot Objective (Equation 7).
    *   **Step 1.4:** In each optimization step, for a validation batch, layer-wise merging coefficients are calculated using the current polynomial parameters: $\alpha_k(l) = \sum_j \theta_{k,j} (l/L)^j$.
    *   **Step 1.5:** The model weights are assembled layer-wise, the forward pass is executed, loss is evaluated, and the optimizer updates $\Theta$.
    *   **Step 1.6:** Once optimization converges, the optimal polynomial parameters $\Theta^*$ are frozen.

2.  **Inference / Deployment Phase:**
    *   **Step 2.1:** The final merged weights are compiled statically layer-wise using the optimal polynomial coefficients: $W_{\text{final}}^{(l)} = W_0^{(l)} + \sum_k \alpha_k^*(l) V_k^{(l)}$.
    *   **Step 2.2:** The compiled, static model is deployed.
    *   **Step 2.3:** During inference, incoming samples $x$ flow through the network under the compiled weights $W_{\text{final}}^{(l)}$ with **zero test-time optimization compute, zero memory overhead, and absolute stability**.
