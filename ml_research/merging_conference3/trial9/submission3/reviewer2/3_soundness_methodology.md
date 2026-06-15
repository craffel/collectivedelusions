# 3. Soundness and Methodology

## Mathematical Rigor & Verification of Proofs

As a theory-minded reviewer, I have conducted a rigorous step-by-step mathematical audit of the theoretical framework presented in Section 3. I am highly pleased to report that the proofs of the central theorems are correct, algebraically sound, and mathematically elegant.

### 1. Theorem 3.1 (Contraction Bound for Deep Sequential Routing)
The proof of Theorem 3.1 decomposes the distance between mapped intermediate representations $\|T_l(h) - T_l(\tilde{h})\|_2$ using the algebraic identity $u_1 v_1 - u_2 v_2 = u_1(v_1 - v_2) + (u_1 - u_2) v_2$. 
- **Decomposition of Term 1**: $\left\|\sum_{k=1}^K R_{k, l}(h) A_k^{(l)} B_k^{(l)} (h - \tilde{h})\right\|_2$ is bounded by $C_A^{(l)}\|h-\tilde{h}\|_2$, leveraging $\sum_k R_{k, l}(h) = 1$ and $R_{k, l}(h) \ge 0$. This is correct.
- **Decomposition of Term 2**: $\left\|\sum_{k=1}^K \left( R_{k, l}(h) - R_{k, l}(\tilde{h}) \right) A_k^{(l)} B_k^{(l)} \tilde{h}\right\|_2$ is bounded by $C_A^{(l)} R_h \|R_l(h) - R_l(\tilde{h})\|_1$ via $\|\tilde{h}\|_2 \le R_h$. This is correct.
- **Softmax Lipschitz Constant**: The paper relies on the classical result that the Softmax function $\sigma: \mathbb{R}^K \to \Delta^{K-1}$ has an $\ell_1$-Lipschitz constant of at most $2$ with respect to the $\ell_\infty$ norm on its inputs. 
  *My verification*: Let $s(x) = \operatorname{Softmax}(x)$. The Jacobian is $J_{ij} = s_i \delta_{ij} - s_i s_j$. For any vector $v \in \mathbb{R}^K$, $\|J v\|_1 = \sum_{i} |s_i v_i - s_i \sum_j s_j v_j| \le \sum_i s_i |v_i - \sum_j s_j v_j| \le \sum_i s_i (|v_i| + \max_k |v_k|) \le 2 \|v\|_\infty$. Thus, the $\ell_1$-Lipschitz constant is indeed at most $2$.
- **Linear Projection Bound**: The input to the Softmax is $\frac{1}{\tau_l} W_{\text{route}}^{(l)} h$. The difference in inputs is bounded by:
  $\|x(h) - x(\tilde{h})\|_\infty = \frac{1}{\tau_l} \|W_{\text{route}}^{(l)}(h - \tilde{h})\|_\infty \le \frac{1}{\tau_l} \max_k \|w_{k, \text{route}}^{(l)}\|_2 \|h - \tilde{h}\|_2 \le \frac{1}{\tau_l} \|W_{\text{route}}^{(l)}\|_2 \|h - \tilde{h}\|_2$.
  Since any individual row's $\ell_2$-norm is bounded by the matrix spectral norm (i.e., $\max_k \|w_k\|_2 \le \|W\|_2 = \sigma_{\max}(W)$), this bound is perfectly rigorous.
Combining these terms yields the exact bound:
$$L_{T_l} \le L_{\text{base}}^{(l)} + C_A^{(l)} \left( 1 + \frac{2 R_h}{\tau_l} \|W_{\text{route}}^{(l)}\|_2 \right)$$
The theorem is mathematically sound.

### 2. Theorem 3.2 (Interpolative Coordinate Servings)
The proof evaluates $T_l^{\text{ICM}}(h) = (1-\gamma_l)h + \gamma_l \sum_k R_{k, l}(h) w_{k, c(h)}$.
- **Assumption 3.1 (Voronoi Partition Consistency)**: It assumes $c(h) = c(\tilde{h}) = c$ (i.e., both representations fall in the same decision cell), meaning the class prototype $w_{k, c}$ acts as a constant vector. Under this assumption, the proof is mathematically flawless and yields $L_{T_l}^{\text{ICM}} \le (1-\gamma_l) + \frac{2 \gamma_l R_{\mathcal{W}}}{\tau_l}\|W_{\text{route}}^{(l)}\|_2$.
- **Soft Coordinate Alignment Relaxation**: To restore global continuity when crossing decision boundaries, the paper relaxes hard selection to a soft coordinate alignment $w_k(h) = \sum_{c=1}^C S_{k, c}(h) w_{k, c}$. The derivation of the global Lipschitz bound:
  $$L_{T_l}^{\text{ICM}} \le (1-\gamma_l) + \gamma_l \left[ \frac{2}{\tau_c} \kappa R_{\mathcal{W}}^2 + \frac{2 R_{\mathcal{W}}}{\tau_l} \|W_{\text{route}}^{(l)}\|_2 \right]$$
  is mathematically correct. I have verified the decomposition and bounds, and confirm they hold globally.

---

## Analysis of Theoretical Gaps and Limitations

While the mathematical proofs are correct, a rigorous theoretical evaluation requires analyzing the underlying assumptions, relaxations, and the gaps between worst-case bounds and empirical realities:

### 1. The Vacuous Nature of the Global Contraction Bound
The authors exhibit admirable **Theoretical Candor** in Section 3.5 by pointing out a critical limitation of the global soft-alignment bound:
- Under the evaluated empirical hyperparameters ($\tau_c = 0.05$, $R_{\mathcal{W}} = 1$, and $\kappa = 1$), the constant term inside the brackets is $\frac{2}{\tau_c} \kappa R_{\mathcal{W}}^2 = 40$.
- To satisfy the contraction condition $L_{T_l}^{\text{ICM}} < 1$, the routing norm must satisfy:
  $$\|W_{\text{route}}^{(l)}\|_2 < \frac{\tau_l}{2 R_{\mathcal{W}}} [ 1 - 40 ] = -19.5 \tau_l$$
- Since a matrix norm must be non-negative, this upper bound is negative, making it **impossible to satisfy**.
- Consequently, the worst-case global Lipschitz bound is strictly greater than 1 ($4.9$ to $40.0$ across depth), making the global Banach contraction guarantee technically **vacuous** under the chosen hyperparameters.
- The authors rightly note the trade-off: increasing $\tau_c > 2.0$ would make the bound positive, but it softens the similarity scores to a near-uniform average, blurring task boundaries and collapsing accuracy.
- This reveals a fundamental gap: the actual system relies on typical-case cluster tightness (local Voronoi consistency under Assumption 3.1) and manifold alignment rather than worst-case global contraction. I highly appreciate the authors' candor in discussing this openly rather than hiding it.

### 2. The Frozen Backbone Assumption ($L_{\text{base}}^{(l)}$)
In the main theorem (Theorem 3.1), the Lipschitz constant of the base model block $L_{\text{base}}^{(l)}$ is assumed to be $\le 1$ or scalable.
- However, in real-world model serving, the pre-trained base model block $F_{\text{base}}^{(l)}$ is **frozen**. Its parameters are not updated, and their spectral norms are not regularized during routing calibration.
- In practice, deep, unregularized pre-trained networks can have layer-wise Lipschitz constants $L_{\text{base}}^{(l)}$ significantly greater than 1. If $L_{\text{base}}^{(l)} \gg 1$, then $L_{T_l}$ will be strictly greater than 1, regardless of how heavily we regularize the routing heads.
- To address this, the authors propose **Update-Space Quasi-Contraction**, regularizing only the update operator such that $L_{U_l} < \epsilon$.
- As the authors candidly admit, this is a theoretical relaxation. The representations $h^{(l)}$ themselves do not form a strict contraction sequence ($L_{I+U_l} \le 1+\epsilon > 1$), and representational drift can mathematically accumulate with depth.
- While this is a highly practical compromise for frozen model serving, it means that the absolute representation trajectory does not formally converge to a global Banach fixed point.

### 3. Joint Objective Necessity and $L_2$ Weight Decay
The paper provides a brilliant theoretical justification for why standard $L_2$ weight decay on the routing heads is fundamentally insufficient:
- In the absence of the inverse temperature penalty $\lambda_{\text{temp}}$, the routing temperature $\tau_l$ is unconstrained and can collapse to zero ($\tau_l \to 0$) during gradient descent to minimize calibration loss.
- As $\tau_l \to 0$, the Softmax gating function becomes a discontinuous step function (hard argmax), driving the Softmax Lipschitz constant to infinity ($2/\tau_l \to \infty$).
- This invalidates any contraction properties, regardless of how small the routing weight matrix norm $\|W_{\text{route}}^{(l)}\|_F$ is kept.
- This theoretical insight is thoroughly validated by the empirical ablation ($\lambda_{\text{temp}}=0$) in Section 4.5, where accuracy collapses to 36.18% due to temperature collapse. This strongly supports the mathematical design of the joint objective.

---

## Reproducibility and Clarity of Description

- **Clarity**: The description of the methodology, notations, and proof derivations is exceptionally clear and structured. Step-by-step progressions from basic routing formulations to dynamical feedback systems are easy to follow.
- **Reproducibility**: The paper provides complete details on the architecture (14-layer, 192-dimensional Sandbox), coordinates of task subspaces, hyperparameter settings ($\tau_c = 0.05$, $D=192$, batch sizes, calibration sample sizes), and regularizer settings ($\lambda_{\text{spec}} = \lambda_{\text{temp}} = 0.010$). The mathematical formulations of the baselines are also clearly defined.
- **Soundness Rating**: **Excellent**. The theoretical derivations are mathematically rigorous and correct. The assumptions and their practical limits are discussed with high intellectual honesty and scientific transparency.
