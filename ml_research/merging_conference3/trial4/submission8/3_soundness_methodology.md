# Methodological Soundness: CR-PolySACM

We evaluate the mathematical rigor, physical representation, and stability of the CR-PolySACM framework.

---

## 1. Mathematical Rigor of the Noise Decomposition
In Section 3.2, the author presents a highly rigorous mathematical formulation of post-training quantization (PTQ) noise within a low-dimensional parameter space. The high-dimensional weight-space quantization noise $\delta \in \mathbb{R}^D$ is decomposed into an in-subspace projected perturbation and an orthogonal complement:
$$
\delta = J_{\mathbf{p}} \boldsymbol{\epsilon} + \delta_{\perp}
$$
where $J_{\mathbf{p}} \in \mathbb{R}^{D \times 3K}$ is the Jacobian matrix of merged parameters with respect to the polynomial coefficients, $\boldsymbol{\epsilon} \in \mathbb{R}^{3K}$ is the projected coefficient perturbation, and $\delta_{\perp} \in \mathbb{R}^D$ is the out-of-subspace noise component orthogonal to the task-vector subspace (i.e., $J_{\mathbf{p}}^T \delta_{\perp} = 0$).

Substituting this decomposition into a second-order Taylor expansion of the multi-task loss function yields:
$$
\Delta \mathcal{L} \approx \nabla_W \mathcal{L}^T \delta_{\perp} + \frac{1}{2} \boldsymbol{\epsilon}^T \mathcal{H}_{\mathbf{p}} \boldsymbol{\epsilon} + \frac{1}{2} \delta_{\perp}^T \mathcal{H}_W \delta_{\perp}
$$
where $\mathcal{H}_{\mathbf{p}} = \nabla^2_{\mathbf{p}} F(\mathbf{p}^*) \in \mathbb{R}^{3K \times 3K}$ is the polynomial coefficient-space Hessian matrix. This utilizes the standard Gauss-Newton approximation $\mathcal{H}_{\mathbf{p}} \approx J_{\mathbf{p}}^T \mathcal{H}_W J_{\mathbf{p}}$, which is mathematically solid and standard under smooth parameter mapping convergence.

This derivation yields a critical, high-signal theoretical insight: **test-time adaptive merging can only minimize and flat-map the second-order in-subspace error ($\frac{1}{2} \boldsymbol{\epsilon}^T \mathcal{H}_{\mathbf{p}} \boldsymbol{\epsilon}$). It has zero control over the out-of-subspace noise $\delta_{\perp}$.** Under aggressive low-precision quantization (e.g., INT4), out-of-subspace noise dominates and structurally destroys the model's representations, explaining why unconstrained TTA methods collapse and establishing a hard ceiling on model-merging robustness.

---

## 2. Transductive Gradient Generalization Gap
The paper provides a thorough analysis of the transductive generalization gap. At convergence of the test-time adaptation on the calibration stream, the calibration gradient vanishes: $\nabla_{\mathbf{p}} F_{\text{cal}}(\mathbf{p}^*) = 0$. The test stream gradient is bounded by:
$$
\|\nabla_{\mathbf{p}} F_{\text{test}}(\mathbf{p}^*)\|_2 = \|\nabla_{\mathbf{p}} F_{\text{test}}(\mathbf{p}^*) - \nabla_{\mathbf{p}} F_{\text{cal}}(\mathbf{p}^*)\|_2
$$
Because the structural polynomial parameterization restricts the search space to a tiny 12-dimensional subspace, this transductive generalization gap is extremely small and generalization is highly stable.

The author honestly discusses the impact of calibration stream size ($N$) in Appendix A.1. While reducing $N$ to extreme low-data regimes like $N=16$ ($B=4$ samples per task) can expand this transductive gradient deviation, the tight 12-dimensional parameter constraint keeps the generalization gap bounded. They identify an empirical threshold of $N < 8$, below which a single batch lacks sufficient multi-task representations, causing local gradient divergence. This level of analysis is highly scholarly and mathematically sound.

---

## 3. Task-Vector Norm Scale Pathology and Clipping-Regularization
The physical and mathematical justification of the task-vector norm scale pathology is exceptionally strong. Diagnostic measurements on the ViT-Tiny backbone reveal a massive, 50-fold discrepancy in task-vector norms: intermediate transformer blocks have norms ranging from $0.40$ to $0.68$, while the final layer normalization group (group 13) has a norm of only $0.014$ to $0.020$.

The author proves that a uniform coefficient perturbation results in a weight-space perturbation at Layer 13 that is over $100\times$ smaller than at other layers. Consequently, standard flatness optimization is blind to the highly sensitive final layer norm. Conversely, unmitigated scale-invariant normalization scales the perturbation of Layer 13 by over $2,500$ to $5,000$ times, triggering immediate gradient explosion and representational collapse.

The proposed Clipping-Regularized SACM (CR-SACM) successfully resolves this dilemma by clipping task-vector norms to a robust minimum threshold ($\beta = 0.10$):
$$
V_{\text{clipped}, k}^l = \max\left( \|\tau_k^l\|_2, \beta \right)
$$
This bounds the perturbation multiplier of low-norm layers to a stable and robust $1/0.1^2 = 100\times$, preventing division-by-zero or gradient explosion while successfully restoring optimizer sensitivity.

---

## 4. Stability of Sigmoid Parameterization and Boundary Clamping
The paper includes a thorough discussion of the potential pitfalls of the logistic sigmoid parameterization and the boundary clamping operator.
- **Sigmoid Saturation:** The author notes that the gradient updates with respect to the polynomial parameters scale with the derivative of the sigmoid: $\sigma'(\cdot) = \lambda_k^l(1 - \lambda_k^l)$. If a blending coefficient converges extremely close to the boundaries $0.0$ or $1.0$, the derivative approaches zero, which could theoretically cause gradient vanishing. However, they empirically confirm that during the short 40 steps of TTA, coefficients remain well within the active interior region (typically between $0.15$ and $0.85$), avoiding saturation.
- **Boundary Clamping:** The clamping operator $\text{clamp}\left( \lambda_k^l + \epsilon_k^l, 0.0, 1.0 \right)$ restricts the perturbed coefficients to the valid simplex domain. The author shows that less than 2% of the blending coefficients experience active boundary clamping during the entire adaptation trajectory, primarily occurring in early layers where task vectors are highly aligned or near-zero, confirming that clamping is highly stable and does not introduce optimization oscillations or gradient discontinuities.
- **Asymptotic Stability:** Appendix A.2 extends the trajectory analysis up to $T=150$ steps, confirming that adjacent layer polynomial constraints prevent boundary contact, showing asymptotic convergence in the safe interior region without parameter freezing.
