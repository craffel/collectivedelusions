# Evaluation Part 3: Soundness and Methodology

## Mathematical Soundness and Theoretical Gaps

The submission places a heavy emphasis on its mathematical formulation, presenting several proofs to guarantee representation stability. However, a rigorous mathematical audit reveals critical gaps and inconsistencies between the theoretical assertions and the empirical execution.

---

### 1. Vacuous Global Bound under Empirical Hyperparameters
In the analysis of "Soft Coordinate Alignment" and the global Lipschitz bound (Section 3.4), the authors derive a condition for global contraction:
$$\|W_{\text{route}}^{(l)}\|_2 < \frac{\tau_l}{2 R_{\mathcal{W}}} \left[ 1 - \frac{2}{\tau_c} \kappa R_{\mathcal{W}}^2 \right]$$
where $\tau_c$ is the task centroid coordinate scaling, $R_{\mathcal{W}}$ is the routing weight bound, and $\kappa$ is the coordinate alignment factor.

The authors explicitly admit that under their chosen empirical sandbox hyperparameters ($\tau_c = 0.05$, $R_{\mathcal{W}} = 1$, and setting $\kappa = 1$ for worst-case analysis), the term inside the brackets becomes:
$$1 - \frac{2}{0.05} = 1 - 40 = -39$$
This results in the following bound for the routing weight norm:
$$\|W_{\text{route}}^{(l)}\|_2 < -19.5 \tau_l$$
Since the spectral norm of any matrix is strictly non-negative ($\|W\|_2 \ge 0$) and the routing temperature is strictly positive ($\tau_l > 0$), **this condition is mathematically impossible to satisfy**. It requires a non-negative quantity to be strictly less than a negative number.

Consequently, **the theoretical global contraction guarantees are completely vacuous** under the actual experimental settings used to evaluate the model. The global Lipschitz bound ranges from 4.9 to 40.0. The authors attempt to wave this away by stating:
> *"Despite the theoretical violation, representations in practice do not exhibit chaotic divergence because the latent representations remain highly clustered..."*

This is a major methodological failure. If the mathematical guarantees are violated by a factor of 40 in the actual experimental setup, then the theory is decoupled from practice, and any observed stability is an empirical artifact rather than a consequence of the proven contraction mapping.

---

### 2. "Update-Space Quasi-Contraction" is Mathematically Deceptive
Standard deep architectures, including Transformers and ResNets, use identity residual connections where the base backbone is:
$$F_{\text{base}}^{(l)}(h) = h$$
This means $L_{\text{base}} = 1$. The overall layer mapping is:
$$T_l(h) = h + U_l(h)$$
where $U_l(h)$ represents the adapter update. For $T_l(h)$ to be a strict contraction, we must have $L_{T_l} < 1$. Mathematically, this is impossible for any non-trivial $U_l(h)$ because $L_{T_l} \ge L_{\text{base}} - L_{U_l} = 1 - L_{U_l}$, and if we require $L_{T_l} < 1$, it restricts $U_l(h)$ in ways that destroy representation capacity.

The authors propose "Update-Space Quasi-Contraction" as a relaxation, requiring $L_{U_l} < \epsilon$. Under this relaxation, the Lipschitz constant of the full layer is:
$$L_{T_l} \le 1 + \epsilon$$
Calling a mapping with $L \ge 1$ a "quasi-contraction" is mathematically misleading. Any dynamical system governed by a mapping with a Lipschitz constant strictly greater than 1 **can diverge exponentially**. 
More importantly, **all the core benefits of Banach's Fixed-Point Theorem are lost**:
* There is no guarantee of a unique fixed-point trajectory.
* Convergence to a stable state under depth is no longer mathematically guaranteed.
* The system is highly sensitive to the initial representation $h^{(0)}$.

Thus, the entire theoretical foundation of "fixed-point convergence" is discarded in the practical implementation on residual backbones, making the mathematical marketing of the paper highly deceptive.

---

### 3. Inference Temperature Annealing Contradicts the Core Theory
To recover the massive classification accuracy lost due to smooth, contractive regularization, the authors introduce **Adaptive Test-Time Temperature Annealing** (Section 5.6), scaling down the learned temperatures $\tau_l$ by a factor of $\gamma_{\text{scale}} = 0.10$ or $0.01$ during inference.

This post-hoc sharpening creates a fundamental logical contradiction:
* **During Training/Calibration:** The model is regularized with $\lambda_{\text{temp}} \frac{1}{\tau_l^2}$ to force $\tau_l$ to be large, ensuring a smooth Softmax routing function with a small Lipschitz constant.
* **During Inference:** The temperature is scaled down to near-zero, making the Softmax behave like a sharp, step-like argmax function.

The Lipschitz constant of a Softmax routing function is inversely proportional to the temperature, scaling as $\mathcal{O}(1/\tau_l)$. When $\gamma_{\text{scale}} \to 0$:
$$L_{T_l} \to \infty$$
During actual deployment/inference, the routing function is **not** smooth, is **not** a contraction, and exhibits extremely high Lipschitz constants. The very "routing jitter" and "trajectory instability" that the paper claims to solve are fully re-introduced at test time. 

The paper fails to provide any trajectory stability plots or empirical Gating Depth-Variance (GDV) measurements for the annealed test-time models. It is highly likely that the test-time models exhibit the exact same chaotic routing jitter as the unregularized baselines, meaning the entire contraction-regularization framework is functionally bypassed to achieve competitive accuracy.

---

### 4. Unrealistic Representation Domain Assumptions
Theorem 3.1 relies on the assumption of a bounded domain: $\|h\|_2 \le R_h$.
In actual residual networks, activation magnitudes tend to grow with depth $l$ as features accumulate. If $R_h^{(l)}$ grows across depth, then to maintain a constant Lipschitz bound, the routing projection weights $W_{\text{route}}^{(l)}$ must be regularized exponentially more stringently at later layers.

The proposed objective (Equation 19) applies a **uniform regularization strength** ($\lambda_{\text{spec}}$ and $\lambda_{\text{temp}}$) across all layers. This is mathematically inconsistent with a depth-dependent growth of $\|h\|_2$, meaning the contraction bounds are likely violated in the deeper layers of the network even under training conditions.
