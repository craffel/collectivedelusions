# 3. Soundness and Methodology Check

## Mathematical Rigor and Correctness
The mathematical derivation of PAC-Bayes Merge is elegant and structurally sound.
- **Analytical KL Divergence:** The closed-form derivation of the Kullback-Leibler divergence between two isotropic Gaussian distributions (Equation 10) is correct:
  $$D_{\text{KL}}(Q \parallel P) = \frac{\|\Theta - \Theta_{\text{uniform}}\|_2^2}{2 \sigma_0^2} + \frac{K(d+1)}{2} \left( \frac{\sigma^2}{\sigma_0^2} - 1 - \ln \frac{\sigma^2}{\sigma_0^2} \right)$$
- **Derivation of the Quadratic $L_2$ Penalty:** By transitioning from McAllester's non-linear square-root bound to Alquier's linear PAC-Bayesian bound, the authors successfully linearize the bound. Minimizing this linearized bound with respect to the posterior mean $\Theta$ mathematically yields the quadratic $L_2$ Consensus-Pulling penalty centered at $\Theta_{\text{uniform}}$:
  $$\min_{\Theta} \mathcal{L}_{\text{total}}(\Theta) = \mathcal{L}_{\text{ce}}(\Theta) + \lambda_{\text{PAC}} \|\Theta - \Theta_{\text{uniform}}\|_2^2$$
  This step is mathematically rigorous and standard in the statistical learning theory literature.

## Modeling Assumptions and Limitations
While the mathematics are correct, there are several key theoretical/methodological assumptions and limitations:

### 1. The Bounded Loss Assumption
Alquier's PAC-Bayesian bound strictly assumes a $[0, 1]$-bounded loss function. 
- In practice, the multi-task cross-entropy loss is unbounded $[0, \infty)$.
- The authors address this by clipping the cross-entropy loss at $L_{\max} = 5.0$ and scaling it by $1 / L_{\max}$ to map it to $[0, 1]$.
- In the physical PyTorch code, they minimize the unrescaled clipped loss directly, arguing that the $1 / L_{\max}$ scaling factor is mathematically absorbed into the learning rate and the regularization hyperparameter $\lambda_{\text{PAC}}$. While this is a standard practical bridge, it represents a minor gap between the strict assumptions of the mathematical theorem and the empirical execution.

### 2. SWA Equivalence and Single-Basin Assumption (Theorem 3.1)
Theorem 3.1 proves that uniform weight merging acts as a parametric low-pass filter that reduces SGD noise variance by a factor of $K$.
- This proof assumes that the fine-tuned expert task vectors $V_k$ are corrupted by independent, zero-mean SGD noise centered around a single local basin of attraction.
- As the authors themselves acknowledge, this assumption is highly unrealistic for disparate tasks (e.g., MNIST, SVHN, CIFAR-10) because independently trained models generally lie in structurally distinct, disconnected basins of attraction separated by high non-convex barrier boundaries (the Linear Mode Connectivity barrier).
- Therefore, Theorem 3.1 is primarily a stylized, conceptual caricature rather than a mathematically realistic representation of multi-task weight-space merging.

### 3. Local-to-Global Curvature Mismatch in FIM Regularization
In Appendix B, the authors derive a non-isotropic variant using diagonal Gaussians, where the prior variances are set inversely proportional to the empirical Fisher Information Matrix (FIM) diagonal:
$$\sigma_{0, k, j}^2 = \frac{\tau^2}{F_{k, j} + \epsilon}$$
- This FIM is evaluated **locally** at the initial uniform consensus point $\Theta_{\text{uniform}}$ before optimization begins.
- As optimization progresses and the trajectory parameters $\Theta$ drift away from uniform to fit the task-specific headers on the calibration data, the local curvature approximation at $\Theta_{\text{uniform}}$ can deviate significantly from the true global curvature.
- This local-to-global curvature mismatch represents a major methodological limitation, preventing the non-isotropic prior from adapting to the changing loss landscape, which explains why the FIM variant underperforms the isotropic model in the main experiments.

### 4. Deterministic Compiled Model vs. Randomized PAC-Bayes Bounds
There is a fundamental theoretical gap between the PAC-Bayesian theorem (which bounds the expected risk of a *randomized* classifier $G_Q$) and the "Deterministic Compiled" model evaluated at the posterior mean $\Theta^*$.
- Because of neural network non-convexity, the risk at the mean $R(G_{\mathbb{E}[\tilde{\Theta}]})$ is not theoretically bounded by the randomized expected risk $\mathbb{E}_{\tilde{\Theta} \sim Q}[R(G_{\tilde{\Theta}})]$.
- The authors justify their compiled deployment by demonstrating empirically that the deterministic model performs on par with the randomized posterior ensemble, suggesting a highly flat loss landscape. While this is an excellent empirical observation, it is a local property of this specific, toy MLP architecture and does not resolve the underlying theoretical mismatch in a generalizable way.
