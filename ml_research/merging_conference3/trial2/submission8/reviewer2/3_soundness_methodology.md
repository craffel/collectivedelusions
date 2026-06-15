# Evaluation of Technical Soundness and Methodology

## 1. Description Clarity and Reproducibility
The methodology in Section 3 and the Appendix is written with high clarity and mathematical precision. 
- **Formulation:** The definitions of task-vector extraction, Task Arithmetic merging, SAM optimization (first-order approximation), Uniform Pruning (NP-BTVP-U), and Saliency-Based Pruning (NP-BTVP-S) are mathematically complete, unambiguous, and easy to follow.
- **Reproducibility:** The paper provides a comprehensive list of hyperparameters, including the learning rate ($10^{-5}$), number of epochs (5), backbone model (CLIP ViT-B/32 using `open_clip` weights), and SAM perturbation radius ($\rho = 0.002$). The datasets used (MNIST, FashionMNIST, CIFAR-10, SVHN) and the specific training/test split sizes (1024 samples) are standard and readily accessible, ensuring a high level of reproducibility.

## 2. Technical and Mathematical Soundness

### A. The "Norm-Preserving" Misnomer
There is a fundamental conceptual contradiction in the paper's naming and framing:
- The framework is named **Norm-Preserved** Budgeted Task-Vector Pruning (NP-BTVP).
- Yet, the authors formally derive in Appendix Sections 1.1 and 1.2 that the expected $L_1$ norm of the rescaled task vector is **significantly larger** than that of the original dense vector:
  - Under a Laplace distribution: $\frac{\mathbb{E}[\|\tilde{\tau}^{(p)}\|_1]}{\mathbb{E}[\|\tau\|_1]} = 1 - \ln p \approx 3.30$ (for $p = 0.10$).
  - Under a Gaussian distribution: $\frac{\mathbb{E}[\|\tilde{\tau}^{(p)}\|_1]}{\mathbb{E}[\|\tau\|_1]} \approx 2.58$ (for $p = 0.10$).
- Thus, the $1/p$ scaling factor does **not** preserve the norm. Instead, it over-scales the remaining largest magnitude updates, boosting the expected $L_1$ norm by 2.58x to 3.30x.
- While the authors acknowledge this in Section 3.3 and Appendix Section 1.1, calling the framework "Norm-Preserved" remains a conceptual misnomer. It should be more accurately framed as "Signal-Boosted" or "Over-scaled" pruning.

### B. Resolving the "Minor Performance Gap" Paradox
In Appendix Section 1.3, the authors derive the expected $L_2$ reconstruction error:
$$\mathbb{E}[\|\tilde{\tau}^{(p)} - \tau\|_2^2] = d \mathbb{E}\left[ \tau_i^2 \cdot \mathbb{I}(|\tau_i| < t_p) \right] + d \left( \frac{1}{p} - 1 \right)^2 \mathbb{E}\left[ \tau_i^2 \cdot \mathbb{I}(|\tau_i| \geq t_p) \right]$$
They note that for $p = 0.10$, the quadratic multiplier is $(\frac{1}{p} - 1)^2 = 81$, which causes a massive $L_2$ reconstruction error in parameter space. They state:
> *"This elegant trade-off explains the minor performance gap (0.60%) between the rescaled 10% sparse model and the uncompressed dense model..."*

From a theoretical standpoint, this statement is flawed. A massive $L_2$ reconstruction error (81x larger variance in active weights) would normally imply that the weights deviate severely from their optimal values, which should increase, not decrease, the performance gap. The paper fails to provide a rigorous mathematical link explaining why such a massive parameter-space distortion results in only a 0.60% performance drop.

We can formally resolve this paradox by analyzing the final multi-task model in parameter space:
Let the original dense merged model be:
$$\theta_{\text{MTL}} = \theta_{\text{base}} + \sum_{k=1}^K \lambda_k \tau_k$$
Let the rescaled sparse merged model be:
$$\theta_{\text{MTL}}^{(p)} = \theta_{\text{base}} + \sum_{k=1}^K \lambda_k \tilde{\tau}_k^{(p)}$$
The parameter-space difference $\Delta_{\text{MTL}}$ between the dense and sparse merged models is:
$$\Delta_{\text{MTL}} = \theta_{\text{MTL}}^{(p)} - \theta_{\text{MTL}} = \sum_{k=1}^K \lambda_k \left( \tilde{\tau}_k^{(p)} - \tau_k \right)$$
Assuming the tasks are independent, the expected squared $L_2$ distance of the final merged parameters is:
$$\mathbb{E}[\|\Delta_{\text{MTL}}\|_2^2] = \sum_{k=1}^K \lambda_k^2 \mathbb{E}[\|\tilde{\tau}_k^{(p)} - \tau_k\|_2^2]$$
In the paper's experiments, the optimized merging coefficients are small, typically $\lambda_k \approx 0.25$ to $0.30$. 
This means the squared coefficient is $\lambda_k^2 \approx 0.06$ to $0.09$.
Consequently, the massive individual reconstruction error $\mathbb{E}[\|\tilde{\tau}_k^{(p)} - \tau_k\|_2^2]$ is **scaled down by a factor of 10 to 15** in the final merged parameter space:
$$\mathbb{E}[\|\Delta_{\text{MTL}}\|_2^2] \approx 0.09 \sum_{k=1}^K \mathbb{E}[\|\tilde{\tau}_k^{(p)} - \tau_k\|_2^2]$$
Furthermore, because the pre-trained base network $\theta_{\text{base}}$ is highly overparameterized, the loss landscape possesses significant flat dimensions (null spaces of the Hessian). The remaining parameter-space distortion $\Delta_{\text{MTL}}$ is projected onto these flat dimensions, resulting in negligible changes in the objective function and explaining the minor 0.60% performance gap.
This rigorous derivation explains the paradox that the authors hand-waved away as an "elegant trade-off".

### C. Evaluation of the Layer-wise Saliency Metric
The normalized layer-wise update saliency $S_l$ is defined as:
$$S_l = \frac{1}{K \cdot N_l} \sum_{k=1}^K \|\tau_{k, l}\|_1$$
This is a standard first-order heuristic. From a theoretical perspective, using $L_1$ magnitude as a proxy for layer sensitivity is computationally practical ($O(d)$) but mathematically limited:
- It assumes that the loss change is isotropic across layers.
- It ignores the curvature (second-order Hessian diagonal $H_{ii}$) of different layers.
While the authors justify this in Appendix Section 2 by citing the prohibitive computational cost of Hessian estimation on edge devices, this first-order proxy is likely the reason why Saliency-Based Pruning (NP-BTVP-S) fails to outperform global Uniform Pruning (NP-BTVP-U). A layer with a large average update norm $\|\tau_{k, l}\|_1$ might simply have a large scale but reside in a very flat loss valley (low Hessian eigenvalues), making its parameters less sensitive to pruning than a small-magnitude layer with high curvature.

## 3. Summary of Soundness
The methodology is **theoretically sound but conceptually misnamed and mathematically incomplete** in its explanations:
- The derivations under Laplace and Gaussian distributions are mathematically correct and elegant.
- The term "Norm-Preserving" is a misnomer, as it actually boosts the expected $L_1$ norm by ~3x.
- The claim that $L_2$ reconstruction error trade-offs "explain" the minor performance gap is theoretically hand-waved; we have provided the formal mathematical explanation via merging coefficient scaling and Hessian null-space projection.
- The first-order layer-wise saliency heuristic is mathematically simplistic, explaining its lack of empirical edge over the uniform baseline.
- The empirical setup and reproducibility are excellent.
