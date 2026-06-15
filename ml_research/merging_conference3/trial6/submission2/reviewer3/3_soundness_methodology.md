# 3. Soundness and Methodology

## Clarity of the Description
The methodology is described with a high level of mathematical detail, including complete proofs and derivations. The parameter blending formulation and the derivation of the Covariance-Weighted Frobenius Regularization (CFR) penalty are clearly structured. However, several critical assumptions are brushed over or require closer scrutiny.

## Appropriateness of Methods & Potential Technical Flaws

### 1. Representational De-coupling Approximation (Remark 3.2)
In the proof of Theorem 3.1 and the formulation of the CFR matrices, the authors treat the intermediate layer activations $z_i^{(l)}$ as fixed, independent constants. 
* **The Circular Dependency:** For any layer $l > 1$ in a multi-layer dynamic merging architecture, the activation $z_i^{(l)}$ is the output of block $l-1$, which depends directly on the upstream routing parameters $\{w_{l-1, k}, b_{l-1, k}\}$. Consequently, $z_i^{(l)}$ is a function of the router weights $W$.
* **The Flaw:** By treating $z_i^{(l)}$ as fixed constants during the supremum calculation, the proof ignores the downstream impact of changing the upstream routing parameters. The authors attempt to justify this with the "Representational De-coupling Approximation" and show that the empirical drift is small on their specific ViT-Tiny backbone ($\delta_{\text{drift}}^{(11)} = 0.12\%$). 
* **Limitation:** This approximation only holds when the router weights are heavily constrained (under severe regularization) and undergo very small updates. In deeper networks, or with more flexible routers, the cumulative Lipschitz constants of upstream blocks can scale exponentially, causing the representations to drift significantly. In such cases, the de-coupling assumption is violated, making the theoretical Rademacher bound loose or invalid.

### 2. Loss of Representation Scale via Unit-Sphere Normalization
The authors project the representations into a highly compressed $d$-dimensional space ($d=4$) and apply unit-sphere normalization to obtain the final input state:
$$\psi(x_i) = \frac{\tilde{\psi}(x_i)}{\|\tilde{\psi}(x_i)\|_2 + \epsilon}$$
* **The Flaw:** This projection discards all scale, magnitude, and norm information of the input representations.
* **Limitation:** In deep networks, the norm/magnitude of an activation vector carries critical information, such as prediction confidence, task saliency, or whether the input is out-of-distribution (OOD). Forcing the inputs onto a unit sphere to satisfy the Rademacher complexity bound's mathematical constraint ($\|\psi(x_i)\|_2 \le 1$) severely restricts the routing network's expressive capacity.

### 3. The "Dynamic Collapse" Paradox
The core motivation of the paper is to provide a robust **dynamic** model merging framework. 
* **The Flaw:** Under the proposed CFR penalty ($\lambda_{\text{wd}} = 10^{-2}$), the router weights are penalized so heavily that the weight-to-bias ratio drops to $\mathcal{M}_{\text{drift}} \approx 0.012$. The router effectively deactivates its input-dependent weights $w_{l, k}$ and routes based entirely on the learned biases $b_{l, k}$.
* **Methodological Contradiction:** This means the router is functionally static, which is why it achieves a 0.00% collapse impact under batch averaging (a static model is naturally unaffected by averaging across the batch dimension). Achieving "resilience" by destroying the dynamic capabilities of the router is a severe methodological compromise. It makes the entire feature-extraction and projection pipeline (PCA, Block 0 features, unit-sphere normalization) completely redundant.

### 4. Calibration Set Saliency & Covariance Noise
The calibration set is extremely small ($N=64$, with 16 samples per task).
* **The Flaw:** Estimating empirical covariance matrices from 16 samples per task introduces massive estimation noise.
* **Limitation:** The authors admit that at $N \le 32$, standard L2 decay outperforms CFR because the sampling variance of the empirical covariance matrix dominates. Even at $N = 64$, the covariance estimator is highly unstable and prone to overfitting to the specific local features of the tiny calibration split. 

### 5. Weak SVHN Expert Bottleneck
The fine-tuned SVHN expert has an individual test accuracy of only 64.60%. 
* **The Flaw:** This low accuracy limits the performance of all merged models on SVHN to around 25.00%.
* **Limitation:** While the authors state this does not affect the mathematical correctness of the algorithm, a weak task expert makes it difficult to evaluate the true efficacy of the routing network. If the expert itself is heavily degraded, the routing network may learn to default to other task experts, introducing bias and noise into the empirical results.

## Reproducibility
The authors provide details on the experimental setup, hyperparameters, datasets, and architectures (ViT-Tiny). The offline pre-computation of the $d \times d$ covariance matrices is detailed, which suggests that the results are reproducible. However, the theoretical guarantees and empirical results are highly localized to their specific ViT-Tiny backbone and small expert pool.
