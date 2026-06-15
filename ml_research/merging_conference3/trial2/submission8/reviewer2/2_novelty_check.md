# Assessment of Novelty and Theoretical Delta

## 1. Key Claims of Novelty in the Submission
The paper asserts novelty in several areas:
1. **Norm-Preserving Rescaling ($1/p$ factor):** A deterministic rescaling heuristic applied to magnitude-pruned task vectors to counteract update norm shrinkage and boost signal strength.
2. **Adaptive Saliency-Based Pruning (NP-BTVP-S):** A layer-wise budget allocation scheme that dynamically distributes the sparsification budget based on average $L_1$ update magnitudes across layers.
3. **The Saliency Double-Bind:** A conceptualization of the trade-offs in layer-wise budget allocation (inter-layer scale imbalance under global scaling vs. local variance blowup under layer-wise scaling).
4. **De-coupling of Landscape Flatness and Coordinate Sparsification:** The empirical insight that training-stage loss landscape flatness (via SAM) does not provide an additional coordinate-aligned pruning buffer compared to standard AdamW under well-converged regimes.

## 2. Theoretical Analysis of the "Delta" and Mathematical Equivalence
While the paper presents "Norm-Preserving Rescaling" as a novel post-hoc weight sparsification operator, a rigorous mathematical analysis reveals that this mechanism is **formally equivalent to a standard reparameterization of the merging coefficient $\lambda_k$**.

### Mathematical Equivalence Proof:
Let the task vector of expert $k$ be $\tau_k \in \mathbb{R}^d$. Let $M_k^{(p)} \in \{0, 1\}^d$ be the binary mask of magnitude pruning for retention rate $p$.
Under the proposed Uniform Pruning (NP-BTVP-U), the rescaled sparse task vector is defined as:
$$\tilde{\tau}_k^{(p)} = \tau_k \odot M_k^{(p)} \times \frac{1}{p}$$
The final merged model is reconstructed as:
$$\theta_{\text{MTL}}^{(p)} = \theta_{\text{base}} + \sum_{k=1}^K \lambda_k \tilde{\tau}_k^{(p)}$$
Substituting the definition of $\tilde{\tau}_k^{(p)}$ into the merging equation yields:
$$\theta_{\text{MTL}}^{(p)} = \theta_{\text{base}} + \sum_{k=1}^K \lambda_k \left( \tau_k \odot M_k^{(p)} \times \frac{1}{p} \right)$$
Due to the linearity of the scaling and summation operations, this is exactly:
$$\theta_{\text{MTL}}^{(p)} = \theta_{\text{base}} + \sum_{k=1}^K \left( \frac{\lambda_k}{p} \right) \left( \tau_k \odot M_k^{(p)} \right)$$
Let $\bar{\tau}_k^{(p)} = \tau_k \odot M_k^{(p)}$ represent the standard, unrescaled pruned task vector.
Let $\bar{\lambda}_k = \frac{\lambda_k}{p}$ represent a reparameterized merging coefficient.
The equation becomes:
$$\theta_{\text{MTL}}^{(p)} = \theta_{\text{base}} + \sum_{k=1}^K \bar{\lambda}_k \bar{\tau}_k^{(p)}$$

This proves that **the proposed norm-preserving rescaling is mathematically identical to performing unrescaled magnitude-based pruning and scaling up the merging coefficient $\lambda_k$ by a factor of $1/p$**. 

In model merging, the merging coefficients $\lambda_k$ are always swept and optimized post-hoc (e.g., using grid search or validation performance). 
The paper states in Section 4.3 that they "sweep and optimize the merging coefficient $\lambda \in [0.1, 1.0]$ with a step size of $0.1$". 
Under a retention rate of $p = 0.10$, the optimal unrescaled merging coefficient would be $\bar{\lambda}_k = \frac{\lambda_k}{0.10} = 10 \lambda_k$. Since $\lambda_k$ was optimized around $0.3$, the optimal $\bar{\lambda}_k$ should be approximately $3.0$.
However, because the authors restricted their hyperparameter search for the unrescaled baseline to the interval $[0.1, 1.0]$, the baseline was artificially prevented from reaching its optimal scale. The "severe performance collapse" of unrescaled pruning (80.45% accuracy) is therefore **not a fundamental physical phenomenon of update norm shrinkage, but rather a direct artifact of an restricted hyperparameter search space**. If the baseline sweep for $\bar{\lambda}_k$ had been extended to $[0.1, 10.0]$, the unrescaled baseline would have achieved the exact same performance as the rescaled method.

Thus, the theoretical novelty of the "Norm-Preserving Rescaling" is minimal; it is a change of variables in the hyperparameter sweep rather than a new mathematical operator.

## 3. Analysis of the Saliency Double-Bind
The "Saliency Double-Bind" describes why Adaptive Saliency-Based Pruning (NP-BTVP-S) fails to outperform Uniform Pruning:
- Under global scaling, low-saliency layers are silenced because their effective budget $p_l \ll p$ scale down their updates relative to $1/p$.
- Under layer-wise scaling, low-saliency layers suffer from variance/noise blowup because $1/p_l \gg 1/p$ scales up minor updates.

This is a valid and mathematically sound observation of the interaction between layer-wise budget allocation and reciprocal scaling. However, because the authors conclude that global Uniform Pruning (which avoids this double-bind) is the optimal, most stable, and preferred approach, the "saliency-based pruning" contribution is effectively neutralized. The paper proposes a complex layer-wise scheme only to show that a trivial uniform baseline with a scaled merging coefficient is superior.

## 4. Geometric Separation (SAM vs. AdamW)
The empirical finding that loss landscape flatness (SAM) does not provide a coordinate-aligned pruning buffer is intriguing. Theoretically, we can explain this as follows:
- SAM optimizes for isotropic flatness within a Euclidean ball, restricting the spectral norm (maximum eigenvalue) of the Hessian: $\lambda_{\max}(H)$.
- Coordinate-wise magnitude pruning is a highly non-isotropic, coordinate-aligned projection that zeroes out parameters where the update $|\tau_i|$ is small.
- The perturbation introduced by pruning is $\Delta \theta = -\tau_i$ for pruned coordinates. Because magnitude pruning specifically selects the smallest coordinates, the $L_2$ norm of this perturbation $\|\Delta \theta\|_2$ is extremely small, even under high sparsity (e.g., 90%).
- As a result, the Taylor expansion of the loss change $\Delta L \approx \frac{1}{2} \Delta \theta^T H \Delta \theta \le \frac{1}{2} \lambda_{\max}(H) \|\Delta \theta\|_2^2$ is extremely small for both SAM (small $\lambda_{\max}(H)$) and standard AdamW (where the perturbation $\|\Delta \theta\|_2$ is already very small).
- Thus, the loss is insensitive to this specific coordinate-aligned perturbation in both optimization regimes, explaining why standard AdamW and SAM show nearly identical resilience to magnitude pruning.

This theoretical analysis demystifies the empirical observation and shows that the lack of additional buffer from SAM is a natural consequence of the mathematical properties of magnitude pruning on well-converged overparameterized networks.

## 5. Summary of Novelty
The paper's novelty is characterized as **incremental**. 
- The "Norm-Preserving Rescaling" is mathematically equivalent to scaling the merging coefficient $\lambda_k$, which means the main performance boost over the unrescaled baseline is an artifact of the restricted search space of the baseline.
- The Adaptive Saliency-Based Pruning is shown to be inferior/equivalent to the uniform baseline.
- The empirical results on SAM vs. AdamW are interesting, but are easily explained by standard optimization theory and the geometric properties of magnitude pruning on overparameterized models.
- The practical synergy with INT8 quantization is a solid engineering contribution, but does not represent a significant theoretical breakthrough.
