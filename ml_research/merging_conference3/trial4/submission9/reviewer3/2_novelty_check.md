# 2. Novelty and Literature Check

## Key Novel Aspects
The paper presents **Exclusive Parameter Merging (EPM)** as a training-free coordinate-level routing operator that addresses spatial weight-space interference. The primary claimed novelties are:
1. **Soft Exclusive Parameter Allocation (Soft-EPA):** A method to soft-route each coordinate to its "dominant" task expert based on scaled magnitude, while keeping non-dominant tasks active at a lower level ($\gamma$).
2. **Task Vector Standardization:** Using standard deviation ($\sigma_k$) to normalize different experts' task vector magnitudes before coordinate-level routing, preventing tasks with larger gradients from dominating the model.
3. **Dynamic Coherence Scheduling (DCS):** Scaling $\gamma(p)$ dynamically as a quadratic function of target sparsity $p$.
4. **Task-Level Coefficient Tuning (TLC-Tune):** Optimizing a tiny $K$-dimensional global scale space using a zero-order (1+1) Evolution Strategy.

## Critical Analysis of Novelty and the 'Delta' from Prior Work
From a rigorous, adversarial perspective, the mathematical and conceptual "delta" from existing literature is highly incremental:

### 1. Soft Exclusive Parameter Allocation (Soft-EPA) vs. Existing Coordinate-wise Merging
* Coordinate-wise selection and pruning are well-established concepts in model merging. Methods like **TIES-Merging** trim coordinates based on absolute magnitude and resolve conflicts via sign voting, while **DARE** drops coordinates randomly and rescales. 
* EPM's routing rule (Equation 10) is a straightforward extension. Under the hood, Soft-EPA is simply a soft mask. The authors present an "elegant mathematical identity" in Equation 12, showing that Soft-EPA is a convex/linear combination of hard coordinate-wise exclusivity ($\gamma = 0$) and standard Task Arithmetic ($\gamma = 1$). 
* Rather than a major breakthrough, this "elegant identity" reveals that the method is simply a standard linear interpolation between two known methods: hard-assignment pruning and standard Task Arithmetic.

### 2. Task Vector Standardization vs. Traditional Normalization
* The use of standard deviation ($\sigma_k$) to normalize task vectors before comparison is a basic, standard statistical technique (Z-score normalization without mean centering, since means of task vectors are typically near zero). 
* Normalizing parameter updates to account for varying gradient magnitudes has been used widely in federated learning and gradient-based optimization (e.g., AdaGrad, RMSprop, Adam) and in model merging (e.g., Fisher-weighted averaging or standard magnitude-based normalization). 
* Applying this classic statistical trick to coordinate-level routing is an intuitive application, but lacks fundamental theoretical novelty.

### 3. Dynamic Coherence Scheduling (DCS)
* The dynamic scheduling rule $\gamma(p) = \gamma_0 + (1 - \gamma_0) \cdot p^2$ is a purely empirical polynomial heuristic. 
* There is no formal mathematical derivation or theoretical justification for why the schedule should be quadratic in $p$ other than the fact that it happens to work well in their specific ViT-Tiny grid search. Calling this "Dynamic Coherence Scheduling" overstates a simple polynomial hyperparameter sweep.

### 4. Task-Level Coefficient Tuning (TLC-Tune)
* Scaling task vectors by task-specific global factors ($\lambda_k$) and tuning them on a small validation split is standard practice. Task Arithmetic routinely sweeps and tunes the global scaling factor $\lambda$. 
* Moving from a 1-dimensional sweep ($\lambda$) to a $K$-dimensional search space ($\Lambda$) using a simple black-box (1+1) Evolution Strategy is a direct and standard generalization, not a fundamentally novel optimization paradigm.

## Characterization of Novelty
**Incremental.** 
The proposed framework is a combination of existing, standard concepts: coordinate-level thresholding (TIES-Merging), simple Z-score standard deviation normalization, global scale sweeping (Task Arithmetic), and an empirical quadratic scheduling heuristic. While the specific combination of these blocks is functional, it represents a highly incremental step in weight-space composition design rather than a significant conceptual departure from the state-of-the-art.
